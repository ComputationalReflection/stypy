
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.dist
2: 
3: Provides the Distribution class, which represents the module distribution
4: being built/installed/distributed.
5: '''
6: 
7: __revision__ = "$Id$"
8: 
9: import sys, os, re
10: from email import message_from_file
11: 
12: try:
13:     import warnings
14: except ImportError:
15:     warnings = None
16: 
17: from distutils.errors import (DistutilsOptionError, DistutilsArgError,
18:                               DistutilsModuleError, DistutilsClassError)
19: from distutils.fancy_getopt import FancyGetopt, translate_longopt
20: from distutils.util import check_environ, strtobool, rfc822_escape
21: from distutils import log
22: from distutils.debug import DEBUG
23: 
24: # Encoding used for the PKG-INFO files
25: PKG_INFO_ENCODING = 'utf-8'
26: 
27: # Regex to define acceptable Distutils command names.  This is not *quite*
28: # the same as a Python NAME -- I don't allow leading underscores.  The fact
29: # that they're very similar is no coincidence; the default naming scheme is
30: # to look for a Python module named after the command.
31: command_re = re.compile (r'^[a-zA-Z]([a-zA-Z0-9_]*)$')
32: 
33: 
34: class Distribution:
35:     '''The core of the Distutils.  Most of the work hiding behind 'setup'
36:     is really done within a Distribution instance, which farms the work out
37:     to the Distutils commands specified on the command line.
38: 
39:     Setup scripts will almost never instantiate Distribution directly,
40:     unless the 'setup()' function is totally inadequate to their needs.
41:     However, it is conceivable that a setup script might wish to subclass
42:     Distribution for some specialized purpose, and then pass the subclass
43:     to 'setup()' as the 'distclass' keyword argument.  If so, it is
44:     necessary to respect the expectations that 'setup' has of Distribution.
45:     See the code for 'setup()', in core.py, for details.
46:     '''
47: 
48: 
49:     # 'global_options' describes the command-line options that may be
50:     # supplied to the setup script prior to any actual commands.
51:     # Eg. "./setup.py -n" or "./setup.py --quiet" both take advantage of
52:     # these global options.  This list should be kept to a bare minimum,
53:     # since every global option is also valid as a command option -- and we
54:     # don't want to pollute the commands with too many options that they
55:     # have minimal control over.
56:     # The fourth entry for verbose means that it can be repeated.
57:     global_options = [('verbose', 'v', "run verbosely (default)", 1),
58:                       ('quiet', 'q', "run quietly (turns verbosity off)"),
59:                       ('dry-run', 'n', "don't actually do anything"),
60:                       ('help', 'h', "show detailed help message"),
61:                       ('no-user-cfg', None,
62:                        'ignore pydistutils.cfg in your home directory'),
63:     ]
64: 
65:     # 'common_usage' is a short (2-3 line) string describing the common
66:     # usage of the setup script.
67:     common_usage = '''\
68: Common commands: (see '--help-commands' for more)
69: 
70:   setup.py build      will build the package underneath 'build/'
71:   setup.py install    will install the package
72: '''
73: 
74:     # options that are not propagated to the commands
75:     display_options = [
76:         ('help-commands', None,
77:          "list all available commands"),
78:         ('name', None,
79:          "print package name"),
80:         ('version', 'V',
81:          "print package version"),
82:         ('fullname', None,
83:          "print <package name>-<version>"),
84:         ('author', None,
85:          "print the author's name"),
86:         ('author-email', None,
87:          "print the author's email address"),
88:         ('maintainer', None,
89:          "print the maintainer's name"),
90:         ('maintainer-email', None,
91:          "print the maintainer's email address"),
92:         ('contact', None,
93:          "print the maintainer's name if known, else the author's"),
94:         ('contact-email', None,
95:          "print the maintainer's email address if known, else the author's"),
96:         ('url', None,
97:          "print the URL for this package"),
98:         ('license', None,
99:          "print the license of the package"),
100:         ('licence', None,
101:          "alias for --license"),
102:         ('description', None,
103:          "print the package description"),
104:         ('long-description', None,
105:          "print the long package description"),
106:         ('platforms', None,
107:          "print the list of platforms"),
108:         ('classifiers', None,
109:          "print the list of classifiers"),
110:         ('keywords', None,
111:          "print the list of keywords"),
112:         ('provides', None,
113:          "print the list of packages/modules provided"),
114:         ('requires', None,
115:          "print the list of packages/modules required"),
116:         ('obsoletes', None,
117:          "print the list of packages/modules made obsolete")
118:         ]
119:     display_option_names = map(lambda x: translate_longopt(x[0]),
120:                                display_options)
121: 
122:     # negative options are options that exclude other options
123:     negative_opt = {'quiet': 'verbose'}
124: 
125: 
126:     # -- Creation/initialization methods -------------------------------
127: 
128:     def __init__ (self, attrs=None):
129:         '''Construct a new Distribution instance: initialize all the
130:         attributes of a Distribution, and then use 'attrs' (a dictionary
131:         mapping attribute names to values) to assign some of those
132:         attributes their "real" values.  (Any attributes not mentioned in
133:         'attrs' will be assigned to some null value: 0, None, an empty list
134:         or dictionary, etc.)  Most importantly, initialize the
135:         'command_obj' attribute to the empty dictionary; this will be
136:         filled in with real command objects by 'parse_command_line()'.
137:         '''
138: 
139:         # Default values for our command-line options
140:         self.verbose = 1
141:         self.dry_run = 0
142:         self.help = 0
143:         for attr in self.display_option_names:
144:             setattr(self, attr, 0)
145: 
146:         # Store the distribution meta-data (name, version, author, and so
147:         # forth) in a separate object -- we're getting to have enough
148:         # information here (and enough command-line options) that it's
149:         # worth it.  Also delegate 'get_XXX()' methods to the 'metadata'
150:         # object in a sneaky and underhanded (but efficient!) way.
151:         self.metadata = DistributionMetadata()
152:         for basename in self.metadata._METHOD_BASENAMES:
153:             method_name = "get_" + basename
154:             setattr(self, method_name, getattr(self.metadata, method_name))
155: 
156:         # 'cmdclass' maps command names to class objects, so we
157:         # can 1) quickly figure out which class to instantiate when
158:         # we need to create a new command object, and 2) have a way
159:         # for the setup script to override command classes
160:         self.cmdclass = {}
161: 
162:         # 'command_packages' is a list of packages in which commands
163:         # are searched for.  The factory for command 'foo' is expected
164:         # to be named 'foo' in the module 'foo' in one of the packages
165:         # named here.  This list is searched from the left; an error
166:         # is raised if no named package provides the command being
167:         # searched for.  (Always access using get_command_packages().)
168:         self.command_packages = None
169: 
170:         # 'script_name' and 'script_args' are usually set to sys.argv[0]
171:         # and sys.argv[1:], but they can be overridden when the caller is
172:         # not necessarily a setup script run from the command-line.
173:         self.script_name = None
174:         self.script_args = None
175: 
176:         # 'command_options' is where we store command options between
177:         # parsing them (from config files, the command-line, etc.) and when
178:         # they are actually needed -- ie. when the command in question is
179:         # instantiated.  It is a dictionary of dictionaries of 2-tuples:
180:         #   command_options = { command_name : { option : (source, value) } }
181:         self.command_options = {}
182: 
183:         # 'dist_files' is the list of (command, pyversion, file) that
184:         # have been created by any dist commands run so far. This is
185:         # filled regardless of whether the run is dry or not. pyversion
186:         # gives sysconfig.get_python_version() if the dist file is
187:         # specific to a Python version, 'any' if it is good for all
188:         # Python versions on the target platform, and '' for a source
189:         # file. pyversion should not be used to specify minimum or
190:         # maximum required Python versions; use the metainfo for that
191:         # instead.
192:         self.dist_files = []
193: 
194:         # These options are really the business of various commands, rather
195:         # than of the Distribution itself.  We provide aliases for them in
196:         # Distribution as a convenience to the developer.
197:         self.packages = None
198:         self.package_data = {}
199:         self.package_dir = None
200:         self.py_modules = None
201:         self.libraries = None
202:         self.headers = None
203:         self.ext_modules = None
204:         self.ext_package = None
205:         self.include_dirs = None
206:         self.extra_path = None
207:         self.scripts = None
208:         self.data_files = None
209:         self.password = ''
210: 
211:         # And now initialize bookkeeping stuff that can't be supplied by
212:         # the caller at all.  'command_obj' maps command names to
213:         # Command instances -- that's how we enforce that every command
214:         # class is a singleton.
215:         self.command_obj = {}
216: 
217:         # 'have_run' maps command names to boolean values; it keeps track
218:         # of whether we have actually run a particular command, to make it
219:         # cheap to "run" a command whenever we think we might need to -- if
220:         # it's already been done, no need for expensive filesystem
221:         # operations, we just check the 'have_run' dictionary and carry on.
222:         # It's only safe to query 'have_run' for a command class that has
223:         # been instantiated -- a false value will be inserted when the
224:         # command object is created, and replaced with a true value when
225:         # the command is successfully run.  Thus it's probably best to use
226:         # '.get()' rather than a straight lookup.
227:         self.have_run = {}
228: 
229:         # Now we'll use the attrs dictionary (ultimately, keyword args from
230:         # the setup script) to possibly override any or all of these
231:         # distribution options.
232: 
233:         if attrs:
234:             # Pull out the set of command options and work on them
235:             # specifically.  Note that this order guarantees that aliased
236:             # command options will override any supplied redundantly
237:             # through the general options dictionary.
238:             options = attrs.get('options')
239:             if options is not None:
240:                 del attrs['options']
241:                 for (command, cmd_options) in options.items():
242:                     opt_dict = self.get_option_dict(command)
243:                     for (opt, val) in cmd_options.items():
244:                         opt_dict[opt] = ("setup script", val)
245: 
246:             if 'licence' in attrs:
247:                 attrs['license'] = attrs['licence']
248:                 del attrs['licence']
249:                 msg = "'licence' distribution option is deprecated; use 'license'"
250:                 if warnings is not None:
251:                     warnings.warn(msg)
252:                 else:
253:                     sys.stderr.write(msg + "\n")
254: 
255:             # Now work on the rest of the attributes.  Any attribute that's
256:             # not already defined is invalid!
257:             for (key, val) in attrs.items():
258:                 if hasattr(self.metadata, "set_" + key):
259:                     getattr(self.metadata, "set_" + key)(val)
260:                 elif hasattr(self.metadata, key):
261:                     setattr(self.metadata, key, val)
262:                 elif hasattr(self, key):
263:                     setattr(self, key, val)
264:                 else:
265:                     msg = "Unknown distribution option: %s" % repr(key)
266:                     if warnings is not None:
267:                         warnings.warn(msg)
268:                     else:
269:                         sys.stderr.write(msg + "\n")
270: 
271:         # no-user-cfg is handled before other command line args
272:         # because other args override the config files, and this
273:         # one is needed before we can load the config files.
274:         # If attrs['script_args'] wasn't passed, assume false.
275:         #
276:         # This also make sure we just look at the global options
277:         self.want_user_cfg = True
278: 
279:         if self.script_args is not None:
280:             for arg in self.script_args:
281:                 if not arg.startswith('-'):
282:                     break
283:                 if arg == '--no-user-cfg':
284:                     self.want_user_cfg = False
285:                     break
286: 
287:         self.finalize_options()
288: 
289:     def get_option_dict(self, command):
290:         '''Get the option dictionary for a given command.  If that
291:         command's option dictionary hasn't been created yet, then create it
292:         and return the new dictionary; otherwise, return the existing
293:         option dictionary.
294:         '''
295:         dict = self.command_options.get(command)
296:         if dict is None:
297:             dict = self.command_options[command] = {}
298:         return dict
299: 
300:     def dump_option_dicts(self, header=None, commands=None, indent=""):
301:         from pprint import pformat
302: 
303:         if commands is None:             # dump all command option dicts
304:             commands = self.command_options.keys()
305:             commands.sort()
306: 
307:         if header is not None:
308:             self.announce(indent + header)
309:             indent = indent + "  "
310: 
311:         if not commands:
312:             self.announce(indent + "no commands known yet")
313:             return
314: 
315:         for cmd_name in commands:
316:             opt_dict = self.command_options.get(cmd_name)
317:             if opt_dict is None:
318:                 self.announce(indent +
319:                               "no option dict for '%s' command" % cmd_name)
320:             else:
321:                 self.announce(indent +
322:                               "option dict for '%s' command:" % cmd_name)
323:                 out = pformat(opt_dict)
324:                 for line in out.split('\n'):
325:                     self.announce(indent + "  " + line)
326: 
327:     # -- Config file finding/parsing methods ---------------------------
328: 
329:     def find_config_files(self):
330:         '''Find as many configuration files as should be processed for this
331:         platform, and return a list of filenames in the order in which they
332:         should be parsed.  The filenames returned are guaranteed to exist
333:         (modulo nasty race conditions).
334: 
335:         There are three possible config files: distutils.cfg in the
336:         Distutils installation directory (ie. where the top-level
337:         Distutils __inst__.py file lives), a file in the user's home
338:         directory named .pydistutils.cfg on Unix and pydistutils.cfg
339:         on Windows/Mac; and setup.cfg in the current directory.
340: 
341:         The file in the user's home directory can be disabled with the
342:         --no-user-cfg option.
343:         '''
344:         files = []
345:         check_environ()
346: 
347:         # Where to look for the system-wide Distutils config file
348:         sys_dir = os.path.dirname(sys.modules['distutils'].__file__)
349: 
350:         # Look for the system config file
351:         sys_file = os.path.join(sys_dir, "distutils.cfg")
352:         if os.path.isfile(sys_file):
353:             files.append(sys_file)
354: 
355:         # What to call the per-user config file
356:         if os.name == 'posix':
357:             user_filename = ".pydistutils.cfg"
358:         else:
359:             user_filename = "pydistutils.cfg"
360: 
361:         # And look for the user config file
362:         if self.want_user_cfg:
363:             user_file = os.path.join(os.path.expanduser('~'), user_filename)
364:             if os.path.isfile(user_file):
365:                 files.append(user_file)
366: 
367:         # All platforms support local setup.cfg
368:         local_file = "setup.cfg"
369:         if os.path.isfile(local_file):
370:             files.append(local_file)
371: 
372:         if DEBUG:
373:             self.announce("using config files: %s" % ', '.join(files))
374: 
375:         return files
376: 
377:     def parse_config_files(self, filenames=None):
378:         from ConfigParser import ConfigParser
379: 
380:         if filenames is None:
381:             filenames = self.find_config_files()
382: 
383:         if DEBUG:
384:             self.announce("Distribution.parse_config_files():")
385: 
386:         parser = ConfigParser()
387:         for filename in filenames:
388:             if DEBUG:
389:                 self.announce("  reading %s" % filename)
390:             parser.read(filename)
391:             for section in parser.sections():
392:                 options = parser.options(section)
393:                 opt_dict = self.get_option_dict(section)
394: 
395:                 for opt in options:
396:                     if opt != '__name__':
397:                         val = parser.get(section,opt)
398:                         opt = opt.replace('-', '_')
399:                         opt_dict[opt] = (filename, val)
400: 
401:             # Make the ConfigParser forget everything (so we retain
402:             # the original filenames that options come from)
403:             parser.__init__()
404: 
405:         # If there was a "global" section in the config file, use it
406:         # to set Distribution options.
407: 
408:         if 'global' in self.command_options:
409:             for (opt, (src, val)) in self.command_options['global'].items():
410:                 alias = self.negative_opt.get(opt)
411:                 try:
412:                     if alias:
413:                         setattr(self, alias, not strtobool(val))
414:                     elif opt in ('verbose', 'dry_run'): # ugh!
415:                         setattr(self, opt, strtobool(val))
416:                     else:
417:                         setattr(self, opt, val)
418:                 except ValueError, msg:
419:                     raise DistutilsOptionError, msg
420: 
421:     # -- Command-line parsing methods ----------------------------------
422: 
423:     def parse_command_line(self):
424:         '''Parse the setup script's command line, taken from the
425:         'script_args' instance attribute (which defaults to 'sys.argv[1:]'
426:         -- see 'setup()' in core.py).  This list is first processed for
427:         "global options" -- options that set attributes of the Distribution
428:         instance.  Then, it is alternately scanned for Distutils commands
429:         and options for that command.  Each new command terminates the
430:         options for the previous command.  The allowed options for a
431:         command are determined by the 'user_options' attribute of the
432:         command class -- thus, we have to be able to load command classes
433:         in order to parse the command line.  Any error in that 'options'
434:         attribute raises DistutilsGetoptError; any error on the
435:         command-line raises DistutilsArgError.  If no Distutils commands
436:         were found on the command line, raises DistutilsArgError.  Return
437:         true if command-line was successfully parsed and we should carry
438:         on with executing commands; false if no errors but we shouldn't
439:         execute commands (currently, this only happens if user asks for
440:         help).
441:         '''
442:         #
443:         # We now have enough information to show the Macintosh dialog
444:         # that allows the user to interactively specify the "command line".
445:         #
446:         toplevel_options = self._get_toplevel_options()
447: 
448:         # We have to parse the command line a bit at a time -- global
449:         # options, then the first command, then its options, and so on --
450:         # because each command will be handled by a different class, and
451:         # the options that are valid for a particular class aren't known
452:         # until we have loaded the command class, which doesn't happen
453:         # until we know what the command is.
454: 
455:         self.commands = []
456:         parser = FancyGetopt(toplevel_options + self.display_options)
457:         parser.set_negative_aliases(self.negative_opt)
458:         parser.set_aliases({'licence': 'license'})
459:         args = parser.getopt(args=self.script_args, object=self)
460:         option_order = parser.get_option_order()
461:         log.set_verbosity(self.verbose)
462: 
463:         # for display options we return immediately
464:         if self.handle_display_options(option_order):
465:             return
466:         while args:
467:             args = self._parse_command_opts(parser, args)
468:             if args is None:            # user asked for help (and got it)
469:                 return
470: 
471:         # Handle the cases of --help as a "global" option, ie.
472:         # "setup.py --help" and "setup.py --help command ...".  For the
473:         # former, we show global options (--verbose, --dry-run, etc.)
474:         # and display-only options (--name, --version, etc.); for the
475:         # latter, we omit the display-only options and show help for
476:         # each command listed on the command line.
477:         if self.help:
478:             self._show_help(parser,
479:                             display_options=len(self.commands) == 0,
480:                             commands=self.commands)
481:             return
482: 
483:         # Oops, no commands found -- an end-user error
484:         if not self.commands:
485:             raise DistutilsArgError, "no commands supplied"
486: 
487:         # All is well: return true
488:         return 1
489: 
490:     def _get_toplevel_options(self):
491:         '''Return the non-display options recognized at the top level.
492: 
493:         This includes options that are recognized *only* at the top
494:         level as well as options recognized for commands.
495:         '''
496:         return self.global_options + [
497:             ("command-packages=", None,
498:              "list of packages that provide distutils commands"),
499:             ]
500: 
501:     def _parse_command_opts(self, parser, args):
502:         '''Parse the command-line options for a single command.
503:         'parser' must be a FancyGetopt instance; 'args' must be the list
504:         of arguments, starting with the current command (whose options
505:         we are about to parse).  Returns a new version of 'args' with
506:         the next command at the front of the list; will be the empty
507:         list if there are no more commands on the command line.  Returns
508:         None if the user asked for help on this command.
509:         '''
510:         # late import because of mutual dependence between these modules
511:         from distutils.cmd import Command
512: 
513:         # Pull the current command from the head of the command line
514:         command = args[0]
515:         if not command_re.match(command):
516:             raise SystemExit, "invalid command name '%s'" % command
517:         self.commands.append(command)
518: 
519:         # Dig up the command class that implements this command, so we
520:         # 1) know that it's a valid command, and 2) know which options
521:         # it takes.
522:         try:
523:             cmd_class = self.get_command_class(command)
524:         except DistutilsModuleError, msg:
525:             raise DistutilsArgError, msg
526: 
527:         # Require that the command class be derived from Command -- want
528:         # to be sure that the basic "command" interface is implemented.
529:         if not issubclass(cmd_class, Command):
530:             raise DistutilsClassError, \
531:                   "command class %s must subclass Command" % cmd_class
532: 
533:         # Also make sure that the command object provides a list of its
534:         # known options.
535:         if not (hasattr(cmd_class, 'user_options') and
536:                 isinstance(cmd_class.user_options, list)):
537:             raise DistutilsClassError, \
538:                   ("command class %s must provide " +
539:                    "'user_options' attribute (a list of tuples)") % \
540:                   cmd_class
541: 
542:         # If the command class has a list of negative alias options,
543:         # merge it in with the global negative aliases.
544:         negative_opt = self.negative_opt
545:         if hasattr(cmd_class, 'negative_opt'):
546:             negative_opt = negative_opt.copy()
547:             negative_opt.update(cmd_class.negative_opt)
548: 
549:         # Check for help_options in command class.  They have a different
550:         # format (tuple of four) so we need to preprocess them here.
551:         if (hasattr(cmd_class, 'help_options') and
552:             isinstance(cmd_class.help_options, list)):
553:             help_options = fix_help_options(cmd_class.help_options)
554:         else:
555:             help_options = []
556: 
557: 
558:         # All commands support the global options too, just by adding
559:         # in 'global_options'.
560:         parser.set_option_table(self.global_options +
561:                                 cmd_class.user_options +
562:                                 help_options)
563:         parser.set_negative_aliases(negative_opt)
564:         (args, opts) = parser.getopt(args[1:])
565:         if hasattr(opts, 'help') and opts.help:
566:             self._show_help(parser, display_options=0, commands=[cmd_class])
567:             return
568: 
569:         if (hasattr(cmd_class, 'help_options') and
570:             isinstance(cmd_class.help_options, list)):
571:             help_option_found=0
572:             for (help_option, short, desc, func) in cmd_class.help_options:
573:                 if hasattr(opts, parser.get_attr_name(help_option)):
574:                     help_option_found=1
575:                     if hasattr(func, '__call__'):
576:                         func()
577:                     else:
578:                         raise DistutilsClassError(
579:                             "invalid help function %r for help option '%s': "
580:                             "must be a callable object (function, etc.)"
581:                             % (func, help_option))
582: 
583:             if help_option_found:
584:                 return
585: 
586:         # Put the options from the command-line into their official
587:         # holding pen, the 'command_options' dictionary.
588:         opt_dict = self.get_option_dict(command)
589:         for (name, value) in vars(opts).items():
590:             opt_dict[name] = ("command line", value)
591: 
592:         return args
593: 
594:     def finalize_options(self):
595:         '''Set final values for all the options on the Distribution
596:         instance, analogous to the .finalize_options() method of Command
597:         objects.
598:         '''
599:         for attr in ('keywords', 'platforms'):
600:             value = getattr(self.metadata, attr)
601:             if value is None:
602:                 continue
603:             if isinstance(value, str):
604:                 value = [elm.strip() for elm in value.split(',')]
605:                 setattr(self.metadata, attr, value)
606: 
607:     def _show_help(self, parser, global_options=1, display_options=1,
608:                    commands=[]):
609:         '''Show help for the setup script command-line in the form of
610:         several lists of command-line options.  'parser' should be a
611:         FancyGetopt instance; do not expect it to be returned in the
612:         same state, as its option table will be reset to make it
613:         generate the correct help text.
614: 
615:         If 'global_options' is true, lists the global options:
616:         --verbose, --dry-run, etc.  If 'display_options' is true, lists
617:         the "display-only" options: --name, --version, etc.  Finally,
618:         lists per-command help for every command name or command class
619:         in 'commands'.
620:         '''
621:         # late import because of mutual dependence between these modules
622:         from distutils.core import gen_usage
623:         from distutils.cmd import Command
624: 
625:         if global_options:
626:             if display_options:
627:                 options = self._get_toplevel_options()
628:             else:
629:                 options = self.global_options
630:             parser.set_option_table(options)
631:             parser.print_help(self.common_usage + "\nGlobal options:")
632:             print('')
633: 
634:         if display_options:
635:             parser.set_option_table(self.display_options)
636:             parser.print_help(
637:                 "Information display options (just display " +
638:                 "information, ignore any commands)")
639:             print('')
640: 
641:         for command in self.commands:
642:             if isinstance(command, type) and issubclass(command, Command):
643:                 klass = command
644:             else:
645:                 klass = self.get_command_class(command)
646:             if (hasattr(klass, 'help_options') and
647:                 isinstance(klass.help_options, list)):
648:                 parser.set_option_table(klass.user_options +
649:                                         fix_help_options(klass.help_options))
650:             else:
651:                 parser.set_option_table(klass.user_options)
652:             parser.print_help("Options for '%s' command:" % klass.__name__)
653:             print('')
654: 
655:         print(gen_usage(self.script_name))
656: 
657:     def handle_display_options(self, option_order):
658:         '''If there were any non-global "display-only" options
659:         (--help-commands or the metadata display options) on the command
660:         line, display the requested info and return true; else return
661:         false.
662:         '''
663:         from distutils.core import gen_usage
664: 
665:         # User just wants a list of commands -- we'll print it out and stop
666:         # processing now (ie. if they ran "setup --help-commands foo bar",
667:         # we ignore "foo bar").
668:         if self.help_commands:
669:             self.print_commands()
670:             print('')
671:             print(gen_usage(self.script_name))
672:             return 1
673: 
674:         # If user supplied any of the "display metadata" options, then
675:         # display that metadata in the order in which the user supplied the
676:         # metadata options.
677:         any_display_options = 0
678:         is_display_option = {}
679:         for option in self.display_options:
680:             is_display_option[option[0]] = 1
681: 
682:         for (opt, val) in option_order:
683:             if val and is_display_option.get(opt):
684:                 opt = translate_longopt(opt)
685:                 value = getattr(self.metadata, "get_"+opt)()
686:                 if opt in ['keywords', 'platforms']:
687:                     print(','.join(value))
688:                 elif opt in ('classifiers', 'provides', 'requires',
689:                              'obsoletes'):
690:                     print('\n'.join(value))
691:                 else:
692:                     print(value)
693:                 any_display_options = 1
694: 
695:         return any_display_options
696: 
697:     def print_command_list(self, commands, header, max_length):
698:         '''Print a subset of the list of all commands -- used by
699:         'print_commands()'.
700:         '''
701:         print(header + ":")
702: 
703:         for cmd in commands:
704:             klass = self.cmdclass.get(cmd)
705:             if not klass:
706:                 klass = self.get_command_class(cmd)
707:             try:
708:                 description = klass.description
709:             except AttributeError:
710:                 description = "(no description available)"
711: 
712:             print("  %-*s  %s" % (max_length, cmd, description))
713: 
714:     def print_commands(self):
715:         '''Print out a help message listing all available commands with a
716:         description of each.  The list is divided into "standard commands"
717:         (listed in distutils.command.__all__) and "extra commands"
718:         (mentioned in self.cmdclass, but not a standard command).  The
719:         descriptions come from the command class attribute
720:         'description'.
721:         '''
722:         import distutils.command
723:         std_commands = distutils.command.__all__
724:         is_std = {}
725:         for cmd in std_commands:
726:             is_std[cmd] = 1
727: 
728:         extra_commands = []
729:         for cmd in self.cmdclass.keys():
730:             if not is_std.get(cmd):
731:                 extra_commands.append(cmd)
732: 
733:         max_length = 0
734:         for cmd in (std_commands + extra_commands):
735:             if len(cmd) > max_length:
736:                 max_length = len(cmd)
737: 
738:         self.print_command_list(std_commands,
739:                                 "Standard commands",
740:                                 max_length)
741:         if extra_commands:
742:             print
743:             self.print_command_list(extra_commands,
744:                                     "Extra commands",
745:                                     max_length)
746: 
747:     def get_command_list(self):
748:         '''Get a list of (command, description) tuples.
749:         The list is divided into "standard commands" (listed in
750:         distutils.command.__all__) and "extra commands" (mentioned in
751:         self.cmdclass, but not a standard command).  The descriptions come
752:         from the command class attribute 'description'.
753:         '''
754:         # Currently this is only used on Mac OS, for the Mac-only GUI
755:         # Distutils interface (by Jack Jansen)
756: 
757:         import distutils.command
758:         std_commands = distutils.command.__all__
759:         is_std = {}
760:         for cmd in std_commands:
761:             is_std[cmd] = 1
762: 
763:         extra_commands = []
764:         for cmd in self.cmdclass.keys():
765:             if not is_std.get(cmd):
766:                 extra_commands.append(cmd)
767: 
768:         rv = []
769:         for cmd in (std_commands + extra_commands):
770:             klass = self.cmdclass.get(cmd)
771:             if not klass:
772:                 klass = self.get_command_class(cmd)
773:             try:
774:                 description = klass.description
775:             except AttributeError:
776:                 description = "(no description available)"
777:             rv.append((cmd, description))
778:         return rv
779: 
780:     # -- Command class/object methods ----------------------------------
781: 
782:     def get_command_packages(self):
783:         '''Return a list of packages from which commands are loaded.'''
784:         pkgs = self.command_packages
785:         if not isinstance(pkgs, list):
786:             if pkgs is None:
787:                 pkgs = ''
788:             pkgs = [pkg.strip() for pkg in pkgs.split(',') if pkg != '']
789:             if "distutils.command" not in pkgs:
790:                 pkgs.insert(0, "distutils.command")
791:             self.command_packages = pkgs
792:         return pkgs
793: 
794:     def get_command_class(self, command):
795:         '''Return the class that implements the Distutils command named by
796:         'command'.  First we check the 'cmdclass' dictionary; if the
797:         command is mentioned there, we fetch the class object from the
798:         dictionary and return it.  Otherwise we load the command module
799:         ("distutils.command." + command) and fetch the command class from
800:         the module.  The loaded class is also stored in 'cmdclass'
801:         to speed future calls to 'get_command_class()'.
802: 
803:         Raises DistutilsModuleError if the expected module could not be
804:         found, or if that module does not define the expected class.
805:         '''
806:         klass = self.cmdclass.get(command)
807:         if klass:
808:             return klass
809: 
810:         for pkgname in self.get_command_packages():
811:             module_name = "%s.%s" % (pkgname, command)
812:             klass_name = command
813: 
814:             try:
815:                 __import__ (module_name)
816:                 module = sys.modules[module_name]
817:             except ImportError:
818:                 continue
819: 
820:             try:
821:                 klass = getattr(module, klass_name)
822:             except AttributeError:
823:                 raise DistutilsModuleError, \
824:                       "invalid command '%s' (no class '%s' in module '%s')" \
825:                       % (command, klass_name, module_name)
826: 
827:             self.cmdclass[command] = klass
828:             return klass
829: 
830:         raise DistutilsModuleError("invalid command '%s'" % command)
831: 
832: 
833:     def get_command_obj(self, command, create=1):
834:         '''Return the command object for 'command'.  Normally this object
835:         is cached on a previous call to 'get_command_obj()'; if no command
836:         object for 'command' is in the cache, then we either create and
837:         return it (if 'create' is true) or return None.
838:         '''
839:         cmd_obj = self.command_obj.get(command)
840:         if not cmd_obj and create:
841:             if DEBUG:
842:                 self.announce("Distribution.get_command_obj(): " \
843:                               "creating '%s' command object" % command)
844: 
845:             klass = self.get_command_class(command)
846:             cmd_obj = self.command_obj[command] = klass(self)
847:             self.have_run[command] = 0
848: 
849:             # Set any options that were supplied in config files
850:             # or on the command line.  (NB. support for error
851:             # reporting is lame here: any errors aren't reported
852:             # until 'finalize_options()' is called, which means
853:             # we won't report the source of the error.)
854:             options = self.command_options.get(command)
855:             if options:
856:                 self._set_command_options(cmd_obj, options)
857: 
858:         return cmd_obj
859: 
860:     def _set_command_options(self, command_obj, option_dict=None):
861:         '''Set the options for 'command_obj' from 'option_dict'.  Basically
862:         this means copying elements of a dictionary ('option_dict') to
863:         attributes of an instance ('command').
864: 
865:         'command_obj' must be a Command instance.  If 'option_dict' is not
866:         supplied, uses the standard option dictionary for this command
867:         (from 'self.command_options').
868:         '''
869:         command_name = command_obj.get_command_name()
870:         if option_dict is None:
871:             option_dict = self.get_option_dict(command_name)
872: 
873:         if DEBUG:
874:             self.announce("  setting options for '%s' command:" % command_name)
875:         for (option, (source, value)) in option_dict.items():
876:             if DEBUG:
877:                 self.announce("    %s = %s (from %s)" % (option, value,
878:                                                          source))
879:             try:
880:                 bool_opts = map(translate_longopt, command_obj.boolean_options)
881:             except AttributeError:
882:                 bool_opts = []
883:             try:
884:                 neg_opt = command_obj.negative_opt
885:             except AttributeError:
886:                 neg_opt = {}
887: 
888:             try:
889:                 is_string = isinstance(value, str)
890:                 if option in neg_opt and is_string:
891:                     setattr(command_obj, neg_opt[option], not strtobool(value))
892:                 elif option in bool_opts and is_string:
893:                     setattr(command_obj, option, strtobool(value))
894:                 elif hasattr(command_obj, option):
895:                     setattr(command_obj, option, value)
896:                 else:
897:                     raise DistutilsOptionError, \
898:                           ("error in %s: command '%s' has no such option '%s'"
899:                            % (source, command_name, option))
900:             except ValueError, msg:
901:                 raise DistutilsOptionError, msg
902: 
903:     def reinitialize_command(self, command, reinit_subcommands=0):
904:         '''Reinitializes a command to the state it was in when first
905:         returned by 'get_command_obj()': ie., initialized but not yet
906:         finalized.  This provides the opportunity to sneak option
907:         values in programmatically, overriding or supplementing
908:         user-supplied values from the config files and command line.
909:         You'll have to re-finalize the command object (by calling
910:         'finalize_options()' or 'ensure_finalized()') before using it for
911:         real.
912: 
913:         'command' should be a command name (string) or command object.  If
914:         'reinit_subcommands' is true, also reinitializes the command's
915:         sub-commands, as declared by the 'sub_commands' class attribute (if
916:         it has one).  See the "install" command for an example.  Only
917:         reinitializes the sub-commands that actually matter, ie. those
918:         whose test predicates return true.
919: 
920:         Returns the reinitialized command object.
921:         '''
922:         from distutils.cmd import Command
923:         if not isinstance(command, Command):
924:             command_name = command
925:             command = self.get_command_obj(command_name)
926:         else:
927:             command_name = command.get_command_name()
928: 
929:         if not command.finalized:
930:             return command
931:         command.initialize_options()
932:         command.finalized = 0
933:         self.have_run[command_name] = 0
934:         self._set_command_options(command)
935: 
936:         if reinit_subcommands:
937:             for sub in command.get_sub_commands():
938:                 self.reinitialize_command(sub, reinit_subcommands)
939: 
940:         return command
941: 
942:     # -- Methods that operate on the Distribution ----------------------
943: 
944:     def announce(self, msg, level=log.INFO):
945:         log.log(level, msg)
946: 
947:     def run_commands(self):
948:         '''Run each command that was seen on the setup script command line.
949:         Uses the list of commands found and cache of command objects
950:         created by 'get_command_obj()'.
951:         '''
952:         for cmd in self.commands:
953:             self.run_command(cmd)
954: 
955:     # -- Methods that operate on its Commands --------------------------
956: 
957:     def run_command(self, command):
958:         '''Do whatever it takes to run a command (including nothing at all,
959:         if the command has already been run).  Specifically: if we have
960:         already created and run the command named by 'command', return
961:         silently without doing anything.  If the command named by 'command'
962:         doesn't even have a command object yet, create one.  Then invoke
963:         'run()' on that command object (or an existing one).
964:         '''
965:         # Already been here, done that? then return silently.
966:         if self.have_run.get(command):
967:             return
968: 
969:         log.info("running %s", command)
970:         cmd_obj = self.get_command_obj(command)
971:         cmd_obj.ensure_finalized()
972:         cmd_obj.run()
973:         self.have_run[command] = 1
974: 
975: 
976:     # -- Distribution query methods ------------------------------------
977: 
978:     def has_pure_modules(self):
979:         return len(self.packages or self.py_modules or []) > 0
980: 
981:     def has_ext_modules(self):
982:         return self.ext_modules and len(self.ext_modules) > 0
983: 
984:     def has_c_libraries(self):
985:         return self.libraries and len(self.libraries) > 0
986: 
987:     def has_modules(self):
988:         return self.has_pure_modules() or self.has_ext_modules()
989: 
990:     def has_headers(self):
991:         return self.headers and len(self.headers) > 0
992: 
993:     def has_scripts(self):
994:         return self.scripts and len(self.scripts) > 0
995: 
996:     def has_data_files(self):
997:         return self.data_files and len(self.data_files) > 0
998: 
999:     def is_pure(self):
1000:         return (self.has_pure_modules() and
1001:                 not self.has_ext_modules() and
1002:                 not self.has_c_libraries())
1003: 
1004:     # -- Metadata query methods ----------------------------------------
1005: 
1006:     # If you're looking for 'get_name()', 'get_version()', and so forth,
1007:     # they are defined in a sneaky way: the constructor binds self.get_XXX
1008:     # to self.metadata.get_XXX.  The actual code is in the
1009:     # DistributionMetadata class, below.
1010: 
1011: class DistributionMetadata:
1012:     '''Dummy class to hold the distribution meta-data: name, version,
1013:     author, and so forth.
1014:     '''
1015: 
1016:     _METHOD_BASENAMES = ("name", "version", "author", "author_email",
1017:                          "maintainer", "maintainer_email", "url",
1018:                          "license", "description", "long_description",
1019:                          "keywords", "platforms", "fullname", "contact",
1020:                          "contact_email", "license", "classifiers",
1021:                          "download_url",
1022:                          # PEP 314
1023:                          "provides", "requires", "obsoletes",
1024:                          )
1025: 
1026:     def __init__(self, path=None):
1027:         if path is not None:
1028:             self.read_pkg_file(open(path))
1029:         else:
1030:             self.name = None
1031:             self.version = None
1032:             self.author = None
1033:             self.author_email = None
1034:             self.maintainer = None
1035:             self.maintainer_email = None
1036:             self.url = None
1037:             self.license = None
1038:             self.description = None
1039:             self.long_description = None
1040:             self.keywords = None
1041:             self.platforms = None
1042:             self.classifiers = None
1043:             self.download_url = None
1044:             # PEP 314
1045:             self.provides = None
1046:             self.requires = None
1047:             self.obsoletes = None
1048: 
1049:     def read_pkg_file(self, file):
1050:         '''Reads the metadata values from a file object.'''
1051:         msg = message_from_file(file)
1052: 
1053:         def _read_field(name):
1054:             value = msg[name]
1055:             if value == 'UNKNOWN':
1056:                 return None
1057:             return value
1058: 
1059:         def _read_list(name):
1060:             values = msg.get_all(name, None)
1061:             if values == []:
1062:                 return None
1063:             return values
1064: 
1065:         metadata_version = msg['metadata-version']
1066:         self.name = _read_field('name')
1067:         self.version = _read_field('version')
1068:         self.description = _read_field('summary')
1069:         # we are filling author only.
1070:         self.author = _read_field('author')
1071:         self.maintainer = None
1072:         self.author_email = _read_field('author-email')
1073:         self.maintainer_email = None
1074:         self.url = _read_field('home-page')
1075:         self.license = _read_field('license')
1076: 
1077:         if 'download-url' in msg:
1078:             self.download_url = _read_field('download-url')
1079:         else:
1080:             self.download_url = None
1081: 
1082:         self.long_description = _read_field('description')
1083:         self.description = _read_field('summary')
1084: 
1085:         if 'keywords' in msg:
1086:             self.keywords = _read_field('keywords').split(',')
1087: 
1088:         self.platforms = _read_list('platform')
1089:         self.classifiers = _read_list('classifier')
1090: 
1091:         # PEP 314 - these fields only exist in 1.1
1092:         if metadata_version == '1.1':
1093:             self.requires = _read_list('requires')
1094:             self.provides = _read_list('provides')
1095:             self.obsoletes = _read_list('obsoletes')
1096:         else:
1097:             self.requires = None
1098:             self.provides = None
1099:             self.obsoletes = None
1100: 
1101:     def write_pkg_info(self, base_dir):
1102:         '''Write the PKG-INFO file into the release tree.
1103:         '''
1104:         pkg_info = open(os.path.join(base_dir, 'PKG-INFO'), 'w')
1105:         try:
1106:             self.write_pkg_file(pkg_info)
1107:         finally:
1108:             pkg_info.close()
1109: 
1110:     def write_pkg_file(self, file):
1111:         '''Write the PKG-INFO format data to a file object.
1112:         '''
1113:         version = '1.0'
1114:         if (self.provides or self.requires or self.obsoletes or
1115:             self.classifiers or self.download_url):
1116:             version = '1.1'
1117: 
1118:         self._write_field(file, 'Metadata-Version', version)
1119:         self._write_field(file, 'Name', self.get_name())
1120:         self._write_field(file, 'Version', self.get_version())
1121:         self._write_field(file, 'Summary', self.get_description())
1122:         self._write_field(file, 'Home-page', self.get_url())
1123:         self._write_field(file, 'Author', self.get_contact())
1124:         self._write_field(file, 'Author-email', self.get_contact_email())
1125:         self._write_field(file, 'License', self.get_license())
1126:         if self.download_url:
1127:             self._write_field(file, 'Download-URL', self.download_url)
1128: 
1129:         long_desc = rfc822_escape(self.get_long_description())
1130:         self._write_field(file, 'Description', long_desc)
1131: 
1132:         keywords = ','.join(self.get_keywords())
1133:         if keywords:
1134:             self._write_field(file, 'Keywords', keywords)
1135: 
1136:         self._write_list(file, 'Platform', self.get_platforms())
1137:         self._write_list(file, 'Classifier', self.get_classifiers())
1138: 
1139:         # PEP 314
1140:         self._write_list(file, 'Requires', self.get_requires())
1141:         self._write_list(file, 'Provides', self.get_provides())
1142:         self._write_list(file, 'Obsoletes', self.get_obsoletes())
1143: 
1144:     def _write_field(self, file, name, value):
1145:         file.write('%s: %s\n' % (name, self._encode_field(value)))
1146: 
1147:     def _write_list (self, file, name, values):
1148:         for value in values:
1149:             self._write_field(file, name, value)
1150: 
1151:     def _encode_field(self, value):
1152:         if value is None:
1153:             return None
1154:         if isinstance(value, unicode):
1155:             return value.encode(PKG_INFO_ENCODING)
1156:         return str(value)
1157: 
1158:     # -- Metadata query methods ----------------------------------------
1159: 
1160:     def get_name(self):
1161:         return self.name or "UNKNOWN"
1162: 
1163:     def get_version(self):
1164:         return self.version or "0.0.0"
1165: 
1166:     def get_fullname(self):
1167:         return "%s-%s" % (self.get_name(), self.get_version())
1168: 
1169:     def get_author(self):
1170:         return self._encode_field(self.author) or "UNKNOWN"
1171: 
1172:     def get_author_email(self):
1173:         return self.author_email or "UNKNOWN"
1174: 
1175:     def get_maintainer(self):
1176:         return self._encode_field(self.maintainer) or "UNKNOWN"
1177: 
1178:     def get_maintainer_email(self):
1179:         return self.maintainer_email or "UNKNOWN"
1180: 
1181:     def get_contact(self):
1182:         return (self._encode_field(self.maintainer) or
1183:                 self._encode_field(self.author) or "UNKNOWN")
1184: 
1185:     def get_contact_email(self):
1186:         return self.maintainer_email or self.author_email or "UNKNOWN"
1187: 
1188:     def get_url(self):
1189:         return self.url or "UNKNOWN"
1190: 
1191:     def get_license(self):
1192:         return self.license or "UNKNOWN"
1193:     get_licence = get_license
1194: 
1195:     def get_description(self):
1196:         return self._encode_field(self.description) or "UNKNOWN"
1197: 
1198:     def get_long_description(self):
1199:         return self._encode_field(self.long_description) or "UNKNOWN"
1200: 
1201:     def get_keywords(self):
1202:         return self.keywords or []
1203: 
1204:     def get_platforms(self):
1205:         return self.platforms or ["UNKNOWN"]
1206: 
1207:     def get_classifiers(self):
1208:         return self.classifiers or []
1209: 
1210:     def get_download_url(self):
1211:         return self.download_url or "UNKNOWN"
1212: 
1213:     # PEP 314
1214:     def get_requires(self):
1215:         return self.requires or []
1216: 
1217:     def set_requires(self, value):
1218:         import distutils.versionpredicate
1219:         for v in value:
1220:             distutils.versionpredicate.VersionPredicate(v)
1221:         self.requires = value
1222: 
1223:     def get_provides(self):
1224:         return self.provides or []
1225: 
1226:     def set_provides(self, value):
1227:         value = [v.strip() for v in value]
1228:         for v in value:
1229:             import distutils.versionpredicate
1230:             distutils.versionpredicate.split_provision(v)
1231:         self.provides = value
1232: 
1233:     def get_obsoletes(self):
1234:         return self.obsoletes or []
1235: 
1236:     def set_obsoletes(self, value):
1237:         import distutils.versionpredicate
1238:         for v in value:
1239:             distutils.versionpredicate.VersionPredicate(v)
1240:         self.obsoletes = value
1241: 
1242: def fix_help_options(options):
1243:     '''Convert a 4-tuple 'help_options' list as found in various command
1244:     classes to the 3-tuple form required by FancyGetopt.
1245:     '''
1246:     new_options = []
1247:     for help_tuple in options:
1248:         new_options.append(help_tuple[0:3])
1249:     return new_options
1250: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.dist\n\nProvides the Distribution class, which represents the module distribution\nbeing built/installed/distributed.\n')

# Assigning a Str to a Name (line 7):

# Assigning a Str to a Name (line 7):
str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_531)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# Multiple import statement. import sys (1/3) (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/3) (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)
# Multiple import statement. import re (3/3) (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from email import message_from_file' statement (line 10)
try:
    from email import message_from_file

except:
    message_from_file = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email', None, module_type_store, ['message_from_file'], [message_from_file])



# SSA begins for try-except statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))

# 'import warnings' statement (line 13)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'warnings', warnings, module_type_store)

# SSA branch for the except part of a try statement (line 12)
# SSA branch for the except 'ImportError' branch of a try statement (line 12)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 15):

# Assigning a Name to a Name (line 15):
# Getting the type of 'None' (line 15)
None_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'None')
# Assigning a type to the variable 'warnings' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'warnings', None_532)
# SSA join for try-except statement (line 12)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.errors import DistutilsOptionError, DistutilsArgError, DistutilsModuleError, DistutilsClassError' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors')

if (type(import_533) is not StypyTypeError):

    if (import_533 != 'pyd_module'):
        __import__(import_533)
        sys_modules_534 = sys.modules[import_533]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', sys_modules_534.module_type_store, module_type_store, ['DistutilsOptionError', 'DistutilsArgError', 'DistutilsModuleError', 'DistutilsClassError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_534, sys_modules_534.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError, DistutilsArgError, DistutilsModuleError, DistutilsClassError

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError', 'DistutilsArgError', 'DistutilsModuleError', 'DistutilsClassError'], [DistutilsOptionError, DistutilsArgError, DistutilsModuleError, DistutilsClassError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', import_533)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.fancy_getopt import FancyGetopt, translate_longopt' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_535 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.fancy_getopt')

if (type(import_535) is not StypyTypeError):

    if (import_535 != 'pyd_module'):
        __import__(import_535)
        sys_modules_536 = sys.modules[import_535]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.fancy_getopt', sys_modules_536.module_type_store, module_type_store, ['FancyGetopt', 'translate_longopt'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_536, sys_modules_536.module_type_store, module_type_store)
    else:
        from distutils.fancy_getopt import FancyGetopt, translate_longopt

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.fancy_getopt', None, module_type_store, ['FancyGetopt', 'translate_longopt'], [FancyGetopt, translate_longopt])

else:
    # Assigning a type to the variable 'distutils.fancy_getopt' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.fancy_getopt', import_535)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils.util import check_environ, strtobool, rfc822_escape' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_537 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util')

if (type(import_537) is not StypyTypeError):

    if (import_537 != 'pyd_module'):
        __import__(import_537)
        sys_modules_538 = sys.modules[import_537]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util', sys_modules_538.module_type_store, module_type_store, ['check_environ', 'strtobool', 'rfc822_escape'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_538, sys_modules_538.module_type_store, module_type_store)
    else:
        from distutils.util import check_environ, strtobool, rfc822_escape

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util', None, module_type_store, ['check_environ', 'strtobool', 'rfc822_escape'], [check_environ, strtobool, rfc822_escape])

else:
    # Assigning a type to the variable 'distutils.util' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util', import_537)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils import log' statement (line 21)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.debug import DEBUG' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_539 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.debug')

if (type(import_539) is not StypyTypeError):

    if (import_539 != 'pyd_module'):
        __import__(import_539)
        sys_modules_540 = sys.modules[import_539]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.debug', sys_modules_540.module_type_store, module_type_store, ['DEBUG'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_540, sys_modules_540.module_type_store, module_type_store)
    else:
        from distutils.debug import DEBUG

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

else:
    # Assigning a type to the variable 'distutils.debug' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.debug', import_539)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


# Assigning a Str to a Name (line 25):

# Assigning a Str to a Name (line 25):
str_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'str', 'utf-8')
# Assigning a type to the variable 'PKG_INFO_ENCODING' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'PKG_INFO_ENCODING', str_541)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to compile(...): (line 31)
# Processing the call arguments (line 31)
str_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'str', '^[a-zA-Z]([a-zA-Z0-9_]*)$')
# Processing the call keyword arguments (line 31)
kwargs_545 = {}
# Getting the type of 're' (line 31)
re_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 're', False)
# Obtaining the member 'compile' of a type (line 31)
compile_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), re_542, 'compile')
# Calling compile(args, kwargs) (line 31)
compile_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), compile_543, *[str_544], **kwargs_545)

# Assigning a type to the variable 'command_re' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'command_re', compile_call_result_546)
# Declaration of the 'Distribution' class

class Distribution:
    str_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', "The core of the Distutils.  Most of the work hiding behind 'setup'\n    is really done within a Distribution instance, which farms the work out\n    to the Distutils commands specified on the command line.\n\n    Setup scripts will almost never instantiate Distribution directly,\n    unless the 'setup()' function is totally inadequate to their needs.\n    However, it is conceivable that a setup script might wish to subclass\n    Distribution for some specialized purpose, and then pass the subclass\n    to 'setup()' as the 'distclass' keyword argument.  If so, it is\n    necessary to respect the expectations that 'setup' has of Distribution.\n    See the code for 'setup()', in core.py, for details.\n    ")
    
    # Assigning a List to a Name (line 57):
    
    # Assigning a Str to a Name (line 67):
    
    # Assigning a List to a Name (line 75):
    
    # Assigning a Call to a Name (line 119):
    
    # Assigning a Dict to a Name (line 123):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 128)
        None_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'None')
        defaults = [None_548]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.__init__', ['attrs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['attrs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'str', 'Construct a new Distribution instance: initialize all the\n        attributes of a Distribution, and then use \'attrs\' (a dictionary\n        mapping attribute names to values) to assign some of those\n        attributes their "real" values.  (Any attributes not mentioned in\n        \'attrs\' will be assigned to some null value: 0, None, an empty list\n        or dictionary, etc.)  Most importantly, initialize the\n        \'command_obj\' attribute to the empty dictionary; this will be\n        filled in with real command objects by \'parse_command_line()\'.\n        ')
        
        # Assigning a Num to a Attribute (line 140):
        
        # Assigning a Num to a Attribute (line 140):
        int_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'int')
        # Getting the type of 'self' (line 140)
        self_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_551, 'verbose', int_550)
        
        # Assigning a Num to a Attribute (line 141):
        
        # Assigning a Num to a Attribute (line 141):
        int_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'int')
        # Getting the type of 'self' (line 141)
        self_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self')
        # Setting the type of the member 'dry_run' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_553, 'dry_run', int_552)
        
        # Assigning a Num to a Attribute (line 142):
        
        # Assigning a Num to a Attribute (line 142):
        int_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'int')
        # Getting the type of 'self' (line 142)
        self_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'help' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_555, 'help', int_554)
        
        # Getting the type of 'self' (line 143)
        self_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'self')
        # Obtaining the member 'display_option_names' of a type (line 143)
        display_option_names_557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 20), self_556, 'display_option_names')
        # Testing the type of a for loop iterable (line 143)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 8), display_option_names_557)
        # Getting the type of the for loop variable (line 143)
        for_loop_var_558 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 8), display_option_names_557)
        # Assigning a type to the variable 'attr' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'attr', for_loop_var_558)
        # SSA begins for a for statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'self' (line 144)
        self_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'self', False)
        # Getting the type of 'attr' (line 144)
        attr_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'attr', False)
        int_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 32), 'int')
        # Processing the call keyword arguments (line 144)
        kwargs_563 = {}
        # Getting the type of 'setattr' (line 144)
        setattr_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 144)
        setattr_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), setattr_559, *[self_560, attr_561, int_562], **kwargs_563)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 151):
        
        # Assigning a Call to a Attribute (line 151):
        
        # Call to DistributionMetadata(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_566 = {}
        # Getting the type of 'DistributionMetadata' (line 151)
        DistributionMetadata_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'DistributionMetadata', False)
        # Calling DistributionMetadata(args, kwargs) (line 151)
        DistributionMetadata_call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), DistributionMetadata_565, *[], **kwargs_566)
        
        # Getting the type of 'self' (line 151)
        self_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'metadata' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_568, 'metadata', DistributionMetadata_call_result_567)
        
        # Getting the type of 'self' (line 152)
        self_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'self')
        # Obtaining the member 'metadata' of a type (line 152)
        metadata_570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), self_569, 'metadata')
        # Obtaining the member '_METHOD_BASENAMES' of a type (line 152)
        _METHOD_BASENAMES_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), metadata_570, '_METHOD_BASENAMES')
        # Testing the type of a for loop iterable (line 152)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 8), _METHOD_BASENAMES_571)
        # Getting the type of the for loop variable (line 152)
        for_loop_var_572 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 8), _METHOD_BASENAMES_571)
        # Assigning a type to the variable 'basename' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'basename', for_loop_var_572)
        # SSA begins for a for statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 153):
        
        # Assigning a BinOp to a Name (line 153):
        str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'str', 'get_')
        # Getting the type of 'basename' (line 153)
        basename_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'basename')
        # Applying the binary operator '+' (line 153)
        result_add_575 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 26), '+', str_573, basename_574)
        
        # Assigning a type to the variable 'method_name' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'method_name', result_add_575)
        
        # Call to setattr(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'self', False)
        # Getting the type of 'method_name' (line 154)
        method_name_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'method_name', False)
        
        # Call to getattr(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 47), 'self', False)
        # Obtaining the member 'metadata' of a type (line 154)
        metadata_581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 47), self_580, 'metadata')
        # Getting the type of 'method_name' (line 154)
        method_name_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 62), 'method_name', False)
        # Processing the call keyword arguments (line 154)
        kwargs_583 = {}
        # Getting the type of 'getattr' (line 154)
        getattr_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'getattr', False)
        # Calling getattr(args, kwargs) (line 154)
        getattr_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 154, 39), getattr_579, *[metadata_581, method_name_582], **kwargs_583)
        
        # Processing the call keyword arguments (line 154)
        kwargs_585 = {}
        # Getting the type of 'setattr' (line 154)
        setattr_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 154)
        setattr_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), setattr_576, *[self_577, method_name_578, getattr_call_result_584], **kwargs_585)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Attribute (line 160):
        
        # Assigning a Dict to a Attribute (line 160):
        
        # Obtaining an instance of the builtin type 'dict' (line 160)
        dict_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 160)
        
        # Getting the type of 'self' (line 160)
        self_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'cmdclass' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_588, 'cmdclass', dict_587)
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'None' (line 168)
        None_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'None')
        # Getting the type of 'self' (line 168)
        self_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'command_packages' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_590, 'command_packages', None_589)
        
        # Assigning a Name to a Attribute (line 173):
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'None' (line 173)
        None_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'None')
        # Getting the type of 'self' (line 173)
        self_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member 'script_name' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_592, 'script_name', None_591)
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'None' (line 174)
        None_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'None')
        # Getting the type of 'self' (line 174)
        self_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'script_args' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_594, 'script_args', None_593)
        
        # Assigning a Dict to a Attribute (line 181):
        
        # Assigning a Dict to a Attribute (line 181):
        
        # Obtaining an instance of the builtin type 'dict' (line 181)
        dict_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 181)
        
        # Getting the type of 'self' (line 181)
        self_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self')
        # Setting the type of the member 'command_options' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_596, 'command_options', dict_595)
        
        # Assigning a List to a Attribute (line 192):
        
        # Assigning a List to a Attribute (line 192):
        
        # Obtaining an instance of the builtin type 'list' (line 192)
        list_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 192)
        
        # Getting the type of 'self' (line 192)
        self_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'dist_files' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_598, 'dist_files', list_597)
        
        # Assigning a Name to a Attribute (line 197):
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'None' (line 197)
        None_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'None')
        # Getting the type of 'self' (line 197)
        self_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self')
        # Setting the type of the member 'packages' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_600, 'packages', None_599)
        
        # Assigning a Dict to a Attribute (line 198):
        
        # Assigning a Dict to a Attribute (line 198):
        
        # Obtaining an instance of the builtin type 'dict' (line 198)
        dict_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 198)
        
        # Getting the type of 'self' (line 198)
        self_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member 'package_data' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_602, 'package_data', dict_601)
        
        # Assigning a Name to a Attribute (line 199):
        
        # Assigning a Name to a Attribute (line 199):
        # Getting the type of 'None' (line 199)
        None_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'None')
        # Getting the type of 'self' (line 199)
        self_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member 'package_dir' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_604, 'package_dir', None_603)
        
        # Assigning a Name to a Attribute (line 200):
        
        # Assigning a Name to a Attribute (line 200):
        # Getting the type of 'None' (line 200)
        None_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'None')
        # Getting the type of 'self' (line 200)
        self_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member 'py_modules' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_606, 'py_modules', None_605)
        
        # Assigning a Name to a Attribute (line 201):
        
        # Assigning a Name to a Attribute (line 201):
        # Getting the type of 'None' (line 201)
        None_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'None')
        # Getting the type of 'self' (line 201)
        self_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_608, 'libraries', None_607)
        
        # Assigning a Name to a Attribute (line 202):
        
        # Assigning a Name to a Attribute (line 202):
        # Getting the type of 'None' (line 202)
        None_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'None')
        # Getting the type of 'self' (line 202)
        self_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member 'headers' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_610, 'headers', None_609)
        
        # Assigning a Name to a Attribute (line 203):
        
        # Assigning a Name to a Attribute (line 203):
        # Getting the type of 'None' (line 203)
        None_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'None')
        # Getting the type of 'self' (line 203)
        self_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self')
        # Setting the type of the member 'ext_modules' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_612, 'ext_modules', None_611)
        
        # Assigning a Name to a Attribute (line 204):
        
        # Assigning a Name to a Attribute (line 204):
        # Getting the type of 'None' (line 204)
        None_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'None')
        # Getting the type of 'self' (line 204)
        self_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'ext_package' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_614, 'ext_package', None_613)
        
        # Assigning a Name to a Attribute (line 205):
        
        # Assigning a Name to a Attribute (line 205):
        # Getting the type of 'None' (line 205)
        None_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'None')
        # Getting the type of 'self' (line 205)
        self_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_616, 'include_dirs', None_615)
        
        # Assigning a Name to a Attribute (line 206):
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'None' (line 206)
        None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'None')
        # Getting the type of 'self' (line 206)
        self_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'extra_path' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_618, 'extra_path', None_617)
        
        # Assigning a Name to a Attribute (line 207):
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'None' (line 207)
        None_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'None')
        # Getting the type of 'self' (line 207)
        self_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'scripts' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_620, 'scripts', None_619)
        
        # Assigning a Name to a Attribute (line 208):
        
        # Assigning a Name to a Attribute (line 208):
        # Getting the type of 'None' (line 208)
        None_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'None')
        # Getting the type of 'self' (line 208)
        self_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'self')
        # Setting the type of the member 'data_files' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), self_622, 'data_files', None_621)
        
        # Assigning a Str to a Attribute (line 209):
        
        # Assigning a Str to a Attribute (line 209):
        str_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 24), 'str', '')
        # Getting the type of 'self' (line 209)
        self_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'self')
        # Setting the type of the member 'password' of a type (line 209)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), self_624, 'password', str_623)
        
        # Assigning a Dict to a Attribute (line 215):
        
        # Assigning a Dict to a Attribute (line 215):
        
        # Obtaining an instance of the builtin type 'dict' (line 215)
        dict_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 215)
        
        # Getting the type of 'self' (line 215)
        self_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'self')
        # Setting the type of the member 'command_obj' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), self_626, 'command_obj', dict_625)
        
        # Assigning a Dict to a Attribute (line 227):
        
        # Assigning a Dict to a Attribute (line 227):
        
        # Obtaining an instance of the builtin type 'dict' (line 227)
        dict_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 227)
        
        # Getting the type of 'self' (line 227)
        self_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'have_run' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_628, 'have_run', dict_627)
        
        # Getting the type of 'attrs' (line 233)
        attrs_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'attrs')
        # Testing the type of an if condition (line 233)
        if_condition_630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), attrs_629)
        # Assigning a type to the variable 'if_condition_630' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_630', if_condition_630)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to get(...): (line 238)
        # Processing the call arguments (line 238)
        str_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 32), 'str', 'options')
        # Processing the call keyword arguments (line 238)
        kwargs_634 = {}
        # Getting the type of 'attrs' (line 238)
        attrs_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'attrs', False)
        # Obtaining the member 'get' of a type (line 238)
        get_632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 22), attrs_631, 'get')
        # Calling get(args, kwargs) (line 238)
        get_call_result_635 = invoke(stypy.reporting.localization.Localization(__file__, 238, 22), get_632, *[str_633], **kwargs_634)
        
        # Assigning a type to the variable 'options' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'options', get_call_result_635)
        
        # Type idiom detected: calculating its left and rigth part (line 239)
        # Getting the type of 'options' (line 239)
        options_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'options')
        # Getting the type of 'None' (line 239)
        None_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'None')
        
        (may_be_638, more_types_in_union_639) = may_not_be_none(options_636, None_637)

        if may_be_638:

            if more_types_in_union_639:
                # Runtime conditional SSA (line 239)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Deleting a member
            # Getting the type of 'attrs' (line 240)
            attrs_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'attrs')
            
            # Obtaining the type of the subscript
            str_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 26), 'str', 'options')
            # Getting the type of 'attrs' (line 240)
            attrs_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'attrs')
            # Obtaining the member '__getitem__' of a type (line 240)
            getitem___643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 20), attrs_642, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 240)
            subscript_call_result_644 = invoke(stypy.reporting.localization.Localization(__file__, 240, 20), getitem___643, str_641)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 16), attrs_640, subscript_call_result_644)
            
            
            # Call to items(...): (line 241)
            # Processing the call keyword arguments (line 241)
            kwargs_647 = {}
            # Getting the type of 'options' (line 241)
            options_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 46), 'options', False)
            # Obtaining the member 'items' of a type (line 241)
            items_646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 46), options_645, 'items')
            # Calling items(args, kwargs) (line 241)
            items_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 241, 46), items_646, *[], **kwargs_647)
            
            # Testing the type of a for loop iterable (line 241)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 16), items_call_result_648)
            # Getting the type of the for loop variable (line 241)
            for_loop_var_649 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 16), items_call_result_648)
            # Assigning a type to the variable 'command' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'command', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 16), for_loop_var_649))
            # Assigning a type to the variable 'cmd_options' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'cmd_options', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 16), for_loop_var_649))
            # SSA begins for a for statement (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 242):
            
            # Assigning a Call to a Name (line 242):
            
            # Call to get_option_dict(...): (line 242)
            # Processing the call arguments (line 242)
            # Getting the type of 'command' (line 242)
            command_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'command', False)
            # Processing the call keyword arguments (line 242)
            kwargs_653 = {}
            # Getting the type of 'self' (line 242)
            self_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 31), 'self', False)
            # Obtaining the member 'get_option_dict' of a type (line 242)
            get_option_dict_651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 31), self_650, 'get_option_dict')
            # Calling get_option_dict(args, kwargs) (line 242)
            get_option_dict_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 242, 31), get_option_dict_651, *[command_652], **kwargs_653)
            
            # Assigning a type to the variable 'opt_dict' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'opt_dict', get_option_dict_call_result_654)
            
            
            # Call to items(...): (line 243)
            # Processing the call keyword arguments (line 243)
            kwargs_657 = {}
            # Getting the type of 'cmd_options' (line 243)
            cmd_options_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'cmd_options', False)
            # Obtaining the member 'items' of a type (line 243)
            items_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 38), cmd_options_655, 'items')
            # Calling items(args, kwargs) (line 243)
            items_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 243, 38), items_656, *[], **kwargs_657)
            
            # Testing the type of a for loop iterable (line 243)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 243, 20), items_call_result_658)
            # Getting the type of the for loop variable (line 243)
            for_loop_var_659 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 243, 20), items_call_result_658)
            # Assigning a type to the variable 'opt' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 20), for_loop_var_659))
            # Assigning a type to the variable 'val' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 20), for_loop_var_659))
            # SSA begins for a for statement (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Tuple to a Subscript (line 244):
            
            # Assigning a Tuple to a Subscript (line 244):
            
            # Obtaining an instance of the builtin type 'tuple' (line 244)
            tuple_660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 244)
            # Adding element type (line 244)
            str_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 41), 'str', 'setup script')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 41), tuple_660, str_661)
            # Adding element type (line 244)
            # Getting the type of 'val' (line 244)
            val_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 57), 'val')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 41), tuple_660, val_662)
            
            # Getting the type of 'opt_dict' (line 244)
            opt_dict_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'opt_dict')
            # Getting the type of 'opt' (line 244)
            opt_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 33), 'opt')
            # Storing an element on a container (line 244)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), opt_dict_663, (opt_664, tuple_660))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_639:
                # SSA join for if statement (line 239)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        str_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 15), 'str', 'licence')
        # Getting the type of 'attrs' (line 246)
        attrs_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'attrs')
        # Applying the binary operator 'in' (line 246)
        result_contains_667 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 15), 'in', str_665, attrs_666)
        
        # Testing the type of an if condition (line 246)
        if_condition_668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), result_contains_667)
        # Assigning a type to the variable 'if_condition_668' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'if_condition_668', if_condition_668)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 247):
        
        # Assigning a Subscript to a Subscript (line 247):
        
        # Obtaining the type of the subscript
        str_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 41), 'str', 'licence')
        # Getting the type of 'attrs' (line 247)
        attrs_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 35), 'attrs')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 35), attrs_670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 247, 35), getitem___671, str_669)
        
        # Getting the type of 'attrs' (line 247)
        attrs_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'attrs')
        str_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 22), 'str', 'license')
        # Storing an element on a container (line 247)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 16), attrs_673, (str_674, subscript_call_result_672))
        # Deleting a member
        # Getting the type of 'attrs' (line 248)
        attrs_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'attrs')
        
        # Obtaining the type of the subscript
        str_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 26), 'str', 'licence')
        # Getting the type of 'attrs' (line 248)
        attrs_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'attrs')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), attrs_677, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 248, 20), getitem___678, str_676)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 16), attrs_675, subscript_call_result_679)
        
        # Assigning a Str to a Name (line 249):
        
        # Assigning a Str to a Name (line 249):
        str_680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 22), 'str', "'licence' distribution option is deprecated; use 'license'")
        # Assigning a type to the variable 'msg' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'msg', str_680)
        
        # Type idiom detected: calculating its left and rigth part (line 250)
        # Getting the type of 'warnings' (line 250)
        warnings_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'warnings')
        # Getting the type of 'None' (line 250)
        None_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 35), 'None')
        
        (may_be_683, more_types_in_union_684) = may_not_be_none(warnings_681, None_682)

        if may_be_683:

            if more_types_in_union_684:
                # Runtime conditional SSA (line 250)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to warn(...): (line 251)
            # Processing the call arguments (line 251)
            # Getting the type of 'msg' (line 251)
            msg_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'msg', False)
            # Processing the call keyword arguments (line 251)
            kwargs_688 = {}
            # Getting the type of 'warnings' (line 251)
            warnings_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 251)
            warn_686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), warnings_685, 'warn')
            # Calling warn(args, kwargs) (line 251)
            warn_call_result_689 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), warn_686, *[msg_687], **kwargs_688)
            

            if more_types_in_union_684:
                # Runtime conditional SSA for else branch (line 250)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_683) or more_types_in_union_684):
            
            # Call to write(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 'msg' (line 253)
            msg_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'msg', False)
            str_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 43), 'str', '\n')
            # Applying the binary operator '+' (line 253)
            result_add_695 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 37), '+', msg_693, str_694)
            
            # Processing the call keyword arguments (line 253)
            kwargs_696 = {}
            # Getting the type of 'sys' (line 253)
            sys_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'sys', False)
            # Obtaining the member 'stderr' of a type (line 253)
            stderr_691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), sys_690, 'stderr')
            # Obtaining the member 'write' of a type (line 253)
            write_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), stderr_691, 'write')
            # Calling write(args, kwargs) (line 253)
            write_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), write_692, *[result_add_695], **kwargs_696)
            

            if (may_be_683 and more_types_in_union_684):
                # SSA join for if statement (line 250)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to items(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_700 = {}
        # Getting the type of 'attrs' (line 257)
        attrs_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'attrs', False)
        # Obtaining the member 'items' of a type (line 257)
        items_699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 30), attrs_698, 'items')
        # Calling items(args, kwargs) (line 257)
        items_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 257, 30), items_699, *[], **kwargs_700)
        
        # Testing the type of a for loop iterable (line 257)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 12), items_call_result_701)
        # Getting the type of the for loop variable (line 257)
        for_loop_var_702 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 12), items_call_result_701)
        # Assigning a type to the variable 'key' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 12), for_loop_var_702))
        # Assigning a type to the variable 'val' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 12), for_loop_var_702))
        # SSA begins for a for statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to hasattr(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'self' (line 258)
        self_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'self', False)
        # Obtaining the member 'metadata' of a type (line 258)
        metadata_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 27), self_704, 'metadata')
        str_706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 42), 'str', 'set_')
        # Getting the type of 'key' (line 258)
        key_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 51), 'key', False)
        # Applying the binary operator '+' (line 258)
        result_add_708 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 42), '+', str_706, key_707)
        
        # Processing the call keyword arguments (line 258)
        kwargs_709 = {}
        # Getting the type of 'hasattr' (line 258)
        hasattr_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 258)
        hasattr_call_result_710 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), hasattr_703, *[metadata_705, result_add_708], **kwargs_709)
        
        # Testing the type of an if condition (line 258)
        if_condition_711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 16), hasattr_call_result_710)
        # Assigning a type to the variable 'if_condition_711' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'if_condition_711', if_condition_711)
        # SSA begins for if statement (line 258)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to (...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'val' (line 259)
        val_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 57), 'val', False)
        # Processing the call keyword arguments (line 259)
        kwargs_721 = {}
        
        # Call to getattr(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'self', False)
        # Obtaining the member 'metadata' of a type (line 259)
        metadata_714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 28), self_713, 'metadata')
        str_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'str', 'set_')
        # Getting the type of 'key' (line 259)
        key_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 52), 'key', False)
        # Applying the binary operator '+' (line 259)
        result_add_717 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 43), '+', str_715, key_716)
        
        # Processing the call keyword arguments (line 259)
        kwargs_718 = {}
        # Getting the type of 'getattr' (line 259)
        getattr_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 259)
        getattr_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), getattr_712, *[metadata_714, result_add_717], **kwargs_718)
        
        # Calling (args, kwargs) (line 259)
        _call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), getattr_call_result_719, *[val_720], **kwargs_721)
        
        # SSA branch for the else part of an if statement (line 258)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to hasattr(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'self' (line 260)
        self_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'self', False)
        # Obtaining the member 'metadata' of a type (line 260)
        metadata_725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 29), self_724, 'metadata')
        # Getting the type of 'key' (line 260)
        key_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 44), 'key', False)
        # Processing the call keyword arguments (line 260)
        kwargs_727 = {}
        # Getting the type of 'hasattr' (line 260)
        hasattr_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 21), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 260)
        hasattr_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 260, 21), hasattr_723, *[metadata_725, key_726], **kwargs_727)
        
        # Testing the type of an if condition (line 260)
        if_condition_729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 21), hasattr_call_result_728)
        # Assigning a type to the variable 'if_condition_729' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 21), 'if_condition_729', if_condition_729)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'self' (line 261)
        self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 28), 'self', False)
        # Obtaining the member 'metadata' of a type (line 261)
        metadata_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 28), self_731, 'metadata')
        # Getting the type of 'key' (line 261)
        key_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 43), 'key', False)
        # Getting the type of 'val' (line 261)
        val_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 48), 'val', False)
        # Processing the call keyword arguments (line 261)
        kwargs_735 = {}
        # Getting the type of 'setattr' (line 261)
        setattr_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 261)
        setattr_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 261, 20), setattr_730, *[metadata_732, key_733, val_734], **kwargs_735)
        
        # SSA branch for the else part of an if statement (line 260)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to hasattr(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'self' (line 262)
        self_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 29), 'self', False)
        # Getting the type of 'key' (line 262)
        key_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 35), 'key', False)
        # Processing the call keyword arguments (line 262)
        kwargs_740 = {}
        # Getting the type of 'hasattr' (line 262)
        hasattr_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 21), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 262)
        hasattr_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 262, 21), hasattr_737, *[self_738, key_739], **kwargs_740)
        
        # Testing the type of an if condition (line 262)
        if_condition_742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 21), hasattr_call_result_741)
        # Assigning a type to the variable 'if_condition_742' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 21), 'if_condition_742', if_condition_742)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'self' (line 263)
        self_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 28), 'self', False)
        # Getting the type of 'key' (line 263)
        key_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 34), 'key', False)
        # Getting the type of 'val' (line 263)
        val_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 39), 'val', False)
        # Processing the call keyword arguments (line 263)
        kwargs_747 = {}
        # Getting the type of 'setattr' (line 263)
        setattr_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 263)
        setattr_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 263, 20), setattr_743, *[self_744, key_745, val_746], **kwargs_747)
        
        # SSA branch for the else part of an if statement (line 262)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 265):
        
        # Assigning a BinOp to a Name (line 265):
        str_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 26), 'str', 'Unknown distribution option: %s')
        
        # Call to repr(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'key' (line 265)
        key_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 67), 'key', False)
        # Processing the call keyword arguments (line 265)
        kwargs_752 = {}
        # Getting the type of 'repr' (line 265)
        repr_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 62), 'repr', False)
        # Calling repr(args, kwargs) (line 265)
        repr_call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 265, 62), repr_750, *[key_751], **kwargs_752)
        
        # Applying the binary operator '%' (line 265)
        result_mod_754 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 26), '%', str_749, repr_call_result_753)
        
        # Assigning a type to the variable 'msg' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'msg', result_mod_754)
        
        # Type idiom detected: calculating its left and rigth part (line 266)
        # Getting the type of 'warnings' (line 266)
        warnings_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'warnings')
        # Getting the type of 'None' (line 266)
        None_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 39), 'None')
        
        (may_be_757, more_types_in_union_758) = may_not_be_none(warnings_755, None_756)

        if may_be_757:

            if more_types_in_union_758:
                # Runtime conditional SSA (line 266)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to warn(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 'msg' (line 267)
            msg_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 38), 'msg', False)
            # Processing the call keyword arguments (line 267)
            kwargs_762 = {}
            # Getting the type of 'warnings' (line 267)
            warnings_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 267)
            warn_760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 24), warnings_759, 'warn')
            # Calling warn(args, kwargs) (line 267)
            warn_call_result_763 = invoke(stypy.reporting.localization.Localization(__file__, 267, 24), warn_760, *[msg_761], **kwargs_762)
            

            if more_types_in_union_758:
                # Runtime conditional SSA for else branch (line 266)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_757) or more_types_in_union_758):
            
            # Call to write(...): (line 269)
            # Processing the call arguments (line 269)
            # Getting the type of 'msg' (line 269)
            msg_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'msg', False)
            str_768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 47), 'str', '\n')
            # Applying the binary operator '+' (line 269)
            result_add_769 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 41), '+', msg_767, str_768)
            
            # Processing the call keyword arguments (line 269)
            kwargs_770 = {}
            # Getting the type of 'sys' (line 269)
            sys_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'sys', False)
            # Obtaining the member 'stderr' of a type (line 269)
            stderr_765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 24), sys_764, 'stderr')
            # Obtaining the member 'write' of a type (line 269)
            write_766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 24), stderr_765, 'write')
            # Calling write(args, kwargs) (line 269)
            write_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 269, 24), write_766, *[result_add_769], **kwargs_770)
            

            if (may_be_757 and more_types_in_union_758):
                # SSA join for if statement (line 266)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 258)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 277):
        
        # Assigning a Name to a Attribute (line 277):
        # Getting the type of 'True' (line 277)
        True_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 29), 'True')
        # Getting the type of 'self' (line 277)
        self_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self')
        # Setting the type of the member 'want_user_cfg' of a type (line 277)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_773, 'want_user_cfg', True_772)
        
        
        # Getting the type of 'self' (line 279)
        self_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'self')
        # Obtaining the member 'script_args' of a type (line 279)
        script_args_775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 11), self_774, 'script_args')
        # Getting the type of 'None' (line 279)
        None_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'None')
        # Applying the binary operator 'isnot' (line 279)
        result_is_not_777 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'isnot', script_args_775, None_776)
        
        # Testing the type of an if condition (line 279)
        if_condition_778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_is_not_777)
        # Assigning a type to the variable 'if_condition_778' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_778', if_condition_778)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 280)
        self_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'self')
        # Obtaining the member 'script_args' of a type (line 280)
        script_args_780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), self_779, 'script_args')
        # Testing the type of a for loop iterable (line 280)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 280, 12), script_args_780)
        # Getting the type of the for loop variable (line 280)
        for_loop_var_781 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 280, 12), script_args_780)
        # Assigning a type to the variable 'arg' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'arg', for_loop_var_781)
        # SSA begins for a for statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to startswith(...): (line 281)
        # Processing the call arguments (line 281)
        str_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'str', '-')
        # Processing the call keyword arguments (line 281)
        kwargs_785 = {}
        # Getting the type of 'arg' (line 281)
        arg_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'arg', False)
        # Obtaining the member 'startswith' of a type (line 281)
        startswith_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), arg_782, 'startswith')
        # Calling startswith(args, kwargs) (line 281)
        startswith_call_result_786 = invoke(stypy.reporting.localization.Localization(__file__, 281, 23), startswith_783, *[str_784], **kwargs_785)
        
        # Applying the 'not' unary operator (line 281)
        result_not__787 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 19), 'not', startswith_call_result_786)
        
        # Testing the type of an if condition (line 281)
        if_condition_788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 16), result_not__787)
        # Assigning a type to the variable 'if_condition_788' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'if_condition_788', if_condition_788)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'arg' (line 283)
        arg_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'arg')
        str_790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 26), 'str', '--no-user-cfg')
        # Applying the binary operator '==' (line 283)
        result_eq_791 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 19), '==', arg_789, str_790)
        
        # Testing the type of an if condition (line 283)
        if_condition_792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 16), result_eq_791)
        # Assigning a type to the variable 'if_condition_792' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'if_condition_792', if_condition_792)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 284):
        
        # Assigning a Name to a Attribute (line 284):
        # Getting the type of 'False' (line 284)
        False_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 41), 'False')
        # Getting the type of 'self' (line 284)
        self_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'self')
        # Setting the type of the member 'want_user_cfg' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 20), self_794, 'want_user_cfg', False_793)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to finalize_options(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_797 = {}
        # Getting the type of 'self' (line 287)
        self_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'finalize_options' of a type (line 287)
        finalize_options_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_795, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 287)
        finalize_options_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), finalize_options_796, *[], **kwargs_797)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_option_dict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_option_dict'
        module_type_store = module_type_store.open_function_context('get_option_dict', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.get_option_dict.__dict__.__setitem__('stypy_localization', localization)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_function_name', 'Distribution.get_option_dict')
        Distribution.get_option_dict.__dict__.__setitem__('stypy_param_names_list', ['command'])
        Distribution.get_option_dict.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.get_option_dict.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.get_option_dict', ['command'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_option_dict', localization, ['command'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_option_dict(...)' code ##################

        str_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, (-1)), 'str', "Get the option dictionary for a given command.  If that\n        command's option dictionary hasn't been created yet, then create it\n        and return the new dictionary; otherwise, return the existing\n        option dictionary.\n        ")
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to get(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'command' (line 295)
        command_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 40), 'command', False)
        # Processing the call keyword arguments (line 295)
        kwargs_804 = {}
        # Getting the type of 'self' (line 295)
        self_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'self', False)
        # Obtaining the member 'command_options' of a type (line 295)
        command_options_801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), self_800, 'command_options')
        # Obtaining the member 'get' of a type (line 295)
        get_802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), command_options_801, 'get')
        # Calling get(args, kwargs) (line 295)
        get_call_result_805 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), get_802, *[command_803], **kwargs_804)
        
        # Assigning a type to the variable 'dict' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'dict', get_call_result_805)
        
        # Type idiom detected: calculating its left and rigth part (line 296)
        # Getting the type of 'dict' (line 296)
        dict_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'dict')
        # Getting the type of 'None' (line 296)
        None_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'None')
        
        (may_be_808, more_types_in_union_809) = may_be_none(dict_806, None_807)

        if may_be_808:

            if more_types_in_union_809:
                # Runtime conditional SSA (line 296)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Multiple assignment of 2 elements.
            
            # Assigning a Dict to a Subscript (line 297):
            
            # Obtaining an instance of the builtin type 'dict' (line 297)
            dict_810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 51), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 297)
            
            # Getting the type of 'self' (line 297)
            self_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'self')
            # Obtaining the member 'command_options' of a type (line 297)
            command_options_812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), self_811, 'command_options')
            # Getting the type of 'command' (line 297)
            command_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 40), 'command')
            # Storing an element on a container (line 297)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 19), command_options_812, (command_813, dict_810))
            
            # Assigning a Subscript to a Name (line 297):
            
            # Obtaining the type of the subscript
            # Getting the type of 'command' (line 297)
            command_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 40), 'command')
            # Getting the type of 'self' (line 297)
            self_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'self')
            # Obtaining the member 'command_options' of a type (line 297)
            command_options_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), self_815, 'command_options')
            # Obtaining the member '__getitem__' of a type (line 297)
            getitem___817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), command_options_816, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 297)
            subscript_call_result_818 = invoke(stypy.reporting.localization.Localization(__file__, 297, 19), getitem___817, command_814)
            
            # Assigning a type to the variable 'dict' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'dict', subscript_call_result_818)

            if more_types_in_union_809:
                # SSA join for if statement (line 296)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'dict' (line 298)
        dict_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'dict')
        # Assigning a type to the variable 'stypy_return_type' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type', dict_819)
        
        # ################# End of 'get_option_dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_option_dict' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_820)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_option_dict'
        return stypy_return_type_820


    @norecursion
    def dump_option_dicts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 300)
        None_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 39), 'None')
        # Getting the type of 'None' (line 300)
        None_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 54), 'None')
        str_823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 67), 'str', '')
        defaults = [None_821, None_822, str_823]
        # Create a new context for function 'dump_option_dicts'
        module_type_store = module_type_store.open_function_context('dump_option_dicts', 300, 4, False)
        # Assigning a type to the variable 'self' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_localization', localization)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_function_name', 'Distribution.dump_option_dicts')
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_param_names_list', ['header', 'commands', 'indent'])
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.dump_option_dicts.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.dump_option_dicts', ['header', 'commands', 'indent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_option_dicts', localization, ['header', 'commands', 'indent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_option_dicts(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 301, 8))
        
        # 'from pprint import pformat' statement (line 301)
        try:
            from pprint import pformat

        except:
            pformat = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 301, 8), 'pprint', None, module_type_store, ['pformat'], [pformat])
        
        
        # Type idiom detected: calculating its left and rigth part (line 303)
        # Getting the type of 'commands' (line 303)
        commands_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'commands')
        # Getting the type of 'None' (line 303)
        None_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'None')
        
        (may_be_826, more_types_in_union_827) = may_be_none(commands_824, None_825)

        if may_be_826:

            if more_types_in_union_827:
                # Runtime conditional SSA (line 303)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 304):
            
            # Assigning a Call to a Name (line 304):
            
            # Call to keys(...): (line 304)
            # Processing the call keyword arguments (line 304)
            kwargs_831 = {}
            # Getting the type of 'self' (line 304)
            self_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'self', False)
            # Obtaining the member 'command_options' of a type (line 304)
            command_options_829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 23), self_828, 'command_options')
            # Obtaining the member 'keys' of a type (line 304)
            keys_830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 23), command_options_829, 'keys')
            # Calling keys(args, kwargs) (line 304)
            keys_call_result_832 = invoke(stypy.reporting.localization.Localization(__file__, 304, 23), keys_830, *[], **kwargs_831)
            
            # Assigning a type to the variable 'commands' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'commands', keys_call_result_832)
            
            # Call to sort(...): (line 305)
            # Processing the call keyword arguments (line 305)
            kwargs_835 = {}
            # Getting the type of 'commands' (line 305)
            commands_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'commands', False)
            # Obtaining the member 'sort' of a type (line 305)
            sort_834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), commands_833, 'sort')
            # Calling sort(args, kwargs) (line 305)
            sort_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), sort_834, *[], **kwargs_835)
            

            if more_types_in_union_827:
                # SSA join for if statement (line 303)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 307)
        # Getting the type of 'header' (line 307)
        header_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'header')
        # Getting the type of 'None' (line 307)
        None_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 25), 'None')
        
        (may_be_839, more_types_in_union_840) = may_not_be_none(header_837, None_838)

        if may_be_839:

            if more_types_in_union_840:
                # Runtime conditional SSA (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to announce(...): (line 308)
            # Processing the call arguments (line 308)
            # Getting the type of 'indent' (line 308)
            indent_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'indent', False)
            # Getting the type of 'header' (line 308)
            header_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'header', False)
            # Applying the binary operator '+' (line 308)
            result_add_845 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 26), '+', indent_843, header_844)
            
            # Processing the call keyword arguments (line 308)
            kwargs_846 = {}
            # Getting the type of 'self' (line 308)
            self_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'self', False)
            # Obtaining the member 'announce' of a type (line 308)
            announce_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), self_841, 'announce')
            # Calling announce(args, kwargs) (line 308)
            announce_call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), announce_842, *[result_add_845], **kwargs_846)
            
            
            # Assigning a BinOp to a Name (line 309):
            
            # Assigning a BinOp to a Name (line 309):
            # Getting the type of 'indent' (line 309)
            indent_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 21), 'indent')
            str_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 30), 'str', '  ')
            # Applying the binary operator '+' (line 309)
            result_add_850 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 21), '+', indent_848, str_849)
            
            # Assigning a type to the variable 'indent' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'indent', result_add_850)

            if more_types_in_union_840:
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'commands' (line 311)
        commands_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'commands')
        # Applying the 'not' unary operator (line 311)
        result_not__852 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 11), 'not', commands_851)
        
        # Testing the type of an if condition (line 311)
        if_condition_853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), result_not__852)
        # Assigning a type to the variable 'if_condition_853' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_853', if_condition_853)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'indent' (line 312)
        indent_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'indent', False)
        str_857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 35), 'str', 'no commands known yet')
        # Applying the binary operator '+' (line 312)
        result_add_858 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 26), '+', indent_856, str_857)
        
        # Processing the call keyword arguments (line 312)
        kwargs_859 = {}
        # Getting the type of 'self' (line 312)
        self_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 312)
        announce_855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), self_854, 'announce')
        # Calling announce(args, kwargs) (line 312)
        announce_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), announce_855, *[result_add_858], **kwargs_859)
        
        # Assigning a type to the variable 'stypy_return_type' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'commands' (line 315)
        commands_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'commands')
        # Testing the type of a for loop iterable (line 315)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 315, 8), commands_861)
        # Getting the type of the for loop variable (line 315)
        for_loop_var_862 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 315, 8), commands_861)
        # Assigning a type to the variable 'cmd_name' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'cmd_name', for_loop_var_862)
        # SSA begins for a for statement (line 315)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to get(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'cmd_name' (line 316)
        cmd_name_866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 48), 'cmd_name', False)
        # Processing the call keyword arguments (line 316)
        kwargs_867 = {}
        # Getting the type of 'self' (line 316)
        self_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'self', False)
        # Obtaining the member 'command_options' of a type (line 316)
        command_options_864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), self_863, 'command_options')
        # Obtaining the member 'get' of a type (line 316)
        get_865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), command_options_864, 'get')
        # Calling get(args, kwargs) (line 316)
        get_call_result_868 = invoke(stypy.reporting.localization.Localization(__file__, 316, 23), get_865, *[cmd_name_866], **kwargs_867)
        
        # Assigning a type to the variable 'opt_dict' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'opt_dict', get_call_result_868)
        
        # Type idiom detected: calculating its left and rigth part (line 317)
        # Getting the type of 'opt_dict' (line 317)
        opt_dict_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'opt_dict')
        # Getting the type of 'None' (line 317)
        None_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 27), 'None')
        
        (may_be_871, more_types_in_union_872) = may_be_none(opt_dict_869, None_870)

        if may_be_871:

            if more_types_in_union_872:
                # Runtime conditional SSA (line 317)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to announce(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 'indent' (line 318)
            indent_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'indent', False)
            str_876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'str', "no option dict for '%s' command")
            # Getting the type of 'cmd_name' (line 319)
            cmd_name_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 66), 'cmd_name', False)
            # Applying the binary operator '%' (line 319)
            result_mod_878 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 30), '%', str_876, cmd_name_877)
            
            # Applying the binary operator '+' (line 318)
            result_add_879 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 30), '+', indent_875, result_mod_878)
            
            # Processing the call keyword arguments (line 318)
            kwargs_880 = {}
            # Getting the type of 'self' (line 318)
            self_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'self', False)
            # Obtaining the member 'announce' of a type (line 318)
            announce_874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), self_873, 'announce')
            # Calling announce(args, kwargs) (line 318)
            announce_call_result_881 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), announce_874, *[result_add_879], **kwargs_880)
            

            if more_types_in_union_872:
                # Runtime conditional SSA for else branch (line 317)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_871) or more_types_in_union_872):
            
            # Call to announce(...): (line 321)
            # Processing the call arguments (line 321)
            # Getting the type of 'indent' (line 321)
            indent_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'indent', False)
            str_885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 30), 'str', "option dict for '%s' command:")
            # Getting the type of 'cmd_name' (line 322)
            cmd_name_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 64), 'cmd_name', False)
            # Applying the binary operator '%' (line 322)
            result_mod_887 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 30), '%', str_885, cmd_name_886)
            
            # Applying the binary operator '+' (line 321)
            result_add_888 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 30), '+', indent_884, result_mod_887)
            
            # Processing the call keyword arguments (line 321)
            kwargs_889 = {}
            # Getting the type of 'self' (line 321)
            self_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'self', False)
            # Obtaining the member 'announce' of a type (line 321)
            announce_883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), self_882, 'announce')
            # Calling announce(args, kwargs) (line 321)
            announce_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), announce_883, *[result_add_888], **kwargs_889)
            
            
            # Assigning a Call to a Name (line 323):
            
            # Assigning a Call to a Name (line 323):
            
            # Call to pformat(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'opt_dict' (line 323)
            opt_dict_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 30), 'opt_dict', False)
            # Processing the call keyword arguments (line 323)
            kwargs_893 = {}
            # Getting the type of 'pformat' (line 323)
            pformat_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'pformat', False)
            # Calling pformat(args, kwargs) (line 323)
            pformat_call_result_894 = invoke(stypy.reporting.localization.Localization(__file__, 323, 22), pformat_891, *[opt_dict_892], **kwargs_893)
            
            # Assigning a type to the variable 'out' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'out', pformat_call_result_894)
            
            
            # Call to split(...): (line 324)
            # Processing the call arguments (line 324)
            str_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 38), 'str', '\n')
            # Processing the call keyword arguments (line 324)
            kwargs_898 = {}
            # Getting the type of 'out' (line 324)
            out_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'out', False)
            # Obtaining the member 'split' of a type (line 324)
            split_896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 28), out_895, 'split')
            # Calling split(args, kwargs) (line 324)
            split_call_result_899 = invoke(stypy.reporting.localization.Localization(__file__, 324, 28), split_896, *[str_897], **kwargs_898)
            
            # Testing the type of a for loop iterable (line 324)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 324, 16), split_call_result_899)
            # Getting the type of the for loop variable (line 324)
            for_loop_var_900 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 324, 16), split_call_result_899)
            # Assigning a type to the variable 'line' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'line', for_loop_var_900)
            # SSA begins for a for statement (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to announce(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'indent' (line 325)
            indent_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'indent', False)
            str_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 43), 'str', '  ')
            # Applying the binary operator '+' (line 325)
            result_add_905 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 34), '+', indent_903, str_904)
            
            # Getting the type of 'line' (line 325)
            line_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 50), 'line', False)
            # Applying the binary operator '+' (line 325)
            result_add_907 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 48), '+', result_add_905, line_906)
            
            # Processing the call keyword arguments (line 325)
            kwargs_908 = {}
            # Getting the type of 'self' (line 325)
            self_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'self', False)
            # Obtaining the member 'announce' of a type (line 325)
            announce_902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 20), self_901, 'announce')
            # Calling announce(args, kwargs) (line 325)
            announce_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), announce_902, *[result_add_907], **kwargs_908)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_871 and more_types_in_union_872):
                # SSA join for if statement (line 317)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dump_option_dicts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_option_dicts' in the type store
        # Getting the type of 'stypy_return_type' (line 300)
        stypy_return_type_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_option_dicts'
        return stypy_return_type_910


    @norecursion
    def find_config_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_config_files'
        module_type_store = module_type_store.open_function_context('find_config_files', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.find_config_files.__dict__.__setitem__('stypy_localization', localization)
        Distribution.find_config_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.find_config_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.find_config_files.__dict__.__setitem__('stypy_function_name', 'Distribution.find_config_files')
        Distribution.find_config_files.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.find_config_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.find_config_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.find_config_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.find_config_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.find_config_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.find_config_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.find_config_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_config_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_config_files(...)' code ##################

        str_911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, (-1)), 'str', "Find as many configuration files as should be processed for this\n        platform, and return a list of filenames in the order in which they\n        should be parsed.  The filenames returned are guaranteed to exist\n        (modulo nasty race conditions).\n\n        There are three possible config files: distutils.cfg in the\n        Distutils installation directory (ie. where the top-level\n        Distutils __inst__.py file lives), a file in the user's home\n        directory named .pydistutils.cfg on Unix and pydistutils.cfg\n        on Windows/Mac; and setup.cfg in the current directory.\n\n        The file in the user's home directory can be disabled with the\n        --no-user-cfg option.\n        ")
        
        # Assigning a List to a Name (line 344):
        
        # Assigning a List to a Name (line 344):
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        
        # Assigning a type to the variable 'files' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'files', list_912)
        
        # Call to check_environ(...): (line 345)
        # Processing the call keyword arguments (line 345)
        kwargs_914 = {}
        # Getting the type of 'check_environ' (line 345)
        check_environ_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'check_environ', False)
        # Calling check_environ(args, kwargs) (line 345)
        check_environ_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), check_environ_913, *[], **kwargs_914)
        
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to dirname(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Obtaining the type of the subscript
        str_919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'str', 'distutils')
        # Getting the type of 'sys' (line 348)
        sys_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), 'sys', False)
        # Obtaining the member 'modules' of a type (line 348)
        modules_921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 34), sys_920, 'modules')
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 34), modules_921, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_923 = invoke(stypy.reporting.localization.Localization(__file__, 348, 34), getitem___922, str_919)
        
        # Obtaining the member '__file__' of a type (line 348)
        file___924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 34), subscript_call_result_923, '__file__')
        # Processing the call keyword arguments (line 348)
        kwargs_925 = {}
        # Getting the type of 'os' (line 348)
        os_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 348)
        path_917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 18), os_916, 'path')
        # Obtaining the member 'dirname' of a type (line 348)
        dirname_918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 18), path_917, 'dirname')
        # Calling dirname(args, kwargs) (line 348)
        dirname_call_result_926 = invoke(stypy.reporting.localization.Localization(__file__, 348, 18), dirname_918, *[file___924], **kwargs_925)
        
        # Assigning a type to the variable 'sys_dir' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'sys_dir', dirname_call_result_926)
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to join(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'sys_dir' (line 351)
        sys_dir_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'sys_dir', False)
        str_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 41), 'str', 'distutils.cfg')
        # Processing the call keyword arguments (line 351)
        kwargs_932 = {}
        # Getting the type of 'os' (line 351)
        os_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 351)
        path_928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 19), os_927, 'path')
        # Obtaining the member 'join' of a type (line 351)
        join_929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 19), path_928, 'join')
        # Calling join(args, kwargs) (line 351)
        join_call_result_933 = invoke(stypy.reporting.localization.Localization(__file__, 351, 19), join_929, *[sys_dir_930, str_931], **kwargs_932)
        
        # Assigning a type to the variable 'sys_file' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'sys_file', join_call_result_933)
        
        
        # Call to isfile(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'sys_file' (line 352)
        sys_file_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 26), 'sys_file', False)
        # Processing the call keyword arguments (line 352)
        kwargs_938 = {}
        # Getting the type of 'os' (line 352)
        os_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 352)
        path_935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 11), os_934, 'path')
        # Obtaining the member 'isfile' of a type (line 352)
        isfile_936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 11), path_935, 'isfile')
        # Calling isfile(args, kwargs) (line 352)
        isfile_call_result_939 = invoke(stypy.reporting.localization.Localization(__file__, 352, 11), isfile_936, *[sys_file_937], **kwargs_938)
        
        # Testing the type of an if condition (line 352)
        if_condition_940 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 8), isfile_call_result_939)
        # Assigning a type to the variable 'if_condition_940' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'if_condition_940', if_condition_940)
        # SSA begins for if statement (line 352)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'sys_file' (line 353)
        sys_file_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 25), 'sys_file', False)
        # Processing the call keyword arguments (line 353)
        kwargs_944 = {}
        # Getting the type of 'files' (line 353)
        files_941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'files', False)
        # Obtaining the member 'append' of a type (line 353)
        append_942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 12), files_941, 'append')
        # Calling append(args, kwargs) (line 353)
        append_call_result_945 = invoke(stypy.reporting.localization.Localization(__file__, 353, 12), append_942, *[sys_file_943], **kwargs_944)
        
        # SSA join for if statement (line 352)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'os' (line 356)
        os_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'os')
        # Obtaining the member 'name' of a type (line 356)
        name_947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 11), os_946, 'name')
        str_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 356)
        result_eq_949 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), '==', name_947, str_948)
        
        # Testing the type of an if condition (line 356)
        if_condition_950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_eq_949)
        # Assigning a type to the variable 'if_condition_950' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_950', if_condition_950)
        # SSA begins for if statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 357):
        
        # Assigning a Str to a Name (line 357):
        str_951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 28), 'str', '.pydistutils.cfg')
        # Assigning a type to the variable 'user_filename' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'user_filename', str_951)
        # SSA branch for the else part of an if statement (line 356)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 359):
        
        # Assigning a Str to a Name (line 359):
        str_952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 28), 'str', 'pydistutils.cfg')
        # Assigning a type to the variable 'user_filename' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'user_filename', str_952)
        # SSA join for if statement (line 356)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 362)
        self_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'self')
        # Obtaining the member 'want_user_cfg' of a type (line 362)
        want_user_cfg_954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 11), self_953, 'want_user_cfg')
        # Testing the type of an if condition (line 362)
        if_condition_955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 8), want_user_cfg_954)
        # Assigning a type to the variable 'if_condition_955' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'if_condition_955', if_condition_955)
        # SSA begins for if statement (line 362)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to join(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Call to expanduser(...): (line 363)
        # Processing the call arguments (line 363)
        str_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 56), 'str', '~')
        # Processing the call keyword arguments (line 363)
        kwargs_963 = {}
        # Getting the type of 'os' (line 363)
        os_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 363)
        path_960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 37), os_959, 'path')
        # Obtaining the member 'expanduser' of a type (line 363)
        expanduser_961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 37), path_960, 'expanduser')
        # Calling expanduser(args, kwargs) (line 363)
        expanduser_call_result_964 = invoke(stypy.reporting.localization.Localization(__file__, 363, 37), expanduser_961, *[str_962], **kwargs_963)
        
        # Getting the type of 'user_filename' (line 363)
        user_filename_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 62), 'user_filename', False)
        # Processing the call keyword arguments (line 363)
        kwargs_966 = {}
        # Getting the type of 'os' (line 363)
        os_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 363)
        path_957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 24), os_956, 'path')
        # Obtaining the member 'join' of a type (line 363)
        join_958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 24), path_957, 'join')
        # Calling join(args, kwargs) (line 363)
        join_call_result_967 = invoke(stypy.reporting.localization.Localization(__file__, 363, 24), join_958, *[expanduser_call_result_964, user_filename_965], **kwargs_966)
        
        # Assigning a type to the variable 'user_file' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'user_file', join_call_result_967)
        
        
        # Call to isfile(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'user_file' (line 364)
        user_file_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'user_file', False)
        # Processing the call keyword arguments (line 364)
        kwargs_972 = {}
        # Getting the type of 'os' (line 364)
        os_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 364)
        path_969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), os_968, 'path')
        # Obtaining the member 'isfile' of a type (line 364)
        isfile_970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), path_969, 'isfile')
        # Calling isfile(args, kwargs) (line 364)
        isfile_call_result_973 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), isfile_970, *[user_file_971], **kwargs_972)
        
        # Testing the type of an if condition (line 364)
        if_condition_974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 12), isfile_call_result_973)
        # Assigning a type to the variable 'if_condition_974' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'if_condition_974', if_condition_974)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'user_file' (line 365)
        user_file_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 29), 'user_file', False)
        # Processing the call keyword arguments (line 365)
        kwargs_978 = {}
        # Getting the type of 'files' (line 365)
        files_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'files', False)
        # Obtaining the member 'append' of a type (line 365)
        append_976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 16), files_975, 'append')
        # Calling append(args, kwargs) (line 365)
        append_call_result_979 = invoke(stypy.reporting.localization.Localization(__file__, 365, 16), append_976, *[user_file_977], **kwargs_978)
        
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 362)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 368):
        
        # Assigning a Str to a Name (line 368):
        str_980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 21), 'str', 'setup.cfg')
        # Assigning a type to the variable 'local_file' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'local_file', str_980)
        
        
        # Call to isfile(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'local_file' (line 369)
        local_file_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'local_file', False)
        # Processing the call keyword arguments (line 369)
        kwargs_985 = {}
        # Getting the type of 'os' (line 369)
        os_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 369)
        path_982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 11), os_981, 'path')
        # Obtaining the member 'isfile' of a type (line 369)
        isfile_983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 11), path_982, 'isfile')
        # Calling isfile(args, kwargs) (line 369)
        isfile_call_result_986 = invoke(stypy.reporting.localization.Localization(__file__, 369, 11), isfile_983, *[local_file_984], **kwargs_985)
        
        # Testing the type of an if condition (line 369)
        if_condition_987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 8), isfile_call_result_986)
        # Assigning a type to the variable 'if_condition_987' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'if_condition_987', if_condition_987)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'local_file' (line 370)
        local_file_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 25), 'local_file', False)
        # Processing the call keyword arguments (line 370)
        kwargs_991 = {}
        # Getting the type of 'files' (line 370)
        files_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'files', False)
        # Obtaining the member 'append' of a type (line 370)
        append_989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), files_988, 'append')
        # Calling append(args, kwargs) (line 370)
        append_call_result_992 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), append_989, *[local_file_990], **kwargs_991)
        
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'DEBUG' (line 372)
        DEBUG_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'DEBUG')
        # Testing the type of an if condition (line 372)
        if_condition_994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), DEBUG_993)
        # Assigning a type to the variable 'if_condition_994' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_994', if_condition_994)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 373)
        # Processing the call arguments (line 373)
        str_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 26), 'str', 'using config files: %s')
        
        # Call to join(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'files' (line 373)
        files_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 63), 'files', False)
        # Processing the call keyword arguments (line 373)
        kwargs_1001 = {}
        str_998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 53), 'str', ', ')
        # Obtaining the member 'join' of a type (line 373)
        join_999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 53), str_998, 'join')
        # Calling join(args, kwargs) (line 373)
        join_call_result_1002 = invoke(stypy.reporting.localization.Localization(__file__, 373, 53), join_999, *[files_1000], **kwargs_1001)
        
        # Applying the binary operator '%' (line 373)
        result_mod_1003 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 26), '%', str_997, join_call_result_1002)
        
        # Processing the call keyword arguments (line 373)
        kwargs_1004 = {}
        # Getting the type of 'self' (line 373)
        self_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 373)
        announce_996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), self_995, 'announce')
        # Calling announce(args, kwargs) (line 373)
        announce_call_result_1005 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), announce_996, *[result_mod_1003], **kwargs_1004)
        
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'files' (line 375)
        files_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'files')
        # Assigning a type to the variable 'stypy_return_type' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'stypy_return_type', files_1006)
        
        # ################# End of 'find_config_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_config_files' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_config_files'
        return stypy_return_type_1007


    @norecursion
    def parse_config_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 377)
        None_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 43), 'None')
        defaults = [None_1008]
        # Create a new context for function 'parse_config_files'
        module_type_store = module_type_store.open_function_context('parse_config_files', 377, 4, False)
        # Assigning a type to the variable 'self' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.parse_config_files.__dict__.__setitem__('stypy_localization', localization)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_function_name', 'Distribution.parse_config_files')
        Distribution.parse_config_files.__dict__.__setitem__('stypy_param_names_list', ['filenames'])
        Distribution.parse_config_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.parse_config_files.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.parse_config_files', ['filenames'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse_config_files', localization, ['filenames'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse_config_files(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 378, 8))
        
        # 'from ConfigParser import ConfigParser' statement (line 378)
        try:
            from ConfigParser import ConfigParser

        except:
            ConfigParser = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 378, 8), 'ConfigParser', None, module_type_store, ['ConfigParser'], [ConfigParser])
        
        
        # Type idiom detected: calculating its left and rigth part (line 380)
        # Getting the type of 'filenames' (line 380)
        filenames_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 11), 'filenames')
        # Getting the type of 'None' (line 380)
        None_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'None')
        
        (may_be_1011, more_types_in_union_1012) = may_be_none(filenames_1009, None_1010)

        if may_be_1011:

            if more_types_in_union_1012:
                # Runtime conditional SSA (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 381):
            
            # Assigning a Call to a Name (line 381):
            
            # Call to find_config_files(...): (line 381)
            # Processing the call keyword arguments (line 381)
            kwargs_1015 = {}
            # Getting the type of 'self' (line 381)
            self_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 24), 'self', False)
            # Obtaining the member 'find_config_files' of a type (line 381)
            find_config_files_1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 24), self_1013, 'find_config_files')
            # Calling find_config_files(args, kwargs) (line 381)
            find_config_files_call_result_1016 = invoke(stypy.reporting.localization.Localization(__file__, 381, 24), find_config_files_1014, *[], **kwargs_1015)
            
            # Assigning a type to the variable 'filenames' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'filenames', find_config_files_call_result_1016)

            if more_types_in_union_1012:
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'DEBUG' (line 383)
        DEBUG_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'DEBUG')
        # Testing the type of an if condition (line 383)
        if_condition_1018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), DEBUG_1017)
        # Assigning a type to the variable 'if_condition_1018' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_1018', if_condition_1018)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 384)
        # Processing the call arguments (line 384)
        str_1021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 26), 'str', 'Distribution.parse_config_files():')
        # Processing the call keyword arguments (line 384)
        kwargs_1022 = {}
        # Getting the type of 'self' (line 384)
        self_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 384)
        announce_1020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 12), self_1019, 'announce')
        # Calling announce(args, kwargs) (line 384)
        announce_call_result_1023 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), announce_1020, *[str_1021], **kwargs_1022)
        
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to ConfigParser(...): (line 386)
        # Processing the call keyword arguments (line 386)
        kwargs_1025 = {}
        # Getting the type of 'ConfigParser' (line 386)
        ConfigParser_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'ConfigParser', False)
        # Calling ConfigParser(args, kwargs) (line 386)
        ConfigParser_call_result_1026 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), ConfigParser_1024, *[], **kwargs_1025)
        
        # Assigning a type to the variable 'parser' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'parser', ConfigParser_call_result_1026)
        
        # Getting the type of 'filenames' (line 387)
        filenames_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'filenames')
        # Testing the type of a for loop iterable (line 387)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 387, 8), filenames_1027)
        # Getting the type of the for loop variable (line 387)
        for_loop_var_1028 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 387, 8), filenames_1027)
        # Assigning a type to the variable 'filename' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'filename', for_loop_var_1028)
        # SSA begins for a for statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'DEBUG' (line 388)
        DEBUG_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'DEBUG')
        # Testing the type of an if condition (line 388)
        if_condition_1030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 12), DEBUG_1029)
        # Assigning a type to the variable 'if_condition_1030' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'if_condition_1030', if_condition_1030)
        # SSA begins for if statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 389)
        # Processing the call arguments (line 389)
        str_1033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 30), 'str', '  reading %s')
        # Getting the type of 'filename' (line 389)
        filename_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 47), 'filename', False)
        # Applying the binary operator '%' (line 389)
        result_mod_1035 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 30), '%', str_1033, filename_1034)
        
        # Processing the call keyword arguments (line 389)
        kwargs_1036 = {}
        # Getting the type of 'self' (line 389)
        self_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'self', False)
        # Obtaining the member 'announce' of a type (line 389)
        announce_1032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), self_1031, 'announce')
        # Calling announce(args, kwargs) (line 389)
        announce_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), announce_1032, *[result_mod_1035], **kwargs_1036)
        
        # SSA join for if statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to read(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'filename' (line 390)
        filename_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'filename', False)
        # Processing the call keyword arguments (line 390)
        kwargs_1041 = {}
        # Getting the type of 'parser' (line 390)
        parser_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'parser', False)
        # Obtaining the member 'read' of a type (line 390)
        read_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), parser_1038, 'read')
        # Calling read(args, kwargs) (line 390)
        read_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), read_1039, *[filename_1040], **kwargs_1041)
        
        
        
        # Call to sections(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_1045 = {}
        # Getting the type of 'parser' (line 391)
        parser_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 27), 'parser', False)
        # Obtaining the member 'sections' of a type (line 391)
        sections_1044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 27), parser_1043, 'sections')
        # Calling sections(args, kwargs) (line 391)
        sections_call_result_1046 = invoke(stypy.reporting.localization.Localization(__file__, 391, 27), sections_1044, *[], **kwargs_1045)
        
        # Testing the type of a for loop iterable (line 391)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 391, 12), sections_call_result_1046)
        # Getting the type of the for loop variable (line 391)
        for_loop_var_1047 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 391, 12), sections_call_result_1046)
        # Assigning a type to the variable 'section' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'section', for_loop_var_1047)
        # SSA begins for a for statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to options(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'section' (line 392)
        section_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 41), 'section', False)
        # Processing the call keyword arguments (line 392)
        kwargs_1051 = {}
        # Getting the type of 'parser' (line 392)
        parser_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 26), 'parser', False)
        # Obtaining the member 'options' of a type (line 392)
        options_1049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 26), parser_1048, 'options')
        # Calling options(args, kwargs) (line 392)
        options_call_result_1052 = invoke(stypy.reporting.localization.Localization(__file__, 392, 26), options_1049, *[section_1050], **kwargs_1051)
        
        # Assigning a type to the variable 'options' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'options', options_call_result_1052)
        
        # Assigning a Call to a Name (line 393):
        
        # Assigning a Call to a Name (line 393):
        
        # Call to get_option_dict(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'section' (line 393)
        section_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 48), 'section', False)
        # Processing the call keyword arguments (line 393)
        kwargs_1056 = {}
        # Getting the type of 'self' (line 393)
        self_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 27), 'self', False)
        # Obtaining the member 'get_option_dict' of a type (line 393)
        get_option_dict_1054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 27), self_1053, 'get_option_dict')
        # Calling get_option_dict(args, kwargs) (line 393)
        get_option_dict_call_result_1057 = invoke(stypy.reporting.localization.Localization(__file__, 393, 27), get_option_dict_1054, *[section_1055], **kwargs_1056)
        
        # Assigning a type to the variable 'opt_dict' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'opt_dict', get_option_dict_call_result_1057)
        
        # Getting the type of 'options' (line 395)
        options_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'options')
        # Testing the type of a for loop iterable (line 395)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 395, 16), options_1058)
        # Getting the type of the for loop variable (line 395)
        for_loop_var_1059 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 395, 16), options_1058)
        # Assigning a type to the variable 'opt' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'opt', for_loop_var_1059)
        # SSA begins for a for statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'opt' (line 396)
        opt_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'opt')
        str_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 30), 'str', '__name__')
        # Applying the binary operator '!=' (line 396)
        result_ne_1062 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 23), '!=', opt_1060, str_1061)
        
        # Testing the type of an if condition (line 396)
        if_condition_1063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 20), result_ne_1062)
        # Assigning a type to the variable 'if_condition_1063' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'if_condition_1063', if_condition_1063)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to get(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'section' (line 397)
        section_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 41), 'section', False)
        # Getting the type of 'opt' (line 397)
        opt_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 49), 'opt', False)
        # Processing the call keyword arguments (line 397)
        kwargs_1068 = {}
        # Getting the type of 'parser' (line 397)
        parser_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 30), 'parser', False)
        # Obtaining the member 'get' of a type (line 397)
        get_1065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 30), parser_1064, 'get')
        # Calling get(args, kwargs) (line 397)
        get_call_result_1069 = invoke(stypy.reporting.localization.Localization(__file__, 397, 30), get_1065, *[section_1066, opt_1067], **kwargs_1068)
        
        # Assigning a type to the variable 'val' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'val', get_call_result_1069)
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to replace(...): (line 398)
        # Processing the call arguments (line 398)
        str_1072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 42), 'str', '-')
        str_1073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 47), 'str', '_')
        # Processing the call keyword arguments (line 398)
        kwargs_1074 = {}
        # Getting the type of 'opt' (line 398)
        opt_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 30), 'opt', False)
        # Obtaining the member 'replace' of a type (line 398)
        replace_1071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 30), opt_1070, 'replace')
        # Calling replace(args, kwargs) (line 398)
        replace_call_result_1075 = invoke(stypy.reporting.localization.Localization(__file__, 398, 30), replace_1071, *[str_1072, str_1073], **kwargs_1074)
        
        # Assigning a type to the variable 'opt' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'opt', replace_call_result_1075)
        
        # Assigning a Tuple to a Subscript (line 399):
        
        # Assigning a Tuple to a Subscript (line 399):
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_1076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        # Getting the type of 'filename' (line 399)
        filename_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 41), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 41), tuple_1076, filename_1077)
        # Adding element type (line 399)
        # Getting the type of 'val' (line 399)
        val_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 51), 'val')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 41), tuple_1076, val_1078)
        
        # Getting the type of 'opt_dict' (line 399)
        opt_dict_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 24), 'opt_dict')
        # Getting the type of 'opt' (line 399)
        opt_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 33), 'opt')
        # Storing an element on a container (line 399)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), opt_dict_1079, (opt_1080, tuple_1076))
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __init__(...): (line 403)
        # Processing the call keyword arguments (line 403)
        kwargs_1083 = {}
        # Getting the type of 'parser' (line 403)
        parser_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'parser', False)
        # Obtaining the member '__init__' of a type (line 403)
        init___1082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 12), parser_1081, '__init__')
        # Calling __init__(args, kwargs) (line 403)
        init___call_result_1084 = invoke(stypy.reporting.localization.Localization(__file__, 403, 12), init___1082, *[], **kwargs_1083)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_1085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 11), 'str', 'global')
        # Getting the type of 'self' (line 408)
        self_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 23), 'self')
        # Obtaining the member 'command_options' of a type (line 408)
        command_options_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 23), self_1086, 'command_options')
        # Applying the binary operator 'in' (line 408)
        result_contains_1088 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 11), 'in', str_1085, command_options_1087)
        
        # Testing the type of an if condition (line 408)
        if_condition_1089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 8), result_contains_1088)
        # Assigning a type to the variable 'if_condition_1089' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'if_condition_1089', if_condition_1089)
        # SSA begins for if statement (line 408)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 409)
        # Processing the call keyword arguments (line 409)
        kwargs_1096 = {}
        
        # Obtaining the type of the subscript
        str_1090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 58), 'str', 'global')
        # Getting the type of 'self' (line 409)
        self_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 37), 'self', False)
        # Obtaining the member 'command_options' of a type (line 409)
        command_options_1092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 37), self_1091, 'command_options')
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___1093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 37), command_options_1092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 409)
        subscript_call_result_1094 = invoke(stypy.reporting.localization.Localization(__file__, 409, 37), getitem___1093, str_1090)
        
        # Obtaining the member 'items' of a type (line 409)
        items_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 37), subscript_call_result_1094, 'items')
        # Calling items(args, kwargs) (line 409)
        items_call_result_1097 = invoke(stypy.reporting.localization.Localization(__file__, 409, 37), items_1095, *[], **kwargs_1096)
        
        # Testing the type of a for loop iterable (line 409)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 409, 12), items_call_result_1097)
        # Getting the type of the for loop variable (line 409)
        for_loop_var_1098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 409, 12), items_call_result_1097)
        # Assigning a type to the variable 'opt' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 12), for_loop_var_1098))
        # Assigning a type to the variable 'src' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'src', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 12), for_loop_var_1098))
        # Assigning a type to the variable 'val' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 12), for_loop_var_1098))
        # SSA begins for a for statement (line 409)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 410):
        
        # Assigning a Call to a Name (line 410):
        
        # Call to get(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'opt' (line 410)
        opt_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 46), 'opt', False)
        # Processing the call keyword arguments (line 410)
        kwargs_1103 = {}
        # Getting the type of 'self' (line 410)
        self_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'self', False)
        # Obtaining the member 'negative_opt' of a type (line 410)
        negative_opt_1100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 24), self_1099, 'negative_opt')
        # Obtaining the member 'get' of a type (line 410)
        get_1101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 24), negative_opt_1100, 'get')
        # Calling get(args, kwargs) (line 410)
        get_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 410, 24), get_1101, *[opt_1102], **kwargs_1103)
        
        # Assigning a type to the variable 'alias' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'alias', get_call_result_1104)
        
        
        # SSA begins for try-except statement (line 411)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Getting the type of 'alias' (line 412)
        alias_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), 'alias')
        # Testing the type of an if condition (line 412)
        if_condition_1106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 412, 20), alias_1105)
        # Assigning a type to the variable 'if_condition_1106' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'if_condition_1106', if_condition_1106)
        # SSA begins for if statement (line 412)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'self' (line 413)
        self_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'self', False)
        # Getting the type of 'alias' (line 413)
        alias_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 38), 'alias', False)
        
        
        # Call to strtobool(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'val' (line 413)
        val_1111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 59), 'val', False)
        # Processing the call keyword arguments (line 413)
        kwargs_1112 = {}
        # Getting the type of 'strtobool' (line 413)
        strtobool_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 49), 'strtobool', False)
        # Calling strtobool(args, kwargs) (line 413)
        strtobool_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 413, 49), strtobool_1110, *[val_1111], **kwargs_1112)
        
        # Applying the 'not' unary operator (line 413)
        result_not__1114 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 45), 'not', strtobool_call_result_1113)
        
        # Processing the call keyword arguments (line 413)
        kwargs_1115 = {}
        # Getting the type of 'setattr' (line 413)
        setattr_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 24), 'setattr', False)
        # Calling setattr(args, kwargs) (line 413)
        setattr_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 413, 24), setattr_1107, *[self_1108, alias_1109, result_not__1114], **kwargs_1115)
        
        # SSA branch for the else part of an if statement (line 412)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'opt' (line 414)
        opt_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 25), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_1118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        str_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 33), 'str', 'verbose')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 33), tuple_1118, str_1119)
        # Adding element type (line 414)
        str_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 44), 'str', 'dry_run')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 33), tuple_1118, str_1120)
        
        # Applying the binary operator 'in' (line 414)
        result_contains_1121 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 25), 'in', opt_1117, tuple_1118)
        
        # Testing the type of an if condition (line 414)
        if_condition_1122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 25), result_contains_1121)
        # Assigning a type to the variable 'if_condition_1122' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 25), 'if_condition_1122', if_condition_1122)
        # SSA begins for if statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'self' (line 415)
        self_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 32), 'self', False)
        # Getting the type of 'opt' (line 415)
        opt_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 38), 'opt', False)
        
        # Call to strtobool(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'val' (line 415)
        val_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 53), 'val', False)
        # Processing the call keyword arguments (line 415)
        kwargs_1128 = {}
        # Getting the type of 'strtobool' (line 415)
        strtobool_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 43), 'strtobool', False)
        # Calling strtobool(args, kwargs) (line 415)
        strtobool_call_result_1129 = invoke(stypy.reporting.localization.Localization(__file__, 415, 43), strtobool_1126, *[val_1127], **kwargs_1128)
        
        # Processing the call keyword arguments (line 415)
        kwargs_1130 = {}
        # Getting the type of 'setattr' (line 415)
        setattr_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 24), 'setattr', False)
        # Calling setattr(args, kwargs) (line 415)
        setattr_call_result_1131 = invoke(stypy.reporting.localization.Localization(__file__, 415, 24), setattr_1123, *[self_1124, opt_1125, strtobool_call_result_1129], **kwargs_1130)
        
        # SSA branch for the else part of an if statement (line 414)
        module_type_store.open_ssa_branch('else')
        
        # Call to setattr(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'self' (line 417)
        self_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 32), 'self', False)
        # Getting the type of 'opt' (line 417)
        opt_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 38), 'opt', False)
        # Getting the type of 'val' (line 417)
        val_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 43), 'val', False)
        # Processing the call keyword arguments (line 417)
        kwargs_1136 = {}
        # Getting the type of 'setattr' (line 417)
        setattr_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 24), 'setattr', False)
        # Calling setattr(args, kwargs) (line 417)
        setattr_call_result_1137 = invoke(stypy.reporting.localization.Localization(__file__, 417, 24), setattr_1132, *[self_1133, opt_1134, val_1135], **kwargs_1136)
        
        # SSA join for if statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 412)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 411)
        # SSA branch for the except 'ValueError' branch of a try statement (line 411)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ValueError' (line 418)
        ValueError_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'ValueError')
        # Assigning a type to the variable 'msg' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'msg', ValueError_1138)
        # Getting the type of 'DistutilsOptionError' (line 419)
        DistutilsOptionError_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 419, 20), DistutilsOptionError_1139, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 411)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 408)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'parse_config_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse_config_files' in the type store
        # Getting the type of 'stypy_return_type' (line 377)
        stypy_return_type_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse_config_files'
        return stypy_return_type_1140


    @norecursion
    def parse_command_line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'parse_command_line'
        module_type_store = module_type_store.open_function_context('parse_command_line', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.parse_command_line.__dict__.__setitem__('stypy_localization', localization)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_function_name', 'Distribution.parse_command_line')
        Distribution.parse_command_line.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.parse_command_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.parse_command_line.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.parse_command_line', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse_command_line', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse_command_line(...)' code ##################

        str_1141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, (-1)), 'str', 'Parse the setup script\'s command line, taken from the\n        \'script_args\' instance attribute (which defaults to \'sys.argv[1:]\'\n        -- see \'setup()\' in core.py).  This list is first processed for\n        "global options" -- options that set attributes of the Distribution\n        instance.  Then, it is alternately scanned for Distutils commands\n        and options for that command.  Each new command terminates the\n        options for the previous command.  The allowed options for a\n        command are determined by the \'user_options\' attribute of the\n        command class -- thus, we have to be able to load command classes\n        in order to parse the command line.  Any error in that \'options\'\n        attribute raises DistutilsGetoptError; any error on the\n        command-line raises DistutilsArgError.  If no Distutils commands\n        were found on the command line, raises DistutilsArgError.  Return\n        true if command-line was successfully parsed and we should carry\n        on with executing commands; false if no errors but we shouldn\'t\n        execute commands (currently, this only happens if user asks for\n        help).\n        ')
        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Call to _get_toplevel_options(...): (line 446)
        # Processing the call keyword arguments (line 446)
        kwargs_1144 = {}
        # Getting the type of 'self' (line 446)
        self_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 27), 'self', False)
        # Obtaining the member '_get_toplevel_options' of a type (line 446)
        _get_toplevel_options_1143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 27), self_1142, '_get_toplevel_options')
        # Calling _get_toplevel_options(args, kwargs) (line 446)
        _get_toplevel_options_call_result_1145 = invoke(stypy.reporting.localization.Localization(__file__, 446, 27), _get_toplevel_options_1143, *[], **kwargs_1144)
        
        # Assigning a type to the variable 'toplevel_options' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'toplevel_options', _get_toplevel_options_call_result_1145)
        
        # Assigning a List to a Attribute (line 455):
        
        # Assigning a List to a Attribute (line 455):
        
        # Obtaining an instance of the builtin type 'list' (line 455)
        list_1146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 455)
        
        # Getting the type of 'self' (line 455)
        self_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member 'commands' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_1147, 'commands', list_1146)
        
        # Assigning a Call to a Name (line 456):
        
        # Assigning a Call to a Name (line 456):
        
        # Call to FancyGetopt(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'toplevel_options' (line 456)
        toplevel_options_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'toplevel_options', False)
        # Getting the type of 'self' (line 456)
        self_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 48), 'self', False)
        # Obtaining the member 'display_options' of a type (line 456)
        display_options_1151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 48), self_1150, 'display_options')
        # Applying the binary operator '+' (line 456)
        result_add_1152 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 29), '+', toplevel_options_1149, display_options_1151)
        
        # Processing the call keyword arguments (line 456)
        kwargs_1153 = {}
        # Getting the type of 'FancyGetopt' (line 456)
        FancyGetopt_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'FancyGetopt', False)
        # Calling FancyGetopt(args, kwargs) (line 456)
        FancyGetopt_call_result_1154 = invoke(stypy.reporting.localization.Localization(__file__, 456, 17), FancyGetopt_1148, *[result_add_1152], **kwargs_1153)
        
        # Assigning a type to the variable 'parser' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'parser', FancyGetopt_call_result_1154)
        
        # Call to set_negative_aliases(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'self' (line 457)
        self_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 36), 'self', False)
        # Obtaining the member 'negative_opt' of a type (line 457)
        negative_opt_1158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 36), self_1157, 'negative_opt')
        # Processing the call keyword arguments (line 457)
        kwargs_1159 = {}
        # Getting the type of 'parser' (line 457)
        parser_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'parser', False)
        # Obtaining the member 'set_negative_aliases' of a type (line 457)
        set_negative_aliases_1156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), parser_1155, 'set_negative_aliases')
        # Calling set_negative_aliases(args, kwargs) (line 457)
        set_negative_aliases_call_result_1160 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), set_negative_aliases_1156, *[negative_opt_1158], **kwargs_1159)
        
        
        # Call to set_aliases(...): (line 458)
        # Processing the call arguments (line 458)
        
        # Obtaining an instance of the builtin type 'dict' (line 458)
        dict_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 458)
        # Adding element type (key, value) (line 458)
        str_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 28), 'str', 'licence')
        str_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 39), 'str', 'license')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 27), dict_1163, (str_1164, str_1165))
        
        # Processing the call keyword arguments (line 458)
        kwargs_1166 = {}
        # Getting the type of 'parser' (line 458)
        parser_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'parser', False)
        # Obtaining the member 'set_aliases' of a type (line 458)
        set_aliases_1162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), parser_1161, 'set_aliases')
        # Calling set_aliases(args, kwargs) (line 458)
        set_aliases_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), set_aliases_1162, *[dict_1163], **kwargs_1166)
        
        
        # Assigning a Call to a Name (line 459):
        
        # Assigning a Call to a Name (line 459):
        
        # Call to getopt(...): (line 459)
        # Processing the call keyword arguments (line 459)
        # Getting the type of 'self' (line 459)
        self_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 34), 'self', False)
        # Obtaining the member 'script_args' of a type (line 459)
        script_args_1171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 34), self_1170, 'script_args')
        keyword_1172 = script_args_1171
        # Getting the type of 'self' (line 459)
        self_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 59), 'self', False)
        keyword_1174 = self_1173
        kwargs_1175 = {'args': keyword_1172, 'object': keyword_1174}
        # Getting the type of 'parser' (line 459)
        parser_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'parser', False)
        # Obtaining the member 'getopt' of a type (line 459)
        getopt_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 15), parser_1168, 'getopt')
        # Calling getopt(args, kwargs) (line 459)
        getopt_call_result_1176 = invoke(stypy.reporting.localization.Localization(__file__, 459, 15), getopt_1169, *[], **kwargs_1175)
        
        # Assigning a type to the variable 'args' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'args', getopt_call_result_1176)
        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Call to get_option_order(...): (line 460)
        # Processing the call keyword arguments (line 460)
        kwargs_1179 = {}
        # Getting the type of 'parser' (line 460)
        parser_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'parser', False)
        # Obtaining the member 'get_option_order' of a type (line 460)
        get_option_order_1178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 23), parser_1177, 'get_option_order')
        # Calling get_option_order(args, kwargs) (line 460)
        get_option_order_call_result_1180 = invoke(stypy.reporting.localization.Localization(__file__, 460, 23), get_option_order_1178, *[], **kwargs_1179)
        
        # Assigning a type to the variable 'option_order' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'option_order', get_option_order_call_result_1180)
        
        # Call to set_verbosity(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'self' (line 461)
        self_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 26), 'self', False)
        # Obtaining the member 'verbose' of a type (line 461)
        verbose_1184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 26), self_1183, 'verbose')
        # Processing the call keyword arguments (line 461)
        kwargs_1185 = {}
        # Getting the type of 'log' (line 461)
        log_1181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'log', False)
        # Obtaining the member 'set_verbosity' of a type (line 461)
        set_verbosity_1182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), log_1181, 'set_verbosity')
        # Calling set_verbosity(args, kwargs) (line 461)
        set_verbosity_call_result_1186 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), set_verbosity_1182, *[verbose_1184], **kwargs_1185)
        
        
        
        # Call to handle_display_options(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'option_order' (line 464)
        option_order_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 39), 'option_order', False)
        # Processing the call keyword arguments (line 464)
        kwargs_1190 = {}
        # Getting the type of 'self' (line 464)
        self_1187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'self', False)
        # Obtaining the member 'handle_display_options' of a type (line 464)
        handle_display_options_1188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 11), self_1187, 'handle_display_options')
        # Calling handle_display_options(args, kwargs) (line 464)
        handle_display_options_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 464, 11), handle_display_options_1188, *[option_order_1189], **kwargs_1190)
        
        # Testing the type of an if condition (line 464)
        if_condition_1192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 8), handle_display_options_call_result_1191)
        # Assigning a type to the variable 'if_condition_1192' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'if_condition_1192', if_condition_1192)
        # SSA begins for if statement (line 464)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 464)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'args' (line 466)
        args_1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 14), 'args')
        # Testing the type of an if condition (line 466)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 8), args_1193)
        # SSA begins for while statement (line 466)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to _parse_command_opts(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'parser' (line 467)
        parser_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 44), 'parser', False)
        # Getting the type of 'args' (line 467)
        args_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 52), 'args', False)
        # Processing the call keyword arguments (line 467)
        kwargs_1198 = {}
        # Getting the type of 'self' (line 467)
        self_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'self', False)
        # Obtaining the member '_parse_command_opts' of a type (line 467)
        _parse_command_opts_1195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), self_1194, '_parse_command_opts')
        # Calling _parse_command_opts(args, kwargs) (line 467)
        _parse_command_opts_call_result_1199 = invoke(stypy.reporting.localization.Localization(__file__, 467, 19), _parse_command_opts_1195, *[parser_1196, args_1197], **kwargs_1198)
        
        # Assigning a type to the variable 'args' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'args', _parse_command_opts_call_result_1199)
        
        # Type idiom detected: calculating its left and rigth part (line 468)
        # Getting the type of 'args' (line 468)
        args_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), 'args')
        # Getting the type of 'None' (line 468)
        None_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 23), 'None')
        
        (may_be_1202, more_types_in_union_1203) = may_be_none(args_1200, None_1201)

        if may_be_1202:

            if more_types_in_union_1203:
                # Runtime conditional SSA (line 468)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 469)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'stypy_return_type', types.NoneType)

            if more_types_in_union_1203:
                # SSA join for if statement (line 468)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for while statement (line 466)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 477)
        self_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'self')
        # Obtaining the member 'help' of a type (line 477)
        help_1205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 11), self_1204, 'help')
        # Testing the type of an if condition (line 477)
        if_condition_1206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 8), help_1205)
        # Assigning a type to the variable 'if_condition_1206' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'if_condition_1206', if_condition_1206)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _show_help(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'parser' (line 478)
        parser_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 28), 'parser', False)
        # Processing the call keyword arguments (line 478)
        
        
        # Call to len(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'self' (line 479)
        self_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 48), 'self', False)
        # Obtaining the member 'commands' of a type (line 479)
        commands_1212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 48), self_1211, 'commands')
        # Processing the call keyword arguments (line 479)
        kwargs_1213 = {}
        # Getting the type of 'len' (line 479)
        len_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 44), 'len', False)
        # Calling len(args, kwargs) (line 479)
        len_call_result_1214 = invoke(stypy.reporting.localization.Localization(__file__, 479, 44), len_1210, *[commands_1212], **kwargs_1213)
        
        int_1215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 66), 'int')
        # Applying the binary operator '==' (line 479)
        result_eq_1216 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 44), '==', len_call_result_1214, int_1215)
        
        keyword_1217 = result_eq_1216
        # Getting the type of 'self' (line 480)
        self_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 37), 'self', False)
        # Obtaining the member 'commands' of a type (line 480)
        commands_1219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 37), self_1218, 'commands')
        keyword_1220 = commands_1219
        kwargs_1221 = {'display_options': keyword_1217, 'commands': keyword_1220}
        # Getting the type of 'self' (line 478)
        self_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '_show_help' of a type (line 478)
        _show_help_1208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_1207, '_show_help')
        # Calling _show_help(args, kwargs) (line 478)
        _show_help_call_result_1222 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), _show_help_1208, *[parser_1209], **kwargs_1221)
        
        # Assigning a type to the variable 'stypy_return_type' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 484)
        self_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 15), 'self')
        # Obtaining the member 'commands' of a type (line 484)
        commands_1224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 15), self_1223, 'commands')
        # Applying the 'not' unary operator (line 484)
        result_not__1225 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 11), 'not', commands_1224)
        
        # Testing the type of an if condition (line 484)
        if_condition_1226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 8), result_not__1225)
        # Assigning a type to the variable 'if_condition_1226' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'if_condition_1226', if_condition_1226)
        # SSA begins for if statement (line 484)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsArgError' (line 485)
        DistutilsArgError_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'DistutilsArgError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 485, 12), DistutilsArgError_1227, 'raise parameter', BaseException)
        # SSA join for if statement (line 484)
        module_type_store = module_type_store.join_ssa_context()
        
        int_1228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'stypy_return_type', int_1228)
        
        # ################# End of 'parse_command_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse_command_line' in the type store
        # Getting the type of 'stypy_return_type' (line 423)
        stypy_return_type_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1229)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse_command_line'
        return stypy_return_type_1229


    @norecursion
    def _get_toplevel_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toplevel_options'
        module_type_store = module_type_store.open_function_context('_get_toplevel_options', 490, 4, False)
        # Assigning a type to the variable 'self' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_localization', localization)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_function_name', 'Distribution._get_toplevel_options')
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution._get_toplevel_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution._get_toplevel_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_toplevel_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_toplevel_options(...)' code ##################

        str_1230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, (-1)), 'str', 'Return the non-display options recognized at the top level.\n\n        This includes options that are recognized *only* at the top\n        level as well as options recognized for commands.\n        ')
        # Getting the type of 'self' (line 496)
        self_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'self')
        # Obtaining the member 'global_options' of a type (line 496)
        global_options_1232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 15), self_1231, 'global_options')
        
        # Obtaining an instance of the builtin type 'list' (line 496)
        list_1233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 496)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'tuple' (line 497)
        tuple_1234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 497)
        # Adding element type (line 497)
        str_1235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 13), 'str', 'command-packages=')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), tuple_1234, str_1235)
        # Adding element type (line 497)
        # Getting the type of 'None' (line 497)
        None_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 34), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), tuple_1234, None_1236)
        # Adding element type (line 497)
        str_1237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 13), 'str', 'list of packages that provide distutils commands')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), tuple_1234, str_1237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 37), list_1233, tuple_1234)
        
        # Applying the binary operator '+' (line 496)
        result_add_1238 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 15), '+', global_options_1232, list_1233)
        
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', result_add_1238)
        
        # ################# End of '_get_toplevel_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toplevel_options' in the type store
        # Getting the type of 'stypy_return_type' (line 490)
        stypy_return_type_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toplevel_options'
        return stypy_return_type_1239


    @norecursion
    def _parse_command_opts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_command_opts'
        module_type_store = module_type_store.open_function_context('_parse_command_opts', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_localization', localization)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_function_name', 'Distribution._parse_command_opts')
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_param_names_list', ['parser', 'args'])
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution._parse_command_opts.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution._parse_command_opts', ['parser', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_command_opts', localization, ['parser', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_command_opts(...)' code ##################

        str_1240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, (-1)), 'str', "Parse the command-line options for a single command.\n        'parser' must be a FancyGetopt instance; 'args' must be the list\n        of arguments, starting with the current command (whose options\n        we are about to parse).  Returns a new version of 'args' with\n        the next command at the front of the list; will be the empty\n        list if there are no more commands on the command line.  Returns\n        None if the user asked for help on this command.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 511, 8))
        
        # 'from distutils.cmd import Command' statement (line 511)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1241 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 511, 8), 'distutils.cmd')

        if (type(import_1241) is not StypyTypeError):

            if (import_1241 != 'pyd_module'):
                __import__(import_1241)
                sys_modules_1242 = sys.modules[import_1241]
                import_from_module(stypy.reporting.localization.Localization(__file__, 511, 8), 'distutils.cmd', sys_modules_1242.module_type_store, module_type_store, ['Command'])
                nest_module(stypy.reporting.localization.Localization(__file__, 511, 8), __file__, sys_modules_1242, sys_modules_1242.module_type_store, module_type_store)
            else:
                from distutils.cmd import Command

                import_from_module(stypy.reporting.localization.Localization(__file__, 511, 8), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

        else:
            # Assigning a type to the variable 'distutils.cmd' (line 511)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'distutils.cmd', import_1241)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Assigning a Subscript to a Name (line 514):
        
        # Assigning a Subscript to a Name (line 514):
        
        # Obtaining the type of the subscript
        int_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 23), 'int')
        # Getting the type of 'args' (line 514)
        args_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 18), 'args')
        # Obtaining the member '__getitem__' of a type (line 514)
        getitem___1245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 18), args_1244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 514)
        subscript_call_result_1246 = invoke(stypy.reporting.localization.Localization(__file__, 514, 18), getitem___1245, int_1243)
        
        # Assigning a type to the variable 'command' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'command', subscript_call_result_1246)
        
        
        
        # Call to match(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 'command' (line 515)
        command_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 32), 'command', False)
        # Processing the call keyword arguments (line 515)
        kwargs_1250 = {}
        # Getting the type of 'command_re' (line 515)
        command_re_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 15), 'command_re', False)
        # Obtaining the member 'match' of a type (line 515)
        match_1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 15), command_re_1247, 'match')
        # Calling match(args, kwargs) (line 515)
        match_call_result_1251 = invoke(stypy.reporting.localization.Localization(__file__, 515, 15), match_1248, *[command_1249], **kwargs_1250)
        
        # Applying the 'not' unary operator (line 515)
        result_not__1252 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 11), 'not', match_call_result_1251)
        
        # Testing the type of an if condition (line 515)
        if_condition_1253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 8), result_not__1252)
        # Assigning a type to the variable 'if_condition_1253' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'if_condition_1253', if_condition_1253)
        # SSA begins for if statement (line 515)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'SystemExit' (line 516)
        SystemExit_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 18), 'SystemExit')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 516, 12), SystemExit_1254, 'raise parameter', BaseException)
        # SSA join for if statement (line 515)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'command' (line 517)
        command_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 29), 'command', False)
        # Processing the call keyword arguments (line 517)
        kwargs_1259 = {}
        # Getting the type of 'self' (line 517)
        self_1255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'self', False)
        # Obtaining the member 'commands' of a type (line 517)
        commands_1256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 8), self_1255, 'commands')
        # Obtaining the member 'append' of a type (line 517)
        append_1257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 8), commands_1256, 'append')
        # Calling append(args, kwargs) (line 517)
        append_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 517, 8), append_1257, *[command_1258], **kwargs_1259)
        
        
        
        # SSA begins for try-except statement (line 522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 523):
        
        # Assigning a Call to a Name (line 523):
        
        # Call to get_command_class(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'command' (line 523)
        command_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 47), 'command', False)
        # Processing the call keyword arguments (line 523)
        kwargs_1264 = {}
        # Getting the type of 'self' (line 523)
        self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'self', False)
        # Obtaining the member 'get_command_class' of a type (line 523)
        get_command_class_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 24), self_1261, 'get_command_class')
        # Calling get_command_class(args, kwargs) (line 523)
        get_command_class_call_result_1265 = invoke(stypy.reporting.localization.Localization(__file__, 523, 24), get_command_class_1262, *[command_1263], **kwargs_1264)
        
        # Assigning a type to the variable 'cmd_class' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'cmd_class', get_command_class_call_result_1265)
        # SSA branch for the except part of a try statement (line 522)
        # SSA branch for the except 'DistutilsModuleError' branch of a try statement (line 522)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsModuleError' (line 524)
        DistutilsModuleError_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 15), 'DistutilsModuleError')
        # Assigning a type to the variable 'msg' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'msg', DistutilsModuleError_1266)
        # Getting the type of 'DistutilsArgError' (line 525)
        DistutilsArgError_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 18), 'DistutilsArgError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 525, 12), DistutilsArgError_1267, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 522)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to issubclass(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'cmd_class' (line 529)
        cmd_class_1269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 26), 'cmd_class', False)
        # Getting the type of 'Command' (line 529)
        Command_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 37), 'Command', False)
        # Processing the call keyword arguments (line 529)
        kwargs_1271 = {}
        # Getting the type of 'issubclass' (line 529)
        issubclass_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 529)
        issubclass_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 529, 15), issubclass_1268, *[cmd_class_1269, Command_1270], **kwargs_1271)
        
        # Applying the 'not' unary operator (line 529)
        result_not__1273 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 11), 'not', issubclass_call_result_1272)
        
        # Testing the type of an if condition (line 529)
        if_condition_1274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 8), result_not__1273)
        # Assigning a type to the variable 'if_condition_1274' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'if_condition_1274', if_condition_1274)
        # SSA begins for if statement (line 529)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsClassError' (line 530)
        DistutilsClassError_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 18), 'DistutilsClassError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 530, 12), DistutilsClassError_1275, 'raise parameter', BaseException)
        # SSA join for if statement (line 529)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'cmd_class' (line 535)
        cmd_class_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 24), 'cmd_class', False)
        str_1278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 35), 'str', 'user_options')
        # Processing the call keyword arguments (line 535)
        kwargs_1279 = {}
        # Getting the type of 'hasattr' (line 535)
        hasattr_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 16), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 535)
        hasattr_call_result_1280 = invoke(stypy.reporting.localization.Localization(__file__, 535, 16), hasattr_1276, *[cmd_class_1277, str_1278], **kwargs_1279)
        
        
        # Call to isinstance(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'cmd_class' (line 536)
        cmd_class_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 27), 'cmd_class', False)
        # Obtaining the member 'user_options' of a type (line 536)
        user_options_1283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 27), cmd_class_1282, 'user_options')
        # Getting the type of 'list' (line 536)
        list_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 51), 'list', False)
        # Processing the call keyword arguments (line 536)
        kwargs_1285 = {}
        # Getting the type of 'isinstance' (line 536)
        isinstance_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 536)
        isinstance_call_result_1286 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), isinstance_1281, *[user_options_1283, list_1284], **kwargs_1285)
        
        # Applying the binary operator 'and' (line 535)
        result_and_keyword_1287 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 16), 'and', hasattr_call_result_1280, isinstance_call_result_1286)
        
        # Applying the 'not' unary operator (line 535)
        result_not__1288 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 11), 'not', result_and_keyword_1287)
        
        # Testing the type of an if condition (line 535)
        if_condition_1289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 8), result_not__1288)
        # Assigning a type to the variable 'if_condition_1289' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'if_condition_1289', if_condition_1289)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsClassError' (line 537)
        DistutilsClassError_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 18), 'DistutilsClassError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 537, 12), DistutilsClassError_1290, 'raise parameter', BaseException)
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 544):
        
        # Assigning a Attribute to a Name (line 544):
        # Getting the type of 'self' (line 544)
        self_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 23), 'self')
        # Obtaining the member 'negative_opt' of a type (line 544)
        negative_opt_1292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 23), self_1291, 'negative_opt')
        # Assigning a type to the variable 'negative_opt' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'negative_opt', negative_opt_1292)
        
        # Type idiom detected: calculating its left and rigth part (line 545)
        str_1293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 30), 'str', 'negative_opt')
        # Getting the type of 'cmd_class' (line 545)
        cmd_class_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'cmd_class')
        
        (may_be_1295, more_types_in_union_1296) = may_provide_member(str_1293, cmd_class_1294)

        if may_be_1295:

            if more_types_in_union_1296:
                # Runtime conditional SSA (line 545)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'cmd_class' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'cmd_class', remove_not_member_provider_from_union(cmd_class_1294, 'negative_opt'))
            
            # Assigning a Call to a Name (line 546):
            
            # Assigning a Call to a Name (line 546):
            
            # Call to copy(...): (line 546)
            # Processing the call keyword arguments (line 546)
            kwargs_1299 = {}
            # Getting the type of 'negative_opt' (line 546)
            negative_opt_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 27), 'negative_opt', False)
            # Obtaining the member 'copy' of a type (line 546)
            copy_1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 27), negative_opt_1297, 'copy')
            # Calling copy(args, kwargs) (line 546)
            copy_call_result_1300 = invoke(stypy.reporting.localization.Localization(__file__, 546, 27), copy_1298, *[], **kwargs_1299)
            
            # Assigning a type to the variable 'negative_opt' (line 546)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'negative_opt', copy_call_result_1300)
            
            # Call to update(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'cmd_class' (line 547)
            cmd_class_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 32), 'cmd_class', False)
            # Obtaining the member 'negative_opt' of a type (line 547)
            negative_opt_1304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 32), cmd_class_1303, 'negative_opt')
            # Processing the call keyword arguments (line 547)
            kwargs_1305 = {}
            # Getting the type of 'negative_opt' (line 547)
            negative_opt_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'negative_opt', False)
            # Obtaining the member 'update' of a type (line 547)
            update_1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 12), negative_opt_1301, 'update')
            # Calling update(args, kwargs) (line 547)
            update_call_result_1306 = invoke(stypy.reporting.localization.Localization(__file__, 547, 12), update_1302, *[negative_opt_1304], **kwargs_1305)
            

            if more_types_in_union_1296:
                # SSA join for if statement (line 545)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'cmd_class' (line 551)
        cmd_class_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'cmd_class', False)
        str_1309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 31), 'str', 'help_options')
        # Processing the call keyword arguments (line 551)
        kwargs_1310 = {}
        # Getting the type of 'hasattr' (line 551)
        hasattr_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 551)
        hasattr_call_result_1311 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), hasattr_1307, *[cmd_class_1308, str_1309], **kwargs_1310)
        
        
        # Call to isinstance(...): (line 552)
        # Processing the call arguments (line 552)
        # Getting the type of 'cmd_class' (line 552)
        cmd_class_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 23), 'cmd_class', False)
        # Obtaining the member 'help_options' of a type (line 552)
        help_options_1314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 23), cmd_class_1313, 'help_options')
        # Getting the type of 'list' (line 552)
        list_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 47), 'list', False)
        # Processing the call keyword arguments (line 552)
        kwargs_1316 = {}
        # Getting the type of 'isinstance' (line 552)
        isinstance_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 552)
        isinstance_call_result_1317 = invoke(stypy.reporting.localization.Localization(__file__, 552, 12), isinstance_1312, *[help_options_1314, list_1315], **kwargs_1316)
        
        # Applying the binary operator 'and' (line 551)
        result_and_keyword_1318 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 12), 'and', hasattr_call_result_1311, isinstance_call_result_1317)
        
        # Testing the type of an if condition (line 551)
        if_condition_1319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 8), result_and_keyword_1318)
        # Assigning a type to the variable 'if_condition_1319' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'if_condition_1319', if_condition_1319)
        # SSA begins for if statement (line 551)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to fix_help_options(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'cmd_class' (line 553)
        cmd_class_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 44), 'cmd_class', False)
        # Obtaining the member 'help_options' of a type (line 553)
        help_options_1322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 44), cmd_class_1321, 'help_options')
        # Processing the call keyword arguments (line 553)
        kwargs_1323 = {}
        # Getting the type of 'fix_help_options' (line 553)
        fix_help_options_1320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 27), 'fix_help_options', False)
        # Calling fix_help_options(args, kwargs) (line 553)
        fix_help_options_call_result_1324 = invoke(stypy.reporting.localization.Localization(__file__, 553, 27), fix_help_options_1320, *[help_options_1322], **kwargs_1323)
        
        # Assigning a type to the variable 'help_options' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'help_options', fix_help_options_call_result_1324)
        # SSA branch for the else part of an if statement (line 551)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 555):
        
        # Assigning a List to a Name (line 555):
        
        # Obtaining an instance of the builtin type 'list' (line 555)
        list_1325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 555)
        
        # Assigning a type to the variable 'help_options' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'help_options', list_1325)
        # SSA join for if statement (line 551)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_option_table(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'self' (line 560)
        self_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 32), 'self', False)
        # Obtaining the member 'global_options' of a type (line 560)
        global_options_1329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 32), self_1328, 'global_options')
        # Getting the type of 'cmd_class' (line 561)
        cmd_class_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 32), 'cmd_class', False)
        # Obtaining the member 'user_options' of a type (line 561)
        user_options_1331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 32), cmd_class_1330, 'user_options')
        # Applying the binary operator '+' (line 560)
        result_add_1332 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 32), '+', global_options_1329, user_options_1331)
        
        # Getting the type of 'help_options' (line 562)
        help_options_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'help_options', False)
        # Applying the binary operator '+' (line 561)
        result_add_1334 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 55), '+', result_add_1332, help_options_1333)
        
        # Processing the call keyword arguments (line 560)
        kwargs_1335 = {}
        # Getting the type of 'parser' (line 560)
        parser_1326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'parser', False)
        # Obtaining the member 'set_option_table' of a type (line 560)
        set_option_table_1327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), parser_1326, 'set_option_table')
        # Calling set_option_table(args, kwargs) (line 560)
        set_option_table_call_result_1336 = invoke(stypy.reporting.localization.Localization(__file__, 560, 8), set_option_table_1327, *[result_add_1334], **kwargs_1335)
        
        
        # Call to set_negative_aliases(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'negative_opt' (line 563)
        negative_opt_1339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 36), 'negative_opt', False)
        # Processing the call keyword arguments (line 563)
        kwargs_1340 = {}
        # Getting the type of 'parser' (line 563)
        parser_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'parser', False)
        # Obtaining the member 'set_negative_aliases' of a type (line 563)
        set_negative_aliases_1338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 8), parser_1337, 'set_negative_aliases')
        # Calling set_negative_aliases(args, kwargs) (line 563)
        set_negative_aliases_call_result_1341 = invoke(stypy.reporting.localization.Localization(__file__, 563, 8), set_negative_aliases_1338, *[negative_opt_1339], **kwargs_1340)
        
        
        # Assigning a Call to a Tuple (line 564):
        
        # Assigning a Subscript to a Name (line 564):
        
        # Obtaining the type of the subscript
        int_1342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        
        # Call to getopt(...): (line 564)
        # Processing the call arguments (line 564)
        
        # Obtaining the type of the subscript
        int_1345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 42), 'int')
        slice_1346 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 564, 37), int_1345, None, None)
        # Getting the type of 'args' (line 564)
        args_1347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 37), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___1348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 37), args_1347, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_1349 = invoke(stypy.reporting.localization.Localization(__file__, 564, 37), getitem___1348, slice_1346)
        
        # Processing the call keyword arguments (line 564)
        kwargs_1350 = {}
        # Getting the type of 'parser' (line 564)
        parser_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 23), 'parser', False)
        # Obtaining the member 'getopt' of a type (line 564)
        getopt_1344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 23), parser_1343, 'getopt')
        # Calling getopt(args, kwargs) (line 564)
        getopt_call_result_1351 = invoke(stypy.reporting.localization.Localization(__file__, 564, 23), getopt_1344, *[subscript_call_result_1349], **kwargs_1350)
        
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___1352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), getopt_call_result_1351, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_1353 = invoke(stypy.reporting.localization.Localization(__file__, 564, 8), getitem___1352, int_1342)
        
        # Assigning a type to the variable 'tuple_var_assignment_528' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_528', subscript_call_result_1353)
        
        # Assigning a Subscript to a Name (line 564):
        
        # Obtaining the type of the subscript
        int_1354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        
        # Call to getopt(...): (line 564)
        # Processing the call arguments (line 564)
        
        # Obtaining the type of the subscript
        int_1357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 42), 'int')
        slice_1358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 564, 37), int_1357, None, None)
        # Getting the type of 'args' (line 564)
        args_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 37), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___1360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 37), args_1359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_1361 = invoke(stypy.reporting.localization.Localization(__file__, 564, 37), getitem___1360, slice_1358)
        
        # Processing the call keyword arguments (line 564)
        kwargs_1362 = {}
        # Getting the type of 'parser' (line 564)
        parser_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 23), 'parser', False)
        # Obtaining the member 'getopt' of a type (line 564)
        getopt_1356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 23), parser_1355, 'getopt')
        # Calling getopt(args, kwargs) (line 564)
        getopt_call_result_1363 = invoke(stypy.reporting.localization.Localization(__file__, 564, 23), getopt_1356, *[subscript_call_result_1361], **kwargs_1362)
        
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___1364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), getopt_call_result_1363, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_1365 = invoke(stypy.reporting.localization.Localization(__file__, 564, 8), getitem___1364, int_1354)
        
        # Assigning a type to the variable 'tuple_var_assignment_529' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_529', subscript_call_result_1365)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'tuple_var_assignment_528' (line 564)
        tuple_var_assignment_528_1366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_528')
        # Assigning a type to the variable 'args' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 9), 'args', tuple_var_assignment_528_1366)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'tuple_var_assignment_529' (line 564)
        tuple_var_assignment_529_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_529')
        # Assigning a type to the variable 'opts' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'opts', tuple_var_assignment_529_1367)
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'opts' (line 565)
        opts_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 19), 'opts', False)
        str_1370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 25), 'str', 'help')
        # Processing the call keyword arguments (line 565)
        kwargs_1371 = {}
        # Getting the type of 'hasattr' (line 565)
        hasattr_1368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 565)
        hasattr_call_result_1372 = invoke(stypy.reporting.localization.Localization(__file__, 565, 11), hasattr_1368, *[opts_1369, str_1370], **kwargs_1371)
        
        # Getting the type of 'opts' (line 565)
        opts_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 37), 'opts')
        # Obtaining the member 'help' of a type (line 565)
        help_1374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 37), opts_1373, 'help')
        # Applying the binary operator 'and' (line 565)
        result_and_keyword_1375 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 11), 'and', hasattr_call_result_1372, help_1374)
        
        # Testing the type of an if condition (line 565)
        if_condition_1376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 8), result_and_keyword_1375)
        # Assigning a type to the variable 'if_condition_1376' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'if_condition_1376', if_condition_1376)
        # SSA begins for if statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _show_help(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'parser' (line 566)
        parser_1379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 28), 'parser', False)
        # Processing the call keyword arguments (line 566)
        int_1380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 52), 'int')
        keyword_1381 = int_1380
        
        # Obtaining an instance of the builtin type 'list' (line 566)
        list_1382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 566)
        # Adding element type (line 566)
        # Getting the type of 'cmd_class' (line 566)
        cmd_class_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 65), 'cmd_class', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 64), list_1382, cmd_class_1383)
        
        keyword_1384 = list_1382
        kwargs_1385 = {'display_options': keyword_1381, 'commands': keyword_1384}
        # Getting the type of 'self' (line 566)
        self_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'self', False)
        # Obtaining the member '_show_help' of a type (line 566)
        _show_help_1378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), self_1377, '_show_help')
        # Calling _show_help(args, kwargs) (line 566)
        _show_help_call_result_1386 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), _show_help_1378, *[parser_1379], **kwargs_1385)
        
        # Assigning a type to the variable 'stypy_return_type' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'cmd_class' (line 569)
        cmd_class_1388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'cmd_class', False)
        str_1389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 31), 'str', 'help_options')
        # Processing the call keyword arguments (line 569)
        kwargs_1390 = {}
        # Getting the type of 'hasattr' (line 569)
        hasattr_1387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 569)
        hasattr_call_result_1391 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), hasattr_1387, *[cmd_class_1388, str_1389], **kwargs_1390)
        
        
        # Call to isinstance(...): (line 570)
        # Processing the call arguments (line 570)
        # Getting the type of 'cmd_class' (line 570)
        cmd_class_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'cmd_class', False)
        # Obtaining the member 'help_options' of a type (line 570)
        help_options_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 23), cmd_class_1393, 'help_options')
        # Getting the type of 'list' (line 570)
        list_1395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 47), 'list', False)
        # Processing the call keyword arguments (line 570)
        kwargs_1396 = {}
        # Getting the type of 'isinstance' (line 570)
        isinstance_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 570)
        isinstance_call_result_1397 = invoke(stypy.reporting.localization.Localization(__file__, 570, 12), isinstance_1392, *[help_options_1394, list_1395], **kwargs_1396)
        
        # Applying the binary operator 'and' (line 569)
        result_and_keyword_1398 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 12), 'and', hasattr_call_result_1391, isinstance_call_result_1397)
        
        # Testing the type of an if condition (line 569)
        if_condition_1399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 8), result_and_keyword_1398)
        # Assigning a type to the variable 'if_condition_1399' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'if_condition_1399', if_condition_1399)
        # SSA begins for if statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 571):
        
        # Assigning a Num to a Name (line 571):
        int_1400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 30), 'int')
        # Assigning a type to the variable 'help_option_found' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'help_option_found', int_1400)
        
        # Getting the type of 'cmd_class' (line 572)
        cmd_class_1401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 52), 'cmd_class')
        # Obtaining the member 'help_options' of a type (line 572)
        help_options_1402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 52), cmd_class_1401, 'help_options')
        # Testing the type of a for loop iterable (line 572)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 572, 12), help_options_1402)
        # Getting the type of the for loop variable (line 572)
        for_loop_var_1403 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 572, 12), help_options_1402)
        # Assigning a type to the variable 'help_option' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'help_option', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 12), for_loop_var_1403))
        # Assigning a type to the variable 'short' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'short', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 12), for_loop_var_1403))
        # Assigning a type to the variable 'desc' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'desc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 12), for_loop_var_1403))
        # Assigning a type to the variable 'func' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 12), for_loop_var_1403))
        # SSA begins for a for statement (line 572)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to hasattr(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'opts' (line 573)
        opts_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'opts', False)
        
        # Call to get_attr_name(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'help_option' (line 573)
        help_option_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 54), 'help_option', False)
        # Processing the call keyword arguments (line 573)
        kwargs_1409 = {}
        # Getting the type of 'parser' (line 573)
        parser_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 33), 'parser', False)
        # Obtaining the member 'get_attr_name' of a type (line 573)
        get_attr_name_1407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 33), parser_1406, 'get_attr_name')
        # Calling get_attr_name(args, kwargs) (line 573)
        get_attr_name_call_result_1410 = invoke(stypy.reporting.localization.Localization(__file__, 573, 33), get_attr_name_1407, *[help_option_1408], **kwargs_1409)
        
        # Processing the call keyword arguments (line 573)
        kwargs_1411 = {}
        # Getting the type of 'hasattr' (line 573)
        hasattr_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 19), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 573)
        hasattr_call_result_1412 = invoke(stypy.reporting.localization.Localization(__file__, 573, 19), hasattr_1404, *[opts_1405, get_attr_name_call_result_1410], **kwargs_1411)
        
        # Testing the type of an if condition (line 573)
        if_condition_1413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 16), hasattr_call_result_1412)
        # Assigning a type to the variable 'if_condition_1413' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'if_condition_1413', if_condition_1413)
        # SSA begins for if statement (line 573)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 574):
        
        # Assigning a Num to a Name (line 574):
        int_1414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 38), 'int')
        # Assigning a type to the variable 'help_option_found' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 20), 'help_option_found', int_1414)
        
        # Type idiom detected: calculating its left and rigth part (line 575)
        str_1415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 37), 'str', '__call__')
        # Getting the type of 'func' (line 575)
        func_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 31), 'func')
        
        (may_be_1417, more_types_in_union_1418) = may_provide_member(str_1415, func_1416)

        if may_be_1417:

            if more_types_in_union_1418:
                # Runtime conditional SSA (line 575)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'func' (line 575)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 20), 'func', remove_not_member_provider_from_union(func_1416, '__call__'))
            
            # Call to func(...): (line 576)
            # Processing the call keyword arguments (line 576)
            kwargs_1420 = {}
            # Getting the type of 'func' (line 576)
            func_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 24), 'func', False)
            # Calling func(args, kwargs) (line 576)
            func_call_result_1421 = invoke(stypy.reporting.localization.Localization(__file__, 576, 24), func_1419, *[], **kwargs_1420)
            

            if more_types_in_union_1418:
                # Runtime conditional SSA for else branch (line 575)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_1417) or more_types_in_union_1418):
            # Assigning a type to the variable 'func' (line 575)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 20), 'func', remove_member_provider_from_union(func_1416, '__call__'))
            
            # Call to DistutilsClassError(...): (line 578)
            # Processing the call arguments (line 578)
            str_1423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 28), 'str', "invalid help function %r for help option '%s': must be a callable object (function, etc.)")
            
            # Obtaining an instance of the builtin type 'tuple' (line 581)
            tuple_1424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 581)
            # Adding element type (line 581)
            # Getting the type of 'func' (line 581)
            func_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 31), 'func', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 31), tuple_1424, func_1425)
            # Adding element type (line 581)
            # Getting the type of 'help_option' (line 581)
            help_option_1426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 37), 'help_option', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 31), tuple_1424, help_option_1426)
            
            # Applying the binary operator '%' (line 579)
            result_mod_1427 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 28), '%', str_1423, tuple_1424)
            
            # Processing the call keyword arguments (line 578)
            kwargs_1428 = {}
            # Getting the type of 'DistutilsClassError' (line 578)
            DistutilsClassError_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 30), 'DistutilsClassError', False)
            # Calling DistutilsClassError(args, kwargs) (line 578)
            DistutilsClassError_call_result_1429 = invoke(stypy.reporting.localization.Localization(__file__, 578, 30), DistutilsClassError_1422, *[result_mod_1427], **kwargs_1428)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 578, 24), DistutilsClassError_call_result_1429, 'raise parameter', BaseException)

            if (may_be_1417 and more_types_in_union_1418):
                # SSA join for if statement (line 575)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 573)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'help_option_found' (line 583)
        help_option_found_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'help_option_found')
        # Testing the type of an if condition (line 583)
        if_condition_1431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 12), help_option_found_1430)
        # Assigning a type to the variable 'if_condition_1431' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'if_condition_1431', if_condition_1431)
        # SSA begins for if statement (line 583)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 583)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 569)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 588):
        
        # Assigning a Call to a Name (line 588):
        
        # Call to get_option_dict(...): (line 588)
        # Processing the call arguments (line 588)
        # Getting the type of 'command' (line 588)
        command_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 40), 'command', False)
        # Processing the call keyword arguments (line 588)
        kwargs_1435 = {}
        # Getting the type of 'self' (line 588)
        self_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 19), 'self', False)
        # Obtaining the member 'get_option_dict' of a type (line 588)
        get_option_dict_1433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 19), self_1432, 'get_option_dict')
        # Calling get_option_dict(args, kwargs) (line 588)
        get_option_dict_call_result_1436 = invoke(stypy.reporting.localization.Localization(__file__, 588, 19), get_option_dict_1433, *[command_1434], **kwargs_1435)
        
        # Assigning a type to the variable 'opt_dict' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'opt_dict', get_option_dict_call_result_1436)
        
        
        # Call to items(...): (line 589)
        # Processing the call keyword arguments (line 589)
        kwargs_1442 = {}
        
        # Call to vars(...): (line 589)
        # Processing the call arguments (line 589)
        # Getting the type of 'opts' (line 589)
        opts_1438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), 'opts', False)
        # Processing the call keyword arguments (line 589)
        kwargs_1439 = {}
        # Getting the type of 'vars' (line 589)
        vars_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 29), 'vars', False)
        # Calling vars(args, kwargs) (line 589)
        vars_call_result_1440 = invoke(stypy.reporting.localization.Localization(__file__, 589, 29), vars_1437, *[opts_1438], **kwargs_1439)
        
        # Obtaining the member 'items' of a type (line 589)
        items_1441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 29), vars_call_result_1440, 'items')
        # Calling items(args, kwargs) (line 589)
        items_call_result_1443 = invoke(stypy.reporting.localization.Localization(__file__, 589, 29), items_1441, *[], **kwargs_1442)
        
        # Testing the type of a for loop iterable (line 589)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 589, 8), items_call_result_1443)
        # Getting the type of the for loop variable (line 589)
        for_loop_var_1444 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 589, 8), items_call_result_1443)
        # Assigning a type to the variable 'name' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 8), for_loop_var_1444))
        # Assigning a type to the variable 'value' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 8), for_loop_var_1444))
        # SSA begins for a for statement (line 589)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Subscript (line 590):
        
        # Assigning a Tuple to a Subscript (line 590):
        
        # Obtaining an instance of the builtin type 'tuple' (line 590)
        tuple_1445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 590)
        # Adding element type (line 590)
        str_1446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'str', 'command line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 30), tuple_1445, str_1446)
        # Adding element type (line 590)
        # Getting the type of 'value' (line 590)
        value_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 46), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 30), tuple_1445, value_1447)
        
        # Getting the type of 'opt_dict' (line 590)
        opt_dict_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'opt_dict')
        # Getting the type of 'name' (line 590)
        name_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 21), 'name')
        # Storing an element on a container (line 590)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), opt_dict_1448, (name_1449, tuple_1445))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'args' (line 592)
        args_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'args')
        # Assigning a type to the variable 'stypy_return_type' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'stypy_return_type', args_1450)
        
        # ################# End of '_parse_command_opts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_command_opts' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1451)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_command_opts'
        return stypy_return_type_1451


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 594, 4, False)
        # Assigning a type to the variable 'self' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        Distribution.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.finalize_options.__dict__.__setitem__('stypy_function_name', 'Distribution.finalize_options')
        Distribution.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        str_1452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, (-1)), 'str', 'Set final values for all the options on the Distribution\n        instance, analogous to the .finalize_options() method of Command\n        objects.\n        ')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 599)
        tuple_1453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 599)
        # Adding element type (line 599)
        str_1454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 21), 'str', 'keywords')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 21), tuple_1453, str_1454)
        # Adding element type (line 599)
        str_1455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 33), 'str', 'platforms')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 21), tuple_1453, str_1455)
        
        # Testing the type of a for loop iterable (line 599)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 599, 8), tuple_1453)
        # Getting the type of the for loop variable (line 599)
        for_loop_var_1456 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 599, 8), tuple_1453)
        # Assigning a type to the variable 'attr' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'attr', for_loop_var_1456)
        # SSA begins for a for statement (line 599)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 600):
        
        # Assigning a Call to a Name (line 600):
        
        # Call to getattr(...): (line 600)
        # Processing the call arguments (line 600)
        # Getting the type of 'self' (line 600)
        self_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 28), 'self', False)
        # Obtaining the member 'metadata' of a type (line 600)
        metadata_1459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 28), self_1458, 'metadata')
        # Getting the type of 'attr' (line 600)
        attr_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 43), 'attr', False)
        # Processing the call keyword arguments (line 600)
        kwargs_1461 = {}
        # Getting the type of 'getattr' (line 600)
        getattr_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 600)
        getattr_call_result_1462 = invoke(stypy.reporting.localization.Localization(__file__, 600, 20), getattr_1457, *[metadata_1459, attr_1460], **kwargs_1461)
        
        # Assigning a type to the variable 'value' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'value', getattr_call_result_1462)
        
        # Type idiom detected: calculating its left and rigth part (line 601)
        # Getting the type of 'value' (line 601)
        value_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 15), 'value')
        # Getting the type of 'None' (line 601)
        None_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 24), 'None')
        
        (may_be_1465, more_types_in_union_1466) = may_be_none(value_1463, None_1464)

        if may_be_1465:

            if more_types_in_union_1466:
                # Runtime conditional SSA (line 601)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_1466:
                # SSA join for if statement (line 601)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 603)
        # Getting the type of 'str' (line 603)
        str_1467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 33), 'str')
        # Getting the type of 'value' (line 603)
        value_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 26), 'value')
        
        (may_be_1469, more_types_in_union_1470) = may_be_subtype(str_1467, value_1468)

        if may_be_1469:

            if more_types_in_union_1470:
                # Runtime conditional SSA (line 603)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'value', remove_not_subtype_from_union(value_1468, str))
            
            # Assigning a ListComp to a Name (line 604):
            
            # Assigning a ListComp to a Name (line 604):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to split(...): (line 604)
            # Processing the call arguments (line 604)
            str_1477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 60), 'str', ',')
            # Processing the call keyword arguments (line 604)
            kwargs_1478 = {}
            # Getting the type of 'value' (line 604)
            value_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 48), 'value', False)
            # Obtaining the member 'split' of a type (line 604)
            split_1476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 48), value_1475, 'split')
            # Calling split(args, kwargs) (line 604)
            split_call_result_1479 = invoke(stypy.reporting.localization.Localization(__file__, 604, 48), split_1476, *[str_1477], **kwargs_1478)
            
            comprehension_1480 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 25), split_call_result_1479)
            # Assigning a type to the variable 'elm' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 25), 'elm', comprehension_1480)
            
            # Call to strip(...): (line 604)
            # Processing the call keyword arguments (line 604)
            kwargs_1473 = {}
            # Getting the type of 'elm' (line 604)
            elm_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 25), 'elm', False)
            # Obtaining the member 'strip' of a type (line 604)
            strip_1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 25), elm_1471, 'strip')
            # Calling strip(args, kwargs) (line 604)
            strip_call_result_1474 = invoke(stypy.reporting.localization.Localization(__file__, 604, 25), strip_1472, *[], **kwargs_1473)
            
            list_1481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 25), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 25), list_1481, strip_call_result_1474)
            # Assigning a type to the variable 'value' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'value', list_1481)
            
            # Call to setattr(...): (line 605)
            # Processing the call arguments (line 605)
            # Getting the type of 'self' (line 605)
            self_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 24), 'self', False)
            # Obtaining the member 'metadata' of a type (line 605)
            metadata_1484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 24), self_1483, 'metadata')
            # Getting the type of 'attr' (line 605)
            attr_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 39), 'attr', False)
            # Getting the type of 'value' (line 605)
            value_1486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 45), 'value', False)
            # Processing the call keyword arguments (line 605)
            kwargs_1487 = {}
            # Getting the type of 'setattr' (line 605)
            setattr_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 605)
            setattr_call_result_1488 = invoke(stypy.reporting.localization.Localization(__file__, 605, 16), setattr_1482, *[metadata_1484, attr_1485, value_1486], **kwargs_1487)
            

            if more_types_in_union_1470:
                # SSA join for if statement (line 603)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 594)
        stypy_return_type_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_1489


    @norecursion
    def _show_help(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_1490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 48), 'int')
        int_1491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 67), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 608)
        list_1492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 608)
        
        defaults = [int_1490, int_1491, list_1492]
        # Create a new context for function '_show_help'
        module_type_store = module_type_store.open_function_context('_show_help', 607, 4, False)
        # Assigning a type to the variable 'self' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution._show_help.__dict__.__setitem__('stypy_localization', localization)
        Distribution._show_help.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution._show_help.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution._show_help.__dict__.__setitem__('stypy_function_name', 'Distribution._show_help')
        Distribution._show_help.__dict__.__setitem__('stypy_param_names_list', ['parser', 'global_options', 'display_options', 'commands'])
        Distribution._show_help.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution._show_help.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution._show_help.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution._show_help.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution._show_help.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution._show_help.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution._show_help', ['parser', 'global_options', 'display_options', 'commands'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_show_help', localization, ['parser', 'global_options', 'display_options', 'commands'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_show_help(...)' code ##################

        str_1493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, (-1)), 'str', 'Show help for the setup script command-line in the form of\n        several lists of command-line options.  \'parser\' should be a\n        FancyGetopt instance; do not expect it to be returned in the\n        same state, as its option table will be reset to make it\n        generate the correct help text.\n\n        If \'global_options\' is true, lists the global options:\n        --verbose, --dry-run, etc.  If \'display_options\' is true, lists\n        the "display-only" options: --name, --version, etc.  Finally,\n        lists per-command help for every command name or command class\n        in \'commands\'.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 622, 8))
        
        # 'from distutils.core import gen_usage' statement (line 622)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1494 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 622, 8), 'distutils.core')

        if (type(import_1494) is not StypyTypeError):

            if (import_1494 != 'pyd_module'):
                __import__(import_1494)
                sys_modules_1495 = sys.modules[import_1494]
                import_from_module(stypy.reporting.localization.Localization(__file__, 622, 8), 'distutils.core', sys_modules_1495.module_type_store, module_type_store, ['gen_usage'])
                nest_module(stypy.reporting.localization.Localization(__file__, 622, 8), __file__, sys_modules_1495, sys_modules_1495.module_type_store, module_type_store)
            else:
                from distutils.core import gen_usage

                import_from_module(stypy.reporting.localization.Localization(__file__, 622, 8), 'distutils.core', None, module_type_store, ['gen_usage'], [gen_usage])

        else:
            # Assigning a type to the variable 'distutils.core' (line 622)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'distutils.core', import_1494)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 623, 8))
        
        # 'from distutils.cmd import Command' statement (line 623)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1496 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 623, 8), 'distutils.cmd')

        if (type(import_1496) is not StypyTypeError):

            if (import_1496 != 'pyd_module'):
                __import__(import_1496)
                sys_modules_1497 = sys.modules[import_1496]
                import_from_module(stypy.reporting.localization.Localization(__file__, 623, 8), 'distutils.cmd', sys_modules_1497.module_type_store, module_type_store, ['Command'])
                nest_module(stypy.reporting.localization.Localization(__file__, 623, 8), __file__, sys_modules_1497, sys_modules_1497.module_type_store, module_type_store)
            else:
                from distutils.cmd import Command

                import_from_module(stypy.reporting.localization.Localization(__file__, 623, 8), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

        else:
            # Assigning a type to the variable 'distutils.cmd' (line 623)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'distutils.cmd', import_1496)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'global_options' (line 625)
        global_options_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), 'global_options')
        # Testing the type of an if condition (line 625)
        if_condition_1499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 8), global_options_1498)
        # Assigning a type to the variable 'if_condition_1499' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'if_condition_1499', if_condition_1499)
        # SSA begins for if statement (line 625)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'display_options' (line 626)
        display_options_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 15), 'display_options')
        # Testing the type of an if condition (line 626)
        if_condition_1501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 12), display_options_1500)
        # Assigning a type to the variable 'if_condition_1501' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'if_condition_1501', if_condition_1501)
        # SSA begins for if statement (line 626)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 627):
        
        # Assigning a Call to a Name (line 627):
        
        # Call to _get_toplevel_options(...): (line 627)
        # Processing the call keyword arguments (line 627)
        kwargs_1504 = {}
        # Getting the type of 'self' (line 627)
        self_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 26), 'self', False)
        # Obtaining the member '_get_toplevel_options' of a type (line 627)
        _get_toplevel_options_1503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 26), self_1502, '_get_toplevel_options')
        # Calling _get_toplevel_options(args, kwargs) (line 627)
        _get_toplevel_options_call_result_1505 = invoke(stypy.reporting.localization.Localization(__file__, 627, 26), _get_toplevel_options_1503, *[], **kwargs_1504)
        
        # Assigning a type to the variable 'options' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'options', _get_toplevel_options_call_result_1505)
        # SSA branch for the else part of an if statement (line 626)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 629):
        
        # Assigning a Attribute to a Name (line 629):
        # Getting the type of 'self' (line 629)
        self_1506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 26), 'self')
        # Obtaining the member 'global_options' of a type (line 629)
        global_options_1507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 26), self_1506, 'global_options')
        # Assigning a type to the variable 'options' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 'options', global_options_1507)
        # SSA join for if statement (line 626)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_option_table(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'options' (line 630)
        options_1510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 36), 'options', False)
        # Processing the call keyword arguments (line 630)
        kwargs_1511 = {}
        # Getting the type of 'parser' (line 630)
        parser_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'parser', False)
        # Obtaining the member 'set_option_table' of a type (line 630)
        set_option_table_1509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 12), parser_1508, 'set_option_table')
        # Calling set_option_table(args, kwargs) (line 630)
        set_option_table_call_result_1512 = invoke(stypy.reporting.localization.Localization(__file__, 630, 12), set_option_table_1509, *[options_1510], **kwargs_1511)
        
        
        # Call to print_help(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'self' (line 631)
        self_1515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 30), 'self', False)
        # Obtaining the member 'common_usage' of a type (line 631)
        common_usage_1516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 30), self_1515, 'common_usage')
        str_1517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 50), 'str', '\nGlobal options:')
        # Applying the binary operator '+' (line 631)
        result_add_1518 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 30), '+', common_usage_1516, str_1517)
        
        # Processing the call keyword arguments (line 631)
        kwargs_1519 = {}
        # Getting the type of 'parser' (line 631)
        parser_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'parser', False)
        # Obtaining the member 'print_help' of a type (line 631)
        print_help_1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 12), parser_1513, 'print_help')
        # Calling print_help(args, kwargs) (line 631)
        print_help_call_result_1520 = invoke(stypy.reporting.localization.Localization(__file__, 631, 12), print_help_1514, *[result_add_1518], **kwargs_1519)
        
        str_1521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 18), 'str', '')
        # SSA join for if statement (line 625)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'display_options' (line 634)
        display_options_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 11), 'display_options')
        # Testing the type of an if condition (line 634)
        if_condition_1523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 634, 8), display_options_1522)
        # Assigning a type to the variable 'if_condition_1523' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'if_condition_1523', if_condition_1523)
        # SSA begins for if statement (line 634)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_option_table(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'self' (line 635)
        self_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 36), 'self', False)
        # Obtaining the member 'display_options' of a type (line 635)
        display_options_1527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 36), self_1526, 'display_options')
        # Processing the call keyword arguments (line 635)
        kwargs_1528 = {}
        # Getting the type of 'parser' (line 635)
        parser_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'parser', False)
        # Obtaining the member 'set_option_table' of a type (line 635)
        set_option_table_1525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 12), parser_1524, 'set_option_table')
        # Calling set_option_table(args, kwargs) (line 635)
        set_option_table_call_result_1529 = invoke(stypy.reporting.localization.Localization(__file__, 635, 12), set_option_table_1525, *[display_options_1527], **kwargs_1528)
        
        
        # Call to print_help(...): (line 636)
        # Processing the call arguments (line 636)
        str_1532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 16), 'str', 'Information display options (just display ')
        str_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 16), 'str', 'information, ignore any commands)')
        # Applying the binary operator '+' (line 637)
        result_add_1534 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 16), '+', str_1532, str_1533)
        
        # Processing the call keyword arguments (line 636)
        kwargs_1535 = {}
        # Getting the type of 'parser' (line 636)
        parser_1530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'parser', False)
        # Obtaining the member 'print_help' of a type (line 636)
        print_help_1531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), parser_1530, 'print_help')
        # Calling print_help(args, kwargs) (line 636)
        print_help_call_result_1536 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), print_help_1531, *[result_add_1534], **kwargs_1535)
        
        str_1537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 18), 'str', '')
        # SSA join for if statement (line 634)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 641)
        self_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 23), 'self')
        # Obtaining the member 'commands' of a type (line 641)
        commands_1539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 23), self_1538, 'commands')
        # Testing the type of a for loop iterable (line 641)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 641, 8), commands_1539)
        # Getting the type of the for loop variable (line 641)
        for_loop_var_1540 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 641, 8), commands_1539)
        # Assigning a type to the variable 'command' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'command', for_loop_var_1540)
        # SSA begins for a for statement (line 641)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 642)
        # Processing the call arguments (line 642)
        # Getting the type of 'command' (line 642)
        command_1542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 26), 'command', False)
        # Getting the type of 'type' (line 642)
        type_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 35), 'type', False)
        # Processing the call keyword arguments (line 642)
        kwargs_1544 = {}
        # Getting the type of 'isinstance' (line 642)
        isinstance_1541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 642)
        isinstance_call_result_1545 = invoke(stypy.reporting.localization.Localization(__file__, 642, 15), isinstance_1541, *[command_1542, type_1543], **kwargs_1544)
        
        
        # Call to issubclass(...): (line 642)
        # Processing the call arguments (line 642)
        # Getting the type of 'command' (line 642)
        command_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 56), 'command', False)
        # Getting the type of 'Command' (line 642)
        Command_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 65), 'Command', False)
        # Processing the call keyword arguments (line 642)
        kwargs_1549 = {}
        # Getting the type of 'issubclass' (line 642)
        issubclass_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 45), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 642)
        issubclass_call_result_1550 = invoke(stypy.reporting.localization.Localization(__file__, 642, 45), issubclass_1546, *[command_1547, Command_1548], **kwargs_1549)
        
        # Applying the binary operator 'and' (line 642)
        result_and_keyword_1551 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 15), 'and', isinstance_call_result_1545, issubclass_call_result_1550)
        
        # Testing the type of an if condition (line 642)
        if_condition_1552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 642, 12), result_and_keyword_1551)
        # Assigning a type to the variable 'if_condition_1552' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'if_condition_1552', if_condition_1552)
        # SSA begins for if statement (line 642)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 643):
        
        # Assigning a Name to a Name (line 643):
        # Getting the type of 'command' (line 643)
        command_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 24), 'command')
        # Assigning a type to the variable 'klass' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 16), 'klass', command_1553)
        # SSA branch for the else part of an if statement (line 642)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 645):
        
        # Assigning a Call to a Name (line 645):
        
        # Call to get_command_class(...): (line 645)
        # Processing the call arguments (line 645)
        # Getting the type of 'command' (line 645)
        command_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 47), 'command', False)
        # Processing the call keyword arguments (line 645)
        kwargs_1557 = {}
        # Getting the type of 'self' (line 645)
        self_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 24), 'self', False)
        # Obtaining the member 'get_command_class' of a type (line 645)
        get_command_class_1555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 24), self_1554, 'get_command_class')
        # Calling get_command_class(args, kwargs) (line 645)
        get_command_class_call_result_1558 = invoke(stypy.reporting.localization.Localization(__file__, 645, 24), get_command_class_1555, *[command_1556], **kwargs_1557)
        
        # Assigning a type to the variable 'klass' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 16), 'klass', get_command_class_call_result_1558)
        # SSA join for if statement (line 642)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'klass' (line 646)
        klass_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 24), 'klass', False)
        str_1561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 31), 'str', 'help_options')
        # Processing the call keyword arguments (line 646)
        kwargs_1562 = {}
        # Getting the type of 'hasattr' (line 646)
        hasattr_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 646)
        hasattr_call_result_1563 = invoke(stypy.reporting.localization.Localization(__file__, 646, 16), hasattr_1559, *[klass_1560, str_1561], **kwargs_1562)
        
        
        # Call to isinstance(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'klass' (line 647)
        klass_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 27), 'klass', False)
        # Obtaining the member 'help_options' of a type (line 647)
        help_options_1566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 27), klass_1565, 'help_options')
        # Getting the type of 'list' (line 647)
        list_1567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 47), 'list', False)
        # Processing the call keyword arguments (line 647)
        kwargs_1568 = {}
        # Getting the type of 'isinstance' (line 647)
        isinstance_1564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 647)
        isinstance_call_result_1569 = invoke(stypy.reporting.localization.Localization(__file__, 647, 16), isinstance_1564, *[help_options_1566, list_1567], **kwargs_1568)
        
        # Applying the binary operator 'and' (line 646)
        result_and_keyword_1570 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 16), 'and', hasattr_call_result_1563, isinstance_call_result_1569)
        
        # Testing the type of an if condition (line 646)
        if_condition_1571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 12), result_and_keyword_1570)
        # Assigning a type to the variable 'if_condition_1571' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'if_condition_1571', if_condition_1571)
        # SSA begins for if statement (line 646)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_option_table(...): (line 648)
        # Processing the call arguments (line 648)
        # Getting the type of 'klass' (line 648)
        klass_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 40), 'klass', False)
        # Obtaining the member 'user_options' of a type (line 648)
        user_options_1575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 40), klass_1574, 'user_options')
        
        # Call to fix_help_options(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'klass' (line 649)
        klass_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 57), 'klass', False)
        # Obtaining the member 'help_options' of a type (line 649)
        help_options_1578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 57), klass_1577, 'help_options')
        # Processing the call keyword arguments (line 649)
        kwargs_1579 = {}
        # Getting the type of 'fix_help_options' (line 649)
        fix_help_options_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 40), 'fix_help_options', False)
        # Calling fix_help_options(args, kwargs) (line 649)
        fix_help_options_call_result_1580 = invoke(stypy.reporting.localization.Localization(__file__, 649, 40), fix_help_options_1576, *[help_options_1578], **kwargs_1579)
        
        # Applying the binary operator '+' (line 648)
        result_add_1581 = python_operator(stypy.reporting.localization.Localization(__file__, 648, 40), '+', user_options_1575, fix_help_options_call_result_1580)
        
        # Processing the call keyword arguments (line 648)
        kwargs_1582 = {}
        # Getting the type of 'parser' (line 648)
        parser_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'parser', False)
        # Obtaining the member 'set_option_table' of a type (line 648)
        set_option_table_1573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 16), parser_1572, 'set_option_table')
        # Calling set_option_table(args, kwargs) (line 648)
        set_option_table_call_result_1583 = invoke(stypy.reporting.localization.Localization(__file__, 648, 16), set_option_table_1573, *[result_add_1581], **kwargs_1582)
        
        # SSA branch for the else part of an if statement (line 646)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_option_table(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'klass' (line 651)
        klass_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 40), 'klass', False)
        # Obtaining the member 'user_options' of a type (line 651)
        user_options_1587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 40), klass_1586, 'user_options')
        # Processing the call keyword arguments (line 651)
        kwargs_1588 = {}
        # Getting the type of 'parser' (line 651)
        parser_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'parser', False)
        # Obtaining the member 'set_option_table' of a type (line 651)
        set_option_table_1585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 16), parser_1584, 'set_option_table')
        # Calling set_option_table(args, kwargs) (line 651)
        set_option_table_call_result_1589 = invoke(stypy.reporting.localization.Localization(__file__, 651, 16), set_option_table_1585, *[user_options_1587], **kwargs_1588)
        
        # SSA join for if statement (line 646)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to print_help(...): (line 652)
        # Processing the call arguments (line 652)
        str_1592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 30), 'str', "Options for '%s' command:")
        # Getting the type of 'klass' (line 652)
        klass_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 60), 'klass', False)
        # Obtaining the member '__name__' of a type (line 652)
        name___1594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 60), klass_1593, '__name__')
        # Applying the binary operator '%' (line 652)
        result_mod_1595 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 30), '%', str_1592, name___1594)
        
        # Processing the call keyword arguments (line 652)
        kwargs_1596 = {}
        # Getting the type of 'parser' (line 652)
        parser_1590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'parser', False)
        # Obtaining the member 'print_help' of a type (line 652)
        print_help_1591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 12), parser_1590, 'print_help')
        # Calling print_help(args, kwargs) (line 652)
        print_help_call_result_1597 = invoke(stypy.reporting.localization.Localization(__file__, 652, 12), print_help_1591, *[result_mod_1595], **kwargs_1596)
        
        str_1598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 18), 'str', '')
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to gen_usage(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'self' (line 655)
        self_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 24), 'self', False)
        # Obtaining the member 'script_name' of a type (line 655)
        script_name_1601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 24), self_1600, 'script_name')
        # Processing the call keyword arguments (line 655)
        kwargs_1602 = {}
        # Getting the type of 'gen_usage' (line 655)
        gen_usage_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'gen_usage', False)
        # Calling gen_usage(args, kwargs) (line 655)
        gen_usage_call_result_1603 = invoke(stypy.reporting.localization.Localization(__file__, 655, 14), gen_usage_1599, *[script_name_1601], **kwargs_1602)
        
        
        # ################# End of '_show_help(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_show_help' in the type store
        # Getting the type of 'stypy_return_type' (line 607)
        stypy_return_type_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_show_help'
        return stypy_return_type_1604


    @norecursion
    def handle_display_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_display_options'
        module_type_store = module_type_store.open_function_context('handle_display_options', 657, 4, False)
        # Assigning a type to the variable 'self' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.handle_display_options.__dict__.__setitem__('stypy_localization', localization)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_function_name', 'Distribution.handle_display_options')
        Distribution.handle_display_options.__dict__.__setitem__('stypy_param_names_list', ['option_order'])
        Distribution.handle_display_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.handle_display_options.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.handle_display_options', ['option_order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_display_options', localization, ['option_order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_display_options(...)' code ##################

        str_1605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, (-1)), 'str', 'If there were any non-global "display-only" options\n        (--help-commands or the metadata display options) on the command\n        line, display the requested info and return true; else return\n        false.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 663, 8))
        
        # 'from distutils.core import gen_usage' statement (line 663)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1606 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 663, 8), 'distutils.core')

        if (type(import_1606) is not StypyTypeError):

            if (import_1606 != 'pyd_module'):
                __import__(import_1606)
                sys_modules_1607 = sys.modules[import_1606]
                import_from_module(stypy.reporting.localization.Localization(__file__, 663, 8), 'distutils.core', sys_modules_1607.module_type_store, module_type_store, ['gen_usage'])
                nest_module(stypy.reporting.localization.Localization(__file__, 663, 8), __file__, sys_modules_1607, sys_modules_1607.module_type_store, module_type_store)
            else:
                from distutils.core import gen_usage

                import_from_module(stypy.reporting.localization.Localization(__file__, 663, 8), 'distutils.core', None, module_type_store, ['gen_usage'], [gen_usage])

        else:
            # Assigning a type to the variable 'distutils.core' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'distutils.core', import_1606)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'self' (line 668)
        self_1608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 11), 'self')
        # Obtaining the member 'help_commands' of a type (line 668)
        help_commands_1609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 11), self_1608, 'help_commands')
        # Testing the type of an if condition (line 668)
        if_condition_1610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 8), help_commands_1609)
        # Assigning a type to the variable 'if_condition_1610' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'if_condition_1610', if_condition_1610)
        # SSA begins for if statement (line 668)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print_commands(...): (line 669)
        # Processing the call keyword arguments (line 669)
        kwargs_1613 = {}
        # Getting the type of 'self' (line 669)
        self_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'self', False)
        # Obtaining the member 'print_commands' of a type (line 669)
        print_commands_1612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 12), self_1611, 'print_commands')
        # Calling print_commands(args, kwargs) (line 669)
        print_commands_call_result_1614 = invoke(stypy.reporting.localization.Localization(__file__, 669, 12), print_commands_1612, *[], **kwargs_1613)
        
        str_1615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 18), 'str', '')
        
        # Call to gen_usage(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'self' (line 671)
        self_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 28), 'self', False)
        # Obtaining the member 'script_name' of a type (line 671)
        script_name_1618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 28), self_1617, 'script_name')
        # Processing the call keyword arguments (line 671)
        kwargs_1619 = {}
        # Getting the type of 'gen_usage' (line 671)
        gen_usage_1616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 18), 'gen_usage', False)
        # Calling gen_usage(args, kwargs) (line 671)
        gen_usage_call_result_1620 = invoke(stypy.reporting.localization.Localization(__file__, 671, 18), gen_usage_1616, *[script_name_1618], **kwargs_1619)
        
        int_1621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'stypy_return_type', int_1621)
        # SSA join for if statement (line 668)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 677):
        
        # Assigning a Num to a Name (line 677):
        int_1622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 30), 'int')
        # Assigning a type to the variable 'any_display_options' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'any_display_options', int_1622)
        
        # Assigning a Dict to a Name (line 678):
        
        # Assigning a Dict to a Name (line 678):
        
        # Obtaining an instance of the builtin type 'dict' (line 678)
        dict_1623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 678)
        
        # Assigning a type to the variable 'is_display_option' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'is_display_option', dict_1623)
        
        # Getting the type of 'self' (line 679)
        self_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 22), 'self')
        # Obtaining the member 'display_options' of a type (line 679)
        display_options_1625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 22), self_1624, 'display_options')
        # Testing the type of a for loop iterable (line 679)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 679, 8), display_options_1625)
        # Getting the type of the for loop variable (line 679)
        for_loop_var_1626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 679, 8), display_options_1625)
        # Assigning a type to the variable 'option' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'option', for_loop_var_1626)
        # SSA begins for a for statement (line 679)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 680):
        
        # Assigning a Num to a Subscript (line 680):
        int_1627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 43), 'int')
        # Getting the type of 'is_display_option' (line 680)
        is_display_option_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'is_display_option')
        
        # Obtaining the type of the subscript
        int_1629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 37), 'int')
        # Getting the type of 'option' (line 680)
        option_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 30), 'option')
        # Obtaining the member '__getitem__' of a type (line 680)
        getitem___1631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 30), option_1630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 680)
        subscript_call_result_1632 = invoke(stypy.reporting.localization.Localization(__file__, 680, 30), getitem___1631, int_1629)
        
        # Storing an element on a container (line 680)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 12), is_display_option_1628, (subscript_call_result_1632, int_1627))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'option_order' (line 682)
        option_order_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 26), 'option_order')
        # Testing the type of a for loop iterable (line 682)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 682, 8), option_order_1633)
        # Getting the type of the for loop variable (line 682)
        for_loop_var_1634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 682, 8), option_order_1633)
        # Assigning a type to the variable 'opt' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 8), for_loop_var_1634))
        # Assigning a type to the variable 'val' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 8), for_loop_var_1634))
        # SSA begins for a for statement (line 682)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'val' (line 683)
        val_1635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'val')
        
        # Call to get(...): (line 683)
        # Processing the call arguments (line 683)
        # Getting the type of 'opt' (line 683)
        opt_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 45), 'opt', False)
        # Processing the call keyword arguments (line 683)
        kwargs_1639 = {}
        # Getting the type of 'is_display_option' (line 683)
        is_display_option_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 23), 'is_display_option', False)
        # Obtaining the member 'get' of a type (line 683)
        get_1637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 23), is_display_option_1636, 'get')
        # Calling get(args, kwargs) (line 683)
        get_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 683, 23), get_1637, *[opt_1638], **kwargs_1639)
        
        # Applying the binary operator 'and' (line 683)
        result_and_keyword_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 15), 'and', val_1635, get_call_result_1640)
        
        # Testing the type of an if condition (line 683)
        if_condition_1642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 12), result_and_keyword_1641)
        # Assigning a type to the variable 'if_condition_1642' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'if_condition_1642', if_condition_1642)
        # SSA begins for if statement (line 683)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 684):
        
        # Assigning a Call to a Name (line 684):
        
        # Call to translate_longopt(...): (line 684)
        # Processing the call arguments (line 684)
        # Getting the type of 'opt' (line 684)
        opt_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 40), 'opt', False)
        # Processing the call keyword arguments (line 684)
        kwargs_1645 = {}
        # Getting the type of 'translate_longopt' (line 684)
        translate_longopt_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 22), 'translate_longopt', False)
        # Calling translate_longopt(args, kwargs) (line 684)
        translate_longopt_call_result_1646 = invoke(stypy.reporting.localization.Localization(__file__, 684, 22), translate_longopt_1643, *[opt_1644], **kwargs_1645)
        
        # Assigning a type to the variable 'opt' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'opt', translate_longopt_call_result_1646)
        
        # Assigning a Call to a Name (line 685):
        
        # Assigning a Call to a Name (line 685):
        
        # Call to (...): (line 685)
        # Processing the call keyword arguments (line 685)
        kwargs_1655 = {}
        
        # Call to getattr(...): (line 685)
        # Processing the call arguments (line 685)
        # Getting the type of 'self' (line 685)
        self_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 32), 'self', False)
        # Obtaining the member 'metadata' of a type (line 685)
        metadata_1649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 32), self_1648, 'metadata')
        str_1650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 47), 'str', 'get_')
        # Getting the type of 'opt' (line 685)
        opt_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 54), 'opt', False)
        # Applying the binary operator '+' (line 685)
        result_add_1652 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 47), '+', str_1650, opt_1651)
        
        # Processing the call keyword arguments (line 685)
        kwargs_1653 = {}
        # Getting the type of 'getattr' (line 685)
        getattr_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 685)
        getattr_call_result_1654 = invoke(stypy.reporting.localization.Localization(__file__, 685, 24), getattr_1647, *[metadata_1649, result_add_1652], **kwargs_1653)
        
        # Calling (args, kwargs) (line 685)
        _call_result_1656 = invoke(stypy.reporting.localization.Localization(__file__, 685, 24), getattr_call_result_1654, *[], **kwargs_1655)
        
        # Assigning a type to the variable 'value' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'value', _call_result_1656)
        
        
        # Getting the type of 'opt' (line 686)
        opt_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'list' (line 686)
        list_1658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 686)
        # Adding element type (line 686)
        str_1659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 27), 'str', 'keywords')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 26), list_1658, str_1659)
        # Adding element type (line 686)
        str_1660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 39), 'str', 'platforms')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 26), list_1658, str_1660)
        
        # Applying the binary operator 'in' (line 686)
        result_contains_1661 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 19), 'in', opt_1657, list_1658)
        
        # Testing the type of an if condition (line 686)
        if_condition_1662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 686, 16), result_contains_1661)
        # Assigning a type to the variable 'if_condition_1662' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 16), 'if_condition_1662', if_condition_1662)
        # SSA begins for if statement (line 686)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 687)
        # Processing the call arguments (line 687)
        # Getting the type of 'value' (line 687)
        value_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 35), 'value', False)
        # Processing the call keyword arguments (line 687)
        kwargs_1666 = {}
        str_1663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 26), 'str', ',')
        # Obtaining the member 'join' of a type (line 687)
        join_1664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 26), str_1663, 'join')
        # Calling join(args, kwargs) (line 687)
        join_call_result_1667 = invoke(stypy.reporting.localization.Localization(__file__, 687, 26), join_1664, *[value_1665], **kwargs_1666)
        
        # SSA branch for the else part of an if statement (line 686)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'opt' (line 688)
        opt_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 21), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 688)
        tuple_1669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 688)
        # Adding element type (line 688)
        str_1670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 29), 'str', 'classifiers')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 29), tuple_1669, str_1670)
        # Adding element type (line 688)
        str_1671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 44), 'str', 'provides')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 29), tuple_1669, str_1671)
        # Adding element type (line 688)
        str_1672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 56), 'str', 'requires')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 29), tuple_1669, str_1672)
        # Adding element type (line 688)
        str_1673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 29), 'str', 'obsoletes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 29), tuple_1669, str_1673)
        
        # Applying the binary operator 'in' (line 688)
        result_contains_1674 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 21), 'in', opt_1668, tuple_1669)
        
        # Testing the type of an if condition (line 688)
        if_condition_1675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 21), result_contains_1674)
        # Assigning a type to the variable 'if_condition_1675' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 21), 'if_condition_1675', if_condition_1675)
        # SSA begins for if statement (line 688)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 690)
        # Processing the call arguments (line 690)
        # Getting the type of 'value' (line 690)
        value_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 36), 'value', False)
        # Processing the call keyword arguments (line 690)
        kwargs_1679 = {}
        str_1676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 26), 'str', '\n')
        # Obtaining the member 'join' of a type (line 690)
        join_1677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 26), str_1676, 'join')
        # Calling join(args, kwargs) (line 690)
        join_call_result_1680 = invoke(stypy.reporting.localization.Localization(__file__, 690, 26), join_1677, *[value_1678], **kwargs_1679)
        
        # SSA branch for the else part of an if statement (line 688)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'value' (line 692)
        value_1681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 26), 'value')
        # SSA join for if statement (line 688)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 686)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 693):
        
        # Assigning a Num to a Name (line 693):
        int_1682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 38), 'int')
        # Assigning a type to the variable 'any_display_options' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'any_display_options', int_1682)
        # SSA join for if statement (line 683)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'any_display_options' (line 695)
        any_display_options_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 15), 'any_display_options')
        # Assigning a type to the variable 'stypy_return_type' (line 695)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'stypy_return_type', any_display_options_1683)
        
        # ################# End of 'handle_display_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_display_options' in the type store
        # Getting the type of 'stypy_return_type' (line 657)
        stypy_return_type_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1684)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_display_options'
        return stypy_return_type_1684


    @norecursion
    def print_command_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_command_list'
        module_type_store = module_type_store.open_function_context('print_command_list', 697, 4, False)
        # Assigning a type to the variable 'self' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.print_command_list.__dict__.__setitem__('stypy_localization', localization)
        Distribution.print_command_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.print_command_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.print_command_list.__dict__.__setitem__('stypy_function_name', 'Distribution.print_command_list')
        Distribution.print_command_list.__dict__.__setitem__('stypy_param_names_list', ['commands', 'header', 'max_length'])
        Distribution.print_command_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.print_command_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.print_command_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.print_command_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.print_command_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.print_command_list.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.print_command_list', ['commands', 'header', 'max_length'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_command_list', localization, ['commands', 'header', 'max_length'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_command_list(...)' code ##################

        str_1685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, (-1)), 'str', "Print a subset of the list of all commands -- used by\n        'print_commands()'.\n        ")
        # Getting the type of 'header' (line 701)
        header_1686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 14), 'header')
        str_1687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 23), 'str', ':')
        # Applying the binary operator '+' (line 701)
        result_add_1688 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 14), '+', header_1686, str_1687)
        
        
        # Getting the type of 'commands' (line 703)
        commands_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 19), 'commands')
        # Testing the type of a for loop iterable (line 703)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 703, 8), commands_1689)
        # Getting the type of the for loop variable (line 703)
        for_loop_var_1690 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 703, 8), commands_1689)
        # Assigning a type to the variable 'cmd' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'cmd', for_loop_var_1690)
        # SSA begins for a for statement (line 703)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 704):
        
        # Assigning a Call to a Name (line 704):
        
        # Call to get(...): (line 704)
        # Processing the call arguments (line 704)
        # Getting the type of 'cmd' (line 704)
        cmd_1694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 38), 'cmd', False)
        # Processing the call keyword arguments (line 704)
        kwargs_1695 = {}
        # Getting the type of 'self' (line 704)
        self_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 20), 'self', False)
        # Obtaining the member 'cmdclass' of a type (line 704)
        cmdclass_1692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 20), self_1691, 'cmdclass')
        # Obtaining the member 'get' of a type (line 704)
        get_1693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 20), cmdclass_1692, 'get')
        # Calling get(args, kwargs) (line 704)
        get_call_result_1696 = invoke(stypy.reporting.localization.Localization(__file__, 704, 20), get_1693, *[cmd_1694], **kwargs_1695)
        
        # Assigning a type to the variable 'klass' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'klass', get_call_result_1696)
        
        
        # Getting the type of 'klass' (line 705)
        klass_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'klass')
        # Applying the 'not' unary operator (line 705)
        result_not__1698 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 15), 'not', klass_1697)
        
        # Testing the type of an if condition (line 705)
        if_condition_1699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 12), result_not__1698)
        # Assigning a type to the variable 'if_condition_1699' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'if_condition_1699', if_condition_1699)
        # SSA begins for if statement (line 705)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 706):
        
        # Assigning a Call to a Name (line 706):
        
        # Call to get_command_class(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'cmd' (line 706)
        cmd_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 47), 'cmd', False)
        # Processing the call keyword arguments (line 706)
        kwargs_1703 = {}
        # Getting the type of 'self' (line 706)
        self_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 24), 'self', False)
        # Obtaining the member 'get_command_class' of a type (line 706)
        get_command_class_1701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 24), self_1700, 'get_command_class')
        # Calling get_command_class(args, kwargs) (line 706)
        get_command_class_call_result_1704 = invoke(stypy.reporting.localization.Localization(__file__, 706, 24), get_command_class_1701, *[cmd_1702], **kwargs_1703)
        
        # Assigning a type to the variable 'klass' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 16), 'klass', get_command_class_call_result_1704)
        # SSA join for if statement (line 705)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 707)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 708):
        
        # Assigning a Attribute to a Name (line 708):
        # Getting the type of 'klass' (line 708)
        klass_1705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 30), 'klass')
        # Obtaining the member 'description' of a type (line 708)
        description_1706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 30), klass_1705, 'description')
        # Assigning a type to the variable 'description' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'description', description_1706)
        # SSA branch for the except part of a try statement (line 707)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 707)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Str to a Name (line 710):
        
        # Assigning a Str to a Name (line 710):
        str_1707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 30), 'str', '(no description available)')
        # Assigning a type to the variable 'description' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 16), 'description', str_1707)
        # SSA join for try-except statement (line 707)
        module_type_store = module_type_store.join_ssa_context()
        
        str_1708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 18), 'str', '  %-*s  %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 712)
        tuple_1709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 712)
        # Adding element type (line 712)
        # Getting the type of 'max_length' (line 712)
        max_length_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 34), 'max_length')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 34), tuple_1709, max_length_1710)
        # Adding element type (line 712)
        # Getting the type of 'cmd' (line 712)
        cmd_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 46), 'cmd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 34), tuple_1709, cmd_1711)
        # Adding element type (line 712)
        # Getting the type of 'description' (line 712)
        description_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 51), 'description')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 34), tuple_1709, description_1712)
        
        # Applying the binary operator '%' (line 712)
        result_mod_1713 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 18), '%', str_1708, tuple_1709)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'print_command_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_command_list' in the type store
        # Getting the type of 'stypy_return_type' (line 697)
        stypy_return_type_1714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1714)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_command_list'
        return stypy_return_type_1714


    @norecursion
    def print_commands(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_commands'
        module_type_store = module_type_store.open_function_context('print_commands', 714, 4, False)
        # Assigning a type to the variable 'self' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.print_commands.__dict__.__setitem__('stypy_localization', localization)
        Distribution.print_commands.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.print_commands.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.print_commands.__dict__.__setitem__('stypy_function_name', 'Distribution.print_commands')
        Distribution.print_commands.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.print_commands.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.print_commands.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.print_commands.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.print_commands.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.print_commands.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.print_commands.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.print_commands', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_commands', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_commands(...)' code ##################

        str_1715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, (-1)), 'str', 'Print out a help message listing all available commands with a\n        description of each.  The list is divided into "standard commands"\n        (listed in distutils.command.__all__) and "extra commands"\n        (mentioned in self.cmdclass, but not a standard command).  The\n        descriptions come from the command class attribute\n        \'description\'.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 722, 8))
        
        # 'import distutils.command' statement (line 722)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 722, 8), 'distutils.command')

        if (type(import_1716) is not StypyTypeError):

            if (import_1716 != 'pyd_module'):
                __import__(import_1716)
                sys_modules_1717 = sys.modules[import_1716]
                import_module(stypy.reporting.localization.Localization(__file__, 722, 8), 'distutils.command', sys_modules_1717.module_type_store, module_type_store)
            else:
                import distutils.command

                import_module(stypy.reporting.localization.Localization(__file__, 722, 8), 'distutils.command', distutils.command, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.command' (line 722)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'distutils.command', import_1716)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Assigning a Attribute to a Name (line 723):
        
        # Assigning a Attribute to a Name (line 723):
        # Getting the type of 'distutils' (line 723)
        distutils_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 23), 'distutils')
        # Obtaining the member 'command' of a type (line 723)
        command_1719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 23), distutils_1718, 'command')
        # Obtaining the member '__all__' of a type (line 723)
        all___1720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 23), command_1719, '__all__')
        # Assigning a type to the variable 'std_commands' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'std_commands', all___1720)
        
        # Assigning a Dict to a Name (line 724):
        
        # Assigning a Dict to a Name (line 724):
        
        # Obtaining an instance of the builtin type 'dict' (line 724)
        dict_1721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 724)
        
        # Assigning a type to the variable 'is_std' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'is_std', dict_1721)
        
        # Getting the type of 'std_commands' (line 725)
        std_commands_1722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 19), 'std_commands')
        # Testing the type of a for loop iterable (line 725)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 725, 8), std_commands_1722)
        # Getting the type of the for loop variable (line 725)
        for_loop_var_1723 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 725, 8), std_commands_1722)
        # Assigning a type to the variable 'cmd' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'cmd', for_loop_var_1723)
        # SSA begins for a for statement (line 725)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 726):
        
        # Assigning a Num to a Subscript (line 726):
        int_1724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 26), 'int')
        # Getting the type of 'is_std' (line 726)
        is_std_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 12), 'is_std')
        # Getting the type of 'cmd' (line 726)
        cmd_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 19), 'cmd')
        # Storing an element on a container (line 726)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 12), is_std_1725, (cmd_1726, int_1724))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 728):
        
        # Assigning a List to a Name (line 728):
        
        # Obtaining an instance of the builtin type 'list' (line 728)
        list_1727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 728)
        
        # Assigning a type to the variable 'extra_commands' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'extra_commands', list_1727)
        
        
        # Call to keys(...): (line 729)
        # Processing the call keyword arguments (line 729)
        kwargs_1731 = {}
        # Getting the type of 'self' (line 729)
        self_1728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 19), 'self', False)
        # Obtaining the member 'cmdclass' of a type (line 729)
        cmdclass_1729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 19), self_1728, 'cmdclass')
        # Obtaining the member 'keys' of a type (line 729)
        keys_1730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 19), cmdclass_1729, 'keys')
        # Calling keys(args, kwargs) (line 729)
        keys_call_result_1732 = invoke(stypy.reporting.localization.Localization(__file__, 729, 19), keys_1730, *[], **kwargs_1731)
        
        # Testing the type of a for loop iterable (line 729)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 729, 8), keys_call_result_1732)
        # Getting the type of the for loop variable (line 729)
        for_loop_var_1733 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 729, 8), keys_call_result_1732)
        # Assigning a type to the variable 'cmd' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'cmd', for_loop_var_1733)
        # SSA begins for a for statement (line 729)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to get(...): (line 730)
        # Processing the call arguments (line 730)
        # Getting the type of 'cmd' (line 730)
        cmd_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 30), 'cmd', False)
        # Processing the call keyword arguments (line 730)
        kwargs_1737 = {}
        # Getting the type of 'is_std' (line 730)
        is_std_1734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 19), 'is_std', False)
        # Obtaining the member 'get' of a type (line 730)
        get_1735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 19), is_std_1734, 'get')
        # Calling get(args, kwargs) (line 730)
        get_call_result_1738 = invoke(stypy.reporting.localization.Localization(__file__, 730, 19), get_1735, *[cmd_1736], **kwargs_1737)
        
        # Applying the 'not' unary operator (line 730)
        result_not__1739 = python_operator(stypy.reporting.localization.Localization(__file__, 730, 15), 'not', get_call_result_1738)
        
        # Testing the type of an if condition (line 730)
        if_condition_1740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 730, 12), result_not__1739)
        # Assigning a type to the variable 'if_condition_1740' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'if_condition_1740', if_condition_1740)
        # SSA begins for if statement (line 730)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 731)
        # Processing the call arguments (line 731)
        # Getting the type of 'cmd' (line 731)
        cmd_1743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 38), 'cmd', False)
        # Processing the call keyword arguments (line 731)
        kwargs_1744 = {}
        # Getting the type of 'extra_commands' (line 731)
        extra_commands_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 16), 'extra_commands', False)
        # Obtaining the member 'append' of a type (line 731)
        append_1742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 16), extra_commands_1741, 'append')
        # Calling append(args, kwargs) (line 731)
        append_call_result_1745 = invoke(stypy.reporting.localization.Localization(__file__, 731, 16), append_1742, *[cmd_1743], **kwargs_1744)
        
        # SSA join for if statement (line 730)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 733):
        
        # Assigning a Num to a Name (line 733):
        int_1746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 21), 'int')
        # Assigning a type to the variable 'max_length' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'max_length', int_1746)
        
        # Getting the type of 'std_commands' (line 734)
        std_commands_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 20), 'std_commands')
        # Getting the type of 'extra_commands' (line 734)
        extra_commands_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 35), 'extra_commands')
        # Applying the binary operator '+' (line 734)
        result_add_1749 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 20), '+', std_commands_1747, extra_commands_1748)
        
        # Testing the type of a for loop iterable (line 734)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 734, 8), result_add_1749)
        # Getting the type of the for loop variable (line 734)
        for_loop_var_1750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 734, 8), result_add_1749)
        # Assigning a type to the variable 'cmd' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'cmd', for_loop_var_1750)
        # SSA begins for a for statement (line 734)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to len(...): (line 735)
        # Processing the call arguments (line 735)
        # Getting the type of 'cmd' (line 735)
        cmd_1752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'cmd', False)
        # Processing the call keyword arguments (line 735)
        kwargs_1753 = {}
        # Getting the type of 'len' (line 735)
        len_1751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 15), 'len', False)
        # Calling len(args, kwargs) (line 735)
        len_call_result_1754 = invoke(stypy.reporting.localization.Localization(__file__, 735, 15), len_1751, *[cmd_1752], **kwargs_1753)
        
        # Getting the type of 'max_length' (line 735)
        max_length_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 26), 'max_length')
        # Applying the binary operator '>' (line 735)
        result_gt_1756 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 15), '>', len_call_result_1754, max_length_1755)
        
        # Testing the type of an if condition (line 735)
        if_condition_1757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 735, 12), result_gt_1756)
        # Assigning a type to the variable 'if_condition_1757' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'if_condition_1757', if_condition_1757)
        # SSA begins for if statement (line 735)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 736):
        
        # Assigning a Call to a Name (line 736):
        
        # Call to len(...): (line 736)
        # Processing the call arguments (line 736)
        # Getting the type of 'cmd' (line 736)
        cmd_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 33), 'cmd', False)
        # Processing the call keyword arguments (line 736)
        kwargs_1760 = {}
        # Getting the type of 'len' (line 736)
        len_1758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 29), 'len', False)
        # Calling len(args, kwargs) (line 736)
        len_call_result_1761 = invoke(stypy.reporting.localization.Localization(__file__, 736, 29), len_1758, *[cmd_1759], **kwargs_1760)
        
        # Assigning a type to the variable 'max_length' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 16), 'max_length', len_call_result_1761)
        # SSA join for if statement (line 735)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to print_command_list(...): (line 738)
        # Processing the call arguments (line 738)
        # Getting the type of 'std_commands' (line 738)
        std_commands_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 32), 'std_commands', False)
        str_1765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 32), 'str', 'Standard commands')
        # Getting the type of 'max_length' (line 740)
        max_length_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 32), 'max_length', False)
        # Processing the call keyword arguments (line 738)
        kwargs_1767 = {}
        # Getting the type of 'self' (line 738)
        self_1762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'self', False)
        # Obtaining the member 'print_command_list' of a type (line 738)
        print_command_list_1763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 8), self_1762, 'print_command_list')
        # Calling print_command_list(args, kwargs) (line 738)
        print_command_list_call_result_1768 = invoke(stypy.reporting.localization.Localization(__file__, 738, 8), print_command_list_1763, *[std_commands_1764, str_1765, max_length_1766], **kwargs_1767)
        
        
        # Getting the type of 'extra_commands' (line 741)
        extra_commands_1769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 11), 'extra_commands')
        # Testing the type of an if condition (line 741)
        if_condition_1770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 741, 8), extra_commands_1769)
        # Assigning a type to the variable 'if_condition_1770' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'if_condition_1770', if_condition_1770)
        # SSA begins for if statement (line 741)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print_command_list(...): (line 743)
        # Processing the call arguments (line 743)
        # Getting the type of 'extra_commands' (line 743)
        extra_commands_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 36), 'extra_commands', False)
        str_1774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 36), 'str', 'Extra commands')
        # Getting the type of 'max_length' (line 745)
        max_length_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 36), 'max_length', False)
        # Processing the call keyword arguments (line 743)
        kwargs_1776 = {}
        # Getting the type of 'self' (line 743)
        self_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'self', False)
        # Obtaining the member 'print_command_list' of a type (line 743)
        print_command_list_1772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 12), self_1771, 'print_command_list')
        # Calling print_command_list(args, kwargs) (line 743)
        print_command_list_call_result_1777 = invoke(stypy.reporting.localization.Localization(__file__, 743, 12), print_command_list_1772, *[extra_commands_1773, str_1774, max_length_1775], **kwargs_1776)
        
        # SSA join for if statement (line 741)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'print_commands(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_commands' in the type store
        # Getting the type of 'stypy_return_type' (line 714)
        stypy_return_type_1778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_commands'
        return stypy_return_type_1778


    @norecursion
    def get_command_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_command_list'
        module_type_store = module_type_store.open_function_context('get_command_list', 747, 4, False)
        # Assigning a type to the variable 'self' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.get_command_list.__dict__.__setitem__('stypy_localization', localization)
        Distribution.get_command_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.get_command_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.get_command_list.__dict__.__setitem__('stypy_function_name', 'Distribution.get_command_list')
        Distribution.get_command_list.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.get_command_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.get_command_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.get_command_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.get_command_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.get_command_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.get_command_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.get_command_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_command_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_command_list(...)' code ##################

        str_1779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', 'Get a list of (command, description) tuples.\n        The list is divided into "standard commands" (listed in\n        distutils.command.__all__) and "extra commands" (mentioned in\n        self.cmdclass, but not a standard command).  The descriptions come\n        from the command class attribute \'description\'.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 757, 8))
        
        # 'import distutils.command' statement (line 757)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 757, 8), 'distutils.command')

        if (type(import_1780) is not StypyTypeError):

            if (import_1780 != 'pyd_module'):
                __import__(import_1780)
                sys_modules_1781 = sys.modules[import_1780]
                import_module(stypy.reporting.localization.Localization(__file__, 757, 8), 'distutils.command', sys_modules_1781.module_type_store, module_type_store)
            else:
                import distutils.command

                import_module(stypy.reporting.localization.Localization(__file__, 757, 8), 'distutils.command', distutils.command, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.command' (line 757)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'distutils.command', import_1780)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Assigning a Attribute to a Name (line 758):
        
        # Assigning a Attribute to a Name (line 758):
        # Getting the type of 'distutils' (line 758)
        distutils_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 23), 'distutils')
        # Obtaining the member 'command' of a type (line 758)
        command_1783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 23), distutils_1782, 'command')
        # Obtaining the member '__all__' of a type (line 758)
        all___1784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 23), command_1783, '__all__')
        # Assigning a type to the variable 'std_commands' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'std_commands', all___1784)
        
        # Assigning a Dict to a Name (line 759):
        
        # Assigning a Dict to a Name (line 759):
        
        # Obtaining an instance of the builtin type 'dict' (line 759)
        dict_1785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 759)
        
        # Assigning a type to the variable 'is_std' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'is_std', dict_1785)
        
        # Getting the type of 'std_commands' (line 760)
        std_commands_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'std_commands')
        # Testing the type of a for loop iterable (line 760)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 760, 8), std_commands_1786)
        # Getting the type of the for loop variable (line 760)
        for_loop_var_1787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 760, 8), std_commands_1786)
        # Assigning a type to the variable 'cmd' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'cmd', for_loop_var_1787)
        # SSA begins for a for statement (line 760)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 761):
        
        # Assigning a Num to a Subscript (line 761):
        int_1788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 26), 'int')
        # Getting the type of 'is_std' (line 761)
        is_std_1789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'is_std')
        # Getting the type of 'cmd' (line 761)
        cmd_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 19), 'cmd')
        # Storing an element on a container (line 761)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 12), is_std_1789, (cmd_1790, int_1788))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 763):
        
        # Assigning a List to a Name (line 763):
        
        # Obtaining an instance of the builtin type 'list' (line 763)
        list_1791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 763)
        
        # Assigning a type to the variable 'extra_commands' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 8), 'extra_commands', list_1791)
        
        
        # Call to keys(...): (line 764)
        # Processing the call keyword arguments (line 764)
        kwargs_1795 = {}
        # Getting the type of 'self' (line 764)
        self_1792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 19), 'self', False)
        # Obtaining the member 'cmdclass' of a type (line 764)
        cmdclass_1793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 19), self_1792, 'cmdclass')
        # Obtaining the member 'keys' of a type (line 764)
        keys_1794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 19), cmdclass_1793, 'keys')
        # Calling keys(args, kwargs) (line 764)
        keys_call_result_1796 = invoke(stypy.reporting.localization.Localization(__file__, 764, 19), keys_1794, *[], **kwargs_1795)
        
        # Testing the type of a for loop iterable (line 764)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 764, 8), keys_call_result_1796)
        # Getting the type of the for loop variable (line 764)
        for_loop_var_1797 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 764, 8), keys_call_result_1796)
        # Assigning a type to the variable 'cmd' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'cmd', for_loop_var_1797)
        # SSA begins for a for statement (line 764)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to get(...): (line 765)
        # Processing the call arguments (line 765)
        # Getting the type of 'cmd' (line 765)
        cmd_1800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 30), 'cmd', False)
        # Processing the call keyword arguments (line 765)
        kwargs_1801 = {}
        # Getting the type of 'is_std' (line 765)
        is_std_1798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 19), 'is_std', False)
        # Obtaining the member 'get' of a type (line 765)
        get_1799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 19), is_std_1798, 'get')
        # Calling get(args, kwargs) (line 765)
        get_call_result_1802 = invoke(stypy.reporting.localization.Localization(__file__, 765, 19), get_1799, *[cmd_1800], **kwargs_1801)
        
        # Applying the 'not' unary operator (line 765)
        result_not__1803 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 15), 'not', get_call_result_1802)
        
        # Testing the type of an if condition (line 765)
        if_condition_1804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 765, 12), result_not__1803)
        # Assigning a type to the variable 'if_condition_1804' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'if_condition_1804', if_condition_1804)
        # SSA begins for if statement (line 765)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 766)
        # Processing the call arguments (line 766)
        # Getting the type of 'cmd' (line 766)
        cmd_1807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 38), 'cmd', False)
        # Processing the call keyword arguments (line 766)
        kwargs_1808 = {}
        # Getting the type of 'extra_commands' (line 766)
        extra_commands_1805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 16), 'extra_commands', False)
        # Obtaining the member 'append' of a type (line 766)
        append_1806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 16), extra_commands_1805, 'append')
        # Calling append(args, kwargs) (line 766)
        append_call_result_1809 = invoke(stypy.reporting.localization.Localization(__file__, 766, 16), append_1806, *[cmd_1807], **kwargs_1808)
        
        # SSA join for if statement (line 765)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 768):
        
        # Assigning a List to a Name (line 768):
        
        # Obtaining an instance of the builtin type 'list' (line 768)
        list_1810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 768)
        
        # Assigning a type to the variable 'rv' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'rv', list_1810)
        
        # Getting the type of 'std_commands' (line 769)
        std_commands_1811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 20), 'std_commands')
        # Getting the type of 'extra_commands' (line 769)
        extra_commands_1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 35), 'extra_commands')
        # Applying the binary operator '+' (line 769)
        result_add_1813 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 20), '+', std_commands_1811, extra_commands_1812)
        
        # Testing the type of a for loop iterable (line 769)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 769, 8), result_add_1813)
        # Getting the type of the for loop variable (line 769)
        for_loop_var_1814 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 769, 8), result_add_1813)
        # Assigning a type to the variable 'cmd' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'cmd', for_loop_var_1814)
        # SSA begins for a for statement (line 769)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 770):
        
        # Assigning a Call to a Name (line 770):
        
        # Call to get(...): (line 770)
        # Processing the call arguments (line 770)
        # Getting the type of 'cmd' (line 770)
        cmd_1818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 38), 'cmd', False)
        # Processing the call keyword arguments (line 770)
        kwargs_1819 = {}
        # Getting the type of 'self' (line 770)
        self_1815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 20), 'self', False)
        # Obtaining the member 'cmdclass' of a type (line 770)
        cmdclass_1816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 20), self_1815, 'cmdclass')
        # Obtaining the member 'get' of a type (line 770)
        get_1817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 20), cmdclass_1816, 'get')
        # Calling get(args, kwargs) (line 770)
        get_call_result_1820 = invoke(stypy.reporting.localization.Localization(__file__, 770, 20), get_1817, *[cmd_1818], **kwargs_1819)
        
        # Assigning a type to the variable 'klass' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'klass', get_call_result_1820)
        
        
        # Getting the type of 'klass' (line 771)
        klass_1821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 19), 'klass')
        # Applying the 'not' unary operator (line 771)
        result_not__1822 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 15), 'not', klass_1821)
        
        # Testing the type of an if condition (line 771)
        if_condition_1823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 771, 12), result_not__1822)
        # Assigning a type to the variable 'if_condition_1823' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'if_condition_1823', if_condition_1823)
        # SSA begins for if statement (line 771)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 772):
        
        # Assigning a Call to a Name (line 772):
        
        # Call to get_command_class(...): (line 772)
        # Processing the call arguments (line 772)
        # Getting the type of 'cmd' (line 772)
        cmd_1826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 47), 'cmd', False)
        # Processing the call keyword arguments (line 772)
        kwargs_1827 = {}
        # Getting the type of 'self' (line 772)
        self_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 24), 'self', False)
        # Obtaining the member 'get_command_class' of a type (line 772)
        get_command_class_1825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 24), self_1824, 'get_command_class')
        # Calling get_command_class(args, kwargs) (line 772)
        get_command_class_call_result_1828 = invoke(stypy.reporting.localization.Localization(__file__, 772, 24), get_command_class_1825, *[cmd_1826], **kwargs_1827)
        
        # Assigning a type to the variable 'klass' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 16), 'klass', get_command_class_call_result_1828)
        # SSA join for if statement (line 771)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 773)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 774):
        
        # Assigning a Attribute to a Name (line 774):
        # Getting the type of 'klass' (line 774)
        klass_1829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 30), 'klass')
        # Obtaining the member 'description' of a type (line 774)
        description_1830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 30), klass_1829, 'description')
        # Assigning a type to the variable 'description' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 16), 'description', description_1830)
        # SSA branch for the except part of a try statement (line 773)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 773)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Str to a Name (line 776):
        
        # Assigning a Str to a Name (line 776):
        str_1831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 30), 'str', '(no description available)')
        # Assigning a type to the variable 'description' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 16), 'description', str_1831)
        # SSA join for try-except statement (line 773)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 777)
        # Processing the call arguments (line 777)
        
        # Obtaining an instance of the builtin type 'tuple' (line 777)
        tuple_1834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 777)
        # Adding element type (line 777)
        # Getting the type of 'cmd' (line 777)
        cmd_1835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 23), 'cmd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 23), tuple_1834, cmd_1835)
        # Adding element type (line 777)
        # Getting the type of 'description' (line 777)
        description_1836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 28), 'description', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 23), tuple_1834, description_1836)
        
        # Processing the call keyword arguments (line 777)
        kwargs_1837 = {}
        # Getting the type of 'rv' (line 777)
        rv_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 12), 'rv', False)
        # Obtaining the member 'append' of a type (line 777)
        append_1833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 12), rv_1832, 'append')
        # Calling append(args, kwargs) (line 777)
        append_call_result_1838 = invoke(stypy.reporting.localization.Localization(__file__, 777, 12), append_1833, *[tuple_1834], **kwargs_1837)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'rv' (line 778)
        rv_1839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 15), 'rv')
        # Assigning a type to the variable 'stypy_return_type' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 8), 'stypy_return_type', rv_1839)
        
        # ################# End of 'get_command_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_command_list' in the type store
        # Getting the type of 'stypy_return_type' (line 747)
        stypy_return_type_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_command_list'
        return stypy_return_type_1840


    @norecursion
    def get_command_packages(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_command_packages'
        module_type_store = module_type_store.open_function_context('get_command_packages', 782, 4, False)
        # Assigning a type to the variable 'self' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.get_command_packages.__dict__.__setitem__('stypy_localization', localization)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_function_name', 'Distribution.get_command_packages')
        Distribution.get_command_packages.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.get_command_packages.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.get_command_packages.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.get_command_packages', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_command_packages', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_command_packages(...)' code ##################

        str_1841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 8), 'str', 'Return a list of packages from which commands are loaded.')
        
        # Assigning a Attribute to a Name (line 784):
        
        # Assigning a Attribute to a Name (line 784):
        # Getting the type of 'self' (line 784)
        self_1842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 15), 'self')
        # Obtaining the member 'command_packages' of a type (line 784)
        command_packages_1843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 15), self_1842, 'command_packages')
        # Assigning a type to the variable 'pkgs' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'pkgs', command_packages_1843)
        
        # Type idiom detected: calculating its left and rigth part (line 785)
        # Getting the type of 'list' (line 785)
        list_1844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 32), 'list')
        # Getting the type of 'pkgs' (line 785)
        pkgs_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 26), 'pkgs')
        
        (may_be_1846, more_types_in_union_1847) = may_not_be_subtype(list_1844, pkgs_1845)

        if may_be_1846:

            if more_types_in_union_1847:
                # Runtime conditional SSA (line 785)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'pkgs' (line 785)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'pkgs', remove_subtype_from_union(pkgs_1845, list))
            
            # Type idiom detected: calculating its left and rigth part (line 786)
            # Getting the type of 'pkgs' (line 786)
            pkgs_1848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 15), 'pkgs')
            # Getting the type of 'None' (line 786)
            None_1849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 23), 'None')
            
            (may_be_1850, more_types_in_union_1851) = may_be_none(pkgs_1848, None_1849)

            if may_be_1850:

                if more_types_in_union_1851:
                    # Runtime conditional SSA (line 786)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Str to a Name (line 787):
                
                # Assigning a Str to a Name (line 787):
                str_1852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 23), 'str', '')
                # Assigning a type to the variable 'pkgs' (line 787)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 16), 'pkgs', str_1852)

                if more_types_in_union_1851:
                    # SSA join for if statement (line 786)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a ListComp to a Name (line 788):
            
            # Assigning a ListComp to a Name (line 788):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to split(...): (line 788)
            # Processing the call arguments (line 788)
            str_1862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 54), 'str', ',')
            # Processing the call keyword arguments (line 788)
            kwargs_1863 = {}
            # Getting the type of 'pkgs' (line 788)
            pkgs_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 43), 'pkgs', False)
            # Obtaining the member 'split' of a type (line 788)
            split_1861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 43), pkgs_1860, 'split')
            # Calling split(args, kwargs) (line 788)
            split_call_result_1864 = invoke(stypy.reporting.localization.Localization(__file__, 788, 43), split_1861, *[str_1862], **kwargs_1863)
            
            comprehension_1865 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 20), split_call_result_1864)
            # Assigning a type to the variable 'pkg' (line 788)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 20), 'pkg', comprehension_1865)
            
            # Getting the type of 'pkg' (line 788)
            pkg_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 62), 'pkg')
            str_1858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 69), 'str', '')
            # Applying the binary operator '!=' (line 788)
            result_ne_1859 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 62), '!=', pkg_1857, str_1858)
            
            
            # Call to strip(...): (line 788)
            # Processing the call keyword arguments (line 788)
            kwargs_1855 = {}
            # Getting the type of 'pkg' (line 788)
            pkg_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 20), 'pkg', False)
            # Obtaining the member 'strip' of a type (line 788)
            strip_1854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 20), pkg_1853, 'strip')
            # Calling strip(args, kwargs) (line 788)
            strip_call_result_1856 = invoke(stypy.reporting.localization.Localization(__file__, 788, 20), strip_1854, *[], **kwargs_1855)
            
            list_1866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 20), list_1866, strip_call_result_1856)
            # Assigning a type to the variable 'pkgs' (line 788)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'pkgs', list_1866)
            
            
            str_1867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 15), 'str', 'distutils.command')
            # Getting the type of 'pkgs' (line 789)
            pkgs_1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 42), 'pkgs')
            # Applying the binary operator 'notin' (line 789)
            result_contains_1869 = python_operator(stypy.reporting.localization.Localization(__file__, 789, 15), 'notin', str_1867, pkgs_1868)
            
            # Testing the type of an if condition (line 789)
            if_condition_1870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 789, 12), result_contains_1869)
            # Assigning a type to the variable 'if_condition_1870' (line 789)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'if_condition_1870', if_condition_1870)
            # SSA begins for if statement (line 789)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to insert(...): (line 790)
            # Processing the call arguments (line 790)
            int_1873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 28), 'int')
            str_1874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 31), 'str', 'distutils.command')
            # Processing the call keyword arguments (line 790)
            kwargs_1875 = {}
            # Getting the type of 'pkgs' (line 790)
            pkgs_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 16), 'pkgs', False)
            # Obtaining the member 'insert' of a type (line 790)
            insert_1872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 16), pkgs_1871, 'insert')
            # Calling insert(args, kwargs) (line 790)
            insert_call_result_1876 = invoke(stypy.reporting.localization.Localization(__file__, 790, 16), insert_1872, *[int_1873, str_1874], **kwargs_1875)
            
            # SSA join for if statement (line 789)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 791):
            
            # Assigning a Name to a Attribute (line 791):
            # Getting the type of 'pkgs' (line 791)
            pkgs_1877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 36), 'pkgs')
            # Getting the type of 'self' (line 791)
            self_1878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'self')
            # Setting the type of the member 'command_packages' of a type (line 791)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 12), self_1878, 'command_packages', pkgs_1877)

            if more_types_in_union_1847:
                # SSA join for if statement (line 785)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'pkgs' (line 792)
        pkgs_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 15), 'pkgs')
        # Assigning a type to the variable 'stypy_return_type' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'stypy_return_type', pkgs_1879)
        
        # ################# End of 'get_command_packages(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_command_packages' in the type store
        # Getting the type of 'stypy_return_type' (line 782)
        stypy_return_type_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1880)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_command_packages'
        return stypy_return_type_1880


    @norecursion
    def get_command_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_command_class'
        module_type_store = module_type_store.open_function_context('get_command_class', 794, 4, False)
        # Assigning a type to the variable 'self' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.get_command_class.__dict__.__setitem__('stypy_localization', localization)
        Distribution.get_command_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.get_command_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.get_command_class.__dict__.__setitem__('stypy_function_name', 'Distribution.get_command_class')
        Distribution.get_command_class.__dict__.__setitem__('stypy_param_names_list', ['command'])
        Distribution.get_command_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.get_command_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.get_command_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.get_command_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.get_command_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.get_command_class.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.get_command_class', ['command'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_command_class', localization, ['command'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_command_class(...)' code ##################

        str_1881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, (-1)), 'str', 'Return the class that implements the Distutils command named by\n        \'command\'.  First we check the \'cmdclass\' dictionary; if the\n        command is mentioned there, we fetch the class object from the\n        dictionary and return it.  Otherwise we load the command module\n        ("distutils.command." + command) and fetch the command class from\n        the module.  The loaded class is also stored in \'cmdclass\'\n        to speed future calls to \'get_command_class()\'.\n\n        Raises DistutilsModuleError if the expected module could not be\n        found, or if that module does not define the expected class.\n        ')
        
        # Assigning a Call to a Name (line 806):
        
        # Assigning a Call to a Name (line 806):
        
        # Call to get(...): (line 806)
        # Processing the call arguments (line 806)
        # Getting the type of 'command' (line 806)
        command_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 34), 'command', False)
        # Processing the call keyword arguments (line 806)
        kwargs_1886 = {}
        # Getting the type of 'self' (line 806)
        self_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 16), 'self', False)
        # Obtaining the member 'cmdclass' of a type (line 806)
        cmdclass_1883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 16), self_1882, 'cmdclass')
        # Obtaining the member 'get' of a type (line 806)
        get_1884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 16), cmdclass_1883, 'get')
        # Calling get(args, kwargs) (line 806)
        get_call_result_1887 = invoke(stypy.reporting.localization.Localization(__file__, 806, 16), get_1884, *[command_1885], **kwargs_1886)
        
        # Assigning a type to the variable 'klass' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'klass', get_call_result_1887)
        
        # Getting the type of 'klass' (line 807)
        klass_1888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 11), 'klass')
        # Testing the type of an if condition (line 807)
        if_condition_1889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 807, 8), klass_1888)
        # Assigning a type to the variable 'if_condition_1889' (line 807)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'if_condition_1889', if_condition_1889)
        # SSA begins for if statement (line 807)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'klass' (line 808)
        klass_1890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 19), 'klass')
        # Assigning a type to the variable 'stypy_return_type' (line 808)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 12), 'stypy_return_type', klass_1890)
        # SSA join for if statement (line 807)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_command_packages(...): (line 810)
        # Processing the call keyword arguments (line 810)
        kwargs_1893 = {}
        # Getting the type of 'self' (line 810)
        self_1891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 23), 'self', False)
        # Obtaining the member 'get_command_packages' of a type (line 810)
        get_command_packages_1892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 23), self_1891, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 810)
        get_command_packages_call_result_1894 = invoke(stypy.reporting.localization.Localization(__file__, 810, 23), get_command_packages_1892, *[], **kwargs_1893)
        
        # Testing the type of a for loop iterable (line 810)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 810, 8), get_command_packages_call_result_1894)
        # Getting the type of the for loop variable (line 810)
        for_loop_var_1895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 810, 8), get_command_packages_call_result_1894)
        # Assigning a type to the variable 'pkgname' (line 810)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'pkgname', for_loop_var_1895)
        # SSA begins for a for statement (line 810)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 811):
        
        # Assigning a BinOp to a Name (line 811):
        str_1896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 26), 'str', '%s.%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 811)
        tuple_1897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 811)
        # Adding element type (line 811)
        # Getting the type of 'pkgname' (line 811)
        pkgname_1898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 37), 'pkgname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 37), tuple_1897, pkgname_1898)
        # Adding element type (line 811)
        # Getting the type of 'command' (line 811)
        command_1899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 46), 'command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 37), tuple_1897, command_1899)
        
        # Applying the binary operator '%' (line 811)
        result_mod_1900 = python_operator(stypy.reporting.localization.Localization(__file__, 811, 26), '%', str_1896, tuple_1897)
        
        # Assigning a type to the variable 'module_name' (line 811)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'module_name', result_mod_1900)
        
        # Assigning a Name to a Name (line 812):
        
        # Assigning a Name to a Name (line 812):
        # Getting the type of 'command' (line 812)
        command_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), 'command')
        # Assigning a type to the variable 'klass_name' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'klass_name', command_1901)
        
        
        # SSA begins for try-except statement (line 814)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __import__(...): (line 815)
        # Processing the call arguments (line 815)
        # Getting the type of 'module_name' (line 815)
        module_name_1903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 28), 'module_name', False)
        # Processing the call keyword arguments (line 815)
        kwargs_1904 = {}
        # Getting the type of '__import__' (line 815)
        import___1902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 16), '__import__', False)
        # Calling __import__(args, kwargs) (line 815)
        import___call_result_1905 = invoke(stypy.reporting.localization.Localization(__file__, 815, 16), import___1902, *[module_name_1903], **kwargs_1904)
        
        
        # Assigning a Subscript to a Name (line 816):
        
        # Assigning a Subscript to a Name (line 816):
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 816)
        module_name_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 37), 'module_name')
        # Getting the type of 'sys' (line 816)
        sys_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 25), 'sys')
        # Obtaining the member 'modules' of a type (line 816)
        modules_1908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 25), sys_1907, 'modules')
        # Obtaining the member '__getitem__' of a type (line 816)
        getitem___1909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 25), modules_1908, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 816)
        subscript_call_result_1910 = invoke(stypy.reporting.localization.Localization(__file__, 816, 25), getitem___1909, module_name_1906)
        
        # Assigning a type to the variable 'module' (line 816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 16), 'module', subscript_call_result_1910)
        # SSA branch for the except part of a try statement (line 814)
        # SSA branch for the except 'ImportError' branch of a try statement (line 814)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 814)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 820)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 821):
        
        # Assigning a Call to a Name (line 821):
        
        # Call to getattr(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of 'module' (line 821)
        module_1912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 32), 'module', False)
        # Getting the type of 'klass_name' (line 821)
        klass_name_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 40), 'klass_name', False)
        # Processing the call keyword arguments (line 821)
        kwargs_1914 = {}
        # Getting the type of 'getattr' (line 821)
        getattr_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 821)
        getattr_call_result_1915 = invoke(stypy.reporting.localization.Localization(__file__, 821, 24), getattr_1911, *[module_1912, klass_name_1913], **kwargs_1914)
        
        # Assigning a type to the variable 'klass' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'klass', getattr_call_result_1915)
        # SSA branch for the except part of a try statement (line 820)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 820)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsModuleError' (line 823)
        DistutilsModuleError_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 22), 'DistutilsModuleError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 823, 16), DistutilsModuleError_1916, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 820)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 827):
        
        # Assigning a Name to a Subscript (line 827):
        # Getting the type of 'klass' (line 827)
        klass_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 37), 'klass')
        # Getting the type of 'self' (line 827)
        self_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 12), 'self')
        # Obtaining the member 'cmdclass' of a type (line 827)
        cmdclass_1919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 12), self_1918, 'cmdclass')
        # Getting the type of 'command' (line 827)
        command_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 26), 'command')
        # Storing an element on a container (line 827)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 827, 12), cmdclass_1919, (command_1920, klass_1917))
        # Getting the type of 'klass' (line 828)
        klass_1921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 19), 'klass')
        # Assigning a type to the variable 'stypy_return_type' (line 828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 12), 'stypy_return_type', klass_1921)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to DistutilsModuleError(...): (line 830)
        # Processing the call arguments (line 830)
        str_1923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 35), 'str', "invalid command '%s'")
        # Getting the type of 'command' (line 830)
        command_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 60), 'command', False)
        # Applying the binary operator '%' (line 830)
        result_mod_1925 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 35), '%', str_1923, command_1924)
        
        # Processing the call keyword arguments (line 830)
        kwargs_1926 = {}
        # Getting the type of 'DistutilsModuleError' (line 830)
        DistutilsModuleError_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 14), 'DistutilsModuleError', False)
        # Calling DistutilsModuleError(args, kwargs) (line 830)
        DistutilsModuleError_call_result_1927 = invoke(stypy.reporting.localization.Localization(__file__, 830, 14), DistutilsModuleError_1922, *[result_mod_1925], **kwargs_1926)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 830, 8), DistutilsModuleError_call_result_1927, 'raise parameter', BaseException)
        
        # ################# End of 'get_command_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_command_class' in the type store
        # Getting the type of 'stypy_return_type' (line 794)
        stypy_return_type_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_command_class'
        return stypy_return_type_1928


    @norecursion
    def get_command_obj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_1929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 46), 'int')
        defaults = [int_1929]
        # Create a new context for function 'get_command_obj'
        module_type_store = module_type_store.open_function_context('get_command_obj', 833, 4, False)
        # Assigning a type to the variable 'self' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.get_command_obj.__dict__.__setitem__('stypy_localization', localization)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_function_name', 'Distribution.get_command_obj')
        Distribution.get_command_obj.__dict__.__setitem__('stypy_param_names_list', ['command', 'create'])
        Distribution.get_command_obj.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.get_command_obj.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.get_command_obj', ['command', 'create'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_command_obj', localization, ['command', 'create'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_command_obj(...)' code ##################

        str_1930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, (-1)), 'str', "Return the command object for 'command'.  Normally this object\n        is cached on a previous call to 'get_command_obj()'; if no command\n        object for 'command' is in the cache, then we either create and\n        return it (if 'create' is true) or return None.\n        ")
        
        # Assigning a Call to a Name (line 839):
        
        # Assigning a Call to a Name (line 839):
        
        # Call to get(...): (line 839)
        # Processing the call arguments (line 839)
        # Getting the type of 'command' (line 839)
        command_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 39), 'command', False)
        # Processing the call keyword arguments (line 839)
        kwargs_1935 = {}
        # Getting the type of 'self' (line 839)
        self_1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 18), 'self', False)
        # Obtaining the member 'command_obj' of a type (line 839)
        command_obj_1932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 18), self_1931, 'command_obj')
        # Obtaining the member 'get' of a type (line 839)
        get_1933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 18), command_obj_1932, 'get')
        # Calling get(args, kwargs) (line 839)
        get_call_result_1936 = invoke(stypy.reporting.localization.Localization(__file__, 839, 18), get_1933, *[command_1934], **kwargs_1935)
        
        # Assigning a type to the variable 'cmd_obj' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'cmd_obj', get_call_result_1936)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'cmd_obj' (line 840)
        cmd_obj_1937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 15), 'cmd_obj')
        # Applying the 'not' unary operator (line 840)
        result_not__1938 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 11), 'not', cmd_obj_1937)
        
        # Getting the type of 'create' (line 840)
        create_1939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 27), 'create')
        # Applying the binary operator 'and' (line 840)
        result_and_keyword_1940 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 11), 'and', result_not__1938, create_1939)
        
        # Testing the type of an if condition (line 840)
        if_condition_1941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 840, 8), result_and_keyword_1940)
        # Assigning a type to the variable 'if_condition_1941' (line 840)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'if_condition_1941', if_condition_1941)
        # SSA begins for if statement (line 840)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'DEBUG' (line 841)
        DEBUG_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 15), 'DEBUG')
        # Testing the type of an if condition (line 841)
        if_condition_1943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 841, 12), DEBUG_1942)
        # Assigning a type to the variable 'if_condition_1943' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'if_condition_1943', if_condition_1943)
        # SSA begins for if statement (line 841)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 842)
        # Processing the call arguments (line 842)
        str_1946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 30), 'str', "Distribution.get_command_obj(): creating '%s' command object")
        # Getting the type of 'command' (line 843)
        command_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 63), 'command', False)
        # Applying the binary operator '%' (line 842)
        result_mod_1948 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 30), '%', str_1946, command_1947)
        
        # Processing the call keyword arguments (line 842)
        kwargs_1949 = {}
        # Getting the type of 'self' (line 842)
        self_1944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 16), 'self', False)
        # Obtaining the member 'announce' of a type (line 842)
        announce_1945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 16), self_1944, 'announce')
        # Calling announce(args, kwargs) (line 842)
        announce_call_result_1950 = invoke(stypy.reporting.localization.Localization(__file__, 842, 16), announce_1945, *[result_mod_1948], **kwargs_1949)
        
        # SSA join for if statement (line 841)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 845):
        
        # Assigning a Call to a Name (line 845):
        
        # Call to get_command_class(...): (line 845)
        # Processing the call arguments (line 845)
        # Getting the type of 'command' (line 845)
        command_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 43), 'command', False)
        # Processing the call keyword arguments (line 845)
        kwargs_1954 = {}
        # Getting the type of 'self' (line 845)
        self_1951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 20), 'self', False)
        # Obtaining the member 'get_command_class' of a type (line 845)
        get_command_class_1952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 20), self_1951, 'get_command_class')
        # Calling get_command_class(args, kwargs) (line 845)
        get_command_class_call_result_1955 = invoke(stypy.reporting.localization.Localization(__file__, 845, 20), get_command_class_1952, *[command_1953], **kwargs_1954)
        
        # Assigning a type to the variable 'klass' (line 845)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'klass', get_command_class_call_result_1955)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Subscript (line 846):
        
        # Call to klass(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of 'self' (line 846)
        self_1957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 56), 'self', False)
        # Processing the call keyword arguments (line 846)
        kwargs_1958 = {}
        # Getting the type of 'klass' (line 846)
        klass_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 50), 'klass', False)
        # Calling klass(args, kwargs) (line 846)
        klass_call_result_1959 = invoke(stypy.reporting.localization.Localization(__file__, 846, 50), klass_1956, *[self_1957], **kwargs_1958)
        
        # Getting the type of 'self' (line 846)
        self_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 22), 'self')
        # Obtaining the member 'command_obj' of a type (line 846)
        command_obj_1961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 22), self_1960, 'command_obj')
        # Getting the type of 'command' (line 846)
        command_1962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 39), 'command')
        # Storing an element on a container (line 846)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 22), command_obj_1961, (command_1962, klass_call_result_1959))
        
        # Assigning a Subscript to a Name (line 846):
        
        # Obtaining the type of the subscript
        # Getting the type of 'command' (line 846)
        command_1963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 39), 'command')
        # Getting the type of 'self' (line 846)
        self_1964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 22), 'self')
        # Obtaining the member 'command_obj' of a type (line 846)
        command_obj_1965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 22), self_1964, 'command_obj')
        # Obtaining the member '__getitem__' of a type (line 846)
        getitem___1966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 22), command_obj_1965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 846)
        subscript_call_result_1967 = invoke(stypy.reporting.localization.Localization(__file__, 846, 22), getitem___1966, command_1963)
        
        # Assigning a type to the variable 'cmd_obj' (line 846)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'cmd_obj', subscript_call_result_1967)
        
        # Assigning a Num to a Subscript (line 847):
        
        # Assigning a Num to a Subscript (line 847):
        int_1968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 37), 'int')
        # Getting the type of 'self' (line 847)
        self_1969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'self')
        # Obtaining the member 'have_run' of a type (line 847)
        have_run_1970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 12), self_1969, 'have_run')
        # Getting the type of 'command' (line 847)
        command_1971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 26), 'command')
        # Storing an element on a container (line 847)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 847, 12), have_run_1970, (command_1971, int_1968))
        
        # Assigning a Call to a Name (line 854):
        
        # Assigning a Call to a Name (line 854):
        
        # Call to get(...): (line 854)
        # Processing the call arguments (line 854)
        # Getting the type of 'command' (line 854)
        command_1975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 47), 'command', False)
        # Processing the call keyword arguments (line 854)
        kwargs_1976 = {}
        # Getting the type of 'self' (line 854)
        self_1972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 22), 'self', False)
        # Obtaining the member 'command_options' of a type (line 854)
        command_options_1973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 22), self_1972, 'command_options')
        # Obtaining the member 'get' of a type (line 854)
        get_1974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 22), command_options_1973, 'get')
        # Calling get(args, kwargs) (line 854)
        get_call_result_1977 = invoke(stypy.reporting.localization.Localization(__file__, 854, 22), get_1974, *[command_1975], **kwargs_1976)
        
        # Assigning a type to the variable 'options' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'options', get_call_result_1977)
        
        # Getting the type of 'options' (line 855)
        options_1978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 15), 'options')
        # Testing the type of an if condition (line 855)
        if_condition_1979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 855, 12), options_1978)
        # Assigning a type to the variable 'if_condition_1979' (line 855)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 12), 'if_condition_1979', if_condition_1979)
        # SSA begins for if statement (line 855)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _set_command_options(...): (line 856)
        # Processing the call arguments (line 856)
        # Getting the type of 'cmd_obj' (line 856)
        cmd_obj_1982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 42), 'cmd_obj', False)
        # Getting the type of 'options' (line 856)
        options_1983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 51), 'options', False)
        # Processing the call keyword arguments (line 856)
        kwargs_1984 = {}
        # Getting the type of 'self' (line 856)
        self_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 16), 'self', False)
        # Obtaining the member '_set_command_options' of a type (line 856)
        _set_command_options_1981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 16), self_1980, '_set_command_options')
        # Calling _set_command_options(args, kwargs) (line 856)
        _set_command_options_call_result_1985 = invoke(stypy.reporting.localization.Localization(__file__, 856, 16), _set_command_options_1981, *[cmd_obj_1982, options_1983], **kwargs_1984)
        
        # SSA join for if statement (line 855)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 840)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'cmd_obj' (line 858)
        cmd_obj_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 15), 'cmd_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 858)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 858, 8), 'stypy_return_type', cmd_obj_1986)
        
        # ################# End of 'get_command_obj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_command_obj' in the type store
        # Getting the type of 'stypy_return_type' (line 833)
        stypy_return_type_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_command_obj'
        return stypy_return_type_1987


    @norecursion
    def _set_command_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 860)
        None_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 60), 'None')
        defaults = [None_1988]
        # Create a new context for function '_set_command_options'
        module_type_store = module_type_store.open_function_context('_set_command_options', 860, 4, False)
        # Assigning a type to the variable 'self' (line 861)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution._set_command_options.__dict__.__setitem__('stypy_localization', localization)
        Distribution._set_command_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution._set_command_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution._set_command_options.__dict__.__setitem__('stypy_function_name', 'Distribution._set_command_options')
        Distribution._set_command_options.__dict__.__setitem__('stypy_param_names_list', ['command_obj', 'option_dict'])
        Distribution._set_command_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution._set_command_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution._set_command_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution._set_command_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution._set_command_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution._set_command_options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution._set_command_options', ['command_obj', 'option_dict'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_command_options', localization, ['command_obj', 'option_dict'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_command_options(...)' code ##################

        str_1989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, (-1)), 'str', "Set the options for 'command_obj' from 'option_dict'.  Basically\n        this means copying elements of a dictionary ('option_dict') to\n        attributes of an instance ('command').\n\n        'command_obj' must be a Command instance.  If 'option_dict' is not\n        supplied, uses the standard option dictionary for this command\n        (from 'self.command_options').\n        ")
        
        # Assigning a Call to a Name (line 869):
        
        # Assigning a Call to a Name (line 869):
        
        # Call to get_command_name(...): (line 869)
        # Processing the call keyword arguments (line 869)
        kwargs_1992 = {}
        # Getting the type of 'command_obj' (line 869)
        command_obj_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 23), 'command_obj', False)
        # Obtaining the member 'get_command_name' of a type (line 869)
        get_command_name_1991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 23), command_obj_1990, 'get_command_name')
        # Calling get_command_name(args, kwargs) (line 869)
        get_command_name_call_result_1993 = invoke(stypy.reporting.localization.Localization(__file__, 869, 23), get_command_name_1991, *[], **kwargs_1992)
        
        # Assigning a type to the variable 'command_name' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'command_name', get_command_name_call_result_1993)
        
        # Type idiom detected: calculating its left and rigth part (line 870)
        # Getting the type of 'option_dict' (line 870)
        option_dict_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 11), 'option_dict')
        # Getting the type of 'None' (line 870)
        None_1995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 26), 'None')
        
        (may_be_1996, more_types_in_union_1997) = may_be_none(option_dict_1994, None_1995)

        if may_be_1996:

            if more_types_in_union_1997:
                # Runtime conditional SSA (line 870)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 871):
            
            # Assigning a Call to a Name (line 871):
            
            # Call to get_option_dict(...): (line 871)
            # Processing the call arguments (line 871)
            # Getting the type of 'command_name' (line 871)
            command_name_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 47), 'command_name', False)
            # Processing the call keyword arguments (line 871)
            kwargs_2001 = {}
            # Getting the type of 'self' (line 871)
            self_1998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 26), 'self', False)
            # Obtaining the member 'get_option_dict' of a type (line 871)
            get_option_dict_1999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 26), self_1998, 'get_option_dict')
            # Calling get_option_dict(args, kwargs) (line 871)
            get_option_dict_call_result_2002 = invoke(stypy.reporting.localization.Localization(__file__, 871, 26), get_option_dict_1999, *[command_name_2000], **kwargs_2001)
            
            # Assigning a type to the variable 'option_dict' (line 871)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'option_dict', get_option_dict_call_result_2002)

            if more_types_in_union_1997:
                # SSA join for if statement (line 870)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'DEBUG' (line 873)
        DEBUG_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 11), 'DEBUG')
        # Testing the type of an if condition (line 873)
        if_condition_2004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 873, 8), DEBUG_2003)
        # Assigning a type to the variable 'if_condition_2004' (line 873)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 8), 'if_condition_2004', if_condition_2004)
        # SSA begins for if statement (line 873)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 874)
        # Processing the call arguments (line 874)
        str_2007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 26), 'str', "  setting options for '%s' command:")
        # Getting the type of 'command_name' (line 874)
        command_name_2008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 66), 'command_name', False)
        # Applying the binary operator '%' (line 874)
        result_mod_2009 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 26), '%', str_2007, command_name_2008)
        
        # Processing the call keyword arguments (line 874)
        kwargs_2010 = {}
        # Getting the type of 'self' (line 874)
        self_2005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 874)
        announce_2006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 12), self_2005, 'announce')
        # Calling announce(args, kwargs) (line 874)
        announce_call_result_2011 = invoke(stypy.reporting.localization.Localization(__file__, 874, 12), announce_2006, *[result_mod_2009], **kwargs_2010)
        
        # SSA join for if statement (line 873)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to items(...): (line 875)
        # Processing the call keyword arguments (line 875)
        kwargs_2014 = {}
        # Getting the type of 'option_dict' (line 875)
        option_dict_2012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 41), 'option_dict', False)
        # Obtaining the member 'items' of a type (line 875)
        items_2013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 41), option_dict_2012, 'items')
        # Calling items(args, kwargs) (line 875)
        items_call_result_2015 = invoke(stypy.reporting.localization.Localization(__file__, 875, 41), items_2013, *[], **kwargs_2014)
        
        # Testing the type of a for loop iterable (line 875)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 875, 8), items_call_result_2015)
        # Getting the type of the for loop variable (line 875)
        for_loop_var_2016 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 875, 8), items_call_result_2015)
        # Assigning a type to the variable 'option' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'option', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 8), for_loop_var_2016))
        # Assigning a type to the variable 'source' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'source', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 8), for_loop_var_2016))
        # Assigning a type to the variable 'value' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 8), for_loop_var_2016))
        # SSA begins for a for statement (line 875)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'DEBUG' (line 876)
        DEBUG_2017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 15), 'DEBUG')
        # Testing the type of an if condition (line 876)
        if_condition_2018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 876, 12), DEBUG_2017)
        # Assigning a type to the variable 'if_condition_2018' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 12), 'if_condition_2018', if_condition_2018)
        # SSA begins for if statement (line 876)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 877)
        # Processing the call arguments (line 877)
        str_2021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 30), 'str', '    %s = %s (from %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 877)
        tuple_2022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 877)
        # Adding element type (line 877)
        # Getting the type of 'option' (line 877)
        option_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 57), 'option', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 877, 57), tuple_2022, option_2023)
        # Adding element type (line 877)
        # Getting the type of 'value' (line 877)
        value_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 65), 'value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 877, 57), tuple_2022, value_2024)
        # Adding element type (line 877)
        # Getting the type of 'source' (line 878)
        source_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 57), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 877, 57), tuple_2022, source_2025)
        
        # Applying the binary operator '%' (line 877)
        result_mod_2026 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 30), '%', str_2021, tuple_2022)
        
        # Processing the call keyword arguments (line 877)
        kwargs_2027 = {}
        # Getting the type of 'self' (line 877)
        self_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 16), 'self', False)
        # Obtaining the member 'announce' of a type (line 877)
        announce_2020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 16), self_2019, 'announce')
        # Calling announce(args, kwargs) (line 877)
        announce_call_result_2028 = invoke(stypy.reporting.localization.Localization(__file__, 877, 16), announce_2020, *[result_mod_2026], **kwargs_2027)
        
        # SSA join for if statement (line 876)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 879)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 880):
        
        # Assigning a Call to a Name (line 880):
        
        # Call to map(...): (line 880)
        # Processing the call arguments (line 880)
        # Getting the type of 'translate_longopt' (line 880)
        translate_longopt_2030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 32), 'translate_longopt', False)
        # Getting the type of 'command_obj' (line 880)
        command_obj_2031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 51), 'command_obj', False)
        # Obtaining the member 'boolean_options' of a type (line 880)
        boolean_options_2032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 51), command_obj_2031, 'boolean_options')
        # Processing the call keyword arguments (line 880)
        kwargs_2033 = {}
        # Getting the type of 'map' (line 880)
        map_2029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 28), 'map', False)
        # Calling map(args, kwargs) (line 880)
        map_call_result_2034 = invoke(stypy.reporting.localization.Localization(__file__, 880, 28), map_2029, *[translate_longopt_2030, boolean_options_2032], **kwargs_2033)
        
        # Assigning a type to the variable 'bool_opts' (line 880)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'bool_opts', map_call_result_2034)
        # SSA branch for the except part of a try statement (line 879)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 879)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a List to a Name (line 882):
        
        # Assigning a List to a Name (line 882):
        
        # Obtaining an instance of the builtin type 'list' (line 882)
        list_2035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 882)
        
        # Assigning a type to the variable 'bool_opts' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 16), 'bool_opts', list_2035)
        # SSA join for try-except statement (line 879)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 883)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 884):
        
        # Assigning a Attribute to a Name (line 884):
        # Getting the type of 'command_obj' (line 884)
        command_obj_2036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 26), 'command_obj')
        # Obtaining the member 'negative_opt' of a type (line 884)
        negative_opt_2037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 26), command_obj_2036, 'negative_opt')
        # Assigning a type to the variable 'neg_opt' (line 884)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 16), 'neg_opt', negative_opt_2037)
        # SSA branch for the except part of a try statement (line 883)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 883)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Dict to a Name (line 886):
        
        # Assigning a Dict to a Name (line 886):
        
        # Obtaining an instance of the builtin type 'dict' (line 886)
        dict_2038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 886)
        
        # Assigning a type to the variable 'neg_opt' (line 886)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 16), 'neg_opt', dict_2038)
        # SSA join for try-except statement (line 883)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 888)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 889):
        
        # Assigning a Call to a Name (line 889):
        
        # Call to isinstance(...): (line 889)
        # Processing the call arguments (line 889)
        # Getting the type of 'value' (line 889)
        value_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 39), 'value', False)
        # Getting the type of 'str' (line 889)
        str_2041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 46), 'str', False)
        # Processing the call keyword arguments (line 889)
        kwargs_2042 = {}
        # Getting the type of 'isinstance' (line 889)
        isinstance_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 28), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 889)
        isinstance_call_result_2043 = invoke(stypy.reporting.localization.Localization(__file__, 889, 28), isinstance_2039, *[value_2040, str_2041], **kwargs_2042)
        
        # Assigning a type to the variable 'is_string' (line 889)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 16), 'is_string', isinstance_call_result_2043)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'option' (line 890)
        option_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 19), 'option')
        # Getting the type of 'neg_opt' (line 890)
        neg_opt_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 29), 'neg_opt')
        # Applying the binary operator 'in' (line 890)
        result_contains_2046 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 19), 'in', option_2044, neg_opt_2045)
        
        # Getting the type of 'is_string' (line 890)
        is_string_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 41), 'is_string')
        # Applying the binary operator 'and' (line 890)
        result_and_keyword_2048 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 19), 'and', result_contains_2046, is_string_2047)
        
        # Testing the type of an if condition (line 890)
        if_condition_2049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 890, 16), result_and_keyword_2048)
        # Assigning a type to the variable 'if_condition_2049' (line 890)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 16), 'if_condition_2049', if_condition_2049)
        # SSA begins for if statement (line 890)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 891)
        # Processing the call arguments (line 891)
        # Getting the type of 'command_obj' (line 891)
        command_obj_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 28), 'command_obj', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'option' (line 891)
        option_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 49), 'option', False)
        # Getting the type of 'neg_opt' (line 891)
        neg_opt_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 41), 'neg_opt', False)
        # Obtaining the member '__getitem__' of a type (line 891)
        getitem___2054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 41), neg_opt_2053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 891)
        subscript_call_result_2055 = invoke(stypy.reporting.localization.Localization(__file__, 891, 41), getitem___2054, option_2052)
        
        
        
        # Call to strtobool(...): (line 891)
        # Processing the call arguments (line 891)
        # Getting the type of 'value' (line 891)
        value_2057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 72), 'value', False)
        # Processing the call keyword arguments (line 891)
        kwargs_2058 = {}
        # Getting the type of 'strtobool' (line 891)
        strtobool_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 62), 'strtobool', False)
        # Calling strtobool(args, kwargs) (line 891)
        strtobool_call_result_2059 = invoke(stypy.reporting.localization.Localization(__file__, 891, 62), strtobool_2056, *[value_2057], **kwargs_2058)
        
        # Applying the 'not' unary operator (line 891)
        result_not__2060 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 58), 'not', strtobool_call_result_2059)
        
        # Processing the call keyword arguments (line 891)
        kwargs_2061 = {}
        # Getting the type of 'setattr' (line 891)
        setattr_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 891)
        setattr_call_result_2062 = invoke(stypy.reporting.localization.Localization(__file__, 891, 20), setattr_2050, *[command_obj_2051, subscript_call_result_2055, result_not__2060], **kwargs_2061)
        
        # SSA branch for the else part of an if statement (line 890)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'option' (line 892)
        option_2063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 21), 'option')
        # Getting the type of 'bool_opts' (line 892)
        bool_opts_2064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 31), 'bool_opts')
        # Applying the binary operator 'in' (line 892)
        result_contains_2065 = python_operator(stypy.reporting.localization.Localization(__file__, 892, 21), 'in', option_2063, bool_opts_2064)
        
        # Getting the type of 'is_string' (line 892)
        is_string_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 45), 'is_string')
        # Applying the binary operator 'and' (line 892)
        result_and_keyword_2067 = python_operator(stypy.reporting.localization.Localization(__file__, 892, 21), 'and', result_contains_2065, is_string_2066)
        
        # Testing the type of an if condition (line 892)
        if_condition_2068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 892, 21), result_and_keyword_2067)
        # Assigning a type to the variable 'if_condition_2068' (line 892)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 21), 'if_condition_2068', if_condition_2068)
        # SSA begins for if statement (line 892)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 893)
        # Processing the call arguments (line 893)
        # Getting the type of 'command_obj' (line 893)
        command_obj_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 28), 'command_obj', False)
        # Getting the type of 'option' (line 893)
        option_2071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 41), 'option', False)
        
        # Call to strtobool(...): (line 893)
        # Processing the call arguments (line 893)
        # Getting the type of 'value' (line 893)
        value_2073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 59), 'value', False)
        # Processing the call keyword arguments (line 893)
        kwargs_2074 = {}
        # Getting the type of 'strtobool' (line 893)
        strtobool_2072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 49), 'strtobool', False)
        # Calling strtobool(args, kwargs) (line 893)
        strtobool_call_result_2075 = invoke(stypy.reporting.localization.Localization(__file__, 893, 49), strtobool_2072, *[value_2073], **kwargs_2074)
        
        # Processing the call keyword arguments (line 893)
        kwargs_2076 = {}
        # Getting the type of 'setattr' (line 893)
        setattr_2069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 893)
        setattr_call_result_2077 = invoke(stypy.reporting.localization.Localization(__file__, 893, 20), setattr_2069, *[command_obj_2070, option_2071, strtobool_call_result_2075], **kwargs_2076)
        
        # SSA branch for the else part of an if statement (line 892)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to hasattr(...): (line 894)
        # Processing the call arguments (line 894)
        # Getting the type of 'command_obj' (line 894)
        command_obj_2079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 29), 'command_obj', False)
        # Getting the type of 'option' (line 894)
        option_2080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 42), 'option', False)
        # Processing the call keyword arguments (line 894)
        kwargs_2081 = {}
        # Getting the type of 'hasattr' (line 894)
        hasattr_2078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 21), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 894)
        hasattr_call_result_2082 = invoke(stypy.reporting.localization.Localization(__file__, 894, 21), hasattr_2078, *[command_obj_2079, option_2080], **kwargs_2081)
        
        # Testing the type of an if condition (line 894)
        if_condition_2083 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 894, 21), hasattr_call_result_2082)
        # Assigning a type to the variable 'if_condition_2083' (line 894)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 21), 'if_condition_2083', if_condition_2083)
        # SSA begins for if statement (line 894)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 895)
        # Processing the call arguments (line 895)
        # Getting the type of 'command_obj' (line 895)
        command_obj_2085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 28), 'command_obj', False)
        # Getting the type of 'option' (line 895)
        option_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 41), 'option', False)
        # Getting the type of 'value' (line 895)
        value_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 49), 'value', False)
        # Processing the call keyword arguments (line 895)
        kwargs_2088 = {}
        # Getting the type of 'setattr' (line 895)
        setattr_2084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 895)
        setattr_call_result_2089 = invoke(stypy.reporting.localization.Localization(__file__, 895, 20), setattr_2084, *[command_obj_2085, option_2086, value_2087], **kwargs_2088)
        
        # SSA branch for the else part of an if statement (line 894)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'DistutilsOptionError' (line 897)
        DistutilsOptionError_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 26), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 897, 20), DistutilsOptionError_2090, 'raise parameter', BaseException)
        # SSA join for if statement (line 894)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 892)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 890)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 888)
        # SSA branch for the except 'ValueError' branch of a try statement (line 888)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ValueError' (line 900)
        ValueError_2091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 19), 'ValueError')
        # Assigning a type to the variable 'msg' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'msg', ValueError_2091)
        # Getting the type of 'DistutilsOptionError' (line 901)
        DistutilsOptionError_2092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 901, 16), DistutilsOptionError_2092, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 888)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_command_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_command_options' in the type store
        # Getting the type of 'stypy_return_type' (line 860)
        stypy_return_type_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2093)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_command_options'
        return stypy_return_type_2093


    @norecursion
    def reinitialize_command(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_2094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 63), 'int')
        defaults = [int_2094]
        # Create a new context for function 'reinitialize_command'
        module_type_store = module_type_store.open_function_context('reinitialize_command', 903, 4, False)
        # Assigning a type to the variable 'self' (line 904)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_localization', localization)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_function_name', 'Distribution.reinitialize_command')
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_param_names_list', ['command', 'reinit_subcommands'])
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.reinitialize_command.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.reinitialize_command', ['command', 'reinit_subcommands'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reinitialize_command', localization, ['command', 'reinit_subcommands'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reinitialize_command(...)' code ##################

        str_2095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, (-1)), 'str', 'Reinitializes a command to the state it was in when first\n        returned by \'get_command_obj()\': ie., initialized but not yet\n        finalized.  This provides the opportunity to sneak option\n        values in programmatically, overriding or supplementing\n        user-supplied values from the config files and command line.\n        You\'ll have to re-finalize the command object (by calling\n        \'finalize_options()\' or \'ensure_finalized()\') before using it for\n        real.\n\n        \'command\' should be a command name (string) or command object.  If\n        \'reinit_subcommands\' is true, also reinitializes the command\'s\n        sub-commands, as declared by the \'sub_commands\' class attribute (if\n        it has one).  See the "install" command for an example.  Only\n        reinitializes the sub-commands that actually matter, ie. those\n        whose test predicates return true.\n\n        Returns the reinitialized command object.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 922, 8))
        
        # 'from distutils.cmd import Command' statement (line 922)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_2096 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 922, 8), 'distutils.cmd')

        if (type(import_2096) is not StypyTypeError):

            if (import_2096 != 'pyd_module'):
                __import__(import_2096)
                sys_modules_2097 = sys.modules[import_2096]
                import_from_module(stypy.reporting.localization.Localization(__file__, 922, 8), 'distutils.cmd', sys_modules_2097.module_type_store, module_type_store, ['Command'])
                nest_module(stypy.reporting.localization.Localization(__file__, 922, 8), __file__, sys_modules_2097, sys_modules_2097.module_type_store, module_type_store)
            else:
                from distutils.cmd import Command

                import_from_module(stypy.reporting.localization.Localization(__file__, 922, 8), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

        else:
            # Assigning a type to the variable 'distutils.cmd' (line 922)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'distutils.cmd', import_2096)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        
        
        # Call to isinstance(...): (line 923)
        # Processing the call arguments (line 923)
        # Getting the type of 'command' (line 923)
        command_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 26), 'command', False)
        # Getting the type of 'Command' (line 923)
        Command_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 35), 'Command', False)
        # Processing the call keyword arguments (line 923)
        kwargs_2101 = {}
        # Getting the type of 'isinstance' (line 923)
        isinstance_2098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 923)
        isinstance_call_result_2102 = invoke(stypy.reporting.localization.Localization(__file__, 923, 15), isinstance_2098, *[command_2099, Command_2100], **kwargs_2101)
        
        # Applying the 'not' unary operator (line 923)
        result_not__2103 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 11), 'not', isinstance_call_result_2102)
        
        # Testing the type of an if condition (line 923)
        if_condition_2104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 923, 8), result_not__2103)
        # Assigning a type to the variable 'if_condition_2104' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'if_condition_2104', if_condition_2104)
        # SSA begins for if statement (line 923)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 924):
        
        # Assigning a Name to a Name (line 924):
        # Getting the type of 'command' (line 924)
        command_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 27), 'command')
        # Assigning a type to the variable 'command_name' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'command_name', command_2105)
        
        # Assigning a Call to a Name (line 925):
        
        # Assigning a Call to a Name (line 925):
        
        # Call to get_command_obj(...): (line 925)
        # Processing the call arguments (line 925)
        # Getting the type of 'command_name' (line 925)
        command_name_2108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 43), 'command_name', False)
        # Processing the call keyword arguments (line 925)
        kwargs_2109 = {}
        # Getting the type of 'self' (line 925)
        self_2106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 22), 'self', False)
        # Obtaining the member 'get_command_obj' of a type (line 925)
        get_command_obj_2107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 22), self_2106, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 925)
        get_command_obj_call_result_2110 = invoke(stypy.reporting.localization.Localization(__file__, 925, 22), get_command_obj_2107, *[command_name_2108], **kwargs_2109)
        
        # Assigning a type to the variable 'command' (line 925)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'command', get_command_obj_call_result_2110)
        # SSA branch for the else part of an if statement (line 923)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 927):
        
        # Assigning a Call to a Name (line 927):
        
        # Call to get_command_name(...): (line 927)
        # Processing the call keyword arguments (line 927)
        kwargs_2113 = {}
        # Getting the type of 'command' (line 927)
        command_2111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 27), 'command', False)
        # Obtaining the member 'get_command_name' of a type (line 927)
        get_command_name_2112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 27), command_2111, 'get_command_name')
        # Calling get_command_name(args, kwargs) (line 927)
        get_command_name_call_result_2114 = invoke(stypy.reporting.localization.Localization(__file__, 927, 27), get_command_name_2112, *[], **kwargs_2113)
        
        # Assigning a type to the variable 'command_name' (line 927)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 12), 'command_name', get_command_name_call_result_2114)
        # SSA join for if statement (line 923)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'command' (line 929)
        command_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 15), 'command')
        # Obtaining the member 'finalized' of a type (line 929)
        finalized_2116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 15), command_2115, 'finalized')
        # Applying the 'not' unary operator (line 929)
        result_not__2117 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 11), 'not', finalized_2116)
        
        # Testing the type of an if condition (line 929)
        if_condition_2118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 929, 8), result_not__2117)
        # Assigning a type to the variable 'if_condition_2118' (line 929)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 8), 'if_condition_2118', if_condition_2118)
        # SSA begins for if statement (line 929)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'command' (line 930)
        command_2119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 19), 'command')
        # Assigning a type to the variable 'stypy_return_type' (line 930)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'stypy_return_type', command_2119)
        # SSA join for if statement (line 929)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to initialize_options(...): (line 931)
        # Processing the call keyword arguments (line 931)
        kwargs_2122 = {}
        # Getting the type of 'command' (line 931)
        command_2120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'command', False)
        # Obtaining the member 'initialize_options' of a type (line 931)
        initialize_options_2121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 8), command_2120, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 931)
        initialize_options_call_result_2123 = invoke(stypy.reporting.localization.Localization(__file__, 931, 8), initialize_options_2121, *[], **kwargs_2122)
        
        
        # Assigning a Num to a Attribute (line 932):
        
        # Assigning a Num to a Attribute (line 932):
        int_2124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 28), 'int')
        # Getting the type of 'command' (line 932)
        command_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'command')
        # Setting the type of the member 'finalized' of a type (line 932)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 8), command_2125, 'finalized', int_2124)
        
        # Assigning a Num to a Subscript (line 933):
        
        # Assigning a Num to a Subscript (line 933):
        int_2126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 38), 'int')
        # Getting the type of 'self' (line 933)
        self_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 8), 'self')
        # Obtaining the member 'have_run' of a type (line 933)
        have_run_2128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 8), self_2127, 'have_run')
        # Getting the type of 'command_name' (line 933)
        command_name_2129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 22), 'command_name')
        # Storing an element on a container (line 933)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 8), have_run_2128, (command_name_2129, int_2126))
        
        # Call to _set_command_options(...): (line 934)
        # Processing the call arguments (line 934)
        # Getting the type of 'command' (line 934)
        command_2132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 34), 'command', False)
        # Processing the call keyword arguments (line 934)
        kwargs_2133 = {}
        # Getting the type of 'self' (line 934)
        self_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'self', False)
        # Obtaining the member '_set_command_options' of a type (line 934)
        _set_command_options_2131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 8), self_2130, '_set_command_options')
        # Calling _set_command_options(args, kwargs) (line 934)
        _set_command_options_call_result_2134 = invoke(stypy.reporting.localization.Localization(__file__, 934, 8), _set_command_options_2131, *[command_2132], **kwargs_2133)
        
        
        # Getting the type of 'reinit_subcommands' (line 936)
        reinit_subcommands_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 11), 'reinit_subcommands')
        # Testing the type of an if condition (line 936)
        if_condition_2136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 936, 8), reinit_subcommands_2135)
        # Assigning a type to the variable 'if_condition_2136' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'if_condition_2136', if_condition_2136)
        # SSA begins for if statement (line 936)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get_sub_commands(...): (line 937)
        # Processing the call keyword arguments (line 937)
        kwargs_2139 = {}
        # Getting the type of 'command' (line 937)
        command_2137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 23), 'command', False)
        # Obtaining the member 'get_sub_commands' of a type (line 937)
        get_sub_commands_2138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 23), command_2137, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 937)
        get_sub_commands_call_result_2140 = invoke(stypy.reporting.localization.Localization(__file__, 937, 23), get_sub_commands_2138, *[], **kwargs_2139)
        
        # Testing the type of a for loop iterable (line 937)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 937, 12), get_sub_commands_call_result_2140)
        # Getting the type of the for loop variable (line 937)
        for_loop_var_2141 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 937, 12), get_sub_commands_call_result_2140)
        # Assigning a type to the variable 'sub' (line 937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 12), 'sub', for_loop_var_2141)
        # SSA begins for a for statement (line 937)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to reinitialize_command(...): (line 938)
        # Processing the call arguments (line 938)
        # Getting the type of 'sub' (line 938)
        sub_2144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 42), 'sub', False)
        # Getting the type of 'reinit_subcommands' (line 938)
        reinit_subcommands_2145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 47), 'reinit_subcommands', False)
        # Processing the call keyword arguments (line 938)
        kwargs_2146 = {}
        # Getting the type of 'self' (line 938)
        self_2142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 16), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 938)
        reinitialize_command_2143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 16), self_2142, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 938)
        reinitialize_command_call_result_2147 = invoke(stypy.reporting.localization.Localization(__file__, 938, 16), reinitialize_command_2143, *[sub_2144, reinit_subcommands_2145], **kwargs_2146)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 936)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'command' (line 940)
        command_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 15), 'command')
        # Assigning a type to the variable 'stypy_return_type' (line 940)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'stypy_return_type', command_2148)
        
        # ################# End of 'reinitialize_command(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reinitialize_command' in the type store
        # Getting the type of 'stypy_return_type' (line 903)
        stypy_return_type_2149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reinitialize_command'
        return stypy_return_type_2149


    @norecursion
    def announce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'log' (line 944)
        log_2150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 34), 'log')
        # Obtaining the member 'INFO' of a type (line 944)
        INFO_2151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 34), log_2150, 'INFO')
        defaults = [INFO_2151]
        # Create a new context for function 'announce'
        module_type_store = module_type_store.open_function_context('announce', 944, 4, False)
        # Assigning a type to the variable 'self' (line 945)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.announce.__dict__.__setitem__('stypy_localization', localization)
        Distribution.announce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.announce.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.announce.__dict__.__setitem__('stypy_function_name', 'Distribution.announce')
        Distribution.announce.__dict__.__setitem__('stypy_param_names_list', ['msg', 'level'])
        Distribution.announce.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.announce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.announce.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.announce.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.announce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.announce.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.announce', ['msg', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'announce', localization, ['msg', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'announce(...)' code ##################

        
        # Call to log(...): (line 945)
        # Processing the call arguments (line 945)
        # Getting the type of 'level' (line 945)
        level_2154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 16), 'level', False)
        # Getting the type of 'msg' (line 945)
        msg_2155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 23), 'msg', False)
        # Processing the call keyword arguments (line 945)
        kwargs_2156 = {}
        # Getting the type of 'log' (line 945)
        log_2152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 8), 'log', False)
        # Obtaining the member 'log' of a type (line 945)
        log_2153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 945, 8), log_2152, 'log')
        # Calling log(args, kwargs) (line 945)
        log_call_result_2157 = invoke(stypy.reporting.localization.Localization(__file__, 945, 8), log_2153, *[level_2154, msg_2155], **kwargs_2156)
        
        
        # ################# End of 'announce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'announce' in the type store
        # Getting the type of 'stypy_return_type' (line 944)
        stypy_return_type_2158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'announce'
        return stypy_return_type_2158


    @norecursion
    def run_commands(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_commands'
        module_type_store = module_type_store.open_function_context('run_commands', 947, 4, False)
        # Assigning a type to the variable 'self' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.run_commands.__dict__.__setitem__('stypy_localization', localization)
        Distribution.run_commands.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.run_commands.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.run_commands.__dict__.__setitem__('stypy_function_name', 'Distribution.run_commands')
        Distribution.run_commands.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.run_commands.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.run_commands.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.run_commands.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.run_commands.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.run_commands.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.run_commands.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.run_commands', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run_commands', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run_commands(...)' code ##################

        str_2159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, (-1)), 'str', "Run each command that was seen on the setup script command line.\n        Uses the list of commands found and cache of command objects\n        created by 'get_command_obj()'.\n        ")
        
        # Getting the type of 'self' (line 952)
        self_2160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 19), 'self')
        # Obtaining the member 'commands' of a type (line 952)
        commands_2161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 19), self_2160, 'commands')
        # Testing the type of a for loop iterable (line 952)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 952, 8), commands_2161)
        # Getting the type of the for loop variable (line 952)
        for_loop_var_2162 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 952, 8), commands_2161)
        # Assigning a type to the variable 'cmd' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'cmd', for_loop_var_2162)
        # SSA begins for a for statement (line 952)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to run_command(...): (line 953)
        # Processing the call arguments (line 953)
        # Getting the type of 'cmd' (line 953)
        cmd_2165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 29), 'cmd', False)
        # Processing the call keyword arguments (line 953)
        kwargs_2166 = {}
        # Getting the type of 'self' (line 953)
        self_2163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 953)
        run_command_2164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 953, 12), self_2163, 'run_command')
        # Calling run_command(args, kwargs) (line 953)
        run_command_call_result_2167 = invoke(stypy.reporting.localization.Localization(__file__, 953, 12), run_command_2164, *[cmd_2165], **kwargs_2166)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run_commands(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_commands' in the type store
        # Getting the type of 'stypy_return_type' (line 947)
        stypy_return_type_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2168)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_commands'
        return stypy_return_type_2168


    @norecursion
    def run_command(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_command'
        module_type_store = module_type_store.open_function_context('run_command', 957, 4, False)
        # Assigning a type to the variable 'self' (line 958)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.run_command.__dict__.__setitem__('stypy_localization', localization)
        Distribution.run_command.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.run_command.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.run_command.__dict__.__setitem__('stypy_function_name', 'Distribution.run_command')
        Distribution.run_command.__dict__.__setitem__('stypy_param_names_list', ['command'])
        Distribution.run_command.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.run_command.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.run_command.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.run_command.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.run_command.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.run_command.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.run_command', ['command'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run_command', localization, ['command'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run_command(...)' code ##################

        str_2169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, (-1)), 'str', "Do whatever it takes to run a command (including nothing at all,\n        if the command has already been run).  Specifically: if we have\n        already created and run the command named by 'command', return\n        silently without doing anything.  If the command named by 'command'\n        doesn't even have a command object yet, create one.  Then invoke\n        'run()' on that command object (or an existing one).\n        ")
        
        
        # Call to get(...): (line 966)
        # Processing the call arguments (line 966)
        # Getting the type of 'command' (line 966)
        command_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 29), 'command', False)
        # Processing the call keyword arguments (line 966)
        kwargs_2174 = {}
        # Getting the type of 'self' (line 966)
        self_2170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 11), 'self', False)
        # Obtaining the member 'have_run' of a type (line 966)
        have_run_2171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 11), self_2170, 'have_run')
        # Obtaining the member 'get' of a type (line 966)
        get_2172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 11), have_run_2171, 'get')
        # Calling get(args, kwargs) (line 966)
        get_call_result_2175 = invoke(stypy.reporting.localization.Localization(__file__, 966, 11), get_2172, *[command_2173], **kwargs_2174)
        
        # Testing the type of an if condition (line 966)
        if_condition_2176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 966, 8), get_call_result_2175)
        # Assigning a type to the variable 'if_condition_2176' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'if_condition_2176', if_condition_2176)
        # SSA begins for if statement (line 966)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 966)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 969)
        # Processing the call arguments (line 969)
        str_2179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 17), 'str', 'running %s')
        # Getting the type of 'command' (line 969)
        command_2180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 31), 'command', False)
        # Processing the call keyword arguments (line 969)
        kwargs_2181 = {}
        # Getting the type of 'log' (line 969)
        log_2177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 969)
        info_2178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 8), log_2177, 'info')
        # Calling info(args, kwargs) (line 969)
        info_call_result_2182 = invoke(stypy.reporting.localization.Localization(__file__, 969, 8), info_2178, *[str_2179, command_2180], **kwargs_2181)
        
        
        # Assigning a Call to a Name (line 970):
        
        # Assigning a Call to a Name (line 970):
        
        # Call to get_command_obj(...): (line 970)
        # Processing the call arguments (line 970)
        # Getting the type of 'command' (line 970)
        command_2185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 39), 'command', False)
        # Processing the call keyword arguments (line 970)
        kwargs_2186 = {}
        # Getting the type of 'self' (line 970)
        self_2183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 18), 'self', False)
        # Obtaining the member 'get_command_obj' of a type (line 970)
        get_command_obj_2184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 18), self_2183, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 970)
        get_command_obj_call_result_2187 = invoke(stypy.reporting.localization.Localization(__file__, 970, 18), get_command_obj_2184, *[command_2185], **kwargs_2186)
        
        # Assigning a type to the variable 'cmd_obj' (line 970)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 8), 'cmd_obj', get_command_obj_call_result_2187)
        
        # Call to ensure_finalized(...): (line 971)
        # Processing the call keyword arguments (line 971)
        kwargs_2190 = {}
        # Getting the type of 'cmd_obj' (line 971)
        cmd_obj_2188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'cmd_obj', False)
        # Obtaining the member 'ensure_finalized' of a type (line 971)
        ensure_finalized_2189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 8), cmd_obj_2188, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 971)
        ensure_finalized_call_result_2191 = invoke(stypy.reporting.localization.Localization(__file__, 971, 8), ensure_finalized_2189, *[], **kwargs_2190)
        
        
        # Call to run(...): (line 972)
        # Processing the call keyword arguments (line 972)
        kwargs_2194 = {}
        # Getting the type of 'cmd_obj' (line 972)
        cmd_obj_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'cmd_obj', False)
        # Obtaining the member 'run' of a type (line 972)
        run_2193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 8), cmd_obj_2192, 'run')
        # Calling run(args, kwargs) (line 972)
        run_call_result_2195 = invoke(stypy.reporting.localization.Localization(__file__, 972, 8), run_2193, *[], **kwargs_2194)
        
        
        # Assigning a Num to a Subscript (line 973):
        
        # Assigning a Num to a Subscript (line 973):
        int_2196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 33), 'int')
        # Getting the type of 'self' (line 973)
        self_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'self')
        # Obtaining the member 'have_run' of a type (line 973)
        have_run_2198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 8), self_2197, 'have_run')
        # Getting the type of 'command' (line 973)
        command_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 22), 'command')
        # Storing an element on a container (line 973)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 8), have_run_2198, (command_2199, int_2196))
        
        # ################# End of 'run_command(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_command' in the type store
        # Getting the type of 'stypy_return_type' (line 957)
        stypy_return_type_2200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_command'
        return stypy_return_type_2200


    @norecursion
    def has_pure_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_pure_modules'
        module_type_store = module_type_store.open_function_context('has_pure_modules', 978, 4, False)
        # Assigning a type to the variable 'self' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_function_name', 'Distribution.has_pure_modules')
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_pure_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_pure_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_pure_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_pure_modules(...)' code ##################

        
        
        # Call to len(...): (line 979)
        # Processing the call arguments (line 979)
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 979)
        self_2202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 19), 'self', False)
        # Obtaining the member 'packages' of a type (line 979)
        packages_2203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 19), self_2202, 'packages')
        # Getting the type of 'self' (line 979)
        self_2204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 36), 'self', False)
        # Obtaining the member 'py_modules' of a type (line 979)
        py_modules_2205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 36), self_2204, 'py_modules')
        # Applying the binary operator 'or' (line 979)
        result_or_keyword_2206 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 19), 'or', packages_2203, py_modules_2205)
        
        # Obtaining an instance of the builtin type 'list' (line 979)
        list_2207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 979)
        
        # Applying the binary operator 'or' (line 979)
        result_or_keyword_2208 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 19), 'or', result_or_keyword_2206, list_2207)
        
        # Processing the call keyword arguments (line 979)
        kwargs_2209 = {}
        # Getting the type of 'len' (line 979)
        len_2201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 15), 'len', False)
        # Calling len(args, kwargs) (line 979)
        len_call_result_2210 = invoke(stypy.reporting.localization.Localization(__file__, 979, 15), len_2201, *[result_or_keyword_2208], **kwargs_2209)
        
        int_2211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 61), 'int')
        # Applying the binary operator '>' (line 979)
        result_gt_2212 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 15), '>', len_call_result_2210, int_2211)
        
        # Assigning a type to the variable 'stypy_return_type' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'stypy_return_type', result_gt_2212)
        
        # ################# End of 'has_pure_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_pure_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 978)
        stypy_return_type_2213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_pure_modules'
        return stypy_return_type_2213


    @norecursion
    def has_ext_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_ext_modules'
        module_type_store = module_type_store.open_function_context('has_ext_modules', 981, 4, False)
        # Assigning a type to the variable 'self' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_function_name', 'Distribution.has_ext_modules')
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_ext_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_ext_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_ext_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_ext_modules(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 982)
        self_2214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 15), 'self')
        # Obtaining the member 'ext_modules' of a type (line 982)
        ext_modules_2215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 15), self_2214, 'ext_modules')
        
        
        # Call to len(...): (line 982)
        # Processing the call arguments (line 982)
        # Getting the type of 'self' (line 982)
        self_2217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 40), 'self', False)
        # Obtaining the member 'ext_modules' of a type (line 982)
        ext_modules_2218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 40), self_2217, 'ext_modules')
        # Processing the call keyword arguments (line 982)
        kwargs_2219 = {}
        # Getting the type of 'len' (line 982)
        len_2216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 36), 'len', False)
        # Calling len(args, kwargs) (line 982)
        len_call_result_2220 = invoke(stypy.reporting.localization.Localization(__file__, 982, 36), len_2216, *[ext_modules_2218], **kwargs_2219)
        
        int_2221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 60), 'int')
        # Applying the binary operator '>' (line 982)
        result_gt_2222 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 36), '>', len_call_result_2220, int_2221)
        
        # Applying the binary operator 'and' (line 982)
        result_and_keyword_2223 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 15), 'and', ext_modules_2215, result_gt_2222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 8), 'stypy_return_type', result_and_keyword_2223)
        
        # ################# End of 'has_ext_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_ext_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 981)
        stypy_return_type_2224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_ext_modules'
        return stypy_return_type_2224


    @norecursion
    def has_c_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_c_libraries'
        module_type_store = module_type_store.open_function_context('has_c_libraries', 984, 4, False)
        # Assigning a type to the variable 'self' (line 985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_function_name', 'Distribution.has_c_libraries')
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_c_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_c_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_c_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_c_libraries(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 985)
        self_2225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 15), 'self')
        # Obtaining the member 'libraries' of a type (line 985)
        libraries_2226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 985, 15), self_2225, 'libraries')
        
        
        # Call to len(...): (line 985)
        # Processing the call arguments (line 985)
        # Getting the type of 'self' (line 985)
        self_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 38), 'self', False)
        # Obtaining the member 'libraries' of a type (line 985)
        libraries_2229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 985, 38), self_2228, 'libraries')
        # Processing the call keyword arguments (line 985)
        kwargs_2230 = {}
        # Getting the type of 'len' (line 985)
        len_2227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 34), 'len', False)
        # Calling len(args, kwargs) (line 985)
        len_call_result_2231 = invoke(stypy.reporting.localization.Localization(__file__, 985, 34), len_2227, *[libraries_2229], **kwargs_2230)
        
        int_2232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 56), 'int')
        # Applying the binary operator '>' (line 985)
        result_gt_2233 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 34), '>', len_call_result_2231, int_2232)
        
        # Applying the binary operator 'and' (line 985)
        result_and_keyword_2234 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 15), 'and', libraries_2226, result_gt_2233)
        
        # Assigning a type to the variable 'stypy_return_type' (line 985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 8), 'stypy_return_type', result_and_keyword_2234)
        
        # ################# End of 'has_c_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_c_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 984)
        stypy_return_type_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_c_libraries'
        return stypy_return_type_2235


    @norecursion
    def has_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_modules'
        module_type_store = module_type_store.open_function_context('has_modules', 987, 4, False)
        # Assigning a type to the variable 'self' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_modules.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_modules.__dict__.__setitem__('stypy_function_name', 'Distribution.has_modules')
        Distribution.has_modules.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_modules(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to has_pure_modules(...): (line 988)
        # Processing the call keyword arguments (line 988)
        kwargs_2238 = {}
        # Getting the type of 'self' (line 988)
        self_2236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 15), 'self', False)
        # Obtaining the member 'has_pure_modules' of a type (line 988)
        has_pure_modules_2237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 15), self_2236, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 988)
        has_pure_modules_call_result_2239 = invoke(stypy.reporting.localization.Localization(__file__, 988, 15), has_pure_modules_2237, *[], **kwargs_2238)
        
        
        # Call to has_ext_modules(...): (line 988)
        # Processing the call keyword arguments (line 988)
        kwargs_2242 = {}
        # Getting the type of 'self' (line 988)
        self_2240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 42), 'self', False)
        # Obtaining the member 'has_ext_modules' of a type (line 988)
        has_ext_modules_2241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 42), self_2240, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 988)
        has_ext_modules_call_result_2243 = invoke(stypy.reporting.localization.Localization(__file__, 988, 42), has_ext_modules_2241, *[], **kwargs_2242)
        
        # Applying the binary operator 'or' (line 988)
        result_or_keyword_2244 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 15), 'or', has_pure_modules_call_result_2239, has_ext_modules_call_result_2243)
        
        # Assigning a type to the variable 'stypy_return_type' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 8), 'stypy_return_type', result_or_keyword_2244)
        
        # ################# End of 'has_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 987)
        stypy_return_type_2245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_modules'
        return stypy_return_type_2245


    @norecursion
    def has_headers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_headers'
        module_type_store = module_type_store.open_function_context('has_headers', 990, 4, False)
        # Assigning a type to the variable 'self' (line 991)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_headers.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_headers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_headers.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_headers.__dict__.__setitem__('stypy_function_name', 'Distribution.has_headers')
        Distribution.has_headers.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_headers.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_headers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_headers.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_headers.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_headers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_headers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_headers', [], None, None, defaults, varargs, kwargs)

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

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 991)
        self_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 15), 'self')
        # Obtaining the member 'headers' of a type (line 991)
        headers_2247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 15), self_2246, 'headers')
        
        
        # Call to len(...): (line 991)
        # Processing the call arguments (line 991)
        # Getting the type of 'self' (line 991)
        self_2249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 36), 'self', False)
        # Obtaining the member 'headers' of a type (line 991)
        headers_2250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 36), self_2249, 'headers')
        # Processing the call keyword arguments (line 991)
        kwargs_2251 = {}
        # Getting the type of 'len' (line 991)
        len_2248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 32), 'len', False)
        # Calling len(args, kwargs) (line 991)
        len_call_result_2252 = invoke(stypy.reporting.localization.Localization(__file__, 991, 32), len_2248, *[headers_2250], **kwargs_2251)
        
        int_2253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 52), 'int')
        # Applying the binary operator '>' (line 991)
        result_gt_2254 = python_operator(stypy.reporting.localization.Localization(__file__, 991, 32), '>', len_call_result_2252, int_2253)
        
        # Applying the binary operator 'and' (line 991)
        result_and_keyword_2255 = python_operator(stypy.reporting.localization.Localization(__file__, 991, 15), 'and', headers_2247, result_gt_2254)
        
        # Assigning a type to the variable 'stypy_return_type' (line 991)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 8), 'stypy_return_type', result_and_keyword_2255)
        
        # ################# End of 'has_headers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_headers' in the type store
        # Getting the type of 'stypy_return_type' (line 990)
        stypy_return_type_2256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_headers'
        return stypy_return_type_2256


    @norecursion
    def has_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_scripts'
        module_type_store = module_type_store.open_function_context('has_scripts', 993, 4, False)
        # Assigning a type to the variable 'self' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_scripts.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_scripts.__dict__.__setitem__('stypy_function_name', 'Distribution.has_scripts')
        Distribution.has_scripts.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_scripts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_scripts', [], None, None, defaults, varargs, kwargs)

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

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 994)
        self_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 15), 'self')
        # Obtaining the member 'scripts' of a type (line 994)
        scripts_2258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 15), self_2257, 'scripts')
        
        
        # Call to len(...): (line 994)
        # Processing the call arguments (line 994)
        # Getting the type of 'self' (line 994)
        self_2260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 36), 'self', False)
        # Obtaining the member 'scripts' of a type (line 994)
        scripts_2261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 36), self_2260, 'scripts')
        # Processing the call keyword arguments (line 994)
        kwargs_2262 = {}
        # Getting the type of 'len' (line 994)
        len_2259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 32), 'len', False)
        # Calling len(args, kwargs) (line 994)
        len_call_result_2263 = invoke(stypy.reporting.localization.Localization(__file__, 994, 32), len_2259, *[scripts_2261], **kwargs_2262)
        
        int_2264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 52), 'int')
        # Applying the binary operator '>' (line 994)
        result_gt_2265 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 32), '>', len_call_result_2263, int_2264)
        
        # Applying the binary operator 'and' (line 994)
        result_and_keyword_2266 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 15), 'and', scripts_2258, result_gt_2265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 8), 'stypy_return_type', result_and_keyword_2266)
        
        # ################# End of 'has_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 993)
        stypy_return_type_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_scripts'
        return stypy_return_type_2267


    @norecursion
    def has_data_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_data_files'
        module_type_store = module_type_store.open_function_context('has_data_files', 996, 4, False)
        # Assigning a type to the variable 'self' (line 997)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.has_data_files.__dict__.__setitem__('stypy_localization', localization)
        Distribution.has_data_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.has_data_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.has_data_files.__dict__.__setitem__('stypy_function_name', 'Distribution.has_data_files')
        Distribution.has_data_files.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.has_data_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.has_data_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.has_data_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.has_data_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.has_data_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.has_data_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.has_data_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_data_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_data_files(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 997)
        self_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 15), 'self')
        # Obtaining the member 'data_files' of a type (line 997)
        data_files_2269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 997, 15), self_2268, 'data_files')
        
        
        # Call to len(...): (line 997)
        # Processing the call arguments (line 997)
        # Getting the type of 'self' (line 997)
        self_2271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 39), 'self', False)
        # Obtaining the member 'data_files' of a type (line 997)
        data_files_2272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 997, 39), self_2271, 'data_files')
        # Processing the call keyword arguments (line 997)
        kwargs_2273 = {}
        # Getting the type of 'len' (line 997)
        len_2270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 35), 'len', False)
        # Calling len(args, kwargs) (line 997)
        len_call_result_2274 = invoke(stypy.reporting.localization.Localization(__file__, 997, 35), len_2270, *[data_files_2272], **kwargs_2273)
        
        int_2275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 58), 'int')
        # Applying the binary operator '>' (line 997)
        result_gt_2276 = python_operator(stypy.reporting.localization.Localization(__file__, 997, 35), '>', len_call_result_2274, int_2275)
        
        # Applying the binary operator 'and' (line 997)
        result_and_keyword_2277 = python_operator(stypy.reporting.localization.Localization(__file__, 997, 15), 'and', data_files_2269, result_gt_2276)
        
        # Assigning a type to the variable 'stypy_return_type' (line 997)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), 'stypy_return_type', result_and_keyword_2277)
        
        # ################# End of 'has_data_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_data_files' in the type store
        # Getting the type of 'stypy_return_type' (line 996)
        stypy_return_type_2278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2278)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_data_files'
        return stypy_return_type_2278


    @norecursion
    def is_pure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_pure'
        module_type_store = module_type_store.open_function_context('is_pure', 999, 4, False)
        # Assigning a type to the variable 'self' (line 1000)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Distribution.is_pure.__dict__.__setitem__('stypy_localization', localization)
        Distribution.is_pure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Distribution.is_pure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Distribution.is_pure.__dict__.__setitem__('stypy_function_name', 'Distribution.is_pure')
        Distribution.is_pure.__dict__.__setitem__('stypy_param_names_list', [])
        Distribution.is_pure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Distribution.is_pure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Distribution.is_pure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Distribution.is_pure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Distribution.is_pure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Distribution.is_pure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Distribution.is_pure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_pure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_pure(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to has_pure_modules(...): (line 1000)
        # Processing the call keyword arguments (line 1000)
        kwargs_2281 = {}
        # Getting the type of 'self' (line 1000)
        self_2279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 16), 'self', False)
        # Obtaining the member 'has_pure_modules' of a type (line 1000)
        has_pure_modules_2280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 16), self_2279, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 1000)
        has_pure_modules_call_result_2282 = invoke(stypy.reporting.localization.Localization(__file__, 1000, 16), has_pure_modules_2280, *[], **kwargs_2281)
        
        
        
        # Call to has_ext_modules(...): (line 1001)
        # Processing the call keyword arguments (line 1001)
        kwargs_2285 = {}
        # Getting the type of 'self' (line 1001)
        self_2283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 20), 'self', False)
        # Obtaining the member 'has_ext_modules' of a type (line 1001)
        has_ext_modules_2284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 20), self_2283, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 1001)
        has_ext_modules_call_result_2286 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 20), has_ext_modules_2284, *[], **kwargs_2285)
        
        # Applying the 'not' unary operator (line 1001)
        result_not__2287 = python_operator(stypy.reporting.localization.Localization(__file__, 1001, 16), 'not', has_ext_modules_call_result_2286)
        
        # Applying the binary operator 'and' (line 1000)
        result_and_keyword_2288 = python_operator(stypy.reporting.localization.Localization(__file__, 1000, 16), 'and', has_pure_modules_call_result_2282, result_not__2287)
        
        
        # Call to has_c_libraries(...): (line 1002)
        # Processing the call keyword arguments (line 1002)
        kwargs_2291 = {}
        # Getting the type of 'self' (line 1002)
        self_2289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 20), 'self', False)
        # Obtaining the member 'has_c_libraries' of a type (line 1002)
        has_c_libraries_2290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 20), self_2289, 'has_c_libraries')
        # Calling has_c_libraries(args, kwargs) (line 1002)
        has_c_libraries_call_result_2292 = invoke(stypy.reporting.localization.Localization(__file__, 1002, 20), has_c_libraries_2290, *[], **kwargs_2291)
        
        # Applying the 'not' unary operator (line 1002)
        result_not__2293 = python_operator(stypy.reporting.localization.Localization(__file__, 1002, 16), 'not', has_c_libraries_call_result_2292)
        
        # Applying the binary operator 'and' (line 1000)
        result_and_keyword_2294 = python_operator(stypy.reporting.localization.Localization(__file__, 1000, 16), 'and', result_and_keyword_2288, result_not__2293)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1000)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'stypy_return_type', result_and_keyword_2294)
        
        # ################# End of 'is_pure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_pure' in the type store
        # Getting the type of 'stypy_return_type' (line 999)
        stypy_return_type_2295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_pure'
        return stypy_return_type_2295


# Assigning a type to the variable 'Distribution' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'Distribution', Distribution)

# Assigning a List to a Name (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_2296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_2297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
str_2298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'str', 'verbose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), tuple_2297, str_2298)
# Adding element type (line 57)
str_2299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'str', 'v')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), tuple_2297, str_2299)
# Adding element type (line 57)
str_2300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 39), 'str', 'run verbosely (default)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), tuple_2297, str_2300)
# Adding element type (line 57)
int_2301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), tuple_2297, int_2301)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_2296, tuple_2297)
# Adding element type (line 57)

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_2302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)
str_2303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'str', 'quiet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), tuple_2302, str_2303)
# Adding element type (line 58)
str_2304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 32), 'str', 'q')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), tuple_2302, str_2304)
# Adding element type (line 58)
str_2305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 37), 'str', 'run quietly (turns verbosity off)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), tuple_2302, str_2305)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_2296, tuple_2302)
# Adding element type (line 57)

# Obtaining an instance of the builtin type 'tuple' (line 59)
tuple_2306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 59)
# Adding element type (line 59)
str_2307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'str', 'dry-run')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), tuple_2306, str_2307)
# Adding element type (line 59)
str_2308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'str', 'n')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), tuple_2306, str_2308)
# Adding element type (line 59)
str_2309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 39), 'str', "don't actually do anything")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), tuple_2306, str_2309)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_2296, tuple_2306)
# Adding element type (line 57)

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_2310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)
str_2311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'str', 'help')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_2310, str_2311)
# Adding element type (line 60)
str_2312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'str', 'h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_2310, str_2312)
# Adding element type (line 60)
str_2313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'str', 'show detailed help message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_2310, str_2313)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_2296, tuple_2310)
# Adding element type (line 57)

# Obtaining an instance of the builtin type 'tuple' (line 61)
tuple_2314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 61)
# Adding element type (line 61)
str_2315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'str', 'no-user-cfg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_2314, str_2315)
# Adding element type (line 61)
# Getting the type of 'None' (line 61)
None_2316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_2314, None_2316)
# Adding element type (line 61)
str_2317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'str', 'ignore pydistutils.cfg in your home directory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_2314, str_2317)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_2296, tuple_2314)

# Getting the type of 'Distribution'
Distribution_2318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Distribution')
# Setting the type of the member 'global_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Distribution_2318, 'global_options', list_2296)

# Assigning a Str to a Name (line 67):
str_2319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "Common commands: (see '--help-commands' for more)\n\n  setup.py build      will build the package underneath 'build/'\n  setup.py install    will install the package\n")
# Getting the type of 'Distribution'
Distribution_2320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Distribution')
# Setting the type of the member 'common_usage' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Distribution_2320, 'common_usage', str_2319)

# Assigning a List to a Name (line 75):

# Obtaining an instance of the builtin type 'list' (line 75)
list_2321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 75)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_2322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_2323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'str', 'help-commands')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_2322, str_2323)
# Adding element type (line 76)
# Getting the type of 'None' (line 76)
None_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_2322, None_2324)
# Adding element type (line 76)
str_2325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'str', 'list all available commands')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_2322, str_2325)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2322)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_2326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_2327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'str', 'name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_2326, str_2327)
# Adding element type (line 78)
# Getting the type of 'None' (line 78)
None_2328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_2326, None_2328)
# Adding element type (line 78)
str_2329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 9), 'str', 'print package name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_2326, str_2329)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2326)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_2330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_2331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 9), 'str', 'version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 9), tuple_2330, str_2331)
# Adding element type (line 80)
str_2332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'str', 'V')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 9), tuple_2330, str_2332)
# Adding element type (line 80)
str_2333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 9), 'str', 'print package version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 9), tuple_2330, str_2333)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2330)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_2334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
str_2335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'str', 'fullname')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), tuple_2334, str_2335)
# Adding element type (line 82)
# Getting the type of 'None' (line 82)
None_2336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), tuple_2334, None_2336)
# Adding element type (line 82)
str_2337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'str', 'print <package name>-<version>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), tuple_2334, str_2337)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2334)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_2338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
str_2339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'str', 'author')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 9), tuple_2338, str_2339)
# Adding element type (line 84)
# Getting the type of 'None' (line 84)
None_2340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 9), tuple_2338, None_2340)
# Adding element type (line 84)
str_2341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'str', "print the author's name")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 9), tuple_2338, str_2341)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2338)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 86)
tuple_2342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 86)
# Adding element type (line 86)
str_2343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'str', 'author-email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 9), tuple_2342, str_2343)
# Adding element type (line 86)
# Getting the type of 'None' (line 86)
None_2344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 9), tuple_2342, None_2344)
# Adding element type (line 86)
str_2345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 9), 'str', "print the author's email address")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 9), tuple_2342, str_2345)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2342)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 88)
tuple_2346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 88)
# Adding element type (line 88)
str_2347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 9), 'str', 'maintainer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 9), tuple_2346, str_2347)
# Adding element type (line 88)
# Getting the type of 'None' (line 88)
None_2348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 9), tuple_2346, None_2348)
# Adding element type (line 88)
str_2349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'str', "print the maintainer's name")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 9), tuple_2346, str_2349)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2346)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_2350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
str_2351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'str', 'maintainer-email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_2350, str_2351)
# Adding element type (line 90)
# Getting the type of 'None' (line 90)
None_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_2350, None_2352)
# Adding element type (line 90)
str_2353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 9), 'str', "print the maintainer's email address")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_2350, str_2353)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2350)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 92)
tuple_2354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 92)
# Adding element type (line 92)
str_2355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 9), 'str', 'contact')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 9), tuple_2354, str_2355)
# Adding element type (line 92)
# Getting the type of 'None' (line 92)
None_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 9), tuple_2354, None_2356)
# Adding element type (line 92)
str_2357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'str', "print the maintainer's name if known, else the author's")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 9), tuple_2354, str_2357)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2354)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 94)
tuple_2358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 94)
# Adding element type (line 94)
str_2359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'str', 'contact-email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 9), tuple_2358, str_2359)
# Adding element type (line 94)
# Getting the type of 'None' (line 94)
None_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 9), tuple_2358, None_2360)
# Adding element type (line 94)
str_2361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'str', "print the maintainer's email address if known, else the author's")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 9), tuple_2358, str_2361)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2358)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 96)
tuple_2362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 96)
# Adding element type (line 96)
str_2363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 9), 'str', 'url')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 9), tuple_2362, str_2363)
# Adding element type (line 96)
# Getting the type of 'None' (line 96)
None_2364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 9), tuple_2362, None_2364)
# Adding element type (line 96)
str_2365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 9), 'str', 'print the URL for this package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 9), tuple_2362, str_2365)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2362)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 98)
tuple_2366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 98)
# Adding element type (line 98)
str_2367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'str', 'license')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), tuple_2366, str_2367)
# Adding element type (line 98)
# Getting the type of 'None' (line 98)
None_2368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), tuple_2366, None_2368)
# Adding element type (line 98)
str_2369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 9), 'str', 'print the license of the package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), tuple_2366, str_2369)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2366)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 100)
tuple_2370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 100)
# Adding element type (line 100)
str_2371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'str', 'licence')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_2370, str_2371)
# Adding element type (line 100)
# Getting the type of 'None' (line 100)
None_2372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_2370, None_2372)
# Adding element type (line 100)
str_2373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 9), 'str', 'alias for --license')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_2370, str_2373)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2370)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_2374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
str_2375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 9), 'str', 'description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_2374, str_2375)
# Adding element type (line 102)
# Getting the type of 'None' (line 102)
None_2376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_2374, None_2376)
# Adding element type (line 102)
str_2377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'str', 'print the package description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_2374, str_2377)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2374)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_2378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_2379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'str', 'long-description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2378, str_2379)
# Adding element type (line 104)
# Getting the type of 'None' (line 104)
None_2380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2378, None_2380)
# Adding element type (line 104)
str_2381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 9), 'str', 'print the long package description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2378, str_2381)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2378)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_2382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
str_2383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 9), 'str', 'platforms')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 9), tuple_2382, str_2383)
# Adding element type (line 106)
# Getting the type of 'None' (line 106)
None_2384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 9), tuple_2382, None_2384)
# Adding element type (line 106)
str_2385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 9), 'str', 'print the list of platforms')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 9), tuple_2382, str_2385)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2382)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_2386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
str_2387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 9), 'str', 'classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 9), tuple_2386, str_2387)
# Adding element type (line 108)
# Getting the type of 'None' (line 108)
None_2388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 9), tuple_2386, None_2388)
# Adding element type (line 108)
str_2389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 9), 'str', 'print the list of classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 9), tuple_2386, str_2389)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2386)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 110)
tuple_2390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 110)
# Adding element type (line 110)
str_2391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 9), 'str', 'keywords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), tuple_2390, str_2391)
# Adding element type (line 110)
# Getting the type of 'None' (line 110)
None_2392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), tuple_2390, None_2392)
# Adding element type (line 110)
str_2393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 9), 'str', 'print the list of keywords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), tuple_2390, str_2393)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2390)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 112)
tuple_2394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 112)
# Adding element type (line 112)
str_2395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'str', 'provides')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), tuple_2394, str_2395)
# Adding element type (line 112)
# Getting the type of 'None' (line 112)
None_2396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), tuple_2394, None_2396)
# Adding element type (line 112)
str_2397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 9), 'str', 'print the list of packages/modules provided')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), tuple_2394, str_2397)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2394)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_2398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
str_2399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 9), 'str', 'requires')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 9), tuple_2398, str_2399)
# Adding element type (line 114)
# Getting the type of 'None' (line 114)
None_2400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 9), tuple_2398, None_2400)
# Adding element type (line 114)
str_2401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 9), 'str', 'print the list of packages/modules required')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 9), tuple_2398, str_2401)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2398)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 116)
tuple_2402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 116)
# Adding element type (line 116)
str_2403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 9), 'str', 'obsoletes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 9), tuple_2402, str_2403)
# Adding element type (line 116)
# Getting the type of 'None' (line 116)
None_2404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 9), tuple_2402, None_2404)
# Adding element type (line 116)
str_2405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 9), 'str', 'print the list of packages/modules made obsolete')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 9), tuple_2402, str_2405)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), list_2321, tuple_2402)

# Getting the type of 'Distribution'
Distribution_2406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Distribution')
# Setting the type of the member 'display_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Distribution_2406, 'display_options', list_2321)

# Assigning a Call to a Name (line 119):

# Call to map(...): (line 119)
# Processing the call arguments (line 119)

@norecursion
def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_1'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 119, 31, True)
    # Passed parameters checking function
    _stypy_temp_lambda_1.stypy_localization = localization
    _stypy_temp_lambda_1.stypy_type_of_self = None
    _stypy_temp_lambda_1.stypy_type_store = module_type_store
    _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
    _stypy_temp_lambda_1.stypy_param_names_list = ['x']
    _stypy_temp_lambda_1.stypy_varargs_param_name = None
    _stypy_temp_lambda_1.stypy_kwargs_param_name = None
    _stypy_temp_lambda_1.stypy_call_defaults = defaults
    _stypy_temp_lambda_1.stypy_call_varargs = varargs
    _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to translate_longopt(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining the type of the subscript
    int_2409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 61), 'int')
    # Getting the type of 'x' (line 119)
    x_2410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 59), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___2411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 59), x_2410, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_2412 = invoke(stypy.reporting.localization.Localization(__file__, 119, 59), getitem___2411, int_2409)
    
    # Processing the call keyword arguments (line 119)
    kwargs_2413 = {}
    # Getting the type of 'translate_longopt' (line 119)
    translate_longopt_2408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 41), 'translate_longopt', False)
    # Calling translate_longopt(args, kwargs) (line 119)
    translate_longopt_call_result_2414 = invoke(stypy.reporting.localization.Localization(__file__, 119, 41), translate_longopt_2408, *[subscript_call_result_2412], **kwargs_2413)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'stypy_return_type', translate_longopt_call_result_2414)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_1' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_2415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_1'
    return stypy_return_type_2415

# Assigning a type to the variable '_stypy_temp_lambda_1' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
# Getting the type of '_stypy_temp_lambda_1' (line 119)
_stypy_temp_lambda_1_2416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), '_stypy_temp_lambda_1')
# Getting the type of 'Distribution'
Distribution_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Distribution', False)
# Obtaining the member 'display_options' of a type
display_options_2418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Distribution_2417, 'display_options')
# Processing the call keyword arguments (line 119)
kwargs_2419 = {}
# Getting the type of 'map' (line 119)
map_2407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'map', False)
# Calling map(args, kwargs) (line 119)
map_call_result_2420 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), map_2407, *[_stypy_temp_lambda_1_2416, display_options_2418], **kwargs_2419)

# Getting the type of 'Distribution'
Distribution_2421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Distribution')
# Setting the type of the member 'display_option_names' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Distribution_2421, 'display_option_names', map_call_result_2420)

# Assigning a Dict to a Name (line 123):

# Obtaining an instance of the builtin type 'dict' (line 123)
dict_2422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 123)
# Adding element type (key, value) (line 123)
str_2423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'str', 'quiet')
str_2424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'str', 'verbose')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 19), dict_2422, (str_2423, str_2424))

# Getting the type of 'Distribution'
Distribution_2425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Distribution')
# Setting the type of the member 'negative_opt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Distribution_2425, 'negative_opt', dict_2422)
# Declaration of the 'DistributionMetadata' class

class DistributionMetadata:
    str_2426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, (-1)), 'str', 'Dummy class to hold the distribution meta-data: name, version,\n    author, and so forth.\n    ')
    
    # Assigning a Tuple to a Name (line 1016):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1026)
        None_2427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 28), 'None')
        defaults = [None_2427]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1026, 4, False)
        # Assigning a type to the variable 'self' (line 1027)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1027, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.__init__', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 1027)
        # Getting the type of 'path' (line 1027)
        path_2428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 8), 'path')
        # Getting the type of 'None' (line 1027)
        None_2429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 23), 'None')
        
        (may_be_2430, more_types_in_union_2431) = may_not_be_none(path_2428, None_2429)

        if may_be_2430:

            if more_types_in_union_2431:
                # Runtime conditional SSA (line 1027)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to read_pkg_file(...): (line 1028)
            # Processing the call arguments (line 1028)
            
            # Call to open(...): (line 1028)
            # Processing the call arguments (line 1028)
            # Getting the type of 'path' (line 1028)
            path_2435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 36), 'path', False)
            # Processing the call keyword arguments (line 1028)
            kwargs_2436 = {}
            # Getting the type of 'open' (line 1028)
            open_2434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 31), 'open', False)
            # Calling open(args, kwargs) (line 1028)
            open_call_result_2437 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 31), open_2434, *[path_2435], **kwargs_2436)
            
            # Processing the call keyword arguments (line 1028)
            kwargs_2438 = {}
            # Getting the type of 'self' (line 1028)
            self_2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 12), 'self', False)
            # Obtaining the member 'read_pkg_file' of a type (line 1028)
            read_pkg_file_2433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 12), self_2432, 'read_pkg_file')
            # Calling read_pkg_file(args, kwargs) (line 1028)
            read_pkg_file_call_result_2439 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 12), read_pkg_file_2433, *[open_call_result_2437], **kwargs_2438)
            

            if more_types_in_union_2431:
                # Runtime conditional SSA for else branch (line 1027)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_2430) or more_types_in_union_2431):
            
            # Assigning a Name to a Attribute (line 1030):
            
            # Assigning a Name to a Attribute (line 1030):
            # Getting the type of 'None' (line 1030)
            None_2440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 24), 'None')
            # Getting the type of 'self' (line 1030)
            self_2441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 12), 'self')
            # Setting the type of the member 'name' of a type (line 1030)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 12), self_2441, 'name', None_2440)
            
            # Assigning a Name to a Attribute (line 1031):
            
            # Assigning a Name to a Attribute (line 1031):
            # Getting the type of 'None' (line 1031)
            None_2442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 27), 'None')
            # Getting the type of 'self' (line 1031)
            self_2443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 12), 'self')
            # Setting the type of the member 'version' of a type (line 1031)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1031, 12), self_2443, 'version', None_2442)
            
            # Assigning a Name to a Attribute (line 1032):
            
            # Assigning a Name to a Attribute (line 1032):
            # Getting the type of 'None' (line 1032)
            None_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 26), 'None')
            # Getting the type of 'self' (line 1032)
            self_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 12), 'self')
            # Setting the type of the member 'author' of a type (line 1032)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 12), self_2445, 'author', None_2444)
            
            # Assigning a Name to a Attribute (line 1033):
            
            # Assigning a Name to a Attribute (line 1033):
            # Getting the type of 'None' (line 1033)
            None_2446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 32), 'None')
            # Getting the type of 'self' (line 1033)
            self_2447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 12), 'self')
            # Setting the type of the member 'author_email' of a type (line 1033)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 12), self_2447, 'author_email', None_2446)
            
            # Assigning a Name to a Attribute (line 1034):
            
            # Assigning a Name to a Attribute (line 1034):
            # Getting the type of 'None' (line 1034)
            None_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 30), 'None')
            # Getting the type of 'self' (line 1034)
            self_2449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 12), 'self')
            # Setting the type of the member 'maintainer' of a type (line 1034)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 12), self_2449, 'maintainer', None_2448)
            
            # Assigning a Name to a Attribute (line 1035):
            
            # Assigning a Name to a Attribute (line 1035):
            # Getting the type of 'None' (line 1035)
            None_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 36), 'None')
            # Getting the type of 'self' (line 1035)
            self_2451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 12), 'self')
            # Setting the type of the member 'maintainer_email' of a type (line 1035)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 12), self_2451, 'maintainer_email', None_2450)
            
            # Assigning a Name to a Attribute (line 1036):
            
            # Assigning a Name to a Attribute (line 1036):
            # Getting the type of 'None' (line 1036)
            None_2452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 23), 'None')
            # Getting the type of 'self' (line 1036)
            self_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'self')
            # Setting the type of the member 'url' of a type (line 1036)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 12), self_2453, 'url', None_2452)
            
            # Assigning a Name to a Attribute (line 1037):
            
            # Assigning a Name to a Attribute (line 1037):
            # Getting the type of 'None' (line 1037)
            None_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 27), 'None')
            # Getting the type of 'self' (line 1037)
            self_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 12), 'self')
            # Setting the type of the member 'license' of a type (line 1037)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 12), self_2455, 'license', None_2454)
            
            # Assigning a Name to a Attribute (line 1038):
            
            # Assigning a Name to a Attribute (line 1038):
            # Getting the type of 'None' (line 1038)
            None_2456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 31), 'None')
            # Getting the type of 'self' (line 1038)
            self_2457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 12), 'self')
            # Setting the type of the member 'description' of a type (line 1038)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 12), self_2457, 'description', None_2456)
            
            # Assigning a Name to a Attribute (line 1039):
            
            # Assigning a Name to a Attribute (line 1039):
            # Getting the type of 'None' (line 1039)
            None_2458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 36), 'None')
            # Getting the type of 'self' (line 1039)
            self_2459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 12), 'self')
            # Setting the type of the member 'long_description' of a type (line 1039)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 12), self_2459, 'long_description', None_2458)
            
            # Assigning a Name to a Attribute (line 1040):
            
            # Assigning a Name to a Attribute (line 1040):
            # Getting the type of 'None' (line 1040)
            None_2460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 28), 'None')
            # Getting the type of 'self' (line 1040)
            self_2461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 12), 'self')
            # Setting the type of the member 'keywords' of a type (line 1040)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 12), self_2461, 'keywords', None_2460)
            
            # Assigning a Name to a Attribute (line 1041):
            
            # Assigning a Name to a Attribute (line 1041):
            # Getting the type of 'None' (line 1041)
            None_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 29), 'None')
            # Getting the type of 'self' (line 1041)
            self_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 12), 'self')
            # Setting the type of the member 'platforms' of a type (line 1041)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1041, 12), self_2463, 'platforms', None_2462)
            
            # Assigning a Name to a Attribute (line 1042):
            
            # Assigning a Name to a Attribute (line 1042):
            # Getting the type of 'None' (line 1042)
            None_2464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 31), 'None')
            # Getting the type of 'self' (line 1042)
            self_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 12), 'self')
            # Setting the type of the member 'classifiers' of a type (line 1042)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 12), self_2465, 'classifiers', None_2464)
            
            # Assigning a Name to a Attribute (line 1043):
            
            # Assigning a Name to a Attribute (line 1043):
            # Getting the type of 'None' (line 1043)
            None_2466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 32), 'None')
            # Getting the type of 'self' (line 1043)
            self_2467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 12), 'self')
            # Setting the type of the member 'download_url' of a type (line 1043)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 12), self_2467, 'download_url', None_2466)
            
            # Assigning a Name to a Attribute (line 1045):
            
            # Assigning a Name to a Attribute (line 1045):
            # Getting the type of 'None' (line 1045)
            None_2468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 28), 'None')
            # Getting the type of 'self' (line 1045)
            self_2469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 12), 'self')
            # Setting the type of the member 'provides' of a type (line 1045)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 12), self_2469, 'provides', None_2468)
            
            # Assigning a Name to a Attribute (line 1046):
            
            # Assigning a Name to a Attribute (line 1046):
            # Getting the type of 'None' (line 1046)
            None_2470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 28), 'None')
            # Getting the type of 'self' (line 1046)
            self_2471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 12), 'self')
            # Setting the type of the member 'requires' of a type (line 1046)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 12), self_2471, 'requires', None_2470)
            
            # Assigning a Name to a Attribute (line 1047):
            
            # Assigning a Name to a Attribute (line 1047):
            # Getting the type of 'None' (line 1047)
            None_2472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 29), 'None')
            # Getting the type of 'self' (line 1047)
            self_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 12), 'self')
            # Setting the type of the member 'obsoletes' of a type (line 1047)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 12), self_2473, 'obsoletes', None_2472)

            if (may_be_2430 and more_types_in_union_2431):
                # SSA join for if statement (line 1027)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def read_pkg_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_pkg_file'
        module_type_store = module_type_store.open_function_context('read_pkg_file', 1049, 4, False)
        # Assigning a type to the variable 'self' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.read_pkg_file')
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_param_names_list', ['file'])
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.read_pkg_file.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.read_pkg_file', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_pkg_file', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_pkg_file(...)' code ##################

        str_2474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 8), 'str', 'Reads the metadata values from a file object.')
        
        # Assigning a Call to a Name (line 1051):
        
        # Assigning a Call to a Name (line 1051):
        
        # Call to message_from_file(...): (line 1051)
        # Processing the call arguments (line 1051)
        # Getting the type of 'file' (line 1051)
        file_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 32), 'file', False)
        # Processing the call keyword arguments (line 1051)
        kwargs_2477 = {}
        # Getting the type of 'message_from_file' (line 1051)
        message_from_file_2475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 14), 'message_from_file', False)
        # Calling message_from_file(args, kwargs) (line 1051)
        message_from_file_call_result_2478 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 14), message_from_file_2475, *[file_2476], **kwargs_2477)
        
        # Assigning a type to the variable 'msg' (line 1051)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 8), 'msg', message_from_file_call_result_2478)

        @norecursion
        def _read_field(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_read_field'
            module_type_store = module_type_store.open_function_context('_read_field', 1053, 8, False)
            
            # Passed parameters checking function
            _read_field.stypy_localization = localization
            _read_field.stypy_type_of_self = None
            _read_field.stypy_type_store = module_type_store
            _read_field.stypy_function_name = '_read_field'
            _read_field.stypy_param_names_list = ['name']
            _read_field.stypy_varargs_param_name = None
            _read_field.stypy_kwargs_param_name = None
            _read_field.stypy_call_defaults = defaults
            _read_field.stypy_call_varargs = varargs
            _read_field.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_read_field', ['name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_read_field', localization, ['name'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_read_field(...)' code ##################

            
            # Assigning a Subscript to a Name (line 1054):
            
            # Assigning a Subscript to a Name (line 1054):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 1054)
            name_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 24), 'name')
            # Getting the type of 'msg' (line 1054)
            msg_2480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 20), 'msg')
            # Obtaining the member '__getitem__' of a type (line 1054)
            getitem___2481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 20), msg_2480, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1054)
            subscript_call_result_2482 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 20), getitem___2481, name_2479)
            
            # Assigning a type to the variable 'value' (line 1054)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 12), 'value', subscript_call_result_2482)
            
            
            # Getting the type of 'value' (line 1055)
            value_2483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 15), 'value')
            str_2484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 24), 'str', 'UNKNOWN')
            # Applying the binary operator '==' (line 1055)
            result_eq_2485 = python_operator(stypy.reporting.localization.Localization(__file__, 1055, 15), '==', value_2483, str_2484)
            
            # Testing the type of an if condition (line 1055)
            if_condition_2486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1055, 12), result_eq_2485)
            # Assigning a type to the variable 'if_condition_2486' (line 1055)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 12), 'if_condition_2486', if_condition_2486)
            # SSA begins for if statement (line 1055)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 1056)
            None_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 1056)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 16), 'stypy_return_type', None_2487)
            # SSA join for if statement (line 1055)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'value' (line 1057)
            value_2488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 19), 'value')
            # Assigning a type to the variable 'stypy_return_type' (line 1057)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 12), 'stypy_return_type', value_2488)
            
            # ################# End of '_read_field(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_read_field' in the type store
            # Getting the type of 'stypy_return_type' (line 1053)
            stypy_return_type_2489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2489)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_read_field'
            return stypy_return_type_2489

        # Assigning a type to the variable '_read_field' (line 1053)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), '_read_field', _read_field)

        @norecursion
        def _read_list(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_read_list'
            module_type_store = module_type_store.open_function_context('_read_list', 1059, 8, False)
            
            # Passed parameters checking function
            _read_list.stypy_localization = localization
            _read_list.stypy_type_of_self = None
            _read_list.stypy_type_store = module_type_store
            _read_list.stypy_function_name = '_read_list'
            _read_list.stypy_param_names_list = ['name']
            _read_list.stypy_varargs_param_name = None
            _read_list.stypy_kwargs_param_name = None
            _read_list.stypy_call_defaults = defaults
            _read_list.stypy_call_varargs = varargs
            _read_list.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_read_list', ['name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_read_list', localization, ['name'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_read_list(...)' code ##################

            
            # Assigning a Call to a Name (line 1060):
            
            # Assigning a Call to a Name (line 1060):
            
            # Call to get_all(...): (line 1060)
            # Processing the call arguments (line 1060)
            # Getting the type of 'name' (line 1060)
            name_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 33), 'name', False)
            # Getting the type of 'None' (line 1060)
            None_2493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 39), 'None', False)
            # Processing the call keyword arguments (line 1060)
            kwargs_2494 = {}
            # Getting the type of 'msg' (line 1060)
            msg_2490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 21), 'msg', False)
            # Obtaining the member 'get_all' of a type (line 1060)
            get_all_2491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 21), msg_2490, 'get_all')
            # Calling get_all(args, kwargs) (line 1060)
            get_all_call_result_2495 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 21), get_all_2491, *[name_2492, None_2493], **kwargs_2494)
            
            # Assigning a type to the variable 'values' (line 1060)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 12), 'values', get_all_call_result_2495)
            
            
            # Getting the type of 'values' (line 1061)
            values_2496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 15), 'values')
            
            # Obtaining an instance of the builtin type 'list' (line 1061)
            list_2497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 1061)
            
            # Applying the binary operator '==' (line 1061)
            result_eq_2498 = python_operator(stypy.reporting.localization.Localization(__file__, 1061, 15), '==', values_2496, list_2497)
            
            # Testing the type of an if condition (line 1061)
            if_condition_2499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1061, 12), result_eq_2498)
            # Assigning a type to the variable 'if_condition_2499' (line 1061)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 12), 'if_condition_2499', if_condition_2499)
            # SSA begins for if statement (line 1061)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 1062)
            None_2500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 1062)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1062, 16), 'stypy_return_type', None_2500)
            # SSA join for if statement (line 1061)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'values' (line 1063)
            values_2501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 19), 'values')
            # Assigning a type to the variable 'stypy_return_type' (line 1063)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 12), 'stypy_return_type', values_2501)
            
            # ################# End of '_read_list(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_read_list' in the type store
            # Getting the type of 'stypy_return_type' (line 1059)
            stypy_return_type_2502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2502)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_read_list'
            return stypy_return_type_2502

        # Assigning a type to the variable '_read_list' (line 1059)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 8), '_read_list', _read_list)
        
        # Assigning a Subscript to a Name (line 1065):
        
        # Assigning a Subscript to a Name (line 1065):
        
        # Obtaining the type of the subscript
        str_2503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 31), 'str', 'metadata-version')
        # Getting the type of 'msg' (line 1065)
        msg_2504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 27), 'msg')
        # Obtaining the member '__getitem__' of a type (line 1065)
        getitem___2505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 27), msg_2504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1065)
        subscript_call_result_2506 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 27), getitem___2505, str_2503)
        
        # Assigning a type to the variable 'metadata_version' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 8), 'metadata_version', subscript_call_result_2506)
        
        # Assigning a Call to a Attribute (line 1066):
        
        # Assigning a Call to a Attribute (line 1066):
        
        # Call to _read_field(...): (line 1066)
        # Processing the call arguments (line 1066)
        str_2508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 32), 'str', 'name')
        # Processing the call keyword arguments (line 1066)
        kwargs_2509 = {}
        # Getting the type of '_read_field' (line 1066)
        _read_field_2507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 20), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1066)
        _read_field_call_result_2510 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 20), _read_field_2507, *[str_2508], **kwargs_2509)
        
        # Getting the type of 'self' (line 1066)
        self_2511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 8), 'self')
        # Setting the type of the member 'name' of a type (line 1066)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 8), self_2511, 'name', _read_field_call_result_2510)
        
        # Assigning a Call to a Attribute (line 1067):
        
        # Assigning a Call to a Attribute (line 1067):
        
        # Call to _read_field(...): (line 1067)
        # Processing the call arguments (line 1067)
        str_2513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 35), 'str', 'version')
        # Processing the call keyword arguments (line 1067)
        kwargs_2514 = {}
        # Getting the type of '_read_field' (line 1067)
        _read_field_2512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 23), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1067)
        _read_field_call_result_2515 = invoke(stypy.reporting.localization.Localization(__file__, 1067, 23), _read_field_2512, *[str_2513], **kwargs_2514)
        
        # Getting the type of 'self' (line 1067)
        self_2516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'self')
        # Setting the type of the member 'version' of a type (line 1067)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 8), self_2516, 'version', _read_field_call_result_2515)
        
        # Assigning a Call to a Attribute (line 1068):
        
        # Assigning a Call to a Attribute (line 1068):
        
        # Call to _read_field(...): (line 1068)
        # Processing the call arguments (line 1068)
        str_2518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 39), 'str', 'summary')
        # Processing the call keyword arguments (line 1068)
        kwargs_2519 = {}
        # Getting the type of '_read_field' (line 1068)
        _read_field_2517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 27), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1068)
        _read_field_call_result_2520 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 27), _read_field_2517, *[str_2518], **kwargs_2519)
        
        # Getting the type of 'self' (line 1068)
        self_2521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 8), 'self')
        # Setting the type of the member 'description' of a type (line 1068)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1068, 8), self_2521, 'description', _read_field_call_result_2520)
        
        # Assigning a Call to a Attribute (line 1070):
        
        # Assigning a Call to a Attribute (line 1070):
        
        # Call to _read_field(...): (line 1070)
        # Processing the call arguments (line 1070)
        str_2523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 34), 'str', 'author')
        # Processing the call keyword arguments (line 1070)
        kwargs_2524 = {}
        # Getting the type of '_read_field' (line 1070)
        _read_field_2522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 22), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1070)
        _read_field_call_result_2525 = invoke(stypy.reporting.localization.Localization(__file__, 1070, 22), _read_field_2522, *[str_2523], **kwargs_2524)
        
        # Getting the type of 'self' (line 1070)
        self_2526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 8), 'self')
        # Setting the type of the member 'author' of a type (line 1070)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 8), self_2526, 'author', _read_field_call_result_2525)
        
        # Assigning a Name to a Attribute (line 1071):
        
        # Assigning a Name to a Attribute (line 1071):
        # Getting the type of 'None' (line 1071)
        None_2527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 26), 'None')
        # Getting the type of 'self' (line 1071)
        self_2528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 8), 'self')
        # Setting the type of the member 'maintainer' of a type (line 1071)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1071, 8), self_2528, 'maintainer', None_2527)
        
        # Assigning a Call to a Attribute (line 1072):
        
        # Assigning a Call to a Attribute (line 1072):
        
        # Call to _read_field(...): (line 1072)
        # Processing the call arguments (line 1072)
        str_2530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 40), 'str', 'author-email')
        # Processing the call keyword arguments (line 1072)
        kwargs_2531 = {}
        # Getting the type of '_read_field' (line 1072)
        _read_field_2529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 28), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1072)
        _read_field_call_result_2532 = invoke(stypy.reporting.localization.Localization(__file__, 1072, 28), _read_field_2529, *[str_2530], **kwargs_2531)
        
        # Getting the type of 'self' (line 1072)
        self_2533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 8), 'self')
        # Setting the type of the member 'author_email' of a type (line 1072)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1072, 8), self_2533, 'author_email', _read_field_call_result_2532)
        
        # Assigning a Name to a Attribute (line 1073):
        
        # Assigning a Name to a Attribute (line 1073):
        # Getting the type of 'None' (line 1073)
        None_2534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 32), 'None')
        # Getting the type of 'self' (line 1073)
        self_2535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'self')
        # Setting the type of the member 'maintainer_email' of a type (line 1073)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 8), self_2535, 'maintainer_email', None_2534)
        
        # Assigning a Call to a Attribute (line 1074):
        
        # Assigning a Call to a Attribute (line 1074):
        
        # Call to _read_field(...): (line 1074)
        # Processing the call arguments (line 1074)
        str_2537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 31), 'str', 'home-page')
        # Processing the call keyword arguments (line 1074)
        kwargs_2538 = {}
        # Getting the type of '_read_field' (line 1074)
        _read_field_2536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 19), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1074)
        _read_field_call_result_2539 = invoke(stypy.reporting.localization.Localization(__file__, 1074, 19), _read_field_2536, *[str_2537], **kwargs_2538)
        
        # Getting the type of 'self' (line 1074)
        self_2540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 8), 'self')
        # Setting the type of the member 'url' of a type (line 1074)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 8), self_2540, 'url', _read_field_call_result_2539)
        
        # Assigning a Call to a Attribute (line 1075):
        
        # Assigning a Call to a Attribute (line 1075):
        
        # Call to _read_field(...): (line 1075)
        # Processing the call arguments (line 1075)
        str_2542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 35), 'str', 'license')
        # Processing the call keyword arguments (line 1075)
        kwargs_2543 = {}
        # Getting the type of '_read_field' (line 1075)
        _read_field_2541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 23), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1075)
        _read_field_call_result_2544 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 23), _read_field_2541, *[str_2542], **kwargs_2543)
        
        # Getting the type of 'self' (line 1075)
        self_2545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'self')
        # Setting the type of the member 'license' of a type (line 1075)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), self_2545, 'license', _read_field_call_result_2544)
        
        
        str_2546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 11), 'str', 'download-url')
        # Getting the type of 'msg' (line 1077)
        msg_2547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 29), 'msg')
        # Applying the binary operator 'in' (line 1077)
        result_contains_2548 = python_operator(stypy.reporting.localization.Localization(__file__, 1077, 11), 'in', str_2546, msg_2547)
        
        # Testing the type of an if condition (line 1077)
        if_condition_2549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1077, 8), result_contains_2548)
        # Assigning a type to the variable 'if_condition_2549' (line 1077)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'if_condition_2549', if_condition_2549)
        # SSA begins for if statement (line 1077)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 1078):
        
        # Assigning a Call to a Attribute (line 1078):
        
        # Call to _read_field(...): (line 1078)
        # Processing the call arguments (line 1078)
        str_2551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 44), 'str', 'download-url')
        # Processing the call keyword arguments (line 1078)
        kwargs_2552 = {}
        # Getting the type of '_read_field' (line 1078)
        _read_field_2550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 32), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1078)
        _read_field_call_result_2553 = invoke(stypy.reporting.localization.Localization(__file__, 1078, 32), _read_field_2550, *[str_2551], **kwargs_2552)
        
        # Getting the type of 'self' (line 1078)
        self_2554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 12), 'self')
        # Setting the type of the member 'download_url' of a type (line 1078)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1078, 12), self_2554, 'download_url', _read_field_call_result_2553)
        # SSA branch for the else part of an if statement (line 1077)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 1080):
        
        # Assigning a Name to a Attribute (line 1080):
        # Getting the type of 'None' (line 1080)
        None_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 32), 'None')
        # Getting the type of 'self' (line 1080)
        self_2556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 12), 'self')
        # Setting the type of the member 'download_url' of a type (line 1080)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1080, 12), self_2556, 'download_url', None_2555)
        # SSA join for if statement (line 1077)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 1082):
        
        # Assigning a Call to a Attribute (line 1082):
        
        # Call to _read_field(...): (line 1082)
        # Processing the call arguments (line 1082)
        str_2558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 44), 'str', 'description')
        # Processing the call keyword arguments (line 1082)
        kwargs_2559 = {}
        # Getting the type of '_read_field' (line 1082)
        _read_field_2557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 32), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1082)
        _read_field_call_result_2560 = invoke(stypy.reporting.localization.Localization(__file__, 1082, 32), _read_field_2557, *[str_2558], **kwargs_2559)
        
        # Getting the type of 'self' (line 1082)
        self_2561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 8), 'self')
        # Setting the type of the member 'long_description' of a type (line 1082)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1082, 8), self_2561, 'long_description', _read_field_call_result_2560)
        
        # Assigning a Call to a Attribute (line 1083):
        
        # Assigning a Call to a Attribute (line 1083):
        
        # Call to _read_field(...): (line 1083)
        # Processing the call arguments (line 1083)
        str_2563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 39), 'str', 'summary')
        # Processing the call keyword arguments (line 1083)
        kwargs_2564 = {}
        # Getting the type of '_read_field' (line 1083)
        _read_field_2562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 27), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1083)
        _read_field_call_result_2565 = invoke(stypy.reporting.localization.Localization(__file__, 1083, 27), _read_field_2562, *[str_2563], **kwargs_2564)
        
        # Getting the type of 'self' (line 1083)
        self_2566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 8), 'self')
        # Setting the type of the member 'description' of a type (line 1083)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 8), self_2566, 'description', _read_field_call_result_2565)
        
        
        str_2567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 11), 'str', 'keywords')
        # Getting the type of 'msg' (line 1085)
        msg_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 25), 'msg')
        # Applying the binary operator 'in' (line 1085)
        result_contains_2569 = python_operator(stypy.reporting.localization.Localization(__file__, 1085, 11), 'in', str_2567, msg_2568)
        
        # Testing the type of an if condition (line 1085)
        if_condition_2570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1085, 8), result_contains_2569)
        # Assigning a type to the variable 'if_condition_2570' (line 1085)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'if_condition_2570', if_condition_2570)
        # SSA begins for if statement (line 1085)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 1086):
        
        # Assigning a Call to a Attribute (line 1086):
        
        # Call to split(...): (line 1086)
        # Processing the call arguments (line 1086)
        str_2576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 58), 'str', ',')
        # Processing the call keyword arguments (line 1086)
        kwargs_2577 = {}
        
        # Call to _read_field(...): (line 1086)
        # Processing the call arguments (line 1086)
        str_2572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 40), 'str', 'keywords')
        # Processing the call keyword arguments (line 1086)
        kwargs_2573 = {}
        # Getting the type of '_read_field' (line 1086)
        _read_field_2571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 28), '_read_field', False)
        # Calling _read_field(args, kwargs) (line 1086)
        _read_field_call_result_2574 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 28), _read_field_2571, *[str_2572], **kwargs_2573)
        
        # Obtaining the member 'split' of a type (line 1086)
        split_2575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 28), _read_field_call_result_2574, 'split')
        # Calling split(args, kwargs) (line 1086)
        split_call_result_2578 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 28), split_2575, *[str_2576], **kwargs_2577)
        
        # Getting the type of 'self' (line 1086)
        self_2579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 12), 'self')
        # Setting the type of the member 'keywords' of a type (line 1086)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 12), self_2579, 'keywords', split_call_result_2578)
        # SSA join for if statement (line 1085)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 1088):
        
        # Assigning a Call to a Attribute (line 1088):
        
        # Call to _read_list(...): (line 1088)
        # Processing the call arguments (line 1088)
        str_2581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 36), 'str', 'platform')
        # Processing the call keyword arguments (line 1088)
        kwargs_2582 = {}
        # Getting the type of '_read_list' (line 1088)
        _read_list_2580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 25), '_read_list', False)
        # Calling _read_list(args, kwargs) (line 1088)
        _read_list_call_result_2583 = invoke(stypy.reporting.localization.Localization(__file__, 1088, 25), _read_list_2580, *[str_2581], **kwargs_2582)
        
        # Getting the type of 'self' (line 1088)
        self_2584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'self')
        # Setting the type of the member 'platforms' of a type (line 1088)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 8), self_2584, 'platforms', _read_list_call_result_2583)
        
        # Assigning a Call to a Attribute (line 1089):
        
        # Assigning a Call to a Attribute (line 1089):
        
        # Call to _read_list(...): (line 1089)
        # Processing the call arguments (line 1089)
        str_2586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 38), 'str', 'classifier')
        # Processing the call keyword arguments (line 1089)
        kwargs_2587 = {}
        # Getting the type of '_read_list' (line 1089)
        _read_list_2585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 27), '_read_list', False)
        # Calling _read_list(args, kwargs) (line 1089)
        _read_list_call_result_2588 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 27), _read_list_2585, *[str_2586], **kwargs_2587)
        
        # Getting the type of 'self' (line 1089)
        self_2589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'self')
        # Setting the type of the member 'classifiers' of a type (line 1089)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 8), self_2589, 'classifiers', _read_list_call_result_2588)
        
        
        # Getting the type of 'metadata_version' (line 1092)
        metadata_version_2590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 11), 'metadata_version')
        str_2591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 31), 'str', '1.1')
        # Applying the binary operator '==' (line 1092)
        result_eq_2592 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 11), '==', metadata_version_2590, str_2591)
        
        # Testing the type of an if condition (line 1092)
        if_condition_2593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1092, 8), result_eq_2592)
        # Assigning a type to the variable 'if_condition_2593' (line 1092)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'if_condition_2593', if_condition_2593)
        # SSA begins for if statement (line 1092)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 1093):
        
        # Assigning a Call to a Attribute (line 1093):
        
        # Call to _read_list(...): (line 1093)
        # Processing the call arguments (line 1093)
        str_2595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 39), 'str', 'requires')
        # Processing the call keyword arguments (line 1093)
        kwargs_2596 = {}
        # Getting the type of '_read_list' (line 1093)
        _read_list_2594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 28), '_read_list', False)
        # Calling _read_list(args, kwargs) (line 1093)
        _read_list_call_result_2597 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 28), _read_list_2594, *[str_2595], **kwargs_2596)
        
        # Getting the type of 'self' (line 1093)
        self_2598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 12), 'self')
        # Setting the type of the member 'requires' of a type (line 1093)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 12), self_2598, 'requires', _read_list_call_result_2597)
        
        # Assigning a Call to a Attribute (line 1094):
        
        # Assigning a Call to a Attribute (line 1094):
        
        # Call to _read_list(...): (line 1094)
        # Processing the call arguments (line 1094)
        str_2600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 39), 'str', 'provides')
        # Processing the call keyword arguments (line 1094)
        kwargs_2601 = {}
        # Getting the type of '_read_list' (line 1094)
        _read_list_2599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 28), '_read_list', False)
        # Calling _read_list(args, kwargs) (line 1094)
        _read_list_call_result_2602 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 28), _read_list_2599, *[str_2600], **kwargs_2601)
        
        # Getting the type of 'self' (line 1094)
        self_2603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 12), 'self')
        # Setting the type of the member 'provides' of a type (line 1094)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 12), self_2603, 'provides', _read_list_call_result_2602)
        
        # Assigning a Call to a Attribute (line 1095):
        
        # Assigning a Call to a Attribute (line 1095):
        
        # Call to _read_list(...): (line 1095)
        # Processing the call arguments (line 1095)
        str_2605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1095, 40), 'str', 'obsoletes')
        # Processing the call keyword arguments (line 1095)
        kwargs_2606 = {}
        # Getting the type of '_read_list' (line 1095)
        _read_list_2604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 29), '_read_list', False)
        # Calling _read_list(args, kwargs) (line 1095)
        _read_list_call_result_2607 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 29), _read_list_2604, *[str_2605], **kwargs_2606)
        
        # Getting the type of 'self' (line 1095)
        self_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'self')
        # Setting the type of the member 'obsoletes' of a type (line 1095)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 12), self_2608, 'obsoletes', _read_list_call_result_2607)
        # SSA branch for the else part of an if statement (line 1092)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 1097):
        
        # Assigning a Name to a Attribute (line 1097):
        # Getting the type of 'None' (line 1097)
        None_2609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 28), 'None')
        # Getting the type of 'self' (line 1097)
        self_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 12), 'self')
        # Setting the type of the member 'requires' of a type (line 1097)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 12), self_2610, 'requires', None_2609)
        
        # Assigning a Name to a Attribute (line 1098):
        
        # Assigning a Name to a Attribute (line 1098):
        # Getting the type of 'None' (line 1098)
        None_2611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 28), 'None')
        # Getting the type of 'self' (line 1098)
        self_2612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 12), 'self')
        # Setting the type of the member 'provides' of a type (line 1098)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1098, 12), self_2612, 'provides', None_2611)
        
        # Assigning a Name to a Attribute (line 1099):
        
        # Assigning a Name to a Attribute (line 1099):
        # Getting the type of 'None' (line 1099)
        None_2613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 29), 'None')
        # Getting the type of 'self' (line 1099)
        self_2614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 12), 'self')
        # Setting the type of the member 'obsoletes' of a type (line 1099)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 12), self_2614, 'obsoletes', None_2613)
        # SSA join for if statement (line 1092)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'read_pkg_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_pkg_file' in the type store
        # Getting the type of 'stypy_return_type' (line 1049)
        stypy_return_type_2615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_pkg_file'
        return stypy_return_type_2615


    @norecursion
    def write_pkg_info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_pkg_info'
        module_type_store = module_type_store.open_function_context('write_pkg_info', 1101, 4, False)
        # Assigning a type to the variable 'self' (line 1102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.write_pkg_info')
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_param_names_list', ['base_dir'])
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.write_pkg_info.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.write_pkg_info', ['base_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_pkg_info', localization, ['base_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_pkg_info(...)' code ##################

        str_2616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1103, (-1)), 'str', 'Write the PKG-INFO file into the release tree.\n        ')
        
        # Assigning a Call to a Name (line 1104):
        
        # Assigning a Call to a Name (line 1104):
        
        # Call to open(...): (line 1104)
        # Processing the call arguments (line 1104)
        
        # Call to join(...): (line 1104)
        # Processing the call arguments (line 1104)
        # Getting the type of 'base_dir' (line 1104)
        base_dir_2621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 37), 'base_dir', False)
        str_2622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 47), 'str', 'PKG-INFO')
        # Processing the call keyword arguments (line 1104)
        kwargs_2623 = {}
        # Getting the type of 'os' (line 1104)
        os_2618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 1104)
        path_2619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1104, 24), os_2618, 'path')
        # Obtaining the member 'join' of a type (line 1104)
        join_2620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1104, 24), path_2619, 'join')
        # Calling join(args, kwargs) (line 1104)
        join_call_result_2624 = invoke(stypy.reporting.localization.Localization(__file__, 1104, 24), join_2620, *[base_dir_2621, str_2622], **kwargs_2623)
        
        str_2625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 60), 'str', 'w')
        # Processing the call keyword arguments (line 1104)
        kwargs_2626 = {}
        # Getting the type of 'open' (line 1104)
        open_2617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 19), 'open', False)
        # Calling open(args, kwargs) (line 1104)
        open_call_result_2627 = invoke(stypy.reporting.localization.Localization(__file__, 1104, 19), open_2617, *[join_call_result_2624, str_2625], **kwargs_2626)
        
        # Assigning a type to the variable 'pkg_info' (line 1104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 8), 'pkg_info', open_call_result_2627)
        
        # Try-finally block (line 1105)
        
        # Call to write_pkg_file(...): (line 1106)
        # Processing the call arguments (line 1106)
        # Getting the type of 'pkg_info' (line 1106)
        pkg_info_2630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 32), 'pkg_info', False)
        # Processing the call keyword arguments (line 1106)
        kwargs_2631 = {}
        # Getting the type of 'self' (line 1106)
        self_2628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 12), 'self', False)
        # Obtaining the member 'write_pkg_file' of a type (line 1106)
        write_pkg_file_2629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1106, 12), self_2628, 'write_pkg_file')
        # Calling write_pkg_file(args, kwargs) (line 1106)
        write_pkg_file_call_result_2632 = invoke(stypy.reporting.localization.Localization(__file__, 1106, 12), write_pkg_file_2629, *[pkg_info_2630], **kwargs_2631)
        
        
        # finally branch of the try-finally block (line 1105)
        
        # Call to close(...): (line 1108)
        # Processing the call keyword arguments (line 1108)
        kwargs_2635 = {}
        # Getting the type of 'pkg_info' (line 1108)
        pkg_info_2633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 12), 'pkg_info', False)
        # Obtaining the member 'close' of a type (line 1108)
        close_2634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 12), pkg_info_2633, 'close')
        # Calling close(args, kwargs) (line 1108)
        close_call_result_2636 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 12), close_2634, *[], **kwargs_2635)
        
        
        
        # ################# End of 'write_pkg_info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_pkg_info' in the type store
        # Getting the type of 'stypy_return_type' (line 1101)
        stypy_return_type_2637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2637)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_pkg_info'
        return stypy_return_type_2637


    @norecursion
    def write_pkg_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_pkg_file'
        module_type_store = module_type_store.open_function_context('write_pkg_file', 1110, 4, False)
        # Assigning a type to the variable 'self' (line 1111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.write_pkg_file')
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_param_names_list', ['file'])
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.write_pkg_file.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.write_pkg_file', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_pkg_file', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_pkg_file(...)' code ##################

        str_2638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, (-1)), 'str', 'Write the PKG-INFO format data to a file object.\n        ')
        
        # Assigning a Str to a Name (line 1113):
        
        # Assigning a Str to a Name (line 1113):
        str_2639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 18), 'str', '1.0')
        # Assigning a type to the variable 'version' (line 1113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 8), 'version', str_2639)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1114)
        self_2640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 12), 'self')
        # Obtaining the member 'provides' of a type (line 1114)
        provides_2641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 12), self_2640, 'provides')
        # Getting the type of 'self' (line 1114)
        self_2642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 29), 'self')
        # Obtaining the member 'requires' of a type (line 1114)
        requires_2643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 29), self_2642, 'requires')
        # Applying the binary operator 'or' (line 1114)
        result_or_keyword_2644 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 12), 'or', provides_2641, requires_2643)
        # Getting the type of 'self' (line 1114)
        self_2645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 46), 'self')
        # Obtaining the member 'obsoletes' of a type (line 1114)
        obsoletes_2646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 46), self_2645, 'obsoletes')
        # Applying the binary operator 'or' (line 1114)
        result_or_keyword_2647 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 12), 'or', result_or_keyword_2644, obsoletes_2646)
        # Getting the type of 'self' (line 1115)
        self_2648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 12), 'self')
        # Obtaining the member 'classifiers' of a type (line 1115)
        classifiers_2649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1115, 12), self_2648, 'classifiers')
        # Applying the binary operator 'or' (line 1114)
        result_or_keyword_2650 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 12), 'or', result_or_keyword_2647, classifiers_2649)
        # Getting the type of 'self' (line 1115)
        self_2651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 32), 'self')
        # Obtaining the member 'download_url' of a type (line 1115)
        download_url_2652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1115, 32), self_2651, 'download_url')
        # Applying the binary operator 'or' (line 1114)
        result_or_keyword_2653 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 12), 'or', result_or_keyword_2650, download_url_2652)
        
        # Testing the type of an if condition (line 1114)
        if_condition_2654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1114, 8), result_or_keyword_2653)
        # Assigning a type to the variable 'if_condition_2654' (line 1114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 8), 'if_condition_2654', if_condition_2654)
        # SSA begins for if statement (line 1114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 1116):
        
        # Assigning a Str to a Name (line 1116):
        str_2655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 22), 'str', '1.1')
        # Assigning a type to the variable 'version' (line 1116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 12), 'version', str_2655)
        # SSA join for if statement (line 1114)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _write_field(...): (line 1118)
        # Processing the call arguments (line 1118)
        # Getting the type of 'file' (line 1118)
        file_2658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 26), 'file', False)
        str_2659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 32), 'str', 'Metadata-Version')
        # Getting the type of 'version' (line 1118)
        version_2660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 52), 'version', False)
        # Processing the call keyword arguments (line 1118)
        kwargs_2661 = {}
        # Getting the type of 'self' (line 1118)
        self_2656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1118)
        _write_field_2657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 8), self_2656, '_write_field')
        # Calling _write_field(args, kwargs) (line 1118)
        _write_field_call_result_2662 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 8), _write_field_2657, *[file_2658, str_2659, version_2660], **kwargs_2661)
        
        
        # Call to _write_field(...): (line 1119)
        # Processing the call arguments (line 1119)
        # Getting the type of 'file' (line 1119)
        file_2665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 26), 'file', False)
        str_2666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 32), 'str', 'Name')
        
        # Call to get_name(...): (line 1119)
        # Processing the call keyword arguments (line 1119)
        kwargs_2669 = {}
        # Getting the type of 'self' (line 1119)
        self_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 40), 'self', False)
        # Obtaining the member 'get_name' of a type (line 1119)
        get_name_2668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 40), self_2667, 'get_name')
        # Calling get_name(args, kwargs) (line 1119)
        get_name_call_result_2670 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 40), get_name_2668, *[], **kwargs_2669)
        
        # Processing the call keyword arguments (line 1119)
        kwargs_2671 = {}
        # Getting the type of 'self' (line 1119)
        self_2663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1119)
        _write_field_2664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 8), self_2663, '_write_field')
        # Calling _write_field(args, kwargs) (line 1119)
        _write_field_call_result_2672 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 8), _write_field_2664, *[file_2665, str_2666, get_name_call_result_2670], **kwargs_2671)
        
        
        # Call to _write_field(...): (line 1120)
        # Processing the call arguments (line 1120)
        # Getting the type of 'file' (line 1120)
        file_2675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 26), 'file', False)
        str_2676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 32), 'str', 'Version')
        
        # Call to get_version(...): (line 1120)
        # Processing the call keyword arguments (line 1120)
        kwargs_2679 = {}
        # Getting the type of 'self' (line 1120)
        self_2677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 43), 'self', False)
        # Obtaining the member 'get_version' of a type (line 1120)
        get_version_2678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1120, 43), self_2677, 'get_version')
        # Calling get_version(args, kwargs) (line 1120)
        get_version_call_result_2680 = invoke(stypy.reporting.localization.Localization(__file__, 1120, 43), get_version_2678, *[], **kwargs_2679)
        
        # Processing the call keyword arguments (line 1120)
        kwargs_2681 = {}
        # Getting the type of 'self' (line 1120)
        self_2673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1120)
        _write_field_2674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1120, 8), self_2673, '_write_field')
        # Calling _write_field(args, kwargs) (line 1120)
        _write_field_call_result_2682 = invoke(stypy.reporting.localization.Localization(__file__, 1120, 8), _write_field_2674, *[file_2675, str_2676, get_version_call_result_2680], **kwargs_2681)
        
        
        # Call to _write_field(...): (line 1121)
        # Processing the call arguments (line 1121)
        # Getting the type of 'file' (line 1121)
        file_2685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 26), 'file', False)
        str_2686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 32), 'str', 'Summary')
        
        # Call to get_description(...): (line 1121)
        # Processing the call keyword arguments (line 1121)
        kwargs_2689 = {}
        # Getting the type of 'self' (line 1121)
        self_2687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 43), 'self', False)
        # Obtaining the member 'get_description' of a type (line 1121)
        get_description_2688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 43), self_2687, 'get_description')
        # Calling get_description(args, kwargs) (line 1121)
        get_description_call_result_2690 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 43), get_description_2688, *[], **kwargs_2689)
        
        # Processing the call keyword arguments (line 1121)
        kwargs_2691 = {}
        # Getting the type of 'self' (line 1121)
        self_2683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1121)
        _write_field_2684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 8), self_2683, '_write_field')
        # Calling _write_field(args, kwargs) (line 1121)
        _write_field_call_result_2692 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 8), _write_field_2684, *[file_2685, str_2686, get_description_call_result_2690], **kwargs_2691)
        
        
        # Call to _write_field(...): (line 1122)
        # Processing the call arguments (line 1122)
        # Getting the type of 'file' (line 1122)
        file_2695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 26), 'file', False)
        str_2696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 32), 'str', 'Home-page')
        
        # Call to get_url(...): (line 1122)
        # Processing the call keyword arguments (line 1122)
        kwargs_2699 = {}
        # Getting the type of 'self' (line 1122)
        self_2697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 45), 'self', False)
        # Obtaining the member 'get_url' of a type (line 1122)
        get_url_2698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 45), self_2697, 'get_url')
        # Calling get_url(args, kwargs) (line 1122)
        get_url_call_result_2700 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 45), get_url_2698, *[], **kwargs_2699)
        
        # Processing the call keyword arguments (line 1122)
        kwargs_2701 = {}
        # Getting the type of 'self' (line 1122)
        self_2693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1122)
        _write_field_2694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 8), self_2693, '_write_field')
        # Calling _write_field(args, kwargs) (line 1122)
        _write_field_call_result_2702 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 8), _write_field_2694, *[file_2695, str_2696, get_url_call_result_2700], **kwargs_2701)
        
        
        # Call to _write_field(...): (line 1123)
        # Processing the call arguments (line 1123)
        # Getting the type of 'file' (line 1123)
        file_2705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 26), 'file', False)
        str_2706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1123, 32), 'str', 'Author')
        
        # Call to get_contact(...): (line 1123)
        # Processing the call keyword arguments (line 1123)
        kwargs_2709 = {}
        # Getting the type of 'self' (line 1123)
        self_2707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 42), 'self', False)
        # Obtaining the member 'get_contact' of a type (line 1123)
        get_contact_2708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1123, 42), self_2707, 'get_contact')
        # Calling get_contact(args, kwargs) (line 1123)
        get_contact_call_result_2710 = invoke(stypy.reporting.localization.Localization(__file__, 1123, 42), get_contact_2708, *[], **kwargs_2709)
        
        # Processing the call keyword arguments (line 1123)
        kwargs_2711 = {}
        # Getting the type of 'self' (line 1123)
        self_2703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1123)
        _write_field_2704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1123, 8), self_2703, '_write_field')
        # Calling _write_field(args, kwargs) (line 1123)
        _write_field_call_result_2712 = invoke(stypy.reporting.localization.Localization(__file__, 1123, 8), _write_field_2704, *[file_2705, str_2706, get_contact_call_result_2710], **kwargs_2711)
        
        
        # Call to _write_field(...): (line 1124)
        # Processing the call arguments (line 1124)
        # Getting the type of 'file' (line 1124)
        file_2715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 26), 'file', False)
        str_2716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 32), 'str', 'Author-email')
        
        # Call to get_contact_email(...): (line 1124)
        # Processing the call keyword arguments (line 1124)
        kwargs_2719 = {}
        # Getting the type of 'self' (line 1124)
        self_2717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 48), 'self', False)
        # Obtaining the member 'get_contact_email' of a type (line 1124)
        get_contact_email_2718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 48), self_2717, 'get_contact_email')
        # Calling get_contact_email(args, kwargs) (line 1124)
        get_contact_email_call_result_2720 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 48), get_contact_email_2718, *[], **kwargs_2719)
        
        # Processing the call keyword arguments (line 1124)
        kwargs_2721 = {}
        # Getting the type of 'self' (line 1124)
        self_2713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1124)
        _write_field_2714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 8), self_2713, '_write_field')
        # Calling _write_field(args, kwargs) (line 1124)
        _write_field_call_result_2722 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 8), _write_field_2714, *[file_2715, str_2716, get_contact_email_call_result_2720], **kwargs_2721)
        
        
        # Call to _write_field(...): (line 1125)
        # Processing the call arguments (line 1125)
        # Getting the type of 'file' (line 1125)
        file_2725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 26), 'file', False)
        str_2726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 32), 'str', 'License')
        
        # Call to get_license(...): (line 1125)
        # Processing the call keyword arguments (line 1125)
        kwargs_2729 = {}
        # Getting the type of 'self' (line 1125)
        self_2727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 43), 'self', False)
        # Obtaining the member 'get_license' of a type (line 1125)
        get_license_2728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 43), self_2727, 'get_license')
        # Calling get_license(args, kwargs) (line 1125)
        get_license_call_result_2730 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 43), get_license_2728, *[], **kwargs_2729)
        
        # Processing the call keyword arguments (line 1125)
        kwargs_2731 = {}
        # Getting the type of 'self' (line 1125)
        self_2723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1125)
        _write_field_2724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 8), self_2723, '_write_field')
        # Calling _write_field(args, kwargs) (line 1125)
        _write_field_call_result_2732 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 8), _write_field_2724, *[file_2725, str_2726, get_license_call_result_2730], **kwargs_2731)
        
        
        # Getting the type of 'self' (line 1126)
        self_2733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 11), 'self')
        # Obtaining the member 'download_url' of a type (line 1126)
        download_url_2734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 11), self_2733, 'download_url')
        # Testing the type of an if condition (line 1126)
        if_condition_2735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1126, 8), download_url_2734)
        # Assigning a type to the variable 'if_condition_2735' (line 1126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 8), 'if_condition_2735', if_condition_2735)
        # SSA begins for if statement (line 1126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _write_field(...): (line 1127)
        # Processing the call arguments (line 1127)
        # Getting the type of 'file' (line 1127)
        file_2738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 30), 'file', False)
        str_2739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 36), 'str', 'Download-URL')
        # Getting the type of 'self' (line 1127)
        self_2740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 52), 'self', False)
        # Obtaining the member 'download_url' of a type (line 1127)
        download_url_2741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1127, 52), self_2740, 'download_url')
        # Processing the call keyword arguments (line 1127)
        kwargs_2742 = {}
        # Getting the type of 'self' (line 1127)
        self_2736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 12), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1127)
        _write_field_2737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1127, 12), self_2736, '_write_field')
        # Calling _write_field(args, kwargs) (line 1127)
        _write_field_call_result_2743 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 12), _write_field_2737, *[file_2738, str_2739, download_url_2741], **kwargs_2742)
        
        # SSA join for if statement (line 1126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1129):
        
        # Assigning a Call to a Name (line 1129):
        
        # Call to rfc822_escape(...): (line 1129)
        # Processing the call arguments (line 1129)
        
        # Call to get_long_description(...): (line 1129)
        # Processing the call keyword arguments (line 1129)
        kwargs_2747 = {}
        # Getting the type of 'self' (line 1129)
        self_2745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 34), 'self', False)
        # Obtaining the member 'get_long_description' of a type (line 1129)
        get_long_description_2746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1129, 34), self_2745, 'get_long_description')
        # Calling get_long_description(args, kwargs) (line 1129)
        get_long_description_call_result_2748 = invoke(stypy.reporting.localization.Localization(__file__, 1129, 34), get_long_description_2746, *[], **kwargs_2747)
        
        # Processing the call keyword arguments (line 1129)
        kwargs_2749 = {}
        # Getting the type of 'rfc822_escape' (line 1129)
        rfc822_escape_2744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 20), 'rfc822_escape', False)
        # Calling rfc822_escape(args, kwargs) (line 1129)
        rfc822_escape_call_result_2750 = invoke(stypy.reporting.localization.Localization(__file__, 1129, 20), rfc822_escape_2744, *[get_long_description_call_result_2748], **kwargs_2749)
        
        # Assigning a type to the variable 'long_desc' (line 1129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 8), 'long_desc', rfc822_escape_call_result_2750)
        
        # Call to _write_field(...): (line 1130)
        # Processing the call arguments (line 1130)
        # Getting the type of 'file' (line 1130)
        file_2753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 26), 'file', False)
        str_2754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 32), 'str', 'Description')
        # Getting the type of 'long_desc' (line 1130)
        long_desc_2755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 47), 'long_desc', False)
        # Processing the call keyword arguments (line 1130)
        kwargs_2756 = {}
        # Getting the type of 'self' (line 1130)
        self_2751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 8), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1130)
        _write_field_2752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 8), self_2751, '_write_field')
        # Calling _write_field(args, kwargs) (line 1130)
        _write_field_call_result_2757 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 8), _write_field_2752, *[file_2753, str_2754, long_desc_2755], **kwargs_2756)
        
        
        # Assigning a Call to a Name (line 1132):
        
        # Assigning a Call to a Name (line 1132):
        
        # Call to join(...): (line 1132)
        # Processing the call arguments (line 1132)
        
        # Call to get_keywords(...): (line 1132)
        # Processing the call keyword arguments (line 1132)
        kwargs_2762 = {}
        # Getting the type of 'self' (line 1132)
        self_2760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 28), 'self', False)
        # Obtaining the member 'get_keywords' of a type (line 1132)
        get_keywords_2761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 28), self_2760, 'get_keywords')
        # Calling get_keywords(args, kwargs) (line 1132)
        get_keywords_call_result_2763 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 28), get_keywords_2761, *[], **kwargs_2762)
        
        # Processing the call keyword arguments (line 1132)
        kwargs_2764 = {}
        str_2758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 19), 'str', ',')
        # Obtaining the member 'join' of a type (line 1132)
        join_2759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 19), str_2758, 'join')
        # Calling join(args, kwargs) (line 1132)
        join_call_result_2765 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 19), join_2759, *[get_keywords_call_result_2763], **kwargs_2764)
        
        # Assigning a type to the variable 'keywords' (line 1132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 8), 'keywords', join_call_result_2765)
        
        # Getting the type of 'keywords' (line 1133)
        keywords_2766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 11), 'keywords')
        # Testing the type of an if condition (line 1133)
        if_condition_2767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1133, 8), keywords_2766)
        # Assigning a type to the variable 'if_condition_2767' (line 1133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1133, 8), 'if_condition_2767', if_condition_2767)
        # SSA begins for if statement (line 1133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _write_field(...): (line 1134)
        # Processing the call arguments (line 1134)
        # Getting the type of 'file' (line 1134)
        file_2770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 30), 'file', False)
        str_2771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 36), 'str', 'Keywords')
        # Getting the type of 'keywords' (line 1134)
        keywords_2772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 48), 'keywords', False)
        # Processing the call keyword arguments (line 1134)
        kwargs_2773 = {}
        # Getting the type of 'self' (line 1134)
        self_2768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 12), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1134)
        _write_field_2769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1134, 12), self_2768, '_write_field')
        # Calling _write_field(args, kwargs) (line 1134)
        _write_field_call_result_2774 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 12), _write_field_2769, *[file_2770, str_2771, keywords_2772], **kwargs_2773)
        
        # SSA join for if statement (line 1133)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _write_list(...): (line 1136)
        # Processing the call arguments (line 1136)
        # Getting the type of 'file' (line 1136)
        file_2777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 25), 'file', False)
        str_2778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 31), 'str', 'Platform')
        
        # Call to get_platforms(...): (line 1136)
        # Processing the call keyword arguments (line 1136)
        kwargs_2781 = {}
        # Getting the type of 'self' (line 1136)
        self_2779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 43), 'self', False)
        # Obtaining the member 'get_platforms' of a type (line 1136)
        get_platforms_2780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1136, 43), self_2779, 'get_platforms')
        # Calling get_platforms(args, kwargs) (line 1136)
        get_platforms_call_result_2782 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 43), get_platforms_2780, *[], **kwargs_2781)
        
        # Processing the call keyword arguments (line 1136)
        kwargs_2783 = {}
        # Getting the type of 'self' (line 1136)
        self_2775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 8), 'self', False)
        # Obtaining the member '_write_list' of a type (line 1136)
        _write_list_2776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1136, 8), self_2775, '_write_list')
        # Calling _write_list(args, kwargs) (line 1136)
        _write_list_call_result_2784 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 8), _write_list_2776, *[file_2777, str_2778, get_platforms_call_result_2782], **kwargs_2783)
        
        
        # Call to _write_list(...): (line 1137)
        # Processing the call arguments (line 1137)
        # Getting the type of 'file' (line 1137)
        file_2787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 25), 'file', False)
        str_2788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 31), 'str', 'Classifier')
        
        # Call to get_classifiers(...): (line 1137)
        # Processing the call keyword arguments (line 1137)
        kwargs_2791 = {}
        # Getting the type of 'self' (line 1137)
        self_2789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 45), 'self', False)
        # Obtaining the member 'get_classifiers' of a type (line 1137)
        get_classifiers_2790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 45), self_2789, 'get_classifiers')
        # Calling get_classifiers(args, kwargs) (line 1137)
        get_classifiers_call_result_2792 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 45), get_classifiers_2790, *[], **kwargs_2791)
        
        # Processing the call keyword arguments (line 1137)
        kwargs_2793 = {}
        # Getting the type of 'self' (line 1137)
        self_2785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 8), 'self', False)
        # Obtaining the member '_write_list' of a type (line 1137)
        _write_list_2786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 8), self_2785, '_write_list')
        # Calling _write_list(args, kwargs) (line 1137)
        _write_list_call_result_2794 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 8), _write_list_2786, *[file_2787, str_2788, get_classifiers_call_result_2792], **kwargs_2793)
        
        
        # Call to _write_list(...): (line 1140)
        # Processing the call arguments (line 1140)
        # Getting the type of 'file' (line 1140)
        file_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 25), 'file', False)
        str_2798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 31), 'str', 'Requires')
        
        # Call to get_requires(...): (line 1140)
        # Processing the call keyword arguments (line 1140)
        kwargs_2801 = {}
        # Getting the type of 'self' (line 1140)
        self_2799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 43), 'self', False)
        # Obtaining the member 'get_requires' of a type (line 1140)
        get_requires_2800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1140, 43), self_2799, 'get_requires')
        # Calling get_requires(args, kwargs) (line 1140)
        get_requires_call_result_2802 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 43), get_requires_2800, *[], **kwargs_2801)
        
        # Processing the call keyword arguments (line 1140)
        kwargs_2803 = {}
        # Getting the type of 'self' (line 1140)
        self_2795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'self', False)
        # Obtaining the member '_write_list' of a type (line 1140)
        _write_list_2796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1140, 8), self_2795, '_write_list')
        # Calling _write_list(args, kwargs) (line 1140)
        _write_list_call_result_2804 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 8), _write_list_2796, *[file_2797, str_2798, get_requires_call_result_2802], **kwargs_2803)
        
        
        # Call to _write_list(...): (line 1141)
        # Processing the call arguments (line 1141)
        # Getting the type of 'file' (line 1141)
        file_2807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 25), 'file', False)
        str_2808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 31), 'str', 'Provides')
        
        # Call to get_provides(...): (line 1141)
        # Processing the call keyword arguments (line 1141)
        kwargs_2811 = {}
        # Getting the type of 'self' (line 1141)
        self_2809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 43), 'self', False)
        # Obtaining the member 'get_provides' of a type (line 1141)
        get_provides_2810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 43), self_2809, 'get_provides')
        # Calling get_provides(args, kwargs) (line 1141)
        get_provides_call_result_2812 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 43), get_provides_2810, *[], **kwargs_2811)
        
        # Processing the call keyword arguments (line 1141)
        kwargs_2813 = {}
        # Getting the type of 'self' (line 1141)
        self_2805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 8), 'self', False)
        # Obtaining the member '_write_list' of a type (line 1141)
        _write_list_2806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 8), self_2805, '_write_list')
        # Calling _write_list(args, kwargs) (line 1141)
        _write_list_call_result_2814 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 8), _write_list_2806, *[file_2807, str_2808, get_provides_call_result_2812], **kwargs_2813)
        
        
        # Call to _write_list(...): (line 1142)
        # Processing the call arguments (line 1142)
        # Getting the type of 'file' (line 1142)
        file_2817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 25), 'file', False)
        str_2818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 31), 'str', 'Obsoletes')
        
        # Call to get_obsoletes(...): (line 1142)
        # Processing the call keyword arguments (line 1142)
        kwargs_2821 = {}
        # Getting the type of 'self' (line 1142)
        self_2819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 44), 'self', False)
        # Obtaining the member 'get_obsoletes' of a type (line 1142)
        get_obsoletes_2820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 44), self_2819, 'get_obsoletes')
        # Calling get_obsoletes(args, kwargs) (line 1142)
        get_obsoletes_call_result_2822 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 44), get_obsoletes_2820, *[], **kwargs_2821)
        
        # Processing the call keyword arguments (line 1142)
        kwargs_2823 = {}
        # Getting the type of 'self' (line 1142)
        self_2815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 8), 'self', False)
        # Obtaining the member '_write_list' of a type (line 1142)
        _write_list_2816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 8), self_2815, '_write_list')
        # Calling _write_list(args, kwargs) (line 1142)
        _write_list_call_result_2824 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 8), _write_list_2816, *[file_2817, str_2818, get_obsoletes_call_result_2822], **kwargs_2823)
        
        
        # ################# End of 'write_pkg_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_pkg_file' in the type store
        # Getting the type of 'stypy_return_type' (line 1110)
        stypy_return_type_2825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_pkg_file'
        return stypy_return_type_2825


    @norecursion
    def _write_field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_field'
        module_type_store = module_type_store.open_function_context('_write_field', 1144, 4, False)
        # Assigning a type to the variable 'self' (line 1145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata._write_field')
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_param_names_list', ['file', 'name', 'value'])
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata._write_field.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata._write_field', ['file', 'name', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_field', localization, ['file', 'name', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_field(...)' code ##################

        
        # Call to write(...): (line 1145)
        # Processing the call arguments (line 1145)
        str_2828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 19), 'str', '%s: %s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1145)
        tuple_2829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1145)
        # Adding element type (line 1145)
        # Getting the type of 'name' (line 1145)
        name_2830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 33), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1145, 33), tuple_2829, name_2830)
        # Adding element type (line 1145)
        
        # Call to _encode_field(...): (line 1145)
        # Processing the call arguments (line 1145)
        # Getting the type of 'value' (line 1145)
        value_2833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 58), 'value', False)
        # Processing the call keyword arguments (line 1145)
        kwargs_2834 = {}
        # Getting the type of 'self' (line 1145)
        self_2831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 39), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1145)
        _encode_field_2832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 39), self_2831, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1145)
        _encode_field_call_result_2835 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 39), _encode_field_2832, *[value_2833], **kwargs_2834)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1145, 33), tuple_2829, _encode_field_call_result_2835)
        
        # Applying the binary operator '%' (line 1145)
        result_mod_2836 = python_operator(stypy.reporting.localization.Localization(__file__, 1145, 19), '%', str_2828, tuple_2829)
        
        # Processing the call keyword arguments (line 1145)
        kwargs_2837 = {}
        # Getting the type of 'file' (line 1145)
        file_2826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 8), 'file', False)
        # Obtaining the member 'write' of a type (line 1145)
        write_2827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 8), file_2826, 'write')
        # Calling write(args, kwargs) (line 1145)
        write_call_result_2838 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 8), write_2827, *[result_mod_2836], **kwargs_2837)
        
        
        # ################# End of '_write_field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_field' in the type store
        # Getting the type of 'stypy_return_type' (line 1144)
        stypy_return_type_2839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_field'
        return stypy_return_type_2839


    @norecursion
    def _write_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_list'
        module_type_store = module_type_store.open_function_context('_write_list', 1147, 4, False)
        # Assigning a type to the variable 'self' (line 1148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata._write_list')
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_param_names_list', ['file', 'name', 'values'])
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata._write_list.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata._write_list', ['file', 'name', 'values'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_list', localization, ['file', 'name', 'values'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_list(...)' code ##################

        
        # Getting the type of 'values' (line 1148)
        values_2840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 21), 'values')
        # Testing the type of a for loop iterable (line 1148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1148, 8), values_2840)
        # Getting the type of the for loop variable (line 1148)
        for_loop_var_2841 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1148, 8), values_2840)
        # Assigning a type to the variable 'value' (line 1148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 8), 'value', for_loop_var_2841)
        # SSA begins for a for statement (line 1148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _write_field(...): (line 1149)
        # Processing the call arguments (line 1149)
        # Getting the type of 'file' (line 1149)
        file_2844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 30), 'file', False)
        # Getting the type of 'name' (line 1149)
        name_2845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 36), 'name', False)
        # Getting the type of 'value' (line 1149)
        value_2846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 42), 'value', False)
        # Processing the call keyword arguments (line 1149)
        kwargs_2847 = {}
        # Getting the type of 'self' (line 1149)
        self_2842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 12), 'self', False)
        # Obtaining the member '_write_field' of a type (line 1149)
        _write_field_2843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 12), self_2842, '_write_field')
        # Calling _write_field(args, kwargs) (line 1149)
        _write_field_call_result_2848 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 12), _write_field_2843, *[file_2844, name_2845, value_2846], **kwargs_2847)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_list' in the type store
        # Getting the type of 'stypy_return_type' (line 1147)
        stypy_return_type_2849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_list'
        return stypy_return_type_2849


    @norecursion
    def _encode_field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_encode_field'
        module_type_store = module_type_store.open_function_context('_encode_field', 1151, 4, False)
        # Assigning a type to the variable 'self' (line 1152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata._encode_field')
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_param_names_list', ['value'])
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata._encode_field.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata._encode_field', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_encode_field', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_encode_field(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 1152)
        # Getting the type of 'value' (line 1152)
        value_2850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 11), 'value')
        # Getting the type of 'None' (line 1152)
        None_2851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 20), 'None')
        
        (may_be_2852, more_types_in_union_2853) = may_be_none(value_2850, None_2851)

        if may_be_2852:

            if more_types_in_union_2853:
                # Runtime conditional SSA (line 1152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 1153)
            None_2854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'stypy_return_type', None_2854)

            if more_types_in_union_2853:
                # SSA join for if statement (line 1152)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1154)
        # Getting the type of 'unicode' (line 1154)
        unicode_2855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 29), 'unicode')
        # Getting the type of 'value' (line 1154)
        value_2856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 22), 'value')
        
        (may_be_2857, more_types_in_union_2858) = may_be_subtype(unicode_2855, value_2856)

        if may_be_2857:

            if more_types_in_union_2858:
                # Runtime conditional SSA (line 1154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 1154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 8), 'value', remove_not_subtype_from_union(value_2856, unicode))
            
            # Call to encode(...): (line 1155)
            # Processing the call arguments (line 1155)
            # Getting the type of 'PKG_INFO_ENCODING' (line 1155)
            PKG_INFO_ENCODING_2861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 32), 'PKG_INFO_ENCODING', False)
            # Processing the call keyword arguments (line 1155)
            kwargs_2862 = {}
            # Getting the type of 'value' (line 1155)
            value_2859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 19), 'value', False)
            # Obtaining the member 'encode' of a type (line 1155)
            encode_2860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 19), value_2859, 'encode')
            # Calling encode(args, kwargs) (line 1155)
            encode_call_result_2863 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 19), encode_2860, *[PKG_INFO_ENCODING_2861], **kwargs_2862)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 12), 'stypy_return_type', encode_call_result_2863)

            if more_types_in_union_2858:
                # SSA join for if statement (line 1154)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to str(...): (line 1156)
        # Processing the call arguments (line 1156)
        # Getting the type of 'value' (line 1156)
        value_2865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 19), 'value', False)
        # Processing the call keyword arguments (line 1156)
        kwargs_2866 = {}
        # Getting the type of 'str' (line 1156)
        str_2864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 15), 'str', False)
        # Calling str(args, kwargs) (line 1156)
        str_call_result_2867 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 15), str_2864, *[value_2865], **kwargs_2866)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 8), 'stypy_return_type', str_call_result_2867)
        
        # ################# End of '_encode_field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_encode_field' in the type store
        # Getting the type of 'stypy_return_type' (line 1151)
        stypy_return_type_2868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2868)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_encode_field'
        return stypy_return_type_2868


    @norecursion
    def get_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_name'
        module_type_store = module_type_store.open_function_context('get_name', 1160, 4, False)
        # Assigning a type to the variable 'self' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_name')
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_name(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1161)
        self_2869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 15), 'self')
        # Obtaining the member 'name' of a type (line 1161)
        name_2870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 15), self_2869, 'name')
        str_2871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 28), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1161)
        result_or_keyword_2872 = python_operator(stypy.reporting.localization.Localization(__file__, 1161, 15), 'or', name_2870, str_2871)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 8), 'stypy_return_type', result_or_keyword_2872)
        
        # ################# End of 'get_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_name' in the type store
        # Getting the type of 'stypy_return_type' (line 1160)
        stypy_return_type_2873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_name'
        return stypy_return_type_2873


    @norecursion
    def get_version(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_version'
        module_type_store = module_type_store.open_function_context('get_version', 1163, 4, False)
        # Assigning a type to the variable 'self' (line 1164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_version')
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_version.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_version', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_version', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_version(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1164)
        self_2874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 15), 'self')
        # Obtaining the member 'version' of a type (line 1164)
        version_2875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 15), self_2874, 'version')
        str_2876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 31), 'str', '0.0.0')
        # Applying the binary operator 'or' (line 1164)
        result_or_keyword_2877 = python_operator(stypy.reporting.localization.Localization(__file__, 1164, 15), 'or', version_2875, str_2876)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 8), 'stypy_return_type', result_or_keyword_2877)
        
        # ################# End of 'get_version(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_version' in the type store
        # Getting the type of 'stypy_return_type' (line 1163)
        stypy_return_type_2878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2878)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_version'
        return stypy_return_type_2878


    @norecursion
    def get_fullname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_fullname'
        module_type_store = module_type_store.open_function_context('get_fullname', 1166, 4, False)
        # Assigning a type to the variable 'self' (line 1167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_fullname')
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_fullname.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_fullname', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_fullname', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_fullname(...)' code ##################

        str_2879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 15), 'str', '%s-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1167)
        tuple_2880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1167)
        # Adding element type (line 1167)
        
        # Call to get_name(...): (line 1167)
        # Processing the call keyword arguments (line 1167)
        kwargs_2883 = {}
        # Getting the type of 'self' (line 1167)
        self_2881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 26), 'self', False)
        # Obtaining the member 'get_name' of a type (line 1167)
        get_name_2882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 26), self_2881, 'get_name')
        # Calling get_name(args, kwargs) (line 1167)
        get_name_call_result_2884 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 26), get_name_2882, *[], **kwargs_2883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1167, 26), tuple_2880, get_name_call_result_2884)
        # Adding element type (line 1167)
        
        # Call to get_version(...): (line 1167)
        # Processing the call keyword arguments (line 1167)
        kwargs_2887 = {}
        # Getting the type of 'self' (line 1167)
        self_2885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 43), 'self', False)
        # Obtaining the member 'get_version' of a type (line 1167)
        get_version_2886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 43), self_2885, 'get_version')
        # Calling get_version(args, kwargs) (line 1167)
        get_version_call_result_2888 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 43), get_version_2886, *[], **kwargs_2887)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1167, 26), tuple_2880, get_version_call_result_2888)
        
        # Applying the binary operator '%' (line 1167)
        result_mod_2889 = python_operator(stypy.reporting.localization.Localization(__file__, 1167, 15), '%', str_2879, tuple_2880)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 8), 'stypy_return_type', result_mod_2889)
        
        # ################# End of 'get_fullname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_fullname' in the type store
        # Getting the type of 'stypy_return_type' (line 1166)
        stypy_return_type_2890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2890)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_fullname'
        return stypy_return_type_2890


    @norecursion
    def get_author(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_author'
        module_type_store = module_type_store.open_function_context('get_author', 1169, 4, False)
        # Assigning a type to the variable 'self' (line 1170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_author')
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_author.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_author', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_author', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_author(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to _encode_field(...): (line 1170)
        # Processing the call arguments (line 1170)
        # Getting the type of 'self' (line 1170)
        self_2893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 34), 'self', False)
        # Obtaining the member 'author' of a type (line 1170)
        author_2894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1170, 34), self_2893, 'author')
        # Processing the call keyword arguments (line 1170)
        kwargs_2895 = {}
        # Getting the type of 'self' (line 1170)
        self_2891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 15), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1170)
        _encode_field_2892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1170, 15), self_2891, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1170)
        _encode_field_call_result_2896 = invoke(stypy.reporting.localization.Localization(__file__, 1170, 15), _encode_field_2892, *[author_2894], **kwargs_2895)
        
        str_2897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1170, 50), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1170)
        result_or_keyword_2898 = python_operator(stypy.reporting.localization.Localization(__file__, 1170, 15), 'or', _encode_field_call_result_2896, str_2897)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 8), 'stypy_return_type', result_or_keyword_2898)
        
        # ################# End of 'get_author(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_author' in the type store
        # Getting the type of 'stypy_return_type' (line 1169)
        stypy_return_type_2899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_author'
        return stypy_return_type_2899


    @norecursion
    def get_author_email(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_author_email'
        module_type_store = module_type_store.open_function_context('get_author_email', 1172, 4, False)
        # Assigning a type to the variable 'self' (line 1173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_author_email')
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_author_email.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_author_email', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_author_email', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_author_email(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1173)
        self_2900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 15), 'self')
        # Obtaining the member 'author_email' of a type (line 1173)
        author_email_2901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1173, 15), self_2900, 'author_email')
        str_2902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1173, 36), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1173)
        result_or_keyword_2903 = python_operator(stypy.reporting.localization.Localization(__file__, 1173, 15), 'or', author_email_2901, str_2902)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 8), 'stypy_return_type', result_or_keyword_2903)
        
        # ################# End of 'get_author_email(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_author_email' in the type store
        # Getting the type of 'stypy_return_type' (line 1172)
        stypy_return_type_2904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2904)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_author_email'
        return stypy_return_type_2904


    @norecursion
    def get_maintainer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_maintainer'
        module_type_store = module_type_store.open_function_context('get_maintainer', 1175, 4, False)
        # Assigning a type to the variable 'self' (line 1176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_maintainer')
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_maintainer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_maintainer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_maintainer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_maintainer(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to _encode_field(...): (line 1176)
        # Processing the call arguments (line 1176)
        # Getting the type of 'self' (line 1176)
        self_2907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 34), 'self', False)
        # Obtaining the member 'maintainer' of a type (line 1176)
        maintainer_2908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1176, 34), self_2907, 'maintainer')
        # Processing the call keyword arguments (line 1176)
        kwargs_2909 = {}
        # Getting the type of 'self' (line 1176)
        self_2905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 15), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1176)
        _encode_field_2906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1176, 15), self_2905, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1176)
        _encode_field_call_result_2910 = invoke(stypy.reporting.localization.Localization(__file__, 1176, 15), _encode_field_2906, *[maintainer_2908], **kwargs_2909)
        
        str_2911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1176, 54), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1176)
        result_or_keyword_2912 = python_operator(stypy.reporting.localization.Localization(__file__, 1176, 15), 'or', _encode_field_call_result_2910, str_2911)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 8), 'stypy_return_type', result_or_keyword_2912)
        
        # ################# End of 'get_maintainer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_maintainer' in the type store
        # Getting the type of 'stypy_return_type' (line 1175)
        stypy_return_type_2913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2913)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_maintainer'
        return stypy_return_type_2913


    @norecursion
    def get_maintainer_email(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_maintainer_email'
        module_type_store = module_type_store.open_function_context('get_maintainer_email', 1178, 4, False)
        # Assigning a type to the variable 'self' (line 1179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_maintainer_email')
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_maintainer_email.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_maintainer_email', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_maintainer_email', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_maintainer_email(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1179)
        self_2914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1179, 15), 'self')
        # Obtaining the member 'maintainer_email' of a type (line 1179)
        maintainer_email_2915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1179, 15), self_2914, 'maintainer_email')
        str_2916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1179, 40), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1179)
        result_or_keyword_2917 = python_operator(stypy.reporting.localization.Localization(__file__, 1179, 15), 'or', maintainer_email_2915, str_2916)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1179, 8), 'stypy_return_type', result_or_keyword_2917)
        
        # ################# End of 'get_maintainer_email(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_maintainer_email' in the type store
        # Getting the type of 'stypy_return_type' (line 1178)
        stypy_return_type_2918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_maintainer_email'
        return stypy_return_type_2918


    @norecursion
    def get_contact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_contact'
        module_type_store = module_type_store.open_function_context('get_contact', 1181, 4, False)
        # Assigning a type to the variable 'self' (line 1182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_contact')
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_contact.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_contact', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_contact', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_contact(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to _encode_field(...): (line 1182)
        # Processing the call arguments (line 1182)
        # Getting the type of 'self' (line 1182)
        self_2921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 35), 'self', False)
        # Obtaining the member 'maintainer' of a type (line 1182)
        maintainer_2922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1182, 35), self_2921, 'maintainer')
        # Processing the call keyword arguments (line 1182)
        kwargs_2923 = {}
        # Getting the type of 'self' (line 1182)
        self_2919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 16), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1182)
        _encode_field_2920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1182, 16), self_2919, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1182)
        _encode_field_call_result_2924 = invoke(stypy.reporting.localization.Localization(__file__, 1182, 16), _encode_field_2920, *[maintainer_2922], **kwargs_2923)
        
        
        # Call to _encode_field(...): (line 1183)
        # Processing the call arguments (line 1183)
        # Getting the type of 'self' (line 1183)
        self_2927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 35), 'self', False)
        # Obtaining the member 'author' of a type (line 1183)
        author_2928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1183, 35), self_2927, 'author')
        # Processing the call keyword arguments (line 1183)
        kwargs_2929 = {}
        # Getting the type of 'self' (line 1183)
        self_2925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 16), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1183)
        _encode_field_2926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1183, 16), self_2925, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1183)
        _encode_field_call_result_2930 = invoke(stypy.reporting.localization.Localization(__file__, 1183, 16), _encode_field_2926, *[author_2928], **kwargs_2929)
        
        # Applying the binary operator 'or' (line 1182)
        result_or_keyword_2931 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 16), 'or', _encode_field_call_result_2924, _encode_field_call_result_2930)
        str_2932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 51), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1182)
        result_or_keyword_2933 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 16), 'or', result_or_keyword_2931, str_2932)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 8), 'stypy_return_type', result_or_keyword_2933)
        
        # ################# End of 'get_contact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_contact' in the type store
        # Getting the type of 'stypy_return_type' (line 1181)
        stypy_return_type_2934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_contact'
        return stypy_return_type_2934


    @norecursion
    def get_contact_email(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_contact_email'
        module_type_store = module_type_store.open_function_context('get_contact_email', 1185, 4, False)
        # Assigning a type to the variable 'self' (line 1186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_contact_email')
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_contact_email.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_contact_email', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_contact_email', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_contact_email(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1186)
        self_2935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 15), 'self')
        # Obtaining the member 'maintainer_email' of a type (line 1186)
        maintainer_email_2936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 15), self_2935, 'maintainer_email')
        # Getting the type of 'self' (line 1186)
        self_2937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 40), 'self')
        # Obtaining the member 'author_email' of a type (line 1186)
        author_email_2938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 40), self_2937, 'author_email')
        # Applying the binary operator 'or' (line 1186)
        result_or_keyword_2939 = python_operator(stypy.reporting.localization.Localization(__file__, 1186, 15), 'or', maintainer_email_2936, author_email_2938)
        str_2940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 61), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1186)
        result_or_keyword_2941 = python_operator(stypy.reporting.localization.Localization(__file__, 1186, 15), 'or', result_or_keyword_2939, str_2940)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 8), 'stypy_return_type', result_or_keyword_2941)
        
        # ################# End of 'get_contact_email(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_contact_email' in the type store
        # Getting the type of 'stypy_return_type' (line 1185)
        stypy_return_type_2942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2942)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_contact_email'
        return stypy_return_type_2942


    @norecursion
    def get_url(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_url'
        module_type_store = module_type_store.open_function_context('get_url', 1188, 4, False)
        # Assigning a type to the variable 'self' (line 1189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_url')
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_url.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_url', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_url', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_url(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1189)
        self_2943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 15), 'self')
        # Obtaining the member 'url' of a type (line 1189)
        url_2944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1189, 15), self_2943, 'url')
        str_2945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1189, 27), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1189)
        result_or_keyword_2946 = python_operator(stypy.reporting.localization.Localization(__file__, 1189, 15), 'or', url_2944, str_2945)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 8), 'stypy_return_type', result_or_keyword_2946)
        
        # ################# End of 'get_url(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_url' in the type store
        # Getting the type of 'stypy_return_type' (line 1188)
        stypy_return_type_2947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2947)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_url'
        return stypy_return_type_2947


    @norecursion
    def get_license(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_license'
        module_type_store = module_type_store.open_function_context('get_license', 1191, 4, False)
        # Assigning a type to the variable 'self' (line 1192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_license')
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_license.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_license', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_license', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_license(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1192)
        self_2948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 15), 'self')
        # Obtaining the member 'license' of a type (line 1192)
        license_2949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 15), self_2948, 'license')
        str_2950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1192, 31), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1192)
        result_or_keyword_2951 = python_operator(stypy.reporting.localization.Localization(__file__, 1192, 15), 'or', license_2949, str_2950)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1192, 8), 'stypy_return_type', result_or_keyword_2951)
        
        # ################# End of 'get_license(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_license' in the type store
        # Getting the type of 'stypy_return_type' (line 1191)
        stypy_return_type_2952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_license'
        return stypy_return_type_2952

    
    # Assigning a Name to a Name (line 1193):

    @norecursion
    def get_description(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_description'
        module_type_store = module_type_store.open_function_context('get_description', 1195, 4, False)
        # Assigning a type to the variable 'self' (line 1196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_description')
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_description.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_description', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_description', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_description(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to _encode_field(...): (line 1196)
        # Processing the call arguments (line 1196)
        # Getting the type of 'self' (line 1196)
        self_2955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 34), 'self', False)
        # Obtaining the member 'description' of a type (line 1196)
        description_2956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 34), self_2955, 'description')
        # Processing the call keyword arguments (line 1196)
        kwargs_2957 = {}
        # Getting the type of 'self' (line 1196)
        self_2953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 15), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1196)
        _encode_field_2954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 15), self_2953, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1196)
        _encode_field_call_result_2958 = invoke(stypy.reporting.localization.Localization(__file__, 1196, 15), _encode_field_2954, *[description_2956], **kwargs_2957)
        
        str_2959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1196, 55), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1196)
        result_or_keyword_2960 = python_operator(stypy.reporting.localization.Localization(__file__, 1196, 15), 'or', _encode_field_call_result_2958, str_2959)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 8), 'stypy_return_type', result_or_keyword_2960)
        
        # ################# End of 'get_description(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_description' in the type store
        # Getting the type of 'stypy_return_type' (line 1195)
        stypy_return_type_2961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2961)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_description'
        return stypy_return_type_2961


    @norecursion
    def get_long_description(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_long_description'
        module_type_store = module_type_store.open_function_context('get_long_description', 1198, 4, False)
        # Assigning a type to the variable 'self' (line 1199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_long_description')
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_long_description.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_long_description', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_long_description', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_long_description(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to _encode_field(...): (line 1199)
        # Processing the call arguments (line 1199)
        # Getting the type of 'self' (line 1199)
        self_2964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 34), 'self', False)
        # Obtaining the member 'long_description' of a type (line 1199)
        long_description_2965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1199, 34), self_2964, 'long_description')
        # Processing the call keyword arguments (line 1199)
        kwargs_2966 = {}
        # Getting the type of 'self' (line 1199)
        self_2962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 15), 'self', False)
        # Obtaining the member '_encode_field' of a type (line 1199)
        _encode_field_2963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1199, 15), self_2962, '_encode_field')
        # Calling _encode_field(args, kwargs) (line 1199)
        _encode_field_call_result_2967 = invoke(stypy.reporting.localization.Localization(__file__, 1199, 15), _encode_field_2963, *[long_description_2965], **kwargs_2966)
        
        str_2968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 60), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1199)
        result_or_keyword_2969 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 15), 'or', _encode_field_call_result_2967, str_2968)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 8), 'stypy_return_type', result_or_keyword_2969)
        
        # ################# End of 'get_long_description(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_long_description' in the type store
        # Getting the type of 'stypy_return_type' (line 1198)
        stypy_return_type_2970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_long_description'
        return stypy_return_type_2970


    @norecursion
    def get_keywords(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_keywords'
        module_type_store = module_type_store.open_function_context('get_keywords', 1201, 4, False)
        # Assigning a type to the variable 'self' (line 1202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_keywords')
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_keywords.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_keywords', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_keywords', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_keywords(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1202)
        self_2971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 15), 'self')
        # Obtaining the member 'keywords' of a type (line 1202)
        keywords_2972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1202, 15), self_2971, 'keywords')
        
        # Obtaining an instance of the builtin type 'list' (line 1202)
        list_2973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1202, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1202)
        
        # Applying the binary operator 'or' (line 1202)
        result_or_keyword_2974 = python_operator(stypy.reporting.localization.Localization(__file__, 1202, 15), 'or', keywords_2972, list_2973)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 8), 'stypy_return_type', result_or_keyword_2974)
        
        # ################# End of 'get_keywords(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_keywords' in the type store
        # Getting the type of 'stypy_return_type' (line 1201)
        stypy_return_type_2975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_keywords'
        return stypy_return_type_2975


    @norecursion
    def get_platforms(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_platforms'
        module_type_store = module_type_store.open_function_context('get_platforms', 1204, 4, False)
        # Assigning a type to the variable 'self' (line 1205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_platforms')
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_platforms.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_platforms', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_platforms', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_platforms(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1205)
        self_2976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 15), 'self')
        # Obtaining the member 'platforms' of a type (line 1205)
        platforms_2977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 15), self_2976, 'platforms')
        
        # Obtaining an instance of the builtin type 'list' (line 1205)
        list_2978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1205)
        # Adding element type (line 1205)
        str_2979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 34), 'str', 'UNKNOWN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1205, 33), list_2978, str_2979)
        
        # Applying the binary operator 'or' (line 1205)
        result_or_keyword_2980 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 15), 'or', platforms_2977, list_2978)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 8), 'stypy_return_type', result_or_keyword_2980)
        
        # ################# End of 'get_platforms(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_platforms' in the type store
        # Getting the type of 'stypy_return_type' (line 1204)
        stypy_return_type_2981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_platforms'
        return stypy_return_type_2981


    @norecursion
    def get_classifiers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_classifiers'
        module_type_store = module_type_store.open_function_context('get_classifiers', 1207, 4, False)
        # Assigning a type to the variable 'self' (line 1208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_classifiers')
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_classifiers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_classifiers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_classifiers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_classifiers(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1208)
        self_2982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 15), 'self')
        # Obtaining the member 'classifiers' of a type (line 1208)
        classifiers_2983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1208, 15), self_2982, 'classifiers')
        
        # Obtaining an instance of the builtin type 'list' (line 1208)
        list_2984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1208, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1208)
        
        # Applying the binary operator 'or' (line 1208)
        result_or_keyword_2985 = python_operator(stypy.reporting.localization.Localization(__file__, 1208, 15), 'or', classifiers_2983, list_2984)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 8), 'stypy_return_type', result_or_keyword_2985)
        
        # ################# End of 'get_classifiers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_classifiers' in the type store
        # Getting the type of 'stypy_return_type' (line 1207)
        stypy_return_type_2986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2986)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_classifiers'
        return stypy_return_type_2986


    @norecursion
    def get_download_url(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_download_url'
        module_type_store = module_type_store.open_function_context('get_download_url', 1210, 4, False)
        # Assigning a type to the variable 'self' (line 1211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_download_url')
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_download_url.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_download_url', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_download_url', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_download_url(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1211)
        self_2987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 15), 'self')
        # Obtaining the member 'download_url' of a type (line 1211)
        download_url_2988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1211, 15), self_2987, 'download_url')
        str_2989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1211, 36), 'str', 'UNKNOWN')
        # Applying the binary operator 'or' (line 1211)
        result_or_keyword_2990 = python_operator(stypy.reporting.localization.Localization(__file__, 1211, 15), 'or', download_url_2988, str_2989)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 8), 'stypy_return_type', result_or_keyword_2990)
        
        # ################# End of 'get_download_url(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_download_url' in the type store
        # Getting the type of 'stypy_return_type' (line 1210)
        stypy_return_type_2991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2991)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_download_url'
        return stypy_return_type_2991


    @norecursion
    def get_requires(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_requires'
        module_type_store = module_type_store.open_function_context('get_requires', 1214, 4, False)
        # Assigning a type to the variable 'self' (line 1215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_requires')
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_requires.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_requires', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_requires', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_requires(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1215)
        self_2992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 15), 'self')
        # Obtaining the member 'requires' of a type (line 1215)
        requires_2993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1215, 15), self_2992, 'requires')
        
        # Obtaining an instance of the builtin type 'list' (line 1215)
        list_2994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1215, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1215)
        
        # Applying the binary operator 'or' (line 1215)
        result_or_keyword_2995 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 15), 'or', requires_2993, list_2994)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 8), 'stypy_return_type', result_or_keyword_2995)
        
        # ################# End of 'get_requires(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_requires' in the type store
        # Getting the type of 'stypy_return_type' (line 1214)
        stypy_return_type_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2996)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_requires'
        return stypy_return_type_2996


    @norecursion
    def set_requires(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_requires'
        module_type_store = module_type_store.open_function_context('set_requires', 1217, 4, False)
        # Assigning a type to the variable 'self' (line 1218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.set_requires')
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_param_names_list', ['value'])
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.set_requires.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.set_requires', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_requires', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_requires(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1218, 8))
        
        # 'import distutils.versionpredicate' statement (line 1218)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_2997 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1218, 8), 'distutils.versionpredicate')

        if (type(import_2997) is not StypyTypeError):

            if (import_2997 != 'pyd_module'):
                __import__(import_2997)
                sys_modules_2998 = sys.modules[import_2997]
                import_module(stypy.reporting.localization.Localization(__file__, 1218, 8), 'distutils.versionpredicate', sys_modules_2998.module_type_store, module_type_store)
            else:
                import distutils.versionpredicate

                import_module(stypy.reporting.localization.Localization(__file__, 1218, 8), 'distutils.versionpredicate', distutils.versionpredicate, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.versionpredicate' (line 1218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 8), 'distutils.versionpredicate', import_2997)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'value' (line 1219)
        value_2999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 17), 'value')
        # Testing the type of a for loop iterable (line 1219)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1219, 8), value_2999)
        # Getting the type of the for loop variable (line 1219)
        for_loop_var_3000 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1219, 8), value_2999)
        # Assigning a type to the variable 'v' (line 1219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1219, 8), 'v', for_loop_var_3000)
        # SSA begins for a for statement (line 1219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to VersionPredicate(...): (line 1220)
        # Processing the call arguments (line 1220)
        # Getting the type of 'v' (line 1220)
        v_3004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 56), 'v', False)
        # Processing the call keyword arguments (line 1220)
        kwargs_3005 = {}
        # Getting the type of 'distutils' (line 1220)
        distutils_3001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 12), 'distutils', False)
        # Obtaining the member 'versionpredicate' of a type (line 1220)
        versionpredicate_3002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1220, 12), distutils_3001, 'versionpredicate')
        # Obtaining the member 'VersionPredicate' of a type (line 1220)
        VersionPredicate_3003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1220, 12), versionpredicate_3002, 'VersionPredicate')
        # Calling VersionPredicate(args, kwargs) (line 1220)
        VersionPredicate_call_result_3006 = invoke(stypy.reporting.localization.Localization(__file__, 1220, 12), VersionPredicate_3003, *[v_3004], **kwargs_3005)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 1221):
        
        # Assigning a Name to a Attribute (line 1221):
        # Getting the type of 'value' (line 1221)
        value_3007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 24), 'value')
        # Getting the type of 'self' (line 1221)
        self_3008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 8), 'self')
        # Setting the type of the member 'requires' of a type (line 1221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 8), self_3008, 'requires', value_3007)
        
        # ################# End of 'set_requires(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_requires' in the type store
        # Getting the type of 'stypy_return_type' (line 1217)
        stypy_return_type_3009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_requires'
        return stypy_return_type_3009


    @norecursion
    def get_provides(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_provides'
        module_type_store = module_type_store.open_function_context('get_provides', 1223, 4, False)
        # Assigning a type to the variable 'self' (line 1224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_provides')
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_provides.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_provides', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_provides', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_provides(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1224)
        self_3010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 15), 'self')
        # Obtaining the member 'provides' of a type (line 1224)
        provides_3011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1224, 15), self_3010, 'provides')
        
        # Obtaining an instance of the builtin type 'list' (line 1224)
        list_3012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1224)
        
        # Applying the binary operator 'or' (line 1224)
        result_or_keyword_3013 = python_operator(stypy.reporting.localization.Localization(__file__, 1224, 15), 'or', provides_3011, list_3012)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1224, 8), 'stypy_return_type', result_or_keyword_3013)
        
        # ################# End of 'get_provides(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_provides' in the type store
        # Getting the type of 'stypy_return_type' (line 1223)
        stypy_return_type_3014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_provides'
        return stypy_return_type_3014


    @norecursion
    def set_provides(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_provides'
        module_type_store = module_type_store.open_function_context('set_provides', 1226, 4, False)
        # Assigning a type to the variable 'self' (line 1227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.set_provides')
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_param_names_list', ['value'])
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.set_provides.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.set_provides', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_provides', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_provides(...)' code ##################

        
        # Assigning a ListComp to a Name (line 1227):
        
        # Assigning a ListComp to a Name (line 1227):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'value' (line 1227)
        value_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 36), 'value')
        comprehension_3020 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1227, 17), value_3019)
        # Assigning a type to the variable 'v' (line 1227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 17), 'v', comprehension_3020)
        
        # Call to strip(...): (line 1227)
        # Processing the call keyword arguments (line 1227)
        kwargs_3017 = {}
        # Getting the type of 'v' (line 1227)
        v_3015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 17), 'v', False)
        # Obtaining the member 'strip' of a type (line 1227)
        strip_3016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 17), v_3015, 'strip')
        # Calling strip(args, kwargs) (line 1227)
        strip_call_result_3018 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 17), strip_3016, *[], **kwargs_3017)
        
        list_3021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1227, 17), list_3021, strip_call_result_3018)
        # Assigning a type to the variable 'value' (line 1227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'value', list_3021)
        
        # Getting the type of 'value' (line 1228)
        value_3022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 17), 'value')
        # Testing the type of a for loop iterable (line 1228)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1228, 8), value_3022)
        # Getting the type of the for loop variable (line 1228)
        for_loop_var_3023 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1228, 8), value_3022)
        # Assigning a type to the variable 'v' (line 1228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 8), 'v', for_loop_var_3023)
        # SSA begins for a for statement (line 1228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1229, 12))
        
        # 'import distutils.versionpredicate' statement (line 1229)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_3024 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1229, 12), 'distutils.versionpredicate')

        if (type(import_3024) is not StypyTypeError):

            if (import_3024 != 'pyd_module'):
                __import__(import_3024)
                sys_modules_3025 = sys.modules[import_3024]
                import_module(stypy.reporting.localization.Localization(__file__, 1229, 12), 'distutils.versionpredicate', sys_modules_3025.module_type_store, module_type_store)
            else:
                import distutils.versionpredicate

                import_module(stypy.reporting.localization.Localization(__file__, 1229, 12), 'distutils.versionpredicate', distutils.versionpredicate, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.versionpredicate' (line 1229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 12), 'distutils.versionpredicate', import_3024)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Call to split_provision(...): (line 1230)
        # Processing the call arguments (line 1230)
        # Getting the type of 'v' (line 1230)
        v_3029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 55), 'v', False)
        # Processing the call keyword arguments (line 1230)
        kwargs_3030 = {}
        # Getting the type of 'distutils' (line 1230)
        distutils_3026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 12), 'distutils', False)
        # Obtaining the member 'versionpredicate' of a type (line 1230)
        versionpredicate_3027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1230, 12), distutils_3026, 'versionpredicate')
        # Obtaining the member 'split_provision' of a type (line 1230)
        split_provision_3028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1230, 12), versionpredicate_3027, 'split_provision')
        # Calling split_provision(args, kwargs) (line 1230)
        split_provision_call_result_3031 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 12), split_provision_3028, *[v_3029], **kwargs_3030)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 1231):
        
        # Assigning a Name to a Attribute (line 1231):
        # Getting the type of 'value' (line 1231)
        value_3032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 24), 'value')
        # Getting the type of 'self' (line 1231)
        self_3033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 8), 'self')
        # Setting the type of the member 'provides' of a type (line 1231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1231, 8), self_3033, 'provides', value_3032)
        
        # ################# End of 'set_provides(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_provides' in the type store
        # Getting the type of 'stypy_return_type' (line 1226)
        stypy_return_type_3034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3034)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_provides'
        return stypy_return_type_3034


    @norecursion
    def get_obsoletes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_obsoletes'
        module_type_store = module_type_store.open_function_context('get_obsoletes', 1233, 4, False)
        # Assigning a type to the variable 'self' (line 1234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.get_obsoletes')
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.get_obsoletes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.get_obsoletes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_obsoletes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_obsoletes(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 1234)
        self_3035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 15), 'self')
        # Obtaining the member 'obsoletes' of a type (line 1234)
        obsoletes_3036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 15), self_3035, 'obsoletes')
        
        # Obtaining an instance of the builtin type 'list' (line 1234)
        list_3037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1234)
        
        # Applying the binary operator 'or' (line 1234)
        result_or_keyword_3038 = python_operator(stypy.reporting.localization.Localization(__file__, 1234, 15), 'or', obsoletes_3036, list_3037)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 8), 'stypy_return_type', result_or_keyword_3038)
        
        # ################# End of 'get_obsoletes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_obsoletes' in the type store
        # Getting the type of 'stypy_return_type' (line 1233)
        stypy_return_type_3039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_obsoletes'
        return stypy_return_type_3039


    @norecursion
    def set_obsoletes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_obsoletes'
        module_type_store = module_type_store.open_function_context('set_obsoletes', 1236, 4, False)
        # Assigning a type to the variable 'self' (line 1237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_localization', localization)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_function_name', 'DistributionMetadata.set_obsoletes')
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_param_names_list', ['value'])
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionMetadata.set_obsoletes.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionMetadata.set_obsoletes', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_obsoletes', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_obsoletes(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1237, 8))
        
        # 'import distutils.versionpredicate' statement (line 1237)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_3040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1237, 8), 'distutils.versionpredicate')

        if (type(import_3040) is not StypyTypeError):

            if (import_3040 != 'pyd_module'):
                __import__(import_3040)
                sys_modules_3041 = sys.modules[import_3040]
                import_module(stypy.reporting.localization.Localization(__file__, 1237, 8), 'distutils.versionpredicate', sys_modules_3041.module_type_store, module_type_store)
            else:
                import distutils.versionpredicate

                import_module(stypy.reporting.localization.Localization(__file__, 1237, 8), 'distutils.versionpredicate', distutils.versionpredicate, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.versionpredicate' (line 1237)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 8), 'distutils.versionpredicate', import_3040)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'value' (line 1238)
        value_3042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 17), 'value')
        # Testing the type of a for loop iterable (line 1238)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1238, 8), value_3042)
        # Getting the type of the for loop variable (line 1238)
        for_loop_var_3043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1238, 8), value_3042)
        # Assigning a type to the variable 'v' (line 1238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 8), 'v', for_loop_var_3043)
        # SSA begins for a for statement (line 1238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to VersionPredicate(...): (line 1239)
        # Processing the call arguments (line 1239)
        # Getting the type of 'v' (line 1239)
        v_3047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 56), 'v', False)
        # Processing the call keyword arguments (line 1239)
        kwargs_3048 = {}
        # Getting the type of 'distutils' (line 1239)
        distutils_3044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 12), 'distutils', False)
        # Obtaining the member 'versionpredicate' of a type (line 1239)
        versionpredicate_3045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 12), distutils_3044, 'versionpredicate')
        # Obtaining the member 'VersionPredicate' of a type (line 1239)
        VersionPredicate_3046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 12), versionpredicate_3045, 'VersionPredicate')
        # Calling VersionPredicate(args, kwargs) (line 1239)
        VersionPredicate_call_result_3049 = invoke(stypy.reporting.localization.Localization(__file__, 1239, 12), VersionPredicate_3046, *[v_3047], **kwargs_3048)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 1240):
        
        # Assigning a Name to a Attribute (line 1240):
        # Getting the type of 'value' (line 1240)
        value_3050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 25), 'value')
        # Getting the type of 'self' (line 1240)
        self_3051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 8), 'self')
        # Setting the type of the member 'obsoletes' of a type (line 1240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 8), self_3051, 'obsoletes', value_3050)
        
        # ################# End of 'set_obsoletes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_obsoletes' in the type store
        # Getting the type of 'stypy_return_type' (line 1236)
        stypy_return_type_3052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3052)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_obsoletes'
        return stypy_return_type_3052


# Assigning a type to the variable 'DistributionMetadata' (line 1011)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 0), 'DistributionMetadata', DistributionMetadata)

# Assigning a Tuple to a Name (line 1016):

# Obtaining an instance of the builtin type 'tuple' (line 1016)
tuple_3053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1016)
# Adding element type (line 1016)
str_3054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 25), 'str', 'name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3054)
# Adding element type (line 1016)
str_3055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 33), 'str', 'version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3055)
# Adding element type (line 1016)
str_3056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 44), 'str', 'author')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3056)
# Adding element type (line 1016)
str_3057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 54), 'str', 'author_email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3057)
# Adding element type (line 1016)
str_3058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 25), 'str', 'maintainer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3058)
# Adding element type (line 1016)
str_3059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 39), 'str', 'maintainer_email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3059)
# Adding element type (line 1016)
str_3060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 59), 'str', 'url')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3060)
# Adding element type (line 1016)
str_3061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 25), 'str', 'license')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3061)
# Adding element type (line 1016)
str_3062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 36), 'str', 'description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3062)
# Adding element type (line 1016)
str_3063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 51), 'str', 'long_description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3063)
# Adding element type (line 1016)
str_3064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 25), 'str', 'keywords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3064)
# Adding element type (line 1016)
str_3065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 37), 'str', 'platforms')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3065)
# Adding element type (line 1016)
str_3066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 50), 'str', 'fullname')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3066)
# Adding element type (line 1016)
str_3067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 62), 'str', 'contact')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3067)
# Adding element type (line 1016)
str_3068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 25), 'str', 'contact_email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3068)
# Adding element type (line 1016)
str_3069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 42), 'str', 'license')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3069)
# Adding element type (line 1016)
str_3070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 53), 'str', 'classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3070)
# Adding element type (line 1016)
str_3071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 25), 'str', 'download_url')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3071)
# Adding element type (line 1016)
str_3072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 25), 'str', 'provides')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3072)
# Adding element type (line 1016)
str_3073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 37), 'str', 'requires')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3073)
# Adding element type (line 1016)
str_3074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 49), 'str', 'obsoletes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1016, 25), tuple_3053, str_3074)

# Getting the type of 'DistributionMetadata'
DistributionMetadata_3075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DistributionMetadata')
# Setting the type of the member '_METHOD_BASENAMES' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DistributionMetadata_3075, '_METHOD_BASENAMES', tuple_3053)

# Assigning a Name to a Name (line 1193):
# Getting the type of 'DistributionMetadata'
DistributionMetadata_3076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DistributionMetadata')
# Obtaining the member 'get_license' of a type
get_license_3077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DistributionMetadata_3076, 'get_license')
# Getting the type of 'DistributionMetadata'
DistributionMetadata_3078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DistributionMetadata')
# Setting the type of the member 'get_licence' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DistributionMetadata_3078, 'get_licence', get_license_3077)

@norecursion
def fix_help_options(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fix_help_options'
    module_type_store = module_type_store.open_function_context('fix_help_options', 1242, 0, False)
    
    # Passed parameters checking function
    fix_help_options.stypy_localization = localization
    fix_help_options.stypy_type_of_self = None
    fix_help_options.stypy_type_store = module_type_store
    fix_help_options.stypy_function_name = 'fix_help_options'
    fix_help_options.stypy_param_names_list = ['options']
    fix_help_options.stypy_varargs_param_name = None
    fix_help_options.stypy_kwargs_param_name = None
    fix_help_options.stypy_call_defaults = defaults
    fix_help_options.stypy_call_varargs = varargs
    fix_help_options.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fix_help_options', ['options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fix_help_options', localization, ['options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fix_help_options(...)' code ##################

    str_3079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1245, (-1)), 'str', "Convert a 4-tuple 'help_options' list as found in various command\n    classes to the 3-tuple form required by FancyGetopt.\n    ")
    
    # Assigning a List to a Name (line 1246):
    
    # Assigning a List to a Name (line 1246):
    
    # Obtaining an instance of the builtin type 'list' (line 1246)
    list_3080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1246, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1246)
    
    # Assigning a type to the variable 'new_options' (line 1246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1246, 4), 'new_options', list_3080)
    
    # Getting the type of 'options' (line 1247)
    options_3081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 22), 'options')
    # Testing the type of a for loop iterable (line 1247)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1247, 4), options_3081)
    # Getting the type of the for loop variable (line 1247)
    for_loop_var_3082 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1247, 4), options_3081)
    # Assigning a type to the variable 'help_tuple' (line 1247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1247, 4), 'help_tuple', for_loop_var_3082)
    # SSA begins for a for statement (line 1247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 1248)
    # Processing the call arguments (line 1248)
    
    # Obtaining the type of the subscript
    int_3085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 38), 'int')
    int_3086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 40), 'int')
    slice_3087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1248, 27), int_3085, int_3086, None)
    # Getting the type of 'help_tuple' (line 1248)
    help_tuple_3088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 27), 'help_tuple', False)
    # Obtaining the member '__getitem__' of a type (line 1248)
    getitem___3089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 27), help_tuple_3088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1248)
    subscript_call_result_3090 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 27), getitem___3089, slice_3087)
    
    # Processing the call keyword arguments (line 1248)
    kwargs_3091 = {}
    # Getting the type of 'new_options' (line 1248)
    new_options_3083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'new_options', False)
    # Obtaining the member 'append' of a type (line 1248)
    append_3084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 8), new_options_3083, 'append')
    # Calling append(args, kwargs) (line 1248)
    append_call_result_3092 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 8), append_3084, *[subscript_call_result_3090], **kwargs_3091)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'new_options' (line 1249)
    new_options_3093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 11), 'new_options')
    # Assigning a type to the variable 'stypy_return_type' (line 1249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1249, 4), 'stypy_return_type', new_options_3093)
    
    # ################# End of 'fix_help_options(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fix_help_options' in the type store
    # Getting the type of 'stypy_return_type' (line 1242)
    stypy_return_type_3094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3094)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fix_help_options'
    return stypy_return_type_3094

# Assigning a type to the variable 'fix_help_options' (line 1242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 0), 'fix_help_options', fix_help_options)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

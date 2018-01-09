
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.cmd
2: 
3: Provides the Command class, the base class for the command classes
4: in the distutils.command package.
5: '''
6: 
7: __revision__ = "$Id$"
8: 
9: import sys, os, re
10: from distutils.errors import DistutilsOptionError
11: from distutils import util, dir_util, file_util, archive_util, dep_util
12: from distutils import log
13: 
14: class Command:
15:     '''Abstract base class for defining command classes, the "worker bees"
16:     of the Distutils.  A useful analogy for command classes is to think of
17:     them as subroutines with local variables called "options".  The options
18:     are "declared" in 'initialize_options()' and "defined" (given their
19:     final values, aka "finalized") in 'finalize_options()', both of which
20:     must be defined by every command class.  The distinction between the
21:     two is necessary because option values might come from the outside
22:     world (command line, config file, ...), and any options dependent on
23:     other options must be computed *after* these outside influences have
24:     been processed -- hence 'finalize_options()'.  The "body" of the
25:     subroutine, where it does all its work based on the values of its
26:     options, is the 'run()' method, which must also be implemented by every
27:     command class.
28:     '''
29: 
30:     # 'sub_commands' formalizes the notion of a "family" of commands,
31:     # eg. "install" as the parent with sub-commands "install_lib",
32:     # "install_headers", etc.  The parent of a family of commands
33:     # defines 'sub_commands' as a class attribute; it's a list of
34:     #    (command_name : string, predicate : unbound_method | string | None)
35:     # tuples, where 'predicate' is a method of the parent command that
36:     # determines whether the corresponding command is applicable in the
37:     # current situation.  (Eg. we "install_headers" is only applicable if
38:     # we have any C header files to install.)  If 'predicate' is None,
39:     # that command is always applicable.
40:     #
41:     # 'sub_commands' is usually defined at the *end* of a class, because
42:     # predicates can be unbound methods, so they must already have been
43:     # defined.  The canonical example is the "install" command.
44:     sub_commands = []
45: 
46: 
47:     # -- Creation/initialization methods -------------------------------
48: 
49:     def __init__(self, dist):
50:         '''Create and initialize a new Command object.  Most importantly,
51:         invokes the 'initialize_options()' method, which is the real
52:         initializer and depends on the actual command being
53:         instantiated.
54:         '''
55:         # late import because of mutual dependence between these classes
56:         from distutils.dist import Distribution
57: 
58:         if not isinstance(dist, Distribution):
59:             raise TypeError, "dist must be a Distribution instance"
60:         if self.__class__ is Command:
61:             raise RuntimeError, "Command is an abstract class"
62: 
63:         self.distribution = dist
64:         self.initialize_options()
65: 
66:         # Per-command versions of the global flags, so that the user can
67:         # customize Distutils' behaviour command-by-command and let some
68:         # commands fall back on the Distribution's behaviour.  None means
69:         # "not defined, check self.distribution's copy", while 0 or 1 mean
70:         # false and true (duh).  Note that this means figuring out the real
71:         # value of each flag is a touch complicated -- hence "self._dry_run"
72:         # will be handled by __getattr__, below.
73:         # XXX This needs to be fixed.
74:         self._dry_run = None
75: 
76:         # verbose is largely ignored, but needs to be set for
77:         # backwards compatibility (I think)?
78:         self.verbose = dist.verbose
79: 
80:         # Some commands define a 'self.force' option to ignore file
81:         # timestamps, but methods defined *here* assume that
82:         # 'self.force' exists for all commands.  So define it here
83:         # just to be safe.
84:         self.force = None
85: 
86:         # The 'help' flag is just used for command-line parsing, so
87:         # none of that complicated bureaucracy is needed.
88:         self.help = 0
89: 
90:         # 'finalized' records whether or not 'finalize_options()' has been
91:         # called.  'finalize_options()' itself should not pay attention to
92:         # this flag: it is the business of 'ensure_finalized()', which
93:         # always calls 'finalize_options()', to respect/update it.
94:         self.finalized = 0
95: 
96:     # XXX A more explicit way to customize dry_run would be better.
97:     def __getattr__(self, attr):
98:         if attr == 'dry_run':
99:             myval = getattr(self, "_" + attr)
100:             if myval is None:
101:                 return getattr(self.distribution, attr)
102:             else:
103:                 return myval
104:         else:
105:             raise AttributeError, attr
106: 
107:     def ensure_finalized(self):
108:         if not self.finalized:
109:             self.finalize_options()
110:         self.finalized = 1
111: 
112:     # Subclasses must define:
113:     #   initialize_options()
114:     #     provide default values for all options; may be customized by
115:     #     setup script, by options from config file(s), or by command-line
116:     #     options
117:     #   finalize_options()
118:     #     decide on the final values for all options; this is called
119:     #     after all possible intervention from the outside world
120:     #     (command-line, option file, etc.) has been processed
121:     #   run()
122:     #     run the command: do whatever it is we're here to do,
123:     #     controlled by the command's various option values
124: 
125:     def initialize_options(self):
126:         '''Set default values for all the options that this command
127:         supports.  Note that these defaults may be overridden by other
128:         commands, by the setup script, by config files, or by the
129:         command-line.  Thus, this is not the place to code dependencies
130:         between options; generally, 'initialize_options()' implementations
131:         are just a bunch of "self.foo = None" assignments.
132: 
133:         This method must be implemented by all command classes.
134:         '''
135:         raise RuntimeError, \
136:               "abstract method -- subclass %s must override" % self.__class__
137: 
138:     def finalize_options(self):
139:         '''Set final values for all the options that this command supports.
140:         This is always called as late as possible, ie.  after any option
141:         assignments from the command-line or from other commands have been
142:         done.  Thus, this is the place to code option dependencies: if
143:         'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as
144:         long as 'foo' still has the same value it was assigned in
145:         'initialize_options()'.
146: 
147:         This method must be implemented by all command classes.
148:         '''
149:         raise RuntimeError, \
150:               "abstract method -- subclass %s must override" % self.__class__
151: 
152: 
153:     def dump_options(self, header=None, indent=""):
154:         from distutils.fancy_getopt import longopt_xlate
155:         if header is None:
156:             header = "command options for '%s':" % self.get_command_name()
157:         self.announce(indent + header, level=log.INFO)
158:         indent = indent + "  "
159:         for (option, _, _) in self.user_options:
160:             option = option.translate(longopt_xlate)
161:             if option[-1] == "=":
162:                 option = option[:-1]
163:             value = getattr(self, option)
164:             self.announce(indent + "%s = %s" % (option, value),
165:                           level=log.INFO)
166: 
167:     def run(self):
168:         '''A command's raison d'etre: carry out the action it exists to
169:         perform, controlled by the options initialized in
170:         'initialize_options()', customized by other commands, the setup
171:         script, the command-line, and config files, and finalized in
172:         'finalize_options()'.  All terminal output and filesystem
173:         interaction should be done by 'run()'.
174: 
175:         This method must be implemented by all command classes.
176:         '''
177:         raise RuntimeError, \
178:               "abstract method -- subclass %s must override" % self.__class__
179: 
180:     def announce(self, msg, level=1):
181:         '''If the current verbosity level is of greater than or equal to
182:         'level' print 'msg' to stdout.
183:         '''
184:         log.log(level, msg)
185: 
186:     def debug_print(self, msg):
187:         '''Print 'msg' to stdout if the global DEBUG (taken from the
188:         DISTUTILS_DEBUG environment variable) flag is true.
189:         '''
190:         from distutils.debug import DEBUG
191:         if DEBUG:
192:             print msg
193:             sys.stdout.flush()
194: 
195: 
196:     # -- Option validation methods -------------------------------------
197:     # (these are very handy in writing the 'finalize_options()' method)
198:     #
199:     # NB. the general philosophy here is to ensure that a particular option
200:     # value meets certain type and value constraints.  If not, we try to
201:     # force it into conformance (eg. if we expect a list but have a string,
202:     # split the string on comma and/or whitespace).  If we can't force the
203:     # option into conformance, raise DistutilsOptionError.  Thus, command
204:     # classes need do nothing more than (eg.)
205:     #   self.ensure_string_list('foo')
206:     # and they can be guaranteed that thereafter, self.foo will be
207:     # a list of strings.
208: 
209:     def _ensure_stringlike(self, option, what, default=None):
210:         val = getattr(self, option)
211:         if val is None:
212:             setattr(self, option, default)
213:             return default
214:         elif not isinstance(val, str):
215:             raise DistutilsOptionError, \
216:                   "'%s' must be a %s (got `%s`)" % (option, what, val)
217:         return val
218: 
219:     def ensure_string(self, option, default=None):
220:         '''Ensure that 'option' is a string; if not defined, set it to
221:         'default'.
222:         '''
223:         self._ensure_stringlike(option, "string", default)
224: 
225:     def ensure_string_list(self, option):
226:         '''Ensure that 'option' is a list of strings.  If 'option' is
227:         currently a string, we split it either on /,\s*/ or /\s+/, so
228:         "foo bar baz", "foo,bar,baz", and "foo,   bar baz" all become
229:         ["foo", "bar", "baz"].
230:         '''
231:         val = getattr(self, option)
232:         if val is None:
233:             return
234:         elif isinstance(val, str):
235:             setattr(self, option, re.split(r',\s*|\s+', val))
236:         else:
237:             if isinstance(val, list):
238:                 # checks if all elements are str
239:                 ok = 1
240:                 for element in val:
241:                     if not isinstance(element, str):
242:                         ok = 0
243:                         break
244:             else:
245:                 ok = 0
246: 
247:             if not ok:
248:                 raise DistutilsOptionError, \
249:                     "'%s' must be a list of strings (got %r)" % \
250:                         (option, val)
251: 
252: 
253:     def _ensure_tested_string(self, option, tester,
254:                               what, error_fmt, default=None):
255:         val = self._ensure_stringlike(option, what, default)
256:         if val is not None and not tester(val):
257:             raise DistutilsOptionError, \
258:                   ("error in '%s' option: " + error_fmt) % (option, val)
259: 
260:     def ensure_filename(self, option):
261:         '''Ensure that 'option' is the name of an existing file.'''
262:         self._ensure_tested_string(option, os.path.isfile,
263:                                    "filename",
264:                                    "'%s' does not exist or is not a file")
265: 
266:     def ensure_dirname(self, option):
267:         self._ensure_tested_string(option, os.path.isdir,
268:                                    "directory name",
269:                                    "'%s' does not exist or is not a directory")
270: 
271: 
272:     # -- Convenience methods for commands ------------------------------
273: 
274:     def get_command_name(self):
275:         if hasattr(self, 'command_name'):
276:             return self.command_name
277:         else:
278:             return self.__class__.__name__
279: 
280:     def set_undefined_options(self, src_cmd, *option_pairs):
281:         '''Set the values of any "undefined" options from corresponding
282:         option values in some other command object.  "Undefined" here means
283:         "is None", which is the convention used to indicate that an option
284:         has not been changed between 'initialize_options()' and
285:         'finalize_options()'.  Usually called from 'finalize_options()' for
286:         options that depend on some other command rather than another
287:         option of the same command.  'src_cmd' is the other command from
288:         which option values will be taken (a command object will be created
289:         for it if necessary); the remaining arguments are
290:         '(src_option,dst_option)' tuples which mean "take the value of
291:         'src_option' in the 'src_cmd' command object, and copy it to
292:         'dst_option' in the current command object".
293:         '''
294: 
295:         # Option_pairs: list of (src_option, dst_option) tuples
296: 
297:         src_cmd_obj = self.distribution.get_command_obj(src_cmd)
298:         src_cmd_obj.ensure_finalized()
299:         for (src_option, dst_option) in option_pairs:
300:             if getattr(self, dst_option) is None:
301:                 setattr(self, dst_option,
302:                         getattr(src_cmd_obj, src_option))
303: 
304: 
305:     def get_finalized_command(self, command, create=1):
306:         '''Wrapper around Distribution's 'get_command_obj()' method: find
307:         (create if necessary and 'create' is true) the command object for
308:         'command', call its 'ensure_finalized()' method, and return the
309:         finalized command object.
310:         '''
311:         cmd_obj = self.distribution.get_command_obj(command, create)
312:         cmd_obj.ensure_finalized()
313:         return cmd_obj
314: 
315:     # XXX rename to 'get_reinitialized_command()'? (should do the
316:     # same in dist.py, if so)
317:     def reinitialize_command(self, command, reinit_subcommands=0):
318:         return self.distribution.reinitialize_command(
319:             command, reinit_subcommands)
320: 
321:     def run_command(self, command):
322:         '''Run some other command: uses the 'run_command()' method of
323:         Distribution, which creates and finalizes the command object if
324:         necessary and then invokes its 'run()' method.
325:         '''
326:         self.distribution.run_command(command)
327: 
328:     def get_sub_commands(self):
329:         '''Determine the sub-commands that are relevant in the current
330:         distribution (ie., that need to be run).  This is based on the
331:         'sub_commands' class attribute: each tuple in that list may include
332:         a method that we call to determine if the subcommand needs to be
333:         run for the current distribution.  Return a list of command names.
334:         '''
335:         commands = []
336:         for (cmd_name, method) in self.sub_commands:
337:             if method is None or method(self):
338:                 commands.append(cmd_name)
339:         return commands
340: 
341: 
342:     # -- External world manipulation -----------------------------------
343: 
344:     def warn(self, msg):
345:         log.warn("warning: %s: %s\n" %
346:                 (self.get_command_name(), msg))
347: 
348:     def execute(self, func, args, msg=None, level=1):
349:         util.execute(func, args, msg, dry_run=self.dry_run)
350: 
351:     def mkpath(self, name, mode=0777):
352:         dir_util.mkpath(name, mode, dry_run=self.dry_run)
353: 
354:     def copy_file(self, infile, outfile,
355:                    preserve_mode=1, preserve_times=1, link=None, level=1):
356:         '''Copy a file respecting verbose, dry-run and force flags.  (The
357:         former two default to whatever is in the Distribution object, and
358:         the latter defaults to false for commands that don't define it.)'''
359: 
360:         return file_util.copy_file(
361:             infile, outfile,
362:             preserve_mode, preserve_times,
363:             not self.force,
364:             link,
365:             dry_run=self.dry_run)
366: 
367:     def copy_tree(self, infile, outfile,
368:                    preserve_mode=1, preserve_times=1, preserve_symlinks=0,
369:                    level=1):
370:         '''Copy an entire directory tree respecting verbose, dry-run,
371:         and force flags.
372:         '''
373:         return dir_util.copy_tree(
374:             infile, outfile,
375:             preserve_mode,preserve_times,preserve_symlinks,
376:             not self.force,
377:             dry_run=self.dry_run)
378: 
379:     def move_file (self, src, dst, level=1):
380:         '''Move a file respecting dry-run flag.'''
381:         return file_util.move_file(src, dst, dry_run = self.dry_run)
382: 
383:     def spawn (self, cmd, search_path=1, level=1):
384:         '''Spawn an external command respecting dry-run flag.'''
385:         from distutils.spawn import spawn
386:         spawn(cmd, search_path, dry_run= self.dry_run)
387: 
388:     def make_archive(self, base_name, format, root_dir=None, base_dir=None,
389:                      owner=None, group=None):
390:         return archive_util.make_archive(base_name, format, root_dir,
391:                                          base_dir, dry_run=self.dry_run,
392:                                          owner=owner, group=group)
393: 
394:     def make_file(self, infiles, outfile, func, args,
395:                   exec_msg=None, skip_msg=None, level=1):
396:         '''Special case of 'execute()' for operations that process one or
397:         more input files and generate one output file.  Works just like
398:         'execute()', except the operation is skipped and a different
399:         message printed if 'outfile' already exists and is newer than all
400:         files listed in 'infiles'.  If the command defined 'self.force',
401:         and it is true, then the command is unconditionally run -- does no
402:         timestamp checks.
403:         '''
404:         if skip_msg is None:
405:             skip_msg = "skipping %s (inputs unchanged)" % outfile
406: 
407:         # Allow 'infiles' to be a single string
408:         if isinstance(infiles, str):
409:             infiles = (infiles,)
410:         elif not isinstance(infiles, (list, tuple)):
411:             raise TypeError, \
412:                   "'infiles' must be a string, or a list or tuple of strings"
413: 
414:         if exec_msg is None:
415:             exec_msg = "generating %s from %s" % \
416:                        (outfile, ', '.join(infiles))
417: 
418:         # If 'outfile' must be regenerated (either because it doesn't
419:         # exist, is out-of-date, or the 'force' flag is true) then
420:         # perform the action that presumably regenerates it
421:         if self.force or dep_util.newer_group(infiles, outfile):
422:             self.execute(func, args, exec_msg, level)
423: 
424:         # Otherwise, print the "skip" message
425:         else:
426:             log.debug(skip_msg)
427: 
428: # XXX 'install_misc' class not currently used -- it was the base class for
429: # both 'install_scripts' and 'install_data', but they outgrew it.  It might
430: # still be useful for 'install_headers', though, so I'm keeping it around
431: # for the time being.
432: 
433: class install_misc(Command):
434:     '''Common base class for installing some files in a subdirectory.
435:     Currently used by install_data and install_scripts.
436:     '''
437: 
438:     user_options = [('install-dir=', 'd', "directory to install the files to")]
439: 
440:     def initialize_options (self):
441:         self.install_dir = None
442:         self.outfiles = []
443: 
444:     def _install_dir_from(self, dirname):
445:         self.set_undefined_options('install', (dirname, 'install_dir'))
446: 
447:     def _copy_files(self, filelist):
448:         self.outfiles = []
449:         if not filelist:
450:             return
451:         self.mkpath(self.install_dir)
452:         for f in filelist:
453:             self.copy_file(f, self.install_dir)
454:             self.outfiles.append(os.path.join(self.install_dir, f))
455: 
456:     def get_outputs(self):
457:         return self.outfiles
458: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_305500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.cmd\n\nProvides the Command class, the base class for the command classes\nin the distutils.command package.\n')

# Assigning a Str to a Name (line 7):
str_305501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_305501)
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

# 'from distutils.errors import DistutilsOptionError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_305502 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_305502) is not StypyTypeError):

    if (import_305502 != 'pyd_module'):
        __import__(import_305502)
        sys_modules_305503 = sys.modules[import_305502]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_305503.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_305503, sys_modules_305503.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_305502)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils import util, dir_util, file_util, archive_util, dep_util' statement (line 11)
try:
    from distutils import util, dir_util, file_util, archive_util, dep_util

except:
    util = UndefinedType
    dir_util = UndefinedType
    file_util = UndefinedType
    archive_util = UndefinedType
    dep_util = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils', None, module_type_store, ['util', 'dir_util', 'file_util', 'archive_util', 'dep_util'], [util, dir_util, file_util, archive_util, dep_util])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils import log' statement (line 12)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'Command' class

class Command:
    str_305504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', 'Abstract base class for defining command classes, the "worker bees"\n    of the Distutils.  A useful analogy for command classes is to think of\n    them as subroutines with local variables called "options".  The options\n    are "declared" in \'initialize_options()\' and "defined" (given their\n    final values, aka "finalized") in \'finalize_options()\', both of which\n    must be defined by every command class.  The distinction between the\n    two is necessary because option values might come from the outside\n    world (command line, config file, ...), and any options dependent on\n    other options must be computed *after* these outside influences have\n    been processed -- hence \'finalize_options()\'.  The "body" of the\n    subroutine, where it does all its work based on the values of its\n    options, is the \'run()\' method, which must also be implemented by every\n    command class.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.__init__', ['dist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_305505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', "Create and initialize a new Command object.  Most importantly,\n        invokes the 'initialize_options()' method, which is the real\n        initializer and depends on the actual command being\n        instantiated.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 8))
        
        # 'from distutils.dist import Distribution' statement (line 56)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_305506 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 8), 'distutils.dist')

        if (type(import_305506) is not StypyTypeError):

            if (import_305506 != 'pyd_module'):
                __import__(import_305506)
                sys_modules_305507 = sys.modules[import_305506]
                import_from_module(stypy.reporting.localization.Localization(__file__, 56, 8), 'distutils.dist', sys_modules_305507.module_type_store, module_type_store, ['Distribution'])
                nest_module(stypy.reporting.localization.Localization(__file__, 56, 8), __file__, sys_modules_305507, sys_modules_305507.module_type_store, module_type_store)
            else:
                from distutils.dist import Distribution

                import_from_module(stypy.reporting.localization.Localization(__file__, 56, 8), 'distutils.dist', None, module_type_store, ['Distribution'], [Distribution])

        else:
            # Assigning a type to the variable 'distutils.dist' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'distutils.dist', import_305506)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        
        
        # Call to isinstance(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'dist' (line 58)
        dist_305509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'dist', False)
        # Getting the type of 'Distribution' (line 58)
        Distribution_305510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'Distribution', False)
        # Processing the call keyword arguments (line 58)
        kwargs_305511 = {}
        # Getting the type of 'isinstance' (line 58)
        isinstance_305508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 58)
        isinstance_call_result_305512 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), isinstance_305508, *[dist_305509, Distribution_305510], **kwargs_305511)
        
        # Applying the 'not' unary operator (line 58)
        result_not__305513 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), 'not', isinstance_call_result_305512)
        
        # Testing the type of an if condition (line 58)
        if_condition_305514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_not__305513)
        # Assigning a type to the variable 'if_condition_305514' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_305514', if_condition_305514)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'TypeError' (line 59)
        TypeError_305515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'TypeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 59, 12), TypeError_305515, 'raise parameter', BaseException)
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 60)
        self_305516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'self')
        # Obtaining the member '__class__' of a type (line 60)
        class___305517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), self_305516, '__class__')
        # Getting the type of 'Command' (line 60)
        Command_305518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'Command')
        # Applying the binary operator 'is' (line 60)
        result_is__305519 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 11), 'is', class___305517, Command_305518)
        
        # Testing the type of an if condition (line 60)
        if_condition_305520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), result_is__305519)
        # Assigning a type to the variable 'if_condition_305520' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_305520', if_condition_305520)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'RuntimeError' (line 61)
        RuntimeError_305521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'RuntimeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 12), RuntimeError_305521, 'raise parameter', BaseException)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'dist' (line 63)
        dist_305522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'dist')
        # Getting the type of 'self' (line 63)
        self_305523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'distribution' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_305523, 'distribution', dist_305522)
        
        # Call to initialize_options(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_305526 = {}
        # Getting the type of 'self' (line 64)
        self_305524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'initialize_options' of a type (line 64)
        initialize_options_305525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_305524, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 64)
        initialize_options_call_result_305527 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), initialize_options_305525, *[], **kwargs_305526)
        
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'None' (line 74)
        None_305528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'None')
        # Getting the type of 'self' (line 74)
        self_305529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_dry_run' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_305529, '_dry_run', None_305528)
        
        # Assigning a Attribute to a Attribute (line 78):
        # Getting the type of 'dist' (line 78)
        dist_305530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'dist')
        # Obtaining the member 'verbose' of a type (line 78)
        verbose_305531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), dist_305530, 'verbose')
        # Getting the type of 'self' (line 78)
        self_305532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_305532, 'verbose', verbose_305531)
        
        # Assigning a Name to a Attribute (line 84):
        # Getting the type of 'None' (line 84)
        None_305533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'None')
        # Getting the type of 'self' (line 84)
        self_305534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'force' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_305534, 'force', None_305533)
        
        # Assigning a Num to a Attribute (line 88):
        int_305535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'int')
        # Getting the type of 'self' (line 88)
        self_305536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member 'help' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_305536, 'help', int_305535)
        
        # Assigning a Num to a Attribute (line 94):
        int_305537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'int')
        # Getting the type of 'self' (line 94)
        self_305538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'finalized' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_305538, 'finalized', int_305537)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        Command.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.__getattr__.__dict__.__setitem__('stypy_function_name', 'Command.__getattr__')
        Command.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        Command.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 98)
        attr_305539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'attr')
        str_305540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'str', 'dry_run')
        # Applying the binary operator '==' (line 98)
        result_eq_305541 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), '==', attr_305539, str_305540)
        
        # Testing the type of an if condition (line 98)
        if_condition_305542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_eq_305541)
        # Assigning a type to the variable 'if_condition_305542' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_305542', if_condition_305542)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 99):
        
        # Call to getattr(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_305544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'self', False)
        str_305545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 34), 'str', '_')
        # Getting the type of 'attr' (line 99)
        attr_305546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'attr', False)
        # Applying the binary operator '+' (line 99)
        result_add_305547 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 34), '+', str_305545, attr_305546)
        
        # Processing the call keyword arguments (line 99)
        kwargs_305548 = {}
        # Getting the type of 'getattr' (line 99)
        getattr_305543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 99)
        getattr_call_result_305549 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), getattr_305543, *[self_305544, result_add_305547], **kwargs_305548)
        
        # Assigning a type to the variable 'myval' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'myval', getattr_call_result_305549)
        
        # Type idiom detected: calculating its left and rigth part (line 100)
        # Getting the type of 'myval' (line 100)
        myval_305550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'myval')
        # Getting the type of 'None' (line 100)
        None_305551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'None')
        
        (may_be_305552, more_types_in_union_305553) = may_be_none(myval_305550, None_305551)

        if may_be_305552:

            if more_types_in_union_305553:
                # Runtime conditional SSA (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to getattr(...): (line 101)
            # Processing the call arguments (line 101)
            # Getting the type of 'self' (line 101)
            self_305555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'self', False)
            # Obtaining the member 'distribution' of a type (line 101)
            distribution_305556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 31), self_305555, 'distribution')
            # Getting the type of 'attr' (line 101)
            attr_305557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'attr', False)
            # Processing the call keyword arguments (line 101)
            kwargs_305558 = {}
            # Getting the type of 'getattr' (line 101)
            getattr_305554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'getattr', False)
            # Calling getattr(args, kwargs) (line 101)
            getattr_call_result_305559 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), getattr_305554, *[distribution_305556, attr_305557], **kwargs_305558)
            
            # Assigning a type to the variable 'stypy_return_type' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'stypy_return_type', getattr_call_result_305559)

            if more_types_in_union_305553:
                # Runtime conditional SSA for else branch (line 100)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_305552) or more_types_in_union_305553):
            # Getting the type of 'myval' (line 103)
            myval_305560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'myval')
            # Assigning a type to the variable 'stypy_return_type' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'stypy_return_type', myval_305560)

            if (may_be_305552 and more_types_in_union_305553):
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 98)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'AttributeError' (line 105)
        AttributeError_305561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'AttributeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 105, 12), AttributeError_305561, 'raise parameter', BaseException)
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_305562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305562)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_305562


    @norecursion
    def ensure_finalized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ensure_finalized'
        module_type_store = module_type_store.open_function_context('ensure_finalized', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.ensure_finalized.__dict__.__setitem__('stypy_localization', localization)
        Command.ensure_finalized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.ensure_finalized.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.ensure_finalized.__dict__.__setitem__('stypy_function_name', 'Command.ensure_finalized')
        Command.ensure_finalized.__dict__.__setitem__('stypy_param_names_list', [])
        Command.ensure_finalized.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.ensure_finalized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.ensure_finalized.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.ensure_finalized.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.ensure_finalized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.ensure_finalized.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.ensure_finalized', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ensure_finalized', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ensure_finalized(...)' code ##################

        
        
        # Getting the type of 'self' (line 108)
        self_305563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self')
        # Obtaining the member 'finalized' of a type (line 108)
        finalized_305564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_305563, 'finalized')
        # Applying the 'not' unary operator (line 108)
        result_not__305565 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), 'not', finalized_305564)
        
        # Testing the type of an if condition (line 108)
        if_condition_305566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 8), result_not__305565)
        # Assigning a type to the variable 'if_condition_305566' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'if_condition_305566', if_condition_305566)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to finalize_options(...): (line 109)
        # Processing the call keyword arguments (line 109)
        kwargs_305569 = {}
        # Getting the type of 'self' (line 109)
        self_305567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'self', False)
        # Obtaining the member 'finalize_options' of a type (line 109)
        finalize_options_305568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), self_305567, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 109)
        finalize_options_call_result_305570 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), finalize_options_305568, *[], **kwargs_305569)
        
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Attribute (line 110):
        int_305571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'int')
        # Getting the type of 'self' (line 110)
        self_305572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'finalized' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_305572, 'finalized', int_305571)
        
        # ################# End of 'ensure_finalized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ensure_finalized' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_305573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ensure_finalized'
        return stypy_return_type_305573


    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        Command.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.initialize_options.__dict__.__setitem__('stypy_function_name', 'Command.initialize_options')
        Command.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        Command.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        str_305574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', 'Set default values for all the options that this command\n        supports.  Note that these defaults may be overridden by other\n        commands, by the setup script, by config files, or by the\n        command-line.  Thus, this is not the place to code dependencies\n        between options; generally, \'initialize_options()\' implementations\n        are just a bunch of "self.foo = None" assignments.\n\n        This method must be implemented by all command classes.\n        ')
        # Getting the type of 'RuntimeError' (line 135)
        RuntimeError_305575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'RuntimeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 8), RuntimeError_305575, 'raise parameter', BaseException)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_305576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_305576


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        Command.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.finalize_options.__dict__.__setitem__('stypy_function_name', 'Command.finalize_options')
        Command.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        Command.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        str_305577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, (-1)), 'str', "Set final values for all the options that this command supports.\n        This is always called as late as possible, ie.  after any option\n        assignments from the command-line or from other commands have been\n        done.  Thus, this is the place to code option dependencies: if\n        'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as\n        long as 'foo' still has the same value it was assigned in\n        'initialize_options()'.\n\n        This method must be implemented by all command classes.\n        ")
        # Getting the type of 'RuntimeError' (line 149)
        RuntimeError_305578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'RuntimeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 149, 8), RuntimeError_305578, 'raise parameter', BaseException)
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_305579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_305579


    @norecursion
    def dump_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 153)
        None_305580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'None')
        str_305581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 47), 'str', '')
        defaults = [None_305580, str_305581]
        # Create a new context for function 'dump_options'
        module_type_store = module_type_store.open_function_context('dump_options', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.dump_options.__dict__.__setitem__('stypy_localization', localization)
        Command.dump_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.dump_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.dump_options.__dict__.__setitem__('stypy_function_name', 'Command.dump_options')
        Command.dump_options.__dict__.__setitem__('stypy_param_names_list', ['header', 'indent'])
        Command.dump_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.dump_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.dump_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.dump_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.dump_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.dump_options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.dump_options', ['header', 'indent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_options', localization, ['header', 'indent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_options(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 154, 8))
        
        # 'from distutils.fancy_getopt import longopt_xlate' statement (line 154)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_305582 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 154, 8), 'distutils.fancy_getopt')

        if (type(import_305582) is not StypyTypeError):

            if (import_305582 != 'pyd_module'):
                __import__(import_305582)
                sys_modules_305583 = sys.modules[import_305582]
                import_from_module(stypy.reporting.localization.Localization(__file__, 154, 8), 'distutils.fancy_getopt', sys_modules_305583.module_type_store, module_type_store, ['longopt_xlate'])
                nest_module(stypy.reporting.localization.Localization(__file__, 154, 8), __file__, sys_modules_305583, sys_modules_305583.module_type_store, module_type_store)
            else:
                from distutils.fancy_getopt import longopt_xlate

                import_from_module(stypy.reporting.localization.Localization(__file__, 154, 8), 'distutils.fancy_getopt', None, module_type_store, ['longopt_xlate'], [longopt_xlate])

        else:
            # Assigning a type to the variable 'distutils.fancy_getopt' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'distutils.fancy_getopt', import_305582)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Type idiom detected: calculating its left and rigth part (line 155)
        # Getting the type of 'header' (line 155)
        header_305584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'header')
        # Getting the type of 'None' (line 155)
        None_305585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'None')
        
        (may_be_305586, more_types_in_union_305587) = may_be_none(header_305584, None_305585)

        if may_be_305586:

            if more_types_in_union_305587:
                # Runtime conditional SSA (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 156):
            str_305588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'str', "command options for '%s':")
            
            # Call to get_command_name(...): (line 156)
            # Processing the call keyword arguments (line 156)
            kwargs_305591 = {}
            # Getting the type of 'self' (line 156)
            self_305589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 51), 'self', False)
            # Obtaining the member 'get_command_name' of a type (line 156)
            get_command_name_305590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 51), self_305589, 'get_command_name')
            # Calling get_command_name(args, kwargs) (line 156)
            get_command_name_call_result_305592 = invoke(stypy.reporting.localization.Localization(__file__, 156, 51), get_command_name_305590, *[], **kwargs_305591)
            
            # Applying the binary operator '%' (line 156)
            result_mod_305593 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 21), '%', str_305588, get_command_name_call_result_305592)
            
            # Assigning a type to the variable 'header' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'header', result_mod_305593)

            if more_types_in_union_305587:
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to announce(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'indent' (line 157)
        indent_305596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'indent', False)
        # Getting the type of 'header' (line 157)
        header_305597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'header', False)
        # Applying the binary operator '+' (line 157)
        result_add_305598 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 22), '+', indent_305596, header_305597)
        
        # Processing the call keyword arguments (line 157)
        # Getting the type of 'log' (line 157)
        log_305599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 45), 'log', False)
        # Obtaining the member 'INFO' of a type (line 157)
        INFO_305600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 45), log_305599, 'INFO')
        keyword_305601 = INFO_305600
        kwargs_305602 = {'level': keyword_305601}
        # Getting the type of 'self' (line 157)
        self_305594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self', False)
        # Obtaining the member 'announce' of a type (line 157)
        announce_305595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_305594, 'announce')
        # Calling announce(args, kwargs) (line 157)
        announce_call_result_305603 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), announce_305595, *[result_add_305598], **kwargs_305602)
        
        
        # Assigning a BinOp to a Name (line 158):
        # Getting the type of 'indent' (line 158)
        indent_305604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'indent')
        str_305605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'str', '  ')
        # Applying the binary operator '+' (line 158)
        result_add_305606 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 17), '+', indent_305604, str_305605)
        
        # Assigning a type to the variable 'indent' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'indent', result_add_305606)
        
        # Getting the type of 'self' (line 159)
        self_305607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'self')
        # Obtaining the member 'user_options' of a type (line 159)
        user_options_305608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 30), self_305607, 'user_options')
        # Testing the type of a for loop iterable (line 159)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 8), user_options_305608)
        # Getting the type of the for loop variable (line 159)
        for_loop_var_305609 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 8), user_options_305608)
        # Assigning a type to the variable 'option' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'option', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_305609))
        # Assigning a type to the variable '_' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_305609))
        # Assigning a type to the variable '_' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_305609))
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 160):
        
        # Call to translate(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'longopt_xlate' (line 160)
        longopt_xlate_305612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'longopt_xlate', False)
        # Processing the call keyword arguments (line 160)
        kwargs_305613 = {}
        # Getting the type of 'option' (line 160)
        option_305610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'option', False)
        # Obtaining the member 'translate' of a type (line 160)
        translate_305611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 21), option_305610, 'translate')
        # Calling translate(args, kwargs) (line 160)
        translate_call_result_305614 = invoke(stypy.reporting.localization.Localization(__file__, 160, 21), translate_305611, *[longopt_xlate_305612], **kwargs_305613)
        
        # Assigning a type to the variable 'option' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'option', translate_call_result_305614)
        
        
        
        # Obtaining the type of the subscript
        int_305615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'int')
        # Getting the type of 'option' (line 161)
        option_305616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'option')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___305617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), option_305616, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_305618 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), getitem___305617, int_305615)
        
        str_305619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 29), 'str', '=')
        # Applying the binary operator '==' (line 161)
        result_eq_305620 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '==', subscript_call_result_305618, str_305619)
        
        # Testing the type of an if condition (line 161)
        if_condition_305621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_eq_305620)
        # Assigning a type to the variable 'if_condition_305621' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_305621', if_condition_305621)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 162):
        
        # Obtaining the type of the subscript
        int_305622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 33), 'int')
        slice_305623 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 25), None, int_305622, None)
        # Getting the type of 'option' (line 162)
        option_305624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'option')
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___305625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 25), option_305624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_305626 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), getitem___305625, slice_305623)
        
        # Assigning a type to the variable 'option' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'option', subscript_call_result_305626)
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 163):
        
        # Call to getattr(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'self' (line 163)
        self_305628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'self', False)
        # Getting the type of 'option' (line 163)
        option_305629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'option', False)
        # Processing the call keyword arguments (line 163)
        kwargs_305630 = {}
        # Getting the type of 'getattr' (line 163)
        getattr_305627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 163)
        getattr_call_result_305631 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), getattr_305627, *[self_305628, option_305629], **kwargs_305630)
        
        # Assigning a type to the variable 'value' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'value', getattr_call_result_305631)
        
        # Call to announce(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'indent' (line 164)
        indent_305634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'indent', False)
        str_305635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'str', '%s = %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_305636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'option' (line 164)
        option_305637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 48), 'option', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 48), tuple_305636, option_305637)
        # Adding element type (line 164)
        # Getting the type of 'value' (line 164)
        value_305638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 56), 'value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 48), tuple_305636, value_305638)
        
        # Applying the binary operator '%' (line 164)
        result_mod_305639 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 35), '%', str_305635, tuple_305636)
        
        # Applying the binary operator '+' (line 164)
        result_add_305640 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 26), '+', indent_305634, result_mod_305639)
        
        # Processing the call keyword arguments (line 164)
        # Getting the type of 'log' (line 165)
        log_305641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'log', False)
        # Obtaining the member 'INFO' of a type (line 165)
        INFO_305642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), log_305641, 'INFO')
        keyword_305643 = INFO_305642
        kwargs_305644 = {'level': keyword_305643}
        # Getting the type of 'self' (line 164)
        self_305632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 164)
        announce_305633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_305632, 'announce')
        # Calling announce(args, kwargs) (line 164)
        announce_call_result_305645 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), announce_305633, *[result_add_305640], **kwargs_305644)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dump_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_options' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_305646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305646)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_options'
        return stypy_return_type_305646


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.run.__dict__.__setitem__('stypy_localization', localization)
        Command.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.run.__dict__.__setitem__('stypy_function_name', 'Command.run')
        Command.run.__dict__.__setitem__('stypy_param_names_list', [])
        Command.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.run', [], None, None, defaults, varargs, kwargs)

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

        str_305647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', "A command's raison d'etre: carry out the action it exists to\n        perform, controlled by the options initialized in\n        'initialize_options()', customized by other commands, the setup\n        script, the command-line, and config files, and finalized in\n        'finalize_options()'.  All terminal output and filesystem\n        interaction should be done by 'run()'.\n\n        This method must be implemented by all command classes.\n        ")
        # Getting the type of 'RuntimeError' (line 177)
        RuntimeError_305648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 14), 'RuntimeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 177, 8), RuntimeError_305648, 'raise parameter', BaseException)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_305649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_305649


    @norecursion
    def announce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 34), 'int')
        defaults = [int_305650]
        # Create a new context for function 'announce'
        module_type_store = module_type_store.open_function_context('announce', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.announce.__dict__.__setitem__('stypy_localization', localization)
        Command.announce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.announce.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.announce.__dict__.__setitem__('stypy_function_name', 'Command.announce')
        Command.announce.__dict__.__setitem__('stypy_param_names_list', ['msg', 'level'])
        Command.announce.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.announce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.announce.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.announce.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.announce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.announce.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.announce', ['msg', 'level'], None, None, defaults, varargs, kwargs)

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

        str_305651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'str', "If the current verbosity level is of greater than or equal to\n        'level' print 'msg' to stdout.\n        ")
        
        # Call to log(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'level' (line 184)
        level_305654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'level', False)
        # Getting the type of 'msg' (line 184)
        msg_305655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'msg', False)
        # Processing the call keyword arguments (line 184)
        kwargs_305656 = {}
        # Getting the type of 'log' (line 184)
        log_305652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'log', False)
        # Obtaining the member 'log' of a type (line 184)
        log_305653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), log_305652, 'log')
        # Calling log(args, kwargs) (line 184)
        log_call_result_305657 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), log_305653, *[level_305654, msg_305655], **kwargs_305656)
        
        
        # ################# End of 'announce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'announce' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_305658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'announce'
        return stypy_return_type_305658


    @norecursion
    def debug_print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug_print'
        module_type_store = module_type_store.open_function_context('debug_print', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.debug_print.__dict__.__setitem__('stypy_localization', localization)
        Command.debug_print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.debug_print.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.debug_print.__dict__.__setitem__('stypy_function_name', 'Command.debug_print')
        Command.debug_print.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Command.debug_print.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.debug_print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.debug_print.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.debug_print.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.debug_print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.debug_print.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.debug_print', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'debug_print', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'debug_print(...)' code ##################

        str_305659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'str', "Print 'msg' to stdout if the global DEBUG (taken from the\n        DISTUTILS_DEBUG environment variable) flag is true.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 190, 8))
        
        # 'from distutils.debug import DEBUG' statement (line 190)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_305660 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 190, 8), 'distutils.debug')

        if (type(import_305660) is not StypyTypeError):

            if (import_305660 != 'pyd_module'):
                __import__(import_305660)
                sys_modules_305661 = sys.modules[import_305660]
                import_from_module(stypy.reporting.localization.Localization(__file__, 190, 8), 'distutils.debug', sys_modules_305661.module_type_store, module_type_store, ['DEBUG'])
                nest_module(stypy.reporting.localization.Localization(__file__, 190, 8), __file__, sys_modules_305661, sys_modules_305661.module_type_store, module_type_store)
            else:
                from distutils.debug import DEBUG

                import_from_module(stypy.reporting.localization.Localization(__file__, 190, 8), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

        else:
            # Assigning a type to the variable 'distutils.debug' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'distutils.debug', import_305660)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'DEBUG' (line 191)
        DEBUG_305662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'DEBUG')
        # Testing the type of an if condition (line 191)
        if_condition_305663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), DEBUG_305662)
        # Assigning a type to the variable 'if_condition_305663' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'if_condition_305663', if_condition_305663)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'msg' (line 192)
        msg_305664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'msg')
        
        # Call to flush(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_305668 = {}
        # Getting the type of 'sys' (line 193)
        sys_305665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 193)
        stdout_305666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), sys_305665, 'stdout')
        # Obtaining the member 'flush' of a type (line 193)
        flush_305667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), stdout_305666, 'flush')
        # Calling flush(args, kwargs) (line 193)
        flush_call_result_305669 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), flush_305667, *[], **kwargs_305668)
        
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'debug_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug_print' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_305670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug_print'
        return stypy_return_type_305670


    @norecursion
    def _ensure_stringlike(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 209)
        None_305671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 55), 'None')
        defaults = [None_305671]
        # Create a new context for function '_ensure_stringlike'
        module_type_store = module_type_store.open_function_context('_ensure_stringlike', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command._ensure_stringlike.__dict__.__setitem__('stypy_localization', localization)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_function_name', 'Command._ensure_stringlike')
        Command._ensure_stringlike.__dict__.__setitem__('stypy_param_names_list', ['option', 'what', 'default'])
        Command._ensure_stringlike.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command._ensure_stringlike.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command._ensure_stringlike', ['option', 'what', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ensure_stringlike', localization, ['option', 'what', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ensure_stringlike(...)' code ##################

        
        # Assigning a Call to a Name (line 210):
        
        # Call to getattr(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_305673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'self', False)
        # Getting the type of 'option' (line 210)
        option_305674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'option', False)
        # Processing the call keyword arguments (line 210)
        kwargs_305675 = {}
        # Getting the type of 'getattr' (line 210)
        getattr_305672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'getattr', False)
        # Calling getattr(args, kwargs) (line 210)
        getattr_call_result_305676 = invoke(stypy.reporting.localization.Localization(__file__, 210, 14), getattr_305672, *[self_305673, option_305674], **kwargs_305675)
        
        # Assigning a type to the variable 'val' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'val', getattr_call_result_305676)
        
        # Type idiom detected: calculating its left and rigth part (line 211)
        # Getting the type of 'val' (line 211)
        val_305677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'val')
        # Getting the type of 'None' (line 211)
        None_305678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'None')
        
        (may_be_305679, more_types_in_union_305680) = may_be_none(val_305677, None_305678)

        if may_be_305679:

            if more_types_in_union_305680:
                # Runtime conditional SSA (line 211)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 'self' (line 212)
            self_305682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'self', False)
            # Getting the type of 'option' (line 212)
            option_305683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'option', False)
            # Getting the type of 'default' (line 212)
            default_305684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'default', False)
            # Processing the call keyword arguments (line 212)
            kwargs_305685 = {}
            # Getting the type of 'setattr' (line 212)
            setattr_305681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 212)
            setattr_call_result_305686 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), setattr_305681, *[self_305682, option_305683, default_305684], **kwargs_305685)
            
            # Getting the type of 'default' (line 213)
            default_305687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'default')
            # Assigning a type to the variable 'stypy_return_type' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'stypy_return_type', default_305687)

            if more_types_in_union_305680:
                # Runtime conditional SSA for else branch (line 211)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_305679) or more_types_in_union_305680):
            
            # Type idiom detected: calculating its left and rigth part (line 214)
            # Getting the type of 'str' (line 214)
            str_305688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'str')
            # Getting the type of 'val' (line 214)
            val_305689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'val')
            
            (may_be_305690, more_types_in_union_305691) = may_not_be_subtype(str_305688, val_305689)

            if may_be_305690:

                if more_types_in_union_305691:
                    # Runtime conditional SSA (line 214)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'val' (line 214)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 13), 'val', remove_subtype_from_union(val_305689, str))
                # Getting the type of 'DistutilsOptionError' (line 215)
                DistutilsOptionError_305692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 18), 'DistutilsOptionError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 12), DistutilsOptionError_305692, 'raise parameter', BaseException)

                if more_types_in_union_305691:
                    # SSA join for if statement (line 214)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_305679 and more_types_in_union_305680):
                # SSA join for if statement (line 211)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'val' (line 217)
        val_305693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'val')
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', val_305693)
        
        # ################# End of '_ensure_stringlike(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ensure_stringlike' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_305694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305694)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ensure_stringlike'
        return stypy_return_type_305694


    @norecursion
    def ensure_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 219)
        None_305695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 44), 'None')
        defaults = [None_305695]
        # Create a new context for function 'ensure_string'
        module_type_store = module_type_store.open_function_context('ensure_string', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.ensure_string.__dict__.__setitem__('stypy_localization', localization)
        Command.ensure_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.ensure_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.ensure_string.__dict__.__setitem__('stypy_function_name', 'Command.ensure_string')
        Command.ensure_string.__dict__.__setitem__('stypy_param_names_list', ['option', 'default'])
        Command.ensure_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.ensure_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.ensure_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.ensure_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.ensure_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.ensure_string.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.ensure_string', ['option', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ensure_string', localization, ['option', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ensure_string(...)' code ##################

        str_305696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, (-1)), 'str', "Ensure that 'option' is a string; if not defined, set it to\n        'default'.\n        ")
        
        # Call to _ensure_stringlike(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'option' (line 223)
        option_305699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'option', False)
        str_305700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 40), 'str', 'string')
        # Getting the type of 'default' (line 223)
        default_305701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'default', False)
        # Processing the call keyword arguments (line 223)
        kwargs_305702 = {}
        # Getting the type of 'self' (line 223)
        self_305697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member '_ensure_stringlike' of a type (line 223)
        _ensure_stringlike_305698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_305697, '_ensure_stringlike')
        # Calling _ensure_stringlike(args, kwargs) (line 223)
        _ensure_stringlike_call_result_305703 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), _ensure_stringlike_305698, *[option_305699, str_305700, default_305701], **kwargs_305702)
        
        
        # ################# End of 'ensure_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ensure_string' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_305704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305704)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ensure_string'
        return stypy_return_type_305704


    @norecursion
    def ensure_string_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ensure_string_list'
        module_type_store = module_type_store.open_function_context('ensure_string_list', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.ensure_string_list.__dict__.__setitem__('stypy_localization', localization)
        Command.ensure_string_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.ensure_string_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.ensure_string_list.__dict__.__setitem__('stypy_function_name', 'Command.ensure_string_list')
        Command.ensure_string_list.__dict__.__setitem__('stypy_param_names_list', ['option'])
        Command.ensure_string_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.ensure_string_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.ensure_string_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.ensure_string_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.ensure_string_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.ensure_string_list.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.ensure_string_list', ['option'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ensure_string_list', localization, ['option'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ensure_string_list(...)' code ##################

        str_305705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', 'Ensure that \'option\' is a list of strings.  If \'option\' is\n        currently a string, we split it either on /,\\s*/ or /\\s+/, so\n        "foo bar baz", "foo,bar,baz", and "foo,   bar baz" all become\n        ["foo", "bar", "baz"].\n        ')
        
        # Assigning a Call to a Name (line 231):
        
        # Call to getattr(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_305707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'self', False)
        # Getting the type of 'option' (line 231)
        option_305708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 28), 'option', False)
        # Processing the call keyword arguments (line 231)
        kwargs_305709 = {}
        # Getting the type of 'getattr' (line 231)
        getattr_305706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 14), 'getattr', False)
        # Calling getattr(args, kwargs) (line 231)
        getattr_call_result_305710 = invoke(stypy.reporting.localization.Localization(__file__, 231, 14), getattr_305706, *[self_305707, option_305708], **kwargs_305709)
        
        # Assigning a type to the variable 'val' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'val', getattr_call_result_305710)
        
        # Type idiom detected: calculating its left and rigth part (line 232)
        # Getting the type of 'val' (line 232)
        val_305711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'val')
        # Getting the type of 'None' (line 232)
        None_305712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'None')
        
        (may_be_305713, more_types_in_union_305714) = may_be_none(val_305711, None_305712)

        if may_be_305713:

            if more_types_in_union_305714:
                # Runtime conditional SSA (line 232)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 233)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_305714:
                # Runtime conditional SSA for else branch (line 232)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_305713) or more_types_in_union_305714):
            
            # Type idiom detected: calculating its left and rigth part (line 234)
            # Getting the type of 'str' (line 234)
            str_305715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'str')
            # Getting the type of 'val' (line 234)
            val_305716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'val')
            
            (may_be_305717, more_types_in_union_305718) = may_be_subtype(str_305715, val_305716)

            if may_be_305717:

                if more_types_in_union_305718:
                    # Runtime conditional SSA (line 234)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'val' (line 234)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'val', remove_not_subtype_from_union(val_305716, str))
                
                # Call to setattr(...): (line 235)
                # Processing the call arguments (line 235)
                # Getting the type of 'self' (line 235)
                self_305720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'self', False)
                # Getting the type of 'option' (line 235)
                option_305721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'option', False)
                
                # Call to split(...): (line 235)
                # Processing the call arguments (line 235)
                str_305724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 43), 'str', ',\\s*|\\s+')
                # Getting the type of 'val' (line 235)
                val_305725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 56), 'val', False)
                # Processing the call keyword arguments (line 235)
                kwargs_305726 = {}
                # Getting the type of 're' (line 235)
                re_305722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 're', False)
                # Obtaining the member 'split' of a type (line 235)
                split_305723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 34), re_305722, 'split')
                # Calling split(args, kwargs) (line 235)
                split_call_result_305727 = invoke(stypy.reporting.localization.Localization(__file__, 235, 34), split_305723, *[str_305724, val_305725], **kwargs_305726)
                
                # Processing the call keyword arguments (line 235)
                kwargs_305728 = {}
                # Getting the type of 'setattr' (line 235)
                setattr_305719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'setattr', False)
                # Calling setattr(args, kwargs) (line 235)
                setattr_call_result_305729 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), setattr_305719, *[self_305720, option_305721, split_call_result_305727], **kwargs_305728)
                

                if more_types_in_union_305718:
                    # Runtime conditional SSA for else branch (line 234)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_305717) or more_types_in_union_305718):
                # Assigning a type to the variable 'val' (line 234)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'val', remove_subtype_from_union(val_305716, str))
                
                # Type idiom detected: calculating its left and rigth part (line 237)
                # Getting the type of 'list' (line 237)
                list_305730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 31), 'list')
                # Getting the type of 'val' (line 237)
                val_305731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'val')
                
                (may_be_305732, more_types_in_union_305733) = may_be_subtype(list_305730, val_305731)

                if may_be_305732:

                    if more_types_in_union_305733:
                        # Runtime conditional SSA (line 237)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'val' (line 237)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'val', remove_not_subtype_from_union(val_305731, list))
                    
                    # Assigning a Num to a Name (line 239):
                    int_305734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 21), 'int')
                    # Assigning a type to the variable 'ok' (line 239)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'ok', int_305734)
                    
                    # Getting the type of 'val' (line 240)
                    val_305735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'val')
                    # Testing the type of a for loop iterable (line 240)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 16), val_305735)
                    # Getting the type of the for loop variable (line 240)
                    for_loop_var_305736 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 16), val_305735)
                    # Assigning a type to the variable 'element' (line 240)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'element', for_loop_var_305736)
                    # SSA begins for a for statement (line 240)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Type idiom detected: calculating its left and rigth part (line 241)
                    # Getting the type of 'str' (line 241)
                    str_305737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 47), 'str')
                    # Getting the type of 'element' (line 241)
                    element_305738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 38), 'element')
                    
                    (may_be_305739, more_types_in_union_305740) = may_not_be_subtype(str_305737, element_305738)

                    if may_be_305739:

                        if more_types_in_union_305740:
                            # Runtime conditional SSA (line 241)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'element' (line 241)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'element', remove_subtype_from_union(element_305738, str))
                        
                        # Assigning a Num to a Name (line 242):
                        int_305741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 29), 'int')
                        # Assigning a type to the variable 'ok' (line 242)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'ok', int_305741)

                        if more_types_in_union_305740:
                            # SSA join for if statement (line 241)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()
                    

                    if more_types_in_union_305733:
                        # Runtime conditional SSA for else branch (line 237)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_305732) or more_types_in_union_305733):
                    # Assigning a type to the variable 'val' (line 237)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'val', remove_subtype_from_union(val_305731, list))
                    
                    # Assigning a Num to a Name (line 245):
                    int_305742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 21), 'int')
                    # Assigning a type to the variable 'ok' (line 245)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'ok', int_305742)

                    if (may_be_305732 and more_types_in_union_305733):
                        # SSA join for if statement (line 237)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                
                # Getting the type of 'ok' (line 247)
                ok_305743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 19), 'ok')
                # Applying the 'not' unary operator (line 247)
                result_not__305744 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), 'not', ok_305743)
                
                # Testing the type of an if condition (line 247)
                if_condition_305745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 12), result_not__305744)
                # Assigning a type to the variable 'if_condition_305745' (line 247)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'if_condition_305745', if_condition_305745)
                # SSA begins for if statement (line 247)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'DistutilsOptionError' (line 248)
                DistutilsOptionError_305746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'DistutilsOptionError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 248, 16), DistutilsOptionError_305746, 'raise parameter', BaseException)
                # SSA join for if statement (line 247)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_305717 and more_types_in_union_305718):
                    # SSA join for if statement (line 234)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_305713 and more_types_in_union_305714):
                # SSA join for if statement (line 232)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'ensure_string_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ensure_string_list' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_305747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ensure_string_list'
        return stypy_return_type_305747


    @norecursion
    def _ensure_tested_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 254)
        None_305748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 55), 'None')
        defaults = [None_305748]
        # Create a new context for function '_ensure_tested_string'
        module_type_store = module_type_store.open_function_context('_ensure_tested_string', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command._ensure_tested_string.__dict__.__setitem__('stypy_localization', localization)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_function_name', 'Command._ensure_tested_string')
        Command._ensure_tested_string.__dict__.__setitem__('stypy_param_names_list', ['option', 'tester', 'what', 'error_fmt', 'default'])
        Command._ensure_tested_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command._ensure_tested_string.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command._ensure_tested_string', ['option', 'tester', 'what', 'error_fmt', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ensure_tested_string', localization, ['option', 'tester', 'what', 'error_fmt', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ensure_tested_string(...)' code ##################

        
        # Assigning a Call to a Name (line 255):
        
        # Call to _ensure_stringlike(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'option' (line 255)
        option_305751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 38), 'option', False)
        # Getting the type of 'what' (line 255)
        what_305752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 46), 'what', False)
        # Getting the type of 'default' (line 255)
        default_305753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 52), 'default', False)
        # Processing the call keyword arguments (line 255)
        kwargs_305754 = {}
        # Getting the type of 'self' (line 255)
        self_305749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'self', False)
        # Obtaining the member '_ensure_stringlike' of a type (line 255)
        _ensure_stringlike_305750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 14), self_305749, '_ensure_stringlike')
        # Calling _ensure_stringlike(args, kwargs) (line 255)
        _ensure_stringlike_call_result_305755 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), _ensure_stringlike_305750, *[option_305751, what_305752, default_305753], **kwargs_305754)
        
        # Assigning a type to the variable 'val' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'val', _ensure_stringlike_call_result_305755)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'val' (line 256)
        val_305756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'val')
        # Getting the type of 'None' (line 256)
        None_305757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'None')
        # Applying the binary operator 'isnot' (line 256)
        result_is_not_305758 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'isnot', val_305756, None_305757)
        
        
        
        # Call to tester(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'val' (line 256)
        val_305760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 42), 'val', False)
        # Processing the call keyword arguments (line 256)
        kwargs_305761 = {}
        # Getting the type of 'tester' (line 256)
        tester_305759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'tester', False)
        # Calling tester(args, kwargs) (line 256)
        tester_call_result_305762 = invoke(stypy.reporting.localization.Localization(__file__, 256, 35), tester_305759, *[val_305760], **kwargs_305761)
        
        # Applying the 'not' unary operator (line 256)
        result_not__305763 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 31), 'not', tester_call_result_305762)
        
        # Applying the binary operator 'and' (line 256)
        result_and_keyword_305764 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'and', result_is_not_305758, result_not__305763)
        
        # Testing the type of an if condition (line 256)
        if_condition_305765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_and_keyword_305764)
        # Assigning a type to the variable 'if_condition_305765' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_305765', if_condition_305765)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 257)
        DistutilsOptionError_305766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 257, 12), DistutilsOptionError_305766, 'raise parameter', BaseException)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_ensure_tested_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ensure_tested_string' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_305767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ensure_tested_string'
        return stypy_return_type_305767


    @norecursion
    def ensure_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ensure_filename'
        module_type_store = module_type_store.open_function_context('ensure_filename', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.ensure_filename.__dict__.__setitem__('stypy_localization', localization)
        Command.ensure_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.ensure_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.ensure_filename.__dict__.__setitem__('stypy_function_name', 'Command.ensure_filename')
        Command.ensure_filename.__dict__.__setitem__('stypy_param_names_list', ['option'])
        Command.ensure_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.ensure_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.ensure_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.ensure_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.ensure_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.ensure_filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.ensure_filename', ['option'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ensure_filename', localization, ['option'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ensure_filename(...)' code ##################

        str_305768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'str', "Ensure that 'option' is the name of an existing file.")
        
        # Call to _ensure_tested_string(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'option' (line 262)
        option_305771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 35), 'option', False)
        # Getting the type of 'os' (line 262)
        os_305772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 43), 'os', False)
        # Obtaining the member 'path' of a type (line 262)
        path_305773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 43), os_305772, 'path')
        # Obtaining the member 'isfile' of a type (line 262)
        isfile_305774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 43), path_305773, 'isfile')
        str_305775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 35), 'str', 'filename')
        str_305776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 35), 'str', "'%s' does not exist or is not a file")
        # Processing the call keyword arguments (line 262)
        kwargs_305777 = {}
        # Getting the type of 'self' (line 262)
        self_305769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member '_ensure_tested_string' of a type (line 262)
        _ensure_tested_string_305770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_305769, '_ensure_tested_string')
        # Calling _ensure_tested_string(args, kwargs) (line 262)
        _ensure_tested_string_call_result_305778 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), _ensure_tested_string_305770, *[option_305771, isfile_305774, str_305775, str_305776], **kwargs_305777)
        
        
        # ################# End of 'ensure_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ensure_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_305779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ensure_filename'
        return stypy_return_type_305779


    @norecursion
    def ensure_dirname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ensure_dirname'
        module_type_store = module_type_store.open_function_context('ensure_dirname', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.ensure_dirname.__dict__.__setitem__('stypy_localization', localization)
        Command.ensure_dirname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.ensure_dirname.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.ensure_dirname.__dict__.__setitem__('stypy_function_name', 'Command.ensure_dirname')
        Command.ensure_dirname.__dict__.__setitem__('stypy_param_names_list', ['option'])
        Command.ensure_dirname.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.ensure_dirname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.ensure_dirname.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.ensure_dirname.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.ensure_dirname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.ensure_dirname.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.ensure_dirname', ['option'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ensure_dirname', localization, ['option'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ensure_dirname(...)' code ##################

        
        # Call to _ensure_tested_string(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'option' (line 267)
        option_305782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 35), 'option', False)
        # Getting the type of 'os' (line 267)
        os_305783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 43), 'os', False)
        # Obtaining the member 'path' of a type (line 267)
        path_305784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 43), os_305783, 'path')
        # Obtaining the member 'isdir' of a type (line 267)
        isdir_305785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 43), path_305784, 'isdir')
        str_305786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 35), 'str', 'directory name')
        str_305787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 35), 'str', "'%s' does not exist or is not a directory")
        # Processing the call keyword arguments (line 267)
        kwargs_305788 = {}
        # Getting the type of 'self' (line 267)
        self_305780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self', False)
        # Obtaining the member '_ensure_tested_string' of a type (line 267)
        _ensure_tested_string_305781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_305780, '_ensure_tested_string')
        # Calling _ensure_tested_string(args, kwargs) (line 267)
        _ensure_tested_string_call_result_305789 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), _ensure_tested_string_305781, *[option_305782, isdir_305785, str_305786, str_305787], **kwargs_305788)
        
        
        # ################# End of 'ensure_dirname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ensure_dirname' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_305790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ensure_dirname'
        return stypy_return_type_305790


    @norecursion
    def get_command_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_command_name'
        module_type_store = module_type_store.open_function_context('get_command_name', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.get_command_name.__dict__.__setitem__('stypy_localization', localization)
        Command.get_command_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.get_command_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.get_command_name.__dict__.__setitem__('stypy_function_name', 'Command.get_command_name')
        Command.get_command_name.__dict__.__setitem__('stypy_param_names_list', [])
        Command.get_command_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.get_command_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.get_command_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.get_command_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.get_command_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.get_command_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.get_command_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_command_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_command_name(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 275)
        str_305791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 25), 'str', 'command_name')
        # Getting the type of 'self' (line 275)
        self_305792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'self')
        
        (may_be_305793, more_types_in_union_305794) = may_provide_member(str_305791, self_305792)

        if may_be_305793:

            if more_types_in_union_305794:
                # Runtime conditional SSA (line 275)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self', remove_not_member_provider_from_union(self_305792, 'command_name'))
            # Getting the type of 'self' (line 276)
            self_305795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'self')
            # Obtaining the member 'command_name' of a type (line 276)
            command_name_305796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 19), self_305795, 'command_name')
            # Assigning a type to the variable 'stypy_return_type' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'stypy_return_type', command_name_305796)

            if more_types_in_union_305794:
                # Runtime conditional SSA for else branch (line 275)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_305793) or more_types_in_union_305794):
            # Assigning a type to the variable 'self' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self', remove_member_provider_from_union(self_305792, 'command_name'))
            # Getting the type of 'self' (line 278)
            self_305797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'self')
            # Obtaining the member '__class__' of a type (line 278)
            class___305798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), self_305797, '__class__')
            # Obtaining the member '__name__' of a type (line 278)
            name___305799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), class___305798, '__name__')
            # Assigning a type to the variable 'stypy_return_type' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'stypy_return_type', name___305799)

            if (may_be_305793 and more_types_in_union_305794):
                # SSA join for if statement (line 275)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_command_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_command_name' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_305800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305800)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_command_name'
        return stypy_return_type_305800


    @norecursion
    def set_undefined_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_undefined_options'
        module_type_store = module_type_store.open_function_context('set_undefined_options', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.set_undefined_options.__dict__.__setitem__('stypy_localization', localization)
        Command.set_undefined_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.set_undefined_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.set_undefined_options.__dict__.__setitem__('stypy_function_name', 'Command.set_undefined_options')
        Command.set_undefined_options.__dict__.__setitem__('stypy_param_names_list', ['src_cmd'])
        Command.set_undefined_options.__dict__.__setitem__('stypy_varargs_param_name', 'option_pairs')
        Command.set_undefined_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.set_undefined_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.set_undefined_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.set_undefined_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.set_undefined_options.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.set_undefined_options', ['src_cmd'], 'option_pairs', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_undefined_options', localization, ['src_cmd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_undefined_options(...)' code ##################

        str_305801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, (-1)), 'str', 'Set the values of any "undefined" options from corresponding\n        option values in some other command object.  "Undefined" here means\n        "is None", which is the convention used to indicate that an option\n        has not been changed between \'initialize_options()\' and\n        \'finalize_options()\'.  Usually called from \'finalize_options()\' for\n        options that depend on some other command rather than another\n        option of the same command.  \'src_cmd\' is the other command from\n        which option values will be taken (a command object will be created\n        for it if necessary); the remaining arguments are\n        \'(src_option,dst_option)\' tuples which mean "take the value of\n        \'src_option\' in the \'src_cmd\' command object, and copy it to\n        \'dst_option\' in the current command object".\n        ')
        
        # Assigning a Call to a Name (line 297):
        
        # Call to get_command_obj(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'src_cmd' (line 297)
        src_cmd_305805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 56), 'src_cmd', False)
        # Processing the call keyword arguments (line 297)
        kwargs_305806 = {}
        # Getting the type of 'self' (line 297)
        self_305802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'self', False)
        # Obtaining the member 'distribution' of a type (line 297)
        distribution_305803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 22), self_305802, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 297)
        get_command_obj_305804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 22), distribution_305803, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 297)
        get_command_obj_call_result_305807 = invoke(stypy.reporting.localization.Localization(__file__, 297, 22), get_command_obj_305804, *[src_cmd_305805], **kwargs_305806)
        
        # Assigning a type to the variable 'src_cmd_obj' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'src_cmd_obj', get_command_obj_call_result_305807)
        
        # Call to ensure_finalized(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_305810 = {}
        # Getting the type of 'src_cmd_obj' (line 298)
        src_cmd_obj_305808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'src_cmd_obj', False)
        # Obtaining the member 'ensure_finalized' of a type (line 298)
        ensure_finalized_305809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), src_cmd_obj_305808, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 298)
        ensure_finalized_call_result_305811 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), ensure_finalized_305809, *[], **kwargs_305810)
        
        
        # Getting the type of 'option_pairs' (line 299)
        option_pairs_305812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 'option_pairs')
        # Testing the type of a for loop iterable (line 299)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 8), option_pairs_305812)
        # Getting the type of the for loop variable (line 299)
        for_loop_var_305813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 8), option_pairs_305812)
        # Assigning a type to the variable 'src_option' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'src_option', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), for_loop_var_305813))
        # Assigning a type to the variable 'dst_option' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'dst_option', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), for_loop_var_305813))
        # SSA begins for a for statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 300)
        
        # Call to getattr(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'self' (line 300)
        self_305815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'self', False)
        # Getting the type of 'dst_option' (line 300)
        dst_option_305816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 29), 'dst_option', False)
        # Processing the call keyword arguments (line 300)
        kwargs_305817 = {}
        # Getting the type of 'getattr' (line 300)
        getattr_305814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 300)
        getattr_call_result_305818 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), getattr_305814, *[self_305815, dst_option_305816], **kwargs_305817)
        
        # Getting the type of 'None' (line 300)
        None_305819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 44), 'None')
        
        (may_be_305820, more_types_in_union_305821) = may_be_none(getattr_call_result_305818, None_305819)

        if may_be_305820:

            if more_types_in_union_305821:
                # Runtime conditional SSA (line 300)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 'self' (line 301)
            self_305823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'self', False)
            # Getting the type of 'dst_option' (line 301)
            dst_option_305824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'dst_option', False)
            
            # Call to getattr(...): (line 302)
            # Processing the call arguments (line 302)
            # Getting the type of 'src_cmd_obj' (line 302)
            src_cmd_obj_305826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 32), 'src_cmd_obj', False)
            # Getting the type of 'src_option' (line 302)
            src_option_305827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 45), 'src_option', False)
            # Processing the call keyword arguments (line 302)
            kwargs_305828 = {}
            # Getting the type of 'getattr' (line 302)
            getattr_305825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'getattr', False)
            # Calling getattr(args, kwargs) (line 302)
            getattr_call_result_305829 = invoke(stypy.reporting.localization.Localization(__file__, 302, 24), getattr_305825, *[src_cmd_obj_305826, src_option_305827], **kwargs_305828)
            
            # Processing the call keyword arguments (line 301)
            kwargs_305830 = {}
            # Getting the type of 'setattr' (line 301)
            setattr_305822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 301)
            setattr_call_result_305831 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), setattr_305822, *[self_305823, dst_option_305824, getattr_call_result_305829], **kwargs_305830)
            

            if more_types_in_union_305821:
                # SSA join for if statement (line 300)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_undefined_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_undefined_options' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_305832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_undefined_options'
        return stypy_return_type_305832


    @norecursion
    def get_finalized_command(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 52), 'int')
        defaults = [int_305833]
        # Create a new context for function 'get_finalized_command'
        module_type_store = module_type_store.open_function_context('get_finalized_command', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.get_finalized_command.__dict__.__setitem__('stypy_localization', localization)
        Command.get_finalized_command.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.get_finalized_command.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.get_finalized_command.__dict__.__setitem__('stypy_function_name', 'Command.get_finalized_command')
        Command.get_finalized_command.__dict__.__setitem__('stypy_param_names_list', ['command', 'create'])
        Command.get_finalized_command.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.get_finalized_command.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.get_finalized_command.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.get_finalized_command.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.get_finalized_command.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.get_finalized_command.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.get_finalized_command', ['command', 'create'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_finalized_command', localization, ['command', 'create'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_finalized_command(...)' code ##################

        str_305834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, (-1)), 'str', "Wrapper around Distribution's 'get_command_obj()' method: find\n        (create if necessary and 'create' is true) the command object for\n        'command', call its 'ensure_finalized()' method, and return the\n        finalized command object.\n        ")
        
        # Assigning a Call to a Name (line 311):
        
        # Call to get_command_obj(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'command' (line 311)
        command_305838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 52), 'command', False)
        # Getting the type of 'create' (line 311)
        create_305839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 61), 'create', False)
        # Processing the call keyword arguments (line 311)
        kwargs_305840 = {}
        # Getting the type of 'self' (line 311)
        self_305835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'self', False)
        # Obtaining the member 'distribution' of a type (line 311)
        distribution_305836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 18), self_305835, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 311)
        get_command_obj_305837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 18), distribution_305836, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 311)
        get_command_obj_call_result_305841 = invoke(stypy.reporting.localization.Localization(__file__, 311, 18), get_command_obj_305837, *[command_305838, create_305839], **kwargs_305840)
        
        # Assigning a type to the variable 'cmd_obj' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'cmd_obj', get_command_obj_call_result_305841)
        
        # Call to ensure_finalized(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_305844 = {}
        # Getting the type of 'cmd_obj' (line 312)
        cmd_obj_305842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'cmd_obj', False)
        # Obtaining the member 'ensure_finalized' of a type (line 312)
        ensure_finalized_305843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), cmd_obj_305842, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 312)
        ensure_finalized_call_result_305845 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), ensure_finalized_305843, *[], **kwargs_305844)
        
        # Getting the type of 'cmd_obj' (line 313)
        cmd_obj_305846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'cmd_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'stypy_return_type', cmd_obj_305846)
        
        # ################# End of 'get_finalized_command(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_finalized_command' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_305847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_finalized_command'
        return stypy_return_type_305847


    @norecursion
    def reinitialize_command(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 63), 'int')
        defaults = [int_305848]
        # Create a new context for function 'reinitialize_command'
        module_type_store = module_type_store.open_function_context('reinitialize_command', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.reinitialize_command.__dict__.__setitem__('stypy_localization', localization)
        Command.reinitialize_command.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.reinitialize_command.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.reinitialize_command.__dict__.__setitem__('stypy_function_name', 'Command.reinitialize_command')
        Command.reinitialize_command.__dict__.__setitem__('stypy_param_names_list', ['command', 'reinit_subcommands'])
        Command.reinitialize_command.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.reinitialize_command.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.reinitialize_command.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.reinitialize_command.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.reinitialize_command.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.reinitialize_command.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.reinitialize_command', ['command', 'reinit_subcommands'], None, None, defaults, varargs, kwargs)

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

        
        # Call to reinitialize_command(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'command' (line 319)
        command_305852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'command', False)
        # Getting the type of 'reinit_subcommands' (line 319)
        reinit_subcommands_305853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'reinit_subcommands', False)
        # Processing the call keyword arguments (line 318)
        kwargs_305854 = {}
        # Getting the type of 'self' (line 318)
        self_305849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 318)
        distribution_305850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), self_305849, 'distribution')
        # Obtaining the member 'reinitialize_command' of a type (line 318)
        reinitialize_command_305851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), distribution_305850, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 318)
        reinitialize_command_call_result_305855 = invoke(stypy.reporting.localization.Localization(__file__, 318, 15), reinitialize_command_305851, *[command_305852, reinit_subcommands_305853], **kwargs_305854)
        
        # Assigning a type to the variable 'stypy_return_type' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'stypy_return_type', reinitialize_command_call_result_305855)
        
        # ################# End of 'reinitialize_command(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reinitialize_command' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_305856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reinitialize_command'
        return stypy_return_type_305856


    @norecursion
    def run_command(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run_command'
        module_type_store = module_type_store.open_function_context('run_command', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.run_command.__dict__.__setitem__('stypy_localization', localization)
        Command.run_command.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.run_command.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.run_command.__dict__.__setitem__('stypy_function_name', 'Command.run_command')
        Command.run_command.__dict__.__setitem__('stypy_param_names_list', ['command'])
        Command.run_command.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.run_command.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.run_command.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.run_command.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.run_command.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.run_command.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.run_command', ['command'], None, None, defaults, varargs, kwargs)

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

        str_305857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, (-1)), 'str', "Run some other command: uses the 'run_command()' method of\n        Distribution, which creates and finalizes the command object if\n        necessary and then invokes its 'run()' method.\n        ")
        
        # Call to run_command(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'command' (line 326)
        command_305861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'command', False)
        # Processing the call keyword arguments (line 326)
        kwargs_305862 = {}
        # Getting the type of 'self' (line 326)
        self_305858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self', False)
        # Obtaining the member 'distribution' of a type (line 326)
        distribution_305859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_305858, 'distribution')
        # Obtaining the member 'run_command' of a type (line 326)
        run_command_305860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), distribution_305859, 'run_command')
        # Calling run_command(args, kwargs) (line 326)
        run_command_call_result_305863 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), run_command_305860, *[command_305861], **kwargs_305862)
        
        
        # ################# End of 'run_command(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run_command' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_305864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run_command'
        return stypy_return_type_305864


    @norecursion
    def get_sub_commands(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_sub_commands'
        module_type_store = module_type_store.open_function_context('get_sub_commands', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.get_sub_commands.__dict__.__setitem__('stypy_localization', localization)
        Command.get_sub_commands.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.get_sub_commands.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.get_sub_commands.__dict__.__setitem__('stypy_function_name', 'Command.get_sub_commands')
        Command.get_sub_commands.__dict__.__setitem__('stypy_param_names_list', [])
        Command.get_sub_commands.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.get_sub_commands.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.get_sub_commands.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.get_sub_commands.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.get_sub_commands.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.get_sub_commands.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.get_sub_commands', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_sub_commands', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_sub_commands(...)' code ##################

        str_305865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, (-1)), 'str', "Determine the sub-commands that are relevant in the current\n        distribution (ie., that need to be run).  This is based on the\n        'sub_commands' class attribute: each tuple in that list may include\n        a method that we call to determine if the subcommand needs to be\n        run for the current distribution.  Return a list of command names.\n        ")
        
        # Assigning a List to a Name (line 335):
        
        # Obtaining an instance of the builtin type 'list' (line 335)
        list_305866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 335)
        
        # Assigning a type to the variable 'commands' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'commands', list_305866)
        
        # Getting the type of 'self' (line 336)
        self_305867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 34), 'self')
        # Obtaining the member 'sub_commands' of a type (line 336)
        sub_commands_305868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 34), self_305867, 'sub_commands')
        # Testing the type of a for loop iterable (line 336)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 336, 8), sub_commands_305868)
        # Getting the type of the for loop variable (line 336)
        for_loop_var_305869 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 336, 8), sub_commands_305868)
        # Assigning a type to the variable 'cmd_name' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'cmd_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), for_loop_var_305869))
        # Assigning a type to the variable 'method' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), for_loop_var_305869))
        # SSA begins for a for statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'method' (line 337)
        method_305870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'method')
        # Getting the type of 'None' (line 337)
        None_305871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 25), 'None')
        # Applying the binary operator 'is' (line 337)
        result_is__305872 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 15), 'is', method_305870, None_305871)
        
        
        # Call to method(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'self' (line 337)
        self_305874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 40), 'self', False)
        # Processing the call keyword arguments (line 337)
        kwargs_305875 = {}
        # Getting the type of 'method' (line 337)
        method_305873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 33), 'method', False)
        # Calling method(args, kwargs) (line 337)
        method_call_result_305876 = invoke(stypy.reporting.localization.Localization(__file__, 337, 33), method_305873, *[self_305874], **kwargs_305875)
        
        # Applying the binary operator 'or' (line 337)
        result_or_keyword_305877 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 15), 'or', result_is__305872, method_call_result_305876)
        
        # Testing the type of an if condition (line 337)
        if_condition_305878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 12), result_or_keyword_305877)
        # Assigning a type to the variable 'if_condition_305878' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'if_condition_305878', if_condition_305878)
        # SSA begins for if statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'cmd_name' (line 338)
        cmd_name_305881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'cmd_name', False)
        # Processing the call keyword arguments (line 338)
        kwargs_305882 = {}
        # Getting the type of 'commands' (line 338)
        commands_305879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'commands', False)
        # Obtaining the member 'append' of a type (line 338)
        append_305880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 16), commands_305879, 'append')
        # Calling append(args, kwargs) (line 338)
        append_call_result_305883 = invoke(stypy.reporting.localization.Localization(__file__, 338, 16), append_305880, *[cmd_name_305881], **kwargs_305882)
        
        # SSA join for if statement (line 337)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'commands' (line 339)
        commands_305884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'commands')
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', commands_305884)
        
        # ################# End of 'get_sub_commands(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_sub_commands' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_305885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_sub_commands'
        return stypy_return_type_305885


    @norecursion
    def warn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'warn'
        module_type_store = module_type_store.open_function_context('warn', 344, 4, False)
        # Assigning a type to the variable 'self' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.warn.__dict__.__setitem__('stypy_localization', localization)
        Command.warn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.warn.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.warn.__dict__.__setitem__('stypy_function_name', 'Command.warn')
        Command.warn.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Command.warn.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.warn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.warn.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.warn.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.warn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.warn.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.warn', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'warn', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'warn(...)' code ##################

        
        # Call to warn(...): (line 345)
        # Processing the call arguments (line 345)
        str_305888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 17), 'str', 'warning: %s: %s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 346)
        tuple_305889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 346)
        # Adding element type (line 346)
        
        # Call to get_command_name(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_305892 = {}
        # Getting the type of 'self' (line 346)
        self_305890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 17), 'self', False)
        # Obtaining the member 'get_command_name' of a type (line 346)
        get_command_name_305891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 17), self_305890, 'get_command_name')
        # Calling get_command_name(args, kwargs) (line 346)
        get_command_name_call_result_305893 = invoke(stypy.reporting.localization.Localization(__file__, 346, 17), get_command_name_305891, *[], **kwargs_305892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 17), tuple_305889, get_command_name_call_result_305893)
        # Adding element type (line 346)
        # Getting the type of 'msg' (line 346)
        msg_305894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 42), 'msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 17), tuple_305889, msg_305894)
        
        # Applying the binary operator '%' (line 345)
        result_mod_305895 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 17), '%', str_305888, tuple_305889)
        
        # Processing the call keyword arguments (line 345)
        kwargs_305896 = {}
        # Getting the type of 'log' (line 345)
        log_305886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'log', False)
        # Obtaining the member 'warn' of a type (line 345)
        warn_305887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), log_305886, 'warn')
        # Calling warn(args, kwargs) (line 345)
        warn_call_result_305897 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), warn_305887, *[result_mod_305895], **kwargs_305896)
        
        
        # ################# End of 'warn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'warn' in the type store
        # Getting the type of 'stypy_return_type' (line 344)
        stypy_return_type_305898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'warn'
        return stypy_return_type_305898


    @norecursion
    def execute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 348)
        None_305899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 38), 'None')
        int_305900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 50), 'int')
        defaults = [None_305899, int_305900]
        # Create a new context for function 'execute'
        module_type_store = module_type_store.open_function_context('execute', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.execute.__dict__.__setitem__('stypy_localization', localization)
        Command.execute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.execute.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.execute.__dict__.__setitem__('stypy_function_name', 'Command.execute')
        Command.execute.__dict__.__setitem__('stypy_param_names_list', ['func', 'args', 'msg', 'level'])
        Command.execute.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.execute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.execute.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.execute.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.execute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.execute.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.execute', ['func', 'args', 'msg', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'execute', localization, ['func', 'args', 'msg', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'execute(...)' code ##################

        
        # Call to execute(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'func' (line 349)
        func_305903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'func', False)
        # Getting the type of 'args' (line 349)
        args_305904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'args', False)
        # Getting the type of 'msg' (line 349)
        msg_305905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 33), 'msg', False)
        # Processing the call keyword arguments (line 349)
        # Getting the type of 'self' (line 349)
        self_305906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 46), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 349)
        dry_run_305907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 46), self_305906, 'dry_run')
        keyword_305908 = dry_run_305907
        kwargs_305909 = {'dry_run': keyword_305908}
        # Getting the type of 'util' (line 349)
        util_305901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'util', False)
        # Obtaining the member 'execute' of a type (line 349)
        execute_305902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), util_305901, 'execute')
        # Calling execute(args, kwargs) (line 349)
        execute_call_result_305910 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), execute_305902, *[func_305903, args_305904, msg_305905], **kwargs_305909)
        
        
        # ################# End of 'execute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'execute' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_305911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'execute'
        return stypy_return_type_305911


    @norecursion
    def mkpath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 32), 'int')
        defaults = [int_305912]
        # Create a new context for function 'mkpath'
        module_type_store = module_type_store.open_function_context('mkpath', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.mkpath.__dict__.__setitem__('stypy_localization', localization)
        Command.mkpath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.mkpath.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.mkpath.__dict__.__setitem__('stypy_function_name', 'Command.mkpath')
        Command.mkpath.__dict__.__setitem__('stypy_param_names_list', ['name', 'mode'])
        Command.mkpath.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.mkpath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.mkpath.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.mkpath.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.mkpath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.mkpath.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.mkpath', ['name', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mkpath', localization, ['name', 'mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mkpath(...)' code ##################

        
        # Call to mkpath(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'name' (line 352)
        name_305915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'name', False)
        # Getting the type of 'mode' (line 352)
        mode_305916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'mode', False)
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'self' (line 352)
        self_305917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 44), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 352)
        dry_run_305918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 44), self_305917, 'dry_run')
        keyword_305919 = dry_run_305918
        kwargs_305920 = {'dry_run': keyword_305919}
        # Getting the type of 'dir_util' (line 352)
        dir_util_305913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'dir_util', False)
        # Obtaining the member 'mkpath' of a type (line 352)
        mkpath_305914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), dir_util_305913, 'mkpath')
        # Calling mkpath(args, kwargs) (line 352)
        mkpath_call_result_305921 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), mkpath_305914, *[name_305915, mode_305916], **kwargs_305920)
        
        
        # ################# End of 'mkpath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mkpath' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_305922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305922)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mkpath'
        return stypy_return_type_305922


    @norecursion
    def copy_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 33), 'int')
        int_305924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 51), 'int')
        # Getting the type of 'None' (line 355)
        None_305925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 59), 'None')
        int_305926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 71), 'int')
        defaults = [int_305923, int_305924, None_305925, int_305926]
        # Create a new context for function 'copy_file'
        module_type_store = module_type_store.open_function_context('copy_file', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.copy_file.__dict__.__setitem__('stypy_localization', localization)
        Command.copy_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.copy_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.copy_file.__dict__.__setitem__('stypy_function_name', 'Command.copy_file')
        Command.copy_file.__dict__.__setitem__('stypy_param_names_list', ['infile', 'outfile', 'preserve_mode', 'preserve_times', 'link', 'level'])
        Command.copy_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.copy_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.copy_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.copy_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.copy_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.copy_file.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.copy_file', ['infile', 'outfile', 'preserve_mode', 'preserve_times', 'link', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_file', localization, ['infile', 'outfile', 'preserve_mode', 'preserve_times', 'link', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_file(...)' code ##################

        str_305927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, (-1)), 'str', "Copy a file respecting verbose, dry-run and force flags.  (The\n        former two default to whatever is in the Distribution object, and\n        the latter defaults to false for commands that don't define it.)")
        
        # Call to copy_file(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'infile' (line 361)
        infile_305930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'infile', False)
        # Getting the type of 'outfile' (line 361)
        outfile_305931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'outfile', False)
        # Getting the type of 'preserve_mode' (line 362)
        preserve_mode_305932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'preserve_mode', False)
        # Getting the type of 'preserve_times' (line 362)
        preserve_times_305933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 'preserve_times', False)
        
        # Getting the type of 'self' (line 363)
        self_305934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'self', False)
        # Obtaining the member 'force' of a type (line 363)
        force_305935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), self_305934, 'force')
        # Applying the 'not' unary operator (line 363)
        result_not__305936 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 12), 'not', force_305935)
        
        # Getting the type of 'link' (line 364)
        link_305937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'link', False)
        # Processing the call keyword arguments (line 360)
        # Getting the type of 'self' (line 365)
        self_305938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 20), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 365)
        dry_run_305939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 20), self_305938, 'dry_run')
        keyword_305940 = dry_run_305939
        kwargs_305941 = {'dry_run': keyword_305940}
        # Getting the type of 'file_util' (line 360)
        file_util_305928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'file_util', False)
        # Obtaining the member 'copy_file' of a type (line 360)
        copy_file_305929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 15), file_util_305928, 'copy_file')
        # Calling copy_file(args, kwargs) (line 360)
        copy_file_call_result_305942 = invoke(stypy.reporting.localization.Localization(__file__, 360, 15), copy_file_305929, *[infile_305930, outfile_305931, preserve_mode_305932, preserve_times_305933, result_not__305936, link_305937], **kwargs_305941)
        
        # Assigning a type to the variable 'stypy_return_type' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'stypy_return_type', copy_file_call_result_305942)
        
        # ################# End of 'copy_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_file' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_305943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305943)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_file'
        return stypy_return_type_305943


    @norecursion
    def copy_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 33), 'int')
        int_305945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 51), 'int')
        int_305946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 72), 'int')
        int_305947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 25), 'int')
        defaults = [int_305944, int_305945, int_305946, int_305947]
        # Create a new context for function 'copy_tree'
        module_type_store = module_type_store.open_function_context('copy_tree', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.copy_tree.__dict__.__setitem__('stypy_localization', localization)
        Command.copy_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.copy_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.copy_tree.__dict__.__setitem__('stypy_function_name', 'Command.copy_tree')
        Command.copy_tree.__dict__.__setitem__('stypy_param_names_list', ['infile', 'outfile', 'preserve_mode', 'preserve_times', 'preserve_symlinks', 'level'])
        Command.copy_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.copy_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.copy_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.copy_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.copy_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.copy_tree.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.copy_tree', ['infile', 'outfile', 'preserve_mode', 'preserve_times', 'preserve_symlinks', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_tree', localization, ['infile', 'outfile', 'preserve_mode', 'preserve_times', 'preserve_symlinks', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_tree(...)' code ##################

        str_305948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, (-1)), 'str', 'Copy an entire directory tree respecting verbose, dry-run,\n        and force flags.\n        ')
        
        # Call to copy_tree(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'infile' (line 374)
        infile_305951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'infile', False)
        # Getting the type of 'outfile' (line 374)
        outfile_305952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'outfile', False)
        # Getting the type of 'preserve_mode' (line 375)
        preserve_mode_305953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'preserve_mode', False)
        # Getting the type of 'preserve_times' (line 375)
        preserve_times_305954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 26), 'preserve_times', False)
        # Getting the type of 'preserve_symlinks' (line 375)
        preserve_symlinks_305955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 41), 'preserve_symlinks', False)
        
        # Getting the type of 'self' (line 376)
        self_305956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'self', False)
        # Obtaining the member 'force' of a type (line 376)
        force_305957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), self_305956, 'force')
        # Applying the 'not' unary operator (line 376)
        result_not__305958 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 12), 'not', force_305957)
        
        # Processing the call keyword arguments (line 373)
        # Getting the type of 'self' (line 377)
        self_305959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 20), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 377)
        dry_run_305960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 20), self_305959, 'dry_run')
        keyword_305961 = dry_run_305960
        kwargs_305962 = {'dry_run': keyword_305961}
        # Getting the type of 'dir_util' (line 373)
        dir_util_305949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'dir_util', False)
        # Obtaining the member 'copy_tree' of a type (line 373)
        copy_tree_305950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), dir_util_305949, 'copy_tree')
        # Calling copy_tree(args, kwargs) (line 373)
        copy_tree_call_result_305963 = invoke(stypy.reporting.localization.Localization(__file__, 373, 15), copy_tree_305950, *[infile_305951, outfile_305952, preserve_mode_305953, preserve_times_305954, preserve_symlinks_305955, result_not__305958], **kwargs_305962)
        
        # Assigning a type to the variable 'stypy_return_type' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type', copy_tree_call_result_305963)
        
        # ################# End of 'copy_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_305964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305964)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_tree'
        return stypy_return_type_305964


    @norecursion
    def move_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 41), 'int')
        defaults = [int_305965]
        # Create a new context for function 'move_file'
        module_type_store = module_type_store.open_function_context('move_file', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.move_file.__dict__.__setitem__('stypy_localization', localization)
        Command.move_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.move_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.move_file.__dict__.__setitem__('stypy_function_name', 'Command.move_file')
        Command.move_file.__dict__.__setitem__('stypy_param_names_list', ['src', 'dst', 'level'])
        Command.move_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.move_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.move_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.move_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.move_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.move_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.move_file', ['src', 'dst', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'move_file', localization, ['src', 'dst', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'move_file(...)' code ##################

        str_305966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 8), 'str', 'Move a file respecting dry-run flag.')
        
        # Call to move_file(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'src' (line 381)
        src_305969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 35), 'src', False)
        # Getting the type of 'dst' (line 381)
        dst_305970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 40), 'dst', False)
        # Processing the call keyword arguments (line 381)
        # Getting the type of 'self' (line 381)
        self_305971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 55), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 381)
        dry_run_305972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 55), self_305971, 'dry_run')
        keyword_305973 = dry_run_305972
        kwargs_305974 = {'dry_run': keyword_305973}
        # Getting the type of 'file_util' (line 381)
        file_util_305967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'file_util', False)
        # Obtaining the member 'move_file' of a type (line 381)
        move_file_305968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 15), file_util_305967, 'move_file')
        # Calling move_file(args, kwargs) (line 381)
        move_file_call_result_305975 = invoke(stypy.reporting.localization.Localization(__file__, 381, 15), move_file_305968, *[src_305969, dst_305970], **kwargs_305974)
        
        # Assigning a type to the variable 'stypy_return_type' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'stypy_return_type', move_file_call_result_305975)
        
        # ################# End of 'move_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move_file' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_305976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move_file'
        return stypy_return_type_305976


    @norecursion
    def spawn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 38), 'int')
        int_305978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 47), 'int')
        defaults = [int_305977, int_305978]
        # Create a new context for function 'spawn'
        module_type_store = module_type_store.open_function_context('spawn', 383, 4, False)
        # Assigning a type to the variable 'self' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.spawn.__dict__.__setitem__('stypy_localization', localization)
        Command.spawn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.spawn.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.spawn.__dict__.__setitem__('stypy_function_name', 'Command.spawn')
        Command.spawn.__dict__.__setitem__('stypy_param_names_list', ['cmd', 'search_path', 'level'])
        Command.spawn.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.spawn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.spawn.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.spawn.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.spawn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.spawn.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.spawn', ['cmd', 'search_path', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'spawn', localization, ['cmd', 'search_path', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'spawn(...)' code ##################

        str_305979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 8), 'str', 'Spawn an external command respecting dry-run flag.')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 385, 8))
        
        # 'from distutils.spawn import spawn' statement (line 385)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_305980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 385, 8), 'distutils.spawn')

        if (type(import_305980) is not StypyTypeError):

            if (import_305980 != 'pyd_module'):
                __import__(import_305980)
                sys_modules_305981 = sys.modules[import_305980]
                import_from_module(stypy.reporting.localization.Localization(__file__, 385, 8), 'distutils.spawn', sys_modules_305981.module_type_store, module_type_store, ['spawn'])
                nest_module(stypy.reporting.localization.Localization(__file__, 385, 8), __file__, sys_modules_305981, sys_modules_305981.module_type_store, module_type_store)
            else:
                from distutils.spawn import spawn

                import_from_module(stypy.reporting.localization.Localization(__file__, 385, 8), 'distutils.spawn', None, module_type_store, ['spawn'], [spawn])

        else:
            # Assigning a type to the variable 'distutils.spawn' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'distutils.spawn', import_305980)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Call to spawn(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'cmd' (line 386)
        cmd_305983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 14), 'cmd', False)
        # Getting the type of 'search_path' (line 386)
        search_path_305984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'search_path', False)
        # Processing the call keyword arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_305985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 41), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 386)
        dry_run_305986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 41), self_305985, 'dry_run')
        keyword_305987 = dry_run_305986
        kwargs_305988 = {'dry_run': keyword_305987}
        # Getting the type of 'spawn' (line 386)
        spawn_305982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'spawn', False)
        # Calling spawn(args, kwargs) (line 386)
        spawn_call_result_305989 = invoke(stypy.reporting.localization.Localization(__file__, 386, 8), spawn_305982, *[cmd_305983, search_path_305984], **kwargs_305988)
        
        
        # ################# End of 'spawn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'spawn' in the type store
        # Getting the type of 'stypy_return_type' (line 383)
        stypy_return_type_305990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305990)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'spawn'
        return stypy_return_type_305990


    @norecursion
    def make_archive(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 388)
        None_305991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 55), 'None')
        # Getting the type of 'None' (line 388)
        None_305992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 70), 'None')
        # Getting the type of 'None' (line 389)
        None_305993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), 'None')
        # Getting the type of 'None' (line 389)
        None_305994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 39), 'None')
        defaults = [None_305991, None_305992, None_305993, None_305994]
        # Create a new context for function 'make_archive'
        module_type_store = module_type_store.open_function_context('make_archive', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.make_archive.__dict__.__setitem__('stypy_localization', localization)
        Command.make_archive.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.make_archive.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.make_archive.__dict__.__setitem__('stypy_function_name', 'Command.make_archive')
        Command.make_archive.__dict__.__setitem__('stypy_param_names_list', ['base_name', 'format', 'root_dir', 'base_dir', 'owner', 'group'])
        Command.make_archive.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.make_archive.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.make_archive.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.make_archive.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.make_archive.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.make_archive.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.make_archive', ['base_name', 'format', 'root_dir', 'base_dir', 'owner', 'group'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_archive', localization, ['base_name', 'format', 'root_dir', 'base_dir', 'owner', 'group'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_archive(...)' code ##################

        
        # Call to make_archive(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'base_name' (line 390)
        base_name_305997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 41), 'base_name', False)
        # Getting the type of 'format' (line 390)
        format_305998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 52), 'format', False)
        # Getting the type of 'root_dir' (line 390)
        root_dir_305999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 60), 'root_dir', False)
        # Getting the type of 'base_dir' (line 391)
        base_dir_306000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 41), 'base_dir', False)
        # Processing the call keyword arguments (line 390)
        # Getting the type of 'self' (line 391)
        self_306001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 59), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 391)
        dry_run_306002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 59), self_306001, 'dry_run')
        keyword_306003 = dry_run_306002
        # Getting the type of 'owner' (line 392)
        owner_306004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 47), 'owner', False)
        keyword_306005 = owner_306004
        # Getting the type of 'group' (line 392)
        group_306006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 60), 'group', False)
        keyword_306007 = group_306006
        kwargs_306008 = {'owner': keyword_306005, 'group': keyword_306007, 'dry_run': keyword_306003}
        # Getting the type of 'archive_util' (line 390)
        archive_util_305995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'archive_util', False)
        # Obtaining the member 'make_archive' of a type (line 390)
        make_archive_305996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 15), archive_util_305995, 'make_archive')
        # Calling make_archive(args, kwargs) (line 390)
        make_archive_call_result_306009 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), make_archive_305996, *[base_name_305997, format_305998, root_dir_305999, base_dir_306000], **kwargs_306008)
        
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'stypy_return_type', make_archive_call_result_306009)
        
        # ################# End of 'make_archive(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_archive' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_306010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_archive'
        return stypy_return_type_306010


    @norecursion
    def make_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 395)
        None_306011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'None')
        # Getting the type of 'None' (line 395)
        None_306012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 42), 'None')
        int_306013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 54), 'int')
        defaults = [None_306011, None_306012, int_306013]
        # Create a new context for function 'make_file'
        module_type_store = module_type_store.open_function_context('make_file', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Command.make_file.__dict__.__setitem__('stypy_localization', localization)
        Command.make_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Command.make_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        Command.make_file.__dict__.__setitem__('stypy_function_name', 'Command.make_file')
        Command.make_file.__dict__.__setitem__('stypy_param_names_list', ['infiles', 'outfile', 'func', 'args', 'exec_msg', 'skip_msg', 'level'])
        Command.make_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        Command.make_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Command.make_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        Command.make_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        Command.make_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Command.make_file.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Command.make_file', ['infiles', 'outfile', 'func', 'args', 'exec_msg', 'skip_msg', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_file', localization, ['infiles', 'outfile', 'func', 'args', 'exec_msg', 'skip_msg', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_file(...)' code ##################

        str_306014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, (-1)), 'str', "Special case of 'execute()' for operations that process one or\n        more input files and generate one output file.  Works just like\n        'execute()', except the operation is skipped and a different\n        message printed if 'outfile' already exists and is newer than all\n        files listed in 'infiles'.  If the command defined 'self.force',\n        and it is true, then the command is unconditionally run -- does no\n        timestamp checks.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 404)
        # Getting the type of 'skip_msg' (line 404)
        skip_msg_306015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 11), 'skip_msg')
        # Getting the type of 'None' (line 404)
        None_306016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 23), 'None')
        
        (may_be_306017, more_types_in_union_306018) = may_be_none(skip_msg_306015, None_306016)

        if may_be_306017:

            if more_types_in_union_306018:
                # Runtime conditional SSA (line 404)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 405):
            str_306019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 23), 'str', 'skipping %s (inputs unchanged)')
            # Getting the type of 'outfile' (line 405)
            outfile_306020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 58), 'outfile')
            # Applying the binary operator '%' (line 405)
            result_mod_306021 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 23), '%', str_306019, outfile_306020)
            
            # Assigning a type to the variable 'skip_msg' (line 405)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'skip_msg', result_mod_306021)

            if more_types_in_union_306018:
                # SSA join for if statement (line 404)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 408)
        # Getting the type of 'str' (line 408)
        str_306022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 31), 'str')
        # Getting the type of 'infiles' (line 408)
        infiles_306023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 22), 'infiles')
        
        (may_be_306024, more_types_in_union_306025) = may_be_subtype(str_306022, infiles_306023)

        if may_be_306024:

            if more_types_in_union_306025:
                # Runtime conditional SSA (line 408)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'infiles' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'infiles', remove_not_subtype_from_union(infiles_306023, str))
            
            # Assigning a Tuple to a Name (line 409):
            
            # Obtaining an instance of the builtin type 'tuple' (line 409)
            tuple_306026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 409)
            # Adding element type (line 409)
            # Getting the type of 'infiles' (line 409)
            infiles_306027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), 'infiles')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 23), tuple_306026, infiles_306027)
            
            # Assigning a type to the variable 'infiles' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'infiles', tuple_306026)

            if more_types_in_union_306025:
                # Runtime conditional SSA for else branch (line 408)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_306024) or more_types_in_union_306025):
            # Assigning a type to the variable 'infiles' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'infiles', remove_subtype_from_union(infiles_306023, str))
            
            
            
            # Call to isinstance(...): (line 410)
            # Processing the call arguments (line 410)
            # Getting the type of 'infiles' (line 410)
            infiles_306029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), 'infiles', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 410)
            tuple_306030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 410)
            # Adding element type (line 410)
            # Getting the type of 'list' (line 410)
            list_306031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 38), tuple_306030, list_306031)
            # Adding element type (line 410)
            # Getting the type of 'tuple' (line 410)
            tuple_306032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 44), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 38), tuple_306030, tuple_306032)
            
            # Processing the call keyword arguments (line 410)
            kwargs_306033 = {}
            # Getting the type of 'isinstance' (line 410)
            isinstance_306028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 410)
            isinstance_call_result_306034 = invoke(stypy.reporting.localization.Localization(__file__, 410, 17), isinstance_306028, *[infiles_306029, tuple_306030], **kwargs_306033)
            
            # Applying the 'not' unary operator (line 410)
            result_not__306035 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 13), 'not', isinstance_call_result_306034)
            
            # Testing the type of an if condition (line 410)
            if_condition_306036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 13), result_not__306035)
            # Assigning a type to the variable 'if_condition_306036' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 13), 'if_condition_306036', if_condition_306036)
            # SSA begins for if statement (line 410)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'TypeError' (line 411)
            TypeError_306037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 18), 'TypeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 411, 12), TypeError_306037, 'raise parameter', BaseException)
            # SSA join for if statement (line 410)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_306024 and more_types_in_union_306025):
                # SSA join for if statement (line 408)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 414)
        # Getting the type of 'exec_msg' (line 414)
        exec_msg_306038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 11), 'exec_msg')
        # Getting the type of 'None' (line 414)
        None_306039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'None')
        
        (may_be_306040, more_types_in_union_306041) = may_be_none(exec_msg_306038, None_306039)

        if may_be_306040:

            if more_types_in_union_306041:
                # Runtime conditional SSA (line 414)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 415):
            str_306042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 23), 'str', 'generating %s from %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 416)
            tuple_306043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 416)
            # Adding element type (line 416)
            # Getting the type of 'outfile' (line 416)
            outfile_306044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'outfile')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 24), tuple_306043, outfile_306044)
            # Adding element type (line 416)
            
            # Call to join(...): (line 416)
            # Processing the call arguments (line 416)
            # Getting the type of 'infiles' (line 416)
            infiles_306047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 43), 'infiles', False)
            # Processing the call keyword arguments (line 416)
            kwargs_306048 = {}
            str_306045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 33), 'str', ', ')
            # Obtaining the member 'join' of a type (line 416)
            join_306046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 33), str_306045, 'join')
            # Calling join(args, kwargs) (line 416)
            join_call_result_306049 = invoke(stypy.reporting.localization.Localization(__file__, 416, 33), join_306046, *[infiles_306047], **kwargs_306048)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 24), tuple_306043, join_call_result_306049)
            
            # Applying the binary operator '%' (line 415)
            result_mod_306050 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 23), '%', str_306042, tuple_306043)
            
            # Assigning a type to the variable 'exec_msg' (line 415)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'exec_msg', result_mod_306050)

            if more_types_in_union_306041:
                # SSA join for if statement (line 414)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 421)
        self_306051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'self')
        # Obtaining the member 'force' of a type (line 421)
        force_306052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 11), self_306051, 'force')
        
        # Call to newer_group(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'infiles' (line 421)
        infiles_306055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 46), 'infiles', False)
        # Getting the type of 'outfile' (line 421)
        outfile_306056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 55), 'outfile', False)
        # Processing the call keyword arguments (line 421)
        kwargs_306057 = {}
        # Getting the type of 'dep_util' (line 421)
        dep_util_306053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 25), 'dep_util', False)
        # Obtaining the member 'newer_group' of a type (line 421)
        newer_group_306054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 25), dep_util_306053, 'newer_group')
        # Calling newer_group(args, kwargs) (line 421)
        newer_group_call_result_306058 = invoke(stypy.reporting.localization.Localization(__file__, 421, 25), newer_group_306054, *[infiles_306055, outfile_306056], **kwargs_306057)
        
        # Applying the binary operator 'or' (line 421)
        result_or_keyword_306059 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 11), 'or', force_306052, newer_group_call_result_306058)
        
        # Testing the type of an if condition (line 421)
        if_condition_306060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 8), result_or_keyword_306059)
        # Assigning a type to the variable 'if_condition_306060' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'if_condition_306060', if_condition_306060)
        # SSA begins for if statement (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to execute(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'func' (line 422)
        func_306063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 25), 'func', False)
        # Getting the type of 'args' (line 422)
        args_306064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 31), 'args', False)
        # Getting the type of 'exec_msg' (line 422)
        exec_msg_306065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 37), 'exec_msg', False)
        # Getting the type of 'level' (line 422)
        level_306066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 47), 'level', False)
        # Processing the call keyword arguments (line 422)
        kwargs_306067 = {}
        # Getting the type of 'self' (line 422)
        self_306061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 422)
        execute_306062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 12), self_306061, 'execute')
        # Calling execute(args, kwargs) (line 422)
        execute_call_result_306068 = invoke(stypy.reporting.localization.Localization(__file__, 422, 12), execute_306062, *[func_306063, args_306064, exec_msg_306065, level_306066], **kwargs_306067)
        
        # SSA branch for the else part of an if statement (line 421)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'skip_msg' (line 426)
        skip_msg_306071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 22), 'skip_msg', False)
        # Processing the call keyword arguments (line 426)
        kwargs_306072 = {}
        # Getting the type of 'log' (line 426)
        log_306069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 426)
        debug_306070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), log_306069, 'debug')
        # Calling debug(args, kwargs) (line 426)
        debug_call_result_306073 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), debug_306070, *[skip_msg_306071], **kwargs_306072)
        
        # SSA join for if statement (line 421)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'make_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_file' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_306074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_file'
        return stypy_return_type_306074


# Assigning a type to the variable 'Command' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'Command', Command)

# Assigning a List to a Name (line 44):

# Obtaining an instance of the builtin type 'list' (line 44)
list_306075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)

# Getting the type of 'Command'
Command_306076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Command')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Command_306076, 'sub_commands', list_306075)
# Declaration of the 'install_misc' class
# Getting the type of 'Command' (line 433)
Command_306077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), 'Command')

class install_misc(Command_306077, ):
    str_306078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, (-1)), 'str', 'Common base class for installing some files in a subdirectory.\n    Currently used by install_data and install_scripts.\n    ')

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_misc.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_misc.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_misc.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_misc.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_misc.initialize_options')
        install_misc.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_misc.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_misc.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_misc.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_misc.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_misc.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_misc.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_misc.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 441):
        # Getting the type of 'None' (line 441)
        None_306079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'None')
        # Getting the type of 'self' (line 441)
        self_306080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 441)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_306080, 'install_dir', None_306079)
        
        # Assigning a List to a Attribute (line 442):
        
        # Obtaining an instance of the builtin type 'list' (line 442)
        list_306081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 442)
        
        # Getting the type of 'self' (line 442)
        self_306082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 442)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_306082, 'outfiles', list_306081)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 440)
        stypy_return_type_306083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306083)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_306083


    @norecursion
    def _install_dir_from(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_install_dir_from'
        module_type_store = module_type_store.open_function_context('_install_dir_from', 444, 4, False)
        # Assigning a type to the variable 'self' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_misc._install_dir_from.__dict__.__setitem__('stypy_localization', localization)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_function_name', 'install_misc._install_dir_from')
        install_misc._install_dir_from.__dict__.__setitem__('stypy_param_names_list', ['dirname'])
        install_misc._install_dir_from.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_misc._install_dir_from.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_misc._install_dir_from', ['dirname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_install_dir_from', localization, ['dirname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_install_dir_from(...)' code ##################

        
        # Call to set_undefined_options(...): (line 445)
        # Processing the call arguments (line 445)
        str_306086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 445)
        tuple_306087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 445)
        # Adding element type (line 445)
        # Getting the type of 'dirname' (line 445)
        dirname_306088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 47), 'dirname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 47), tuple_306087, dirname_306088)
        # Adding element type (line 445)
        str_306089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 56), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 47), tuple_306087, str_306089)
        
        # Processing the call keyword arguments (line 445)
        kwargs_306090 = {}
        # Getting the type of 'self' (line 445)
        self_306084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 445)
        set_undefined_options_306085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), self_306084, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 445)
        set_undefined_options_call_result_306091 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), set_undefined_options_306085, *[str_306086, tuple_306087], **kwargs_306090)
        
        
        # ################# End of '_install_dir_from(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_install_dir_from' in the type store
        # Getting the type of 'stypy_return_type' (line 444)
        stypy_return_type_306092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306092)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_install_dir_from'
        return stypy_return_type_306092


    @norecursion
    def _copy_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_copy_files'
        module_type_store = module_type_store.open_function_context('_copy_files', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_misc._copy_files.__dict__.__setitem__('stypy_localization', localization)
        install_misc._copy_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_misc._copy_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_misc._copy_files.__dict__.__setitem__('stypy_function_name', 'install_misc._copy_files')
        install_misc._copy_files.__dict__.__setitem__('stypy_param_names_list', ['filelist'])
        install_misc._copy_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_misc._copy_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_misc._copy_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_misc._copy_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_misc._copy_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_misc._copy_files.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_misc._copy_files', ['filelist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_copy_files', localization, ['filelist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_copy_files(...)' code ##################

        
        # Assigning a List to a Attribute (line 448):
        
        # Obtaining an instance of the builtin type 'list' (line 448)
        list_306093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 448)
        
        # Getting the type of 'self' (line 448)
        self_306094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), self_306094, 'outfiles', list_306093)
        
        
        # Getting the type of 'filelist' (line 449)
        filelist_306095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'filelist')
        # Applying the 'not' unary operator (line 449)
        result_not__306096 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 11), 'not', filelist_306095)
        
        # Testing the type of an if condition (line 449)
        if_condition_306097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 8), result_not__306096)
        # Assigning a type to the variable 'if_condition_306097' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'if_condition_306097', if_condition_306097)
        # SSA begins for if statement (line 449)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 449)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'self' (line 451)
        self_306100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 451)
        install_dir_306101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 20), self_306100, 'install_dir')
        # Processing the call keyword arguments (line 451)
        kwargs_306102 = {}
        # Getting the type of 'self' (line 451)
        self_306098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 451)
        mkpath_306099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_306098, 'mkpath')
        # Calling mkpath(args, kwargs) (line 451)
        mkpath_call_result_306103 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), mkpath_306099, *[install_dir_306101], **kwargs_306102)
        
        
        # Getting the type of 'filelist' (line 452)
        filelist_306104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 17), 'filelist')
        # Testing the type of a for loop iterable (line 452)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 452, 8), filelist_306104)
        # Getting the type of the for loop variable (line 452)
        for_loop_var_306105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 452, 8), filelist_306104)
        # Assigning a type to the variable 'f' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'f', for_loop_var_306105)
        # SSA begins for a for statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to copy_file(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'f' (line 453)
        f_306108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 27), 'f', False)
        # Getting the type of 'self' (line 453)
        self_306109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 30), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 453)
        install_dir_306110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 30), self_306109, 'install_dir')
        # Processing the call keyword arguments (line 453)
        kwargs_306111 = {}
        # Getting the type of 'self' (line 453)
        self_306106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 453)
        copy_file_306107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), self_306106, 'copy_file')
        # Calling copy_file(args, kwargs) (line 453)
        copy_file_call_result_306112 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), copy_file_306107, *[f_306108, install_dir_306110], **kwargs_306111)
        
        
        # Call to append(...): (line 454)
        # Processing the call arguments (line 454)
        
        # Call to join(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'self' (line 454)
        self_306119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 46), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 454)
        install_dir_306120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 46), self_306119, 'install_dir')
        # Getting the type of 'f' (line 454)
        f_306121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 64), 'f', False)
        # Processing the call keyword arguments (line 454)
        kwargs_306122 = {}
        # Getting the type of 'os' (line 454)
        os_306116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 454)
        path_306117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 33), os_306116, 'path')
        # Obtaining the member 'join' of a type (line 454)
        join_306118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 33), path_306117, 'join')
        # Calling join(args, kwargs) (line 454)
        join_call_result_306123 = invoke(stypy.reporting.localization.Localization(__file__, 454, 33), join_306118, *[install_dir_306120, f_306121], **kwargs_306122)
        
        # Processing the call keyword arguments (line 454)
        kwargs_306124 = {}
        # Getting the type of 'self' (line 454)
        self_306113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'self', False)
        # Obtaining the member 'outfiles' of a type (line 454)
        outfiles_306114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 12), self_306113, 'outfiles')
        # Obtaining the member 'append' of a type (line 454)
        append_306115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 12), outfiles_306114, 'append')
        # Calling append(args, kwargs) (line 454)
        append_call_result_306125 = invoke(stypy.reporting.localization.Localization(__file__, 454, 12), append_306115, *[join_call_result_306123], **kwargs_306124)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_copy_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_copy_files' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_306126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_copy_files'
        return stypy_return_type_306126


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_misc.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_misc.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_misc.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_misc.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_misc.get_outputs')
        install_misc.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_misc.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_misc.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_misc.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_misc.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_misc.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_misc.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_misc.get_outputs', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 457)
        self_306127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'self')
        # Obtaining the member 'outfiles' of a type (line 457)
        outfiles_306128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), self_306127, 'outfiles')
        # Assigning a type to the variable 'stypy_return_type' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'stypy_return_type', outfiles_306128)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_306129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_306129


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 433, 0, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_misc.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_misc' (line 433)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 0), 'install_misc', install_misc)

# Assigning a List to a Name (line 438):

# Obtaining an instance of the builtin type 'list' (line 438)
list_306130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 438)
# Adding element type (line 438)

# Obtaining an instance of the builtin type 'tuple' (line 438)
tuple_306131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 438)
# Adding element type (line 438)
str_306132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 21), 'str', 'install-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 21), tuple_306131, str_306132)
# Adding element type (line 438)
str_306133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 37), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 21), tuple_306131, str_306133)
# Adding element type (line 438)
str_306134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 42), 'str', 'directory to install the files to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 21), tuple_306131, str_306134)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 19), list_306130, tuple_306131)

# Getting the type of 'install_misc'
install_misc_306135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_misc')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_misc_306135, 'user_options', list_306130)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

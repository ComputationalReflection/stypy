
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.util
2: 
3: Miscellaneous utility functions -- anything that doesn't fit into
4: one of the other *util.py modules.
5: '''
6: 
7: __revision__ = "$Id$"
8: 
9: import sys, os, string, re
10: from distutils.errors import DistutilsPlatformError
11: from distutils.dep_util import newer
12: from distutils.spawn import spawn
13: from distutils import log
14: from distutils.errors import DistutilsByteCompileError
15: 
16: def get_platform ():
17:     '''Return a string that identifies the current platform.  This is used
18:     mainly to distinguish platform-specific build directories and
19:     platform-specific built distributions.  Typically includes the OS name
20:     and version and the architecture (as supplied by 'os.uname()'),
21:     although the exact information included depends on the OS; eg. for IRIX
22:     the architecture isn't particularly important (IRIX only runs on SGI
23:     hardware), but for Linux the kernel version isn't particularly
24:     important.
25: 
26:     Examples of returned values:
27:        linux-i586
28:        linux-alpha (?)
29:        solaris-2.6-sun4u
30:        irix-5.3
31:        irix64-6.2
32: 
33:     Windows will return one of:
34:        win-amd64 (64bit Windows on AMD64 (aka x86_64, Intel64, EM64T, etc)
35:        win-ia64 (64bit Windows on Itanium)
36:        win32 (all others - specifically, sys.platform is returned)
37: 
38:     For other non-POSIX platforms, currently just returns 'sys.platform'.
39:     '''
40:     if os.name == 'nt':
41:         # sniff sys.version for architecture.
42:         prefix = " bit ("
43:         i = string.find(sys.version, prefix)
44:         if i == -1:
45:             return sys.platform
46:         j = string.find(sys.version, ")", i)
47:         look = sys.version[i+len(prefix):j].lower()
48:         if look=='amd64':
49:             return 'win-amd64'
50:         if look=='itanium':
51:             return 'win-ia64'
52:         return sys.platform
53: 
54:     # Set for cross builds explicitly
55:     if "_PYTHON_HOST_PLATFORM" in os.environ:
56:         return os.environ["_PYTHON_HOST_PLATFORM"]
57: 
58:     if os.name != "posix" or not hasattr(os, 'uname'):
59:         # XXX what about the architecture? NT is Intel or Alpha,
60:         # Mac OS is M68k or PPC, etc.
61:         return sys.platform
62: 
63:     # Try to distinguish various flavours of Unix
64: 
65:     (osname, host, release, version, machine) = os.uname()
66: 
67:     # Convert the OS name to lowercase, remove '/' characters
68:     # (to accommodate BSD/OS), and translate spaces (for "Power Macintosh")
69:     osname = string.lower(osname)
70:     osname = string.replace(osname, '/', '')
71:     machine = string.replace(machine, ' ', '_')
72:     machine = string.replace(machine, '/', '-')
73: 
74:     if osname[:5] == "linux":
75:         # At least on Linux/Intel, 'machine' is the processor --
76:         # i386, etc.
77:         # XXX what about Alpha, SPARC, etc?
78:         return  "%s-%s" % (osname, machine)
79:     elif osname[:5] == "sunos":
80:         if release[0] >= "5":           # SunOS 5 == Solaris 2
81:             osname = "solaris"
82:             release = "%d.%s" % (int(release[0]) - 3, release[2:])
83:             # We can't use "platform.architecture()[0]" because a
84:             # bootstrap problem. We use a dict to get an error
85:             # if some suspicious happens.
86:             bitness = {2147483647:"32bit", 9223372036854775807:"64bit"}
87:             machine += ".%s" % bitness[sys.maxint]
88:         # fall through to standard osname-release-machine representation
89:     elif osname[:4] == "irix":              # could be "irix64"!
90:         return "%s-%s" % (osname, release)
91:     elif osname[:3] == "aix":
92:         return "%s-%s.%s" % (osname, version, release)
93:     elif osname[:6] == "cygwin":
94:         osname = "cygwin"
95:         rel_re = re.compile (r'[\d.]+')
96:         m = rel_re.match(release)
97:         if m:
98:             release = m.group()
99:     elif osname[:6] == "darwin":
100:         import _osx_support, distutils.sysconfig
101:         osname, release, machine = _osx_support.get_platform_osx(
102:                                         distutils.sysconfig.get_config_vars(),
103:                                         osname, release, machine)
104: 
105:     return "%s-%s-%s" % (osname, release, machine)
106: 
107: # get_platform ()
108: 
109: 
110: def convert_path (pathname):
111:     '''Return 'pathname' as a name that will work on the native filesystem,
112:     i.e. split it on '/' and put it back together again using the current
113:     directory separator.  Needed because filenames in the setup script are
114:     always supplied in Unix style, and have to be converted to the local
115:     convention before we can actually use them in the filesystem.  Raises
116:     ValueError on non-Unix-ish systems if 'pathname' either starts or
117:     ends with a slash.
118:     '''
119:     if os.sep == '/':
120:         return pathname
121:     if not pathname:
122:         return pathname
123:     if pathname[0] == '/':
124:         raise ValueError, "path '%s' cannot be absolute" % pathname
125:     if pathname[-1] == '/':
126:         raise ValueError, "path '%s' cannot end with '/'" % pathname
127: 
128:     paths = string.split(pathname, '/')
129:     while '.' in paths:
130:         paths.remove('.')
131:     if not paths:
132:         return os.curdir
133:     return os.path.join(*paths)
134: 
135: # convert_path ()
136: 
137: 
138: def change_root (new_root, pathname):
139:     '''Return 'pathname' with 'new_root' prepended.  If 'pathname' is
140:     relative, this is equivalent to "os.path.join(new_root,pathname)".
141:     Otherwise, it requires making 'pathname' relative and then joining the
142:     two, which is tricky on DOS/Windows and Mac OS.
143:     '''
144:     if os.name == 'posix':
145:         if not os.path.isabs(pathname):
146:             return os.path.join(new_root, pathname)
147:         else:
148:             return os.path.join(new_root, pathname[1:])
149: 
150:     elif os.name == 'nt':
151:         (drive, path) = os.path.splitdrive(pathname)
152:         if path[0] == '\\':
153:             path = path[1:]
154:         return os.path.join(new_root, path)
155: 
156:     elif os.name == 'os2':
157:         (drive, path) = os.path.splitdrive(pathname)
158:         if path[0] == os.sep:
159:             path = path[1:]
160:         return os.path.join(new_root, path)
161: 
162:     else:
163:         raise DistutilsPlatformError, \
164:               "nothing known about platform '%s'" % os.name
165: 
166: 
167: _environ_checked = 0
168: def check_environ ():
169:     '''Ensure that 'os.environ' has all the environment variables we
170:     guarantee that users can use in config files, command-line options,
171:     etc.  Currently this includes:
172:       HOME - user's home directory (Unix only)
173:       PLAT - description of the current platform, including hardware
174:              and OS (see 'get_platform()')
175:     '''
176:     global _environ_checked
177:     if _environ_checked:
178:         return
179: 
180:     if os.name == 'posix' and 'HOME' not in os.environ:
181:         import pwd
182:         os.environ['HOME'] = pwd.getpwuid(os.getuid())[5]
183: 
184:     if 'PLAT' not in os.environ:
185:         os.environ['PLAT'] = get_platform()
186: 
187:     _environ_checked = 1
188: 
189: 
190: def subst_vars (s, local_vars):
191:     '''Perform shell/Perl-style variable substitution on 'string'.  Every
192:     occurrence of '$' followed by a name is considered a variable, and
193:     variable is substituted by the value found in the 'local_vars'
194:     dictionary, or in 'os.environ' if it's not in 'local_vars'.
195:     'os.environ' is first checked/augmented to guarantee that it contains
196:     certain values: see 'check_environ()'.  Raise ValueError for any
197:     variables not found in either 'local_vars' or 'os.environ'.
198:     '''
199:     check_environ()
200:     def _subst (match, local_vars=local_vars):
201:         var_name = match.group(1)
202:         if var_name in local_vars:
203:             return str(local_vars[var_name])
204:         else:
205:             return os.environ[var_name]
206: 
207:     try:
208:         return re.sub(r'\$([a-zA-Z_][a-zA-Z_0-9]*)', _subst, s)
209:     except KeyError, var:
210:         raise ValueError, "invalid variable '$%s'" % var
211: 
212: # subst_vars ()
213: 
214: 
215: def grok_environment_error (exc, prefix="error: "):
216:     # Function kept for backward compatibility.
217:     # Used to try clever things with EnvironmentErrors,
218:     # but nowadays str(exception) produces good messages.
219:     return prefix + str(exc)
220: 
221: 
222: # Needed by 'split_quoted()'
223: _wordchars_re = _squote_re = _dquote_re = None
224: def _init_regex():
225:     global _wordchars_re, _squote_re, _dquote_re
226:     _wordchars_re = re.compile(r'[^\\\'\"%s ]*' % string.whitespace)
227:     _squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
228:     _dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')
229: 
230: def split_quoted (s):
231:     '''Split a string up according to Unix shell-like rules for quotes and
232:     backslashes.  In short: words are delimited by spaces, as long as those
233:     spaces are not escaped by a backslash, or inside a quoted string.
234:     Single and double quotes are equivalent, and the quote characters can
235:     be backslash-escaped.  The backslash is stripped from any two-character
236:     escape sequence, leaving only the escaped character.  The quote
237:     characters are stripped from any quoted string.  Returns a list of
238:     words.
239:     '''
240: 
241:     # This is a nice algorithm for splitting up a single string, since it
242:     # doesn't require character-by-character examination.  It was a little
243:     # bit of a brain-bender to get it working right, though...
244:     if _wordchars_re is None: _init_regex()
245: 
246:     s = string.strip(s)
247:     words = []
248:     pos = 0
249: 
250:     while s:
251:         m = _wordchars_re.match(s, pos)
252:         end = m.end()
253:         if end == len(s):
254:             words.append(s[:end])
255:             break
256: 
257:         if s[end] in string.whitespace: # unescaped, unquoted whitespace: now
258:             words.append(s[:end])       # we definitely have a word delimiter
259:             s = string.lstrip(s[end:])
260:             pos = 0
261: 
262:         elif s[end] == '\\':            # preserve whatever is being escaped;
263:                                         # will become part of the current word
264:             s = s[:end] + s[end+1:]
265:             pos = end+1
266: 
267:         else:
268:             if s[end] == "'":           # slurp singly-quoted string
269:                 m = _squote_re.match(s, end)
270:             elif s[end] == '"':         # slurp doubly-quoted string
271:                 m = _dquote_re.match(s, end)
272:             else:
273:                 raise RuntimeError, \
274:                       "this can't happen (bad char '%c')" % s[end]
275: 
276:             if m is None:
277:                 raise ValueError, \
278:                       "bad string (mismatched %s quotes?)" % s[end]
279: 
280:             (beg, end) = m.span()
281:             s = s[:beg] + s[beg+1:end-1] + s[end:]
282:             pos = m.end() - 2
283: 
284:         if pos >= len(s):
285:             words.append(s)
286:             break
287: 
288:     return words
289: 
290: # split_quoted ()
291: 
292: 
293: def execute (func, args, msg=None, verbose=0, dry_run=0):
294:     '''Perform some action that affects the outside world (eg.  by
295:     writing to the filesystem).  Such actions are special because they
296:     are disabled by the 'dry_run' flag.  This method takes care of all
297:     that bureaucracy for you; all you have to do is supply the
298:     function to call and an argument tuple for it (to embody the
299:     "external action" being performed), and an optional message to
300:     print.
301:     '''
302:     if msg is None:
303:         msg = "%s%r" % (func.__name__, args)
304:         if msg[-2:] == ',)':        # correct for singleton tuple
305:             msg = msg[0:-2] + ')'
306: 
307:     log.info(msg)
308:     if not dry_run:
309:         func(*args)
310: 
311: 
312: def strtobool (val):
313:     '''Convert a string representation of truth to true (1) or false (0).
314: 
315:     True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
316:     are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
317:     'val' is anything else.
318:     '''
319:     val = string.lower(val)
320:     if val in ('y', 'yes', 't', 'true', 'on', '1'):
321:         return 1
322:     elif val in ('n', 'no', 'f', 'false', 'off', '0'):
323:         return 0
324:     else:
325:         raise ValueError, "invalid truth value %r" % (val,)
326: 
327: 
328: def byte_compile (py_files,
329:                   optimize=0, force=0,
330:                   prefix=None, base_dir=None,
331:                   verbose=1, dry_run=0,
332:                   direct=None):
333:     '''Byte-compile a collection of Python source files to either .pyc
334:     or .pyo files in the same directory.  'py_files' is a list of files
335:     to compile; any files that don't end in ".py" are silently skipped.
336:     'optimize' must be one of the following:
337:       0 - don't optimize (generate .pyc)
338:       1 - normal optimization (like "python -O")
339:       2 - extra optimization (like "python -OO")
340:     If 'force' is true, all files are recompiled regardless of
341:     timestamps.
342: 
343:     The source filename encoded in each bytecode file defaults to the
344:     filenames listed in 'py_files'; you can modify these with 'prefix' and
345:     'basedir'.  'prefix' is a string that will be stripped off of each
346:     source filename, and 'base_dir' is a directory name that will be
347:     prepended (after 'prefix' is stripped).  You can supply either or both
348:     (or neither) of 'prefix' and 'base_dir', as you wish.
349: 
350:     If 'dry_run' is true, doesn't actually do anything that would
351:     affect the filesystem.
352: 
353:     Byte-compilation is either done directly in this interpreter process
354:     with the standard py_compile module, or indirectly by writing a
355:     temporary script and executing it.  Normally, you should let
356:     'byte_compile()' figure out to use direct compilation or not (see
357:     the source for details).  The 'direct' flag is used by the script
358:     generated in indirect mode; unless you know what you're doing, leave
359:     it set to None.
360:     '''
361:     # nothing is done if sys.dont_write_bytecode is True
362:     if sys.dont_write_bytecode:
363:         raise DistutilsByteCompileError('byte-compiling is disabled.')
364: 
365:     # First, if the caller didn't force us into direct or indirect mode,
366:     # figure out which mode we should be in.  We take a conservative
367:     # approach: choose direct mode *only* if the current interpreter is
368:     # in debug mode and optimize is 0.  If we're not in debug mode (-O
369:     # or -OO), we don't know which level of optimization this
370:     # interpreter is running with, so we can't do direct
371:     # byte-compilation and be certain that it's the right thing.  Thus,
372:     # always compile indirectly if the current interpreter is in either
373:     # optimize mode, or if either optimization level was requested by
374:     # the caller.
375:     if direct is None:
376:         direct = (__debug__ and optimize == 0)
377: 
378:     # "Indirect" byte-compilation: write a temporary script and then
379:     # run it with the appropriate flags.
380:     if not direct:
381:         try:
382:             from tempfile import mkstemp
383:             (script_fd, script_name) = mkstemp(".py")
384:         except ImportError:
385:             from tempfile import mktemp
386:             (script_fd, script_name) = None, mktemp(".py")
387:         log.info("writing byte-compilation script '%s'", script_name)
388:         if not dry_run:
389:             if script_fd is not None:
390:                 script = os.fdopen(script_fd, "w")
391:             else:
392:                 script = open(script_name, "w")
393: 
394:             script.write('''\
395: from distutils.util import byte_compile
396: files = [
397: ''')
398: 
399:             # XXX would be nice to write absolute filenames, just for
400:             # safety's sake (script should be more robust in the face of
401:             # chdir'ing before running it).  But this requires abspath'ing
402:             # 'prefix' as well, and that breaks the hack in build_lib's
403:             # 'byte_compile()' method that carefully tacks on a trailing
404:             # slash (os.sep really) to make sure the prefix here is "just
405:             # right".  This whole prefix business is rather delicate -- the
406:             # problem is that it's really a directory, but I'm treating it
407:             # as a dumb string, so trailing slashes and so forth matter.
408: 
409:             #py_files = map(os.path.abspath, py_files)
410:             #if prefix:
411:             #    prefix = os.path.abspath(prefix)
412: 
413:             script.write(string.join(map(repr, py_files), ",\n") + "]\n")
414:             script.write('''
415: byte_compile(files, optimize=%r, force=%r,
416:              prefix=%r, base_dir=%r,
417:              verbose=%r, dry_run=0,
418:              direct=1)
419: ''' % (optimize, force, prefix, base_dir, verbose))
420: 
421:             script.close()
422: 
423:         cmd = [sys.executable, script_name]
424:         if optimize == 1:
425:             cmd.insert(1, "-O")
426:         elif optimize == 2:
427:             cmd.insert(1, "-OO")
428:         spawn(cmd, dry_run=dry_run)
429:         execute(os.remove, (script_name,), "removing %s" % script_name,
430:                 dry_run=dry_run)
431: 
432:     # "Direct" byte-compilation: use the py_compile module to compile
433:     # right here, right now.  Note that the script generated in indirect
434:     # mode simply calls 'byte_compile()' in direct mode, a weird sort of
435:     # cross-process recursion.  Hey, it works!
436:     else:
437:         from py_compile import compile
438: 
439:         for file in py_files:
440:             if file[-3:] != ".py":
441:                 # This lets us be lazy and not filter filenames in
442:                 # the "install_lib" command.
443:                 continue
444: 
445:             # Terminology from the py_compile module:
446:             #   cfile - byte-compiled file
447:             #   dfile - purported source filename (same as 'file' by default)
448:             cfile = file + (__debug__ and "c" or "o")
449:             dfile = file
450:             if prefix:
451:                 if file[:len(prefix)] != prefix:
452:                     raise ValueError, \
453:                           ("invalid prefix: filename %r doesn't start with %r"
454:                            % (file, prefix))
455:                 dfile = dfile[len(prefix):]
456:             if base_dir:
457:                 dfile = os.path.join(base_dir, dfile)
458: 
459:             cfile_base = os.path.basename(cfile)
460:             if direct:
461:                 if force or newer(file, cfile):
462:                     log.info("byte-compiling %s to %s", file, cfile_base)
463:                     if not dry_run:
464:                         compile(file, cfile, dfile)
465:                 else:
466:                     log.debug("skipping byte-compilation of %s to %s",
467:                               file, cfile_base)
468: 
469: # byte_compile ()
470: 
471: def rfc822_escape (header):
472:     '''Return a version of the string escaped for inclusion in an
473:     RFC-822 header, by ensuring there are 8 spaces space after each newline.
474:     '''
475:     lines = string.split(header, '\n')
476:     header = string.join(lines, '\n' + 8*' ')
477:     return header
478: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_9898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', "distutils.util\n\nMiscellaneous utility functions -- anything that doesn't fit into\none of the other *util.py modules.\n")

# Assigning a Str to a Name (line 7):

# Assigning a Str to a Name (line 7):
str_9899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_9899)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# Multiple import statement. import sys (1/4) (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/4) (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)
# Multiple import statement. import string (3/4) (line 9)
import string

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'string', string, module_type_store)
# Multiple import statement. import re (4/4) (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.errors import DistutilsPlatformError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_9900) is not StypyTypeError):

    if (import_9900 != 'pyd_module'):
        __import__(import_9900)
        sys_modules_9901 = sys.modules[import_9900]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_9901.module_type_store, module_type_store, ['DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_9901, sys_modules_9901.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError'], [DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_9900)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.dep_util import newer' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util')

if (type(import_9902) is not StypyTypeError):

    if (import_9902 != 'pyd_module'):
        __import__(import_9902)
        sys_modules_9903 = sys.modules[import_9902]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util', sys_modules_9903.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_9903, sys_modules_9903.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util', import_9902)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.spawn import spawn' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9904 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.spawn')

if (type(import_9904) is not StypyTypeError):

    if (import_9904 != 'pyd_module'):
        __import__(import_9904)
        sys_modules_9905 = sys.modules[import_9904]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.spawn', sys_modules_9905.module_type_store, module_type_store, ['spawn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_9905, sys_modules_9905.module_type_store, module_type_store)
    else:
        from distutils.spawn import spawn

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.spawn', None, module_type_store, ['spawn'], [spawn])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.spawn', import_9904)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils import log' statement (line 13)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.errors import DistutilsByteCompileError' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9906 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors')

if (type(import_9906) is not StypyTypeError):

    if (import_9906 != 'pyd_module'):
        __import__(import_9906)
        sys_modules_9907 = sys.modules[import_9906]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', sys_modules_9907.module_type_store, module_type_store, ['DistutilsByteCompileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_9907, sys_modules_9907.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsByteCompileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', None, module_type_store, ['DistutilsByteCompileError'], [DistutilsByteCompileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', import_9906)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


@norecursion
def get_platform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_platform'
    module_type_store = module_type_store.open_function_context('get_platform', 16, 0, False)
    
    # Passed parameters checking function
    get_platform.stypy_localization = localization
    get_platform.stypy_type_of_self = None
    get_platform.stypy_type_store = module_type_store
    get_platform.stypy_function_name = 'get_platform'
    get_platform.stypy_param_names_list = []
    get_platform.stypy_varargs_param_name = None
    get_platform.stypy_kwargs_param_name = None
    get_platform.stypy_call_defaults = defaults
    get_platform.stypy_call_varargs = varargs
    get_platform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_platform', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_platform', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_platform(...)' code ##################

    str_9908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', "Return a string that identifies the current platform.  This is used\n    mainly to distinguish platform-specific build directories and\n    platform-specific built distributions.  Typically includes the OS name\n    and version and the architecture (as supplied by 'os.uname()'),\n    although the exact information included depends on the OS; eg. for IRIX\n    the architecture isn't particularly important (IRIX only runs on SGI\n    hardware), but for Linux the kernel version isn't particularly\n    important.\n\n    Examples of returned values:\n       linux-i586\n       linux-alpha (?)\n       solaris-2.6-sun4u\n       irix-5.3\n       irix64-6.2\n\n    Windows will return one of:\n       win-amd64 (64bit Windows on AMD64 (aka x86_64, Intel64, EM64T, etc)\n       win-ia64 (64bit Windows on Itanium)\n       win32 (all others - specifically, sys.platform is returned)\n\n    For other non-POSIX platforms, currently just returns 'sys.platform'.\n    ")
    
    
    # Getting the type of 'os' (line 40)
    os_9909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'os')
    # Obtaining the member 'name' of a type (line 40)
    name_9910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 7), os_9909, 'name')
    str_9911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'str', 'nt')
    # Applying the binary operator '==' (line 40)
    result_eq_9912 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 7), '==', name_9910, str_9911)
    
    # Testing the type of an if condition (line 40)
    if_condition_9913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 4), result_eq_9912)
    # Assigning a type to the variable 'if_condition_9913' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'if_condition_9913', if_condition_9913)
    # SSA begins for if statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 42):
    
    # Assigning a Str to a Name (line 42):
    str_9914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'str', ' bit (')
    # Assigning a type to the variable 'prefix' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'prefix', str_9914)
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to find(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'sys' (line 43)
    sys_9917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'sys', False)
    # Obtaining the member 'version' of a type (line 43)
    version_9918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), sys_9917, 'version')
    # Getting the type of 'prefix' (line 43)
    prefix_9919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 37), 'prefix', False)
    # Processing the call keyword arguments (line 43)
    kwargs_9920 = {}
    # Getting the type of 'string' (line 43)
    string_9915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'string', False)
    # Obtaining the member 'find' of a type (line 43)
    find_9916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), string_9915, 'find')
    # Calling find(args, kwargs) (line 43)
    find_call_result_9921 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), find_9916, *[version_9918, prefix_9919], **kwargs_9920)
    
    # Assigning a type to the variable 'i' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'i', find_call_result_9921)
    
    
    # Getting the type of 'i' (line 44)
    i_9922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'i')
    int_9923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
    # Applying the binary operator '==' (line 44)
    result_eq_9924 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '==', i_9922, int_9923)
    
    # Testing the type of an if condition (line 44)
    if_condition_9925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_eq_9924)
    # Assigning a type to the variable 'if_condition_9925' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_9925', if_condition_9925)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'sys' (line 45)
    sys_9926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'sys')
    # Obtaining the member 'platform' of a type (line 45)
    platform_9927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), sys_9926, 'platform')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', platform_9927)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to find(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'sys' (line 46)
    sys_9930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'sys', False)
    # Obtaining the member 'version' of a type (line 46)
    version_9931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 24), sys_9930, 'version')
    str_9932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'str', ')')
    # Getting the type of 'i' (line 46)
    i_9933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'i', False)
    # Processing the call keyword arguments (line 46)
    kwargs_9934 = {}
    # Getting the type of 'string' (line 46)
    string_9928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'string', False)
    # Obtaining the member 'find' of a type (line 46)
    find_9929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), string_9928, 'find')
    # Calling find(args, kwargs) (line 46)
    find_call_result_9935 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), find_9929, *[version_9931, str_9932, i_9933], **kwargs_9934)
    
    # Assigning a type to the variable 'j' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'j', find_call_result_9935)
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to lower(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_9949 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 47)
    i_9936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'i', False)
    
    # Call to len(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'prefix' (line 47)
    prefix_9938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'prefix', False)
    # Processing the call keyword arguments (line 47)
    kwargs_9939 = {}
    # Getting the type of 'len' (line 47)
    len_9937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'len', False)
    # Calling len(args, kwargs) (line 47)
    len_call_result_9940 = invoke(stypy.reporting.localization.Localization(__file__, 47, 29), len_9937, *[prefix_9938], **kwargs_9939)
    
    # Applying the binary operator '+' (line 47)
    result_add_9941 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 27), '+', i_9936, len_call_result_9940)
    
    # Getting the type of 'j' (line 47)
    j_9942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'j', False)
    slice_9943 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 47, 15), result_add_9941, j_9942, None)
    # Getting the type of 'sys' (line 47)
    sys_9944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'sys', False)
    # Obtaining the member 'version' of a type (line 47)
    version_9945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), sys_9944, 'version')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___9946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), version_9945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_9947 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), getitem___9946, slice_9943)
    
    # Obtaining the member 'lower' of a type (line 47)
    lower_9948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), subscript_call_result_9947, 'lower')
    # Calling lower(args, kwargs) (line 47)
    lower_call_result_9950 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), lower_9948, *[], **kwargs_9949)
    
    # Assigning a type to the variable 'look' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'look', lower_call_result_9950)
    
    
    # Getting the type of 'look' (line 48)
    look_9951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'look')
    str_9952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'str', 'amd64')
    # Applying the binary operator '==' (line 48)
    result_eq_9953 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), '==', look_9951, str_9952)
    
    # Testing the type of an if condition (line 48)
    if_condition_9954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_eq_9953)
    # Assigning a type to the variable 'if_condition_9954' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_9954', if_condition_9954)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_9955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'str', 'win-amd64')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', str_9955)
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'look' (line 50)
    look_9956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'look')
    str_9957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'str', 'itanium')
    # Applying the binary operator '==' (line 50)
    result_eq_9958 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), '==', look_9956, str_9957)
    
    # Testing the type of an if condition (line 50)
    if_condition_9959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_eq_9958)
    # Assigning a type to the variable 'if_condition_9959' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_9959', if_condition_9959)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_9960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'str', 'win-ia64')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'stypy_return_type', str_9960)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sys' (line 52)
    sys_9961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'sys')
    # Obtaining the member 'platform' of a type (line 52)
    platform_9962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), sys_9961, 'platform')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', platform_9962)
    # SSA join for if statement (line 40)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_9963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 7), 'str', '_PYTHON_HOST_PLATFORM')
    # Getting the type of 'os' (line 55)
    os_9964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'os')
    # Obtaining the member 'environ' of a type (line 55)
    environ_9965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 34), os_9964, 'environ')
    # Applying the binary operator 'in' (line 55)
    result_contains_9966 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), 'in', str_9963, environ_9965)
    
    # Testing the type of an if condition (line 55)
    if_condition_9967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), result_contains_9966)
    # Assigning a type to the variable 'if_condition_9967' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_9967', if_condition_9967)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    str_9968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'str', '_PYTHON_HOST_PLATFORM')
    # Getting the type of 'os' (line 56)
    os_9969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'os')
    # Obtaining the member 'environ' of a type (line 56)
    environ_9970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 15), os_9969, 'environ')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___9971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 15), environ_9970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_9972 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), getitem___9971, str_9968)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', subscript_call_result_9972)
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'os' (line 58)
    os_9973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'os')
    # Obtaining the member 'name' of a type (line 58)
    name_9974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 7), os_9973, 'name')
    str_9975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'str', 'posix')
    # Applying the binary operator '!=' (line 58)
    result_ne_9976 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), '!=', name_9974, str_9975)
    
    
    
    # Call to hasattr(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'os' (line 58)
    os_9978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 41), 'os', False)
    str_9979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'str', 'uname')
    # Processing the call keyword arguments (line 58)
    kwargs_9980 = {}
    # Getting the type of 'hasattr' (line 58)
    hasattr_9977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 33), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 58)
    hasattr_call_result_9981 = invoke(stypy.reporting.localization.Localization(__file__, 58, 33), hasattr_9977, *[os_9978, str_9979], **kwargs_9980)
    
    # Applying the 'not' unary operator (line 58)
    result_not__9982 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 29), 'not', hasattr_call_result_9981)
    
    # Applying the binary operator 'or' (line 58)
    result_or_keyword_9983 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), 'or', result_ne_9976, result_not__9982)
    
    # Testing the type of an if condition (line 58)
    if_condition_9984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_or_keyword_9983)
    # Assigning a type to the variable 'if_condition_9984' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_9984', if_condition_9984)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'sys' (line 61)
    sys_9985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'sys')
    # Obtaining the member 'platform' of a type (line 61)
    platform_9986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), sys_9985, 'platform')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', platform_9986)
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_9987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to uname(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_9990 = {}
    # Getting the type of 'os' (line 65)
    os_9988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 48), 'os', False)
    # Obtaining the member 'uname' of a type (line 65)
    uname_9989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 48), os_9988, 'uname')
    # Calling uname(args, kwargs) (line 65)
    uname_call_result_9991 = invoke(stypy.reporting.localization.Localization(__file__, 65, 48), uname_9989, *[], **kwargs_9990)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___9992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), uname_call_result_9991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_9993 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___9992, int_9987)
    
    # Assigning a type to the variable 'tuple_var_assignment_9880' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9880', subscript_call_result_9993)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_9994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to uname(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_9997 = {}
    # Getting the type of 'os' (line 65)
    os_9995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 48), 'os', False)
    # Obtaining the member 'uname' of a type (line 65)
    uname_9996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 48), os_9995, 'uname')
    # Calling uname(args, kwargs) (line 65)
    uname_call_result_9998 = invoke(stypy.reporting.localization.Localization(__file__, 65, 48), uname_9996, *[], **kwargs_9997)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___9999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), uname_call_result_9998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_10000 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___9999, int_9994)
    
    # Assigning a type to the variable 'tuple_var_assignment_9881' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9881', subscript_call_result_10000)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_10001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to uname(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_10004 = {}
    # Getting the type of 'os' (line 65)
    os_10002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 48), 'os', False)
    # Obtaining the member 'uname' of a type (line 65)
    uname_10003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 48), os_10002, 'uname')
    # Calling uname(args, kwargs) (line 65)
    uname_call_result_10005 = invoke(stypy.reporting.localization.Localization(__file__, 65, 48), uname_10003, *[], **kwargs_10004)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___10006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), uname_call_result_10005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_10007 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___10006, int_10001)
    
    # Assigning a type to the variable 'tuple_var_assignment_9882' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9882', subscript_call_result_10007)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_10008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to uname(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_10011 = {}
    # Getting the type of 'os' (line 65)
    os_10009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 48), 'os', False)
    # Obtaining the member 'uname' of a type (line 65)
    uname_10010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 48), os_10009, 'uname')
    # Calling uname(args, kwargs) (line 65)
    uname_call_result_10012 = invoke(stypy.reporting.localization.Localization(__file__, 65, 48), uname_10010, *[], **kwargs_10011)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___10013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), uname_call_result_10012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_10014 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___10013, int_10008)
    
    # Assigning a type to the variable 'tuple_var_assignment_9883' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9883', subscript_call_result_10014)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_10015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to uname(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_10018 = {}
    # Getting the type of 'os' (line 65)
    os_10016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 48), 'os', False)
    # Obtaining the member 'uname' of a type (line 65)
    uname_10017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 48), os_10016, 'uname')
    # Calling uname(args, kwargs) (line 65)
    uname_call_result_10019 = invoke(stypy.reporting.localization.Localization(__file__, 65, 48), uname_10017, *[], **kwargs_10018)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___10020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), uname_call_result_10019, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_10021 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___10020, int_10015)
    
    # Assigning a type to the variable 'tuple_var_assignment_9884' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9884', subscript_call_result_10021)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_9880' (line 65)
    tuple_var_assignment_9880_10022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9880')
    # Assigning a type to the variable 'osname' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 5), 'osname', tuple_var_assignment_9880_10022)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_9881' (line 65)
    tuple_var_assignment_9881_10023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9881')
    # Assigning a type to the variable 'host' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'host', tuple_var_assignment_9881_10023)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_9882' (line 65)
    tuple_var_assignment_9882_10024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9882')
    # Assigning a type to the variable 'release' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'release', tuple_var_assignment_9882_10024)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_9883' (line 65)
    tuple_var_assignment_9883_10025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9883')
    # Assigning a type to the variable 'version' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'version', tuple_var_assignment_9883_10025)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_9884' (line 65)
    tuple_var_assignment_9884_10026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_9884')
    # Assigning a type to the variable 'machine' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 37), 'machine', tuple_var_assignment_9884_10026)
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to lower(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'osname' (line 69)
    osname_10029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'osname', False)
    # Processing the call keyword arguments (line 69)
    kwargs_10030 = {}
    # Getting the type of 'string' (line 69)
    string_10027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'string', False)
    # Obtaining the member 'lower' of a type (line 69)
    lower_10028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), string_10027, 'lower')
    # Calling lower(args, kwargs) (line 69)
    lower_call_result_10031 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), lower_10028, *[osname_10029], **kwargs_10030)
    
    # Assigning a type to the variable 'osname' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'osname', lower_call_result_10031)
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to replace(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'osname' (line 70)
    osname_10034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'osname', False)
    str_10035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'str', '/')
    str_10036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 41), 'str', '')
    # Processing the call keyword arguments (line 70)
    kwargs_10037 = {}
    # Getting the type of 'string' (line 70)
    string_10032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 13), 'string', False)
    # Obtaining the member 'replace' of a type (line 70)
    replace_10033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 13), string_10032, 'replace')
    # Calling replace(args, kwargs) (line 70)
    replace_call_result_10038 = invoke(stypy.reporting.localization.Localization(__file__, 70, 13), replace_10033, *[osname_10034, str_10035, str_10036], **kwargs_10037)
    
    # Assigning a type to the variable 'osname' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'osname', replace_call_result_10038)
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to replace(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'machine' (line 71)
    machine_10041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'machine', False)
    str_10042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'str', ' ')
    str_10043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 43), 'str', '_')
    # Processing the call keyword arguments (line 71)
    kwargs_10044 = {}
    # Getting the type of 'string' (line 71)
    string_10039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'string', False)
    # Obtaining the member 'replace' of a type (line 71)
    replace_10040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 14), string_10039, 'replace')
    # Calling replace(args, kwargs) (line 71)
    replace_call_result_10045 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), replace_10040, *[machine_10041, str_10042, str_10043], **kwargs_10044)
    
    # Assigning a type to the variable 'machine' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'machine', replace_call_result_10045)
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to replace(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'machine' (line 72)
    machine_10048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'machine', False)
    str_10049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'str', '/')
    str_10050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'str', '-')
    # Processing the call keyword arguments (line 72)
    kwargs_10051 = {}
    # Getting the type of 'string' (line 72)
    string_10046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'string', False)
    # Obtaining the member 'replace' of a type (line 72)
    replace_10047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 14), string_10046, 'replace')
    # Calling replace(args, kwargs) (line 72)
    replace_call_result_10052 = invoke(stypy.reporting.localization.Localization(__file__, 72, 14), replace_10047, *[machine_10048, str_10049, str_10050], **kwargs_10051)
    
    # Assigning a type to the variable 'machine' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'machine', replace_call_result_10052)
    
    
    
    # Obtaining the type of the subscript
    int_10053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 15), 'int')
    slice_10054 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 7), None, int_10053, None)
    # Getting the type of 'osname' (line 74)
    osname_10055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'osname')
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___10056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 7), osname_10055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_10057 = invoke(stypy.reporting.localization.Localization(__file__, 74, 7), getitem___10056, slice_10054)
    
    str_10058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'str', 'linux')
    # Applying the binary operator '==' (line 74)
    result_eq_10059 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), '==', subscript_call_result_10057, str_10058)
    
    # Testing the type of an if condition (line 74)
    if_condition_10060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), result_eq_10059)
    # Assigning a type to the variable 'if_condition_10060' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_10060', if_condition_10060)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_10061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'str', '%s-%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_10062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'osname' (line 78)
    osname_10063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'osname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 27), tuple_10062, osname_10063)
    # Adding element type (line 78)
    # Getting the type of 'machine' (line 78)
    machine_10064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'machine')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 27), tuple_10062, machine_10064)
    
    # Applying the binary operator '%' (line 78)
    result_mod_10065 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '%', str_10061, tuple_10062)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', result_mod_10065)
    # SSA branch for the else part of an if statement (line 74)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_10066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'int')
    slice_10067 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 9), None, int_10066, None)
    # Getting the type of 'osname' (line 79)
    osname_10068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'osname')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___10069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 9), osname_10068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_10070 = invoke(stypy.reporting.localization.Localization(__file__, 79, 9), getitem___10069, slice_10067)
    
    str_10071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'str', 'sunos')
    # Applying the binary operator '==' (line 79)
    result_eq_10072 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 9), '==', subscript_call_result_10070, str_10071)
    
    # Testing the type of an if condition (line 79)
    if_condition_10073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 9), result_eq_10072)
    # Assigning a type to the variable 'if_condition_10073' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'if_condition_10073', if_condition_10073)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_10074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'int')
    # Getting the type of 'release' (line 80)
    release_10075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'release')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___10076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), release_10075, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_10077 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), getitem___10076, int_10074)
    
    str_10078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 25), 'str', '5')
    # Applying the binary operator '>=' (line 80)
    result_ge_10079 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), '>=', subscript_call_result_10077, str_10078)
    
    # Testing the type of an if condition (line 80)
    if_condition_10080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_ge_10079)
    # Assigning a type to the variable 'if_condition_10080' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_10080', if_condition_10080)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 81):
    
    # Assigning a Str to a Name (line 81):
    str_10081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'str', 'solaris')
    # Assigning a type to the variable 'osname' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'osname', str_10081)
    
    # Assigning a BinOp to a Name (line 82):
    
    # Assigning a BinOp to a Name (line 82):
    str_10082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'str', '%d.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_10083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    
    # Call to int(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Obtaining the type of the subscript
    int_10085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 45), 'int')
    # Getting the type of 'release' (line 82)
    release_10086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'release', False)
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___10087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 37), release_10086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_10088 = invoke(stypy.reporting.localization.Localization(__file__, 82, 37), getitem___10087, int_10085)
    
    # Processing the call keyword arguments (line 82)
    kwargs_10089 = {}
    # Getting the type of 'int' (line 82)
    int_10084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'int', False)
    # Calling int(args, kwargs) (line 82)
    int_call_result_10090 = invoke(stypy.reporting.localization.Localization(__file__, 82, 33), int_10084, *[subscript_call_result_10088], **kwargs_10089)
    
    int_10091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 51), 'int')
    # Applying the binary operator '-' (line 82)
    result_sub_10092 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 33), '-', int_call_result_10090, int_10091)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 33), tuple_10083, result_sub_10092)
    # Adding element type (line 82)
    
    # Obtaining the type of the subscript
    int_10093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 62), 'int')
    slice_10094 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 54), int_10093, None, None)
    # Getting the type of 'release' (line 82)
    release_10095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 54), 'release')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___10096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 54), release_10095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_10097 = invoke(stypy.reporting.localization.Localization(__file__, 82, 54), getitem___10096, slice_10094)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 33), tuple_10083, subscript_call_result_10097)
    
    # Applying the binary operator '%' (line 82)
    result_mod_10098 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 22), '%', str_10082, tuple_10083)
    
    # Assigning a type to the variable 'release' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'release', result_mod_10098)
    
    # Assigning a Dict to a Name (line 86):
    
    # Assigning a Dict to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'dict' (line 86)
    dict_10099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 86)
    # Adding element type (key, value) (line 86)
    int_10100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'int')
    str_10101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'str', '32bit')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), dict_10099, (int_10100, str_10101))
    # Adding element type (key, value) (line 86)
    long_10102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 43), 'long')
    str_10103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 63), 'str', '64bit')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), dict_10099, (long_10102, str_10103))
    
    # Assigning a type to the variable 'bitness' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'bitness', dict_10099)
    
    # Getting the type of 'machine' (line 87)
    machine_10104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'machine')
    str_10105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'str', '.%s')
    
    # Obtaining the type of the subscript
    # Getting the type of 'sys' (line 87)
    sys_10106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'sys')
    # Obtaining the member 'maxint' of a type (line 87)
    maxint_10107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 39), sys_10106, 'maxint')
    # Getting the type of 'bitness' (line 87)
    bitness_10108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'bitness')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___10109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 31), bitness_10108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_10110 = invoke(stypy.reporting.localization.Localization(__file__, 87, 31), getitem___10109, maxint_10107)
    
    # Applying the binary operator '%' (line 87)
    result_mod_10111 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 23), '%', str_10105, subscript_call_result_10110)
    
    # Applying the binary operator '+=' (line 87)
    result_iadd_10112 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), '+=', machine_10104, result_mod_10111)
    # Assigning a type to the variable 'machine' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'machine', result_iadd_10112)
    
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_10113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'int')
    slice_10114 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 9), None, int_10113, None)
    # Getting the type of 'osname' (line 89)
    osname_10115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'osname')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___10116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 9), osname_10115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_10117 = invoke(stypy.reporting.localization.Localization(__file__, 89, 9), getitem___10116, slice_10114)
    
    str_10118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'str', 'irix')
    # Applying the binary operator '==' (line 89)
    result_eq_10119 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 9), '==', subscript_call_result_10117, str_10118)
    
    # Testing the type of an if condition (line 89)
    if_condition_10120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 9), result_eq_10119)
    # Assigning a type to the variable 'if_condition_10120' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'if_condition_10120', if_condition_10120)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_10121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'str', '%s-%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_10122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'osname' (line 90)
    osname_10123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'osname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 26), tuple_10122, osname_10123)
    # Adding element type (line 90)
    # Getting the type of 'release' (line 90)
    release_10124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'release')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 26), tuple_10122, release_10124)
    
    # Applying the binary operator '%' (line 90)
    result_mod_10125 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '%', str_10121, tuple_10122)
    
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'stypy_return_type', result_mod_10125)
    # SSA branch for the else part of an if statement (line 89)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_10126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'int')
    slice_10127 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 91, 9), None, int_10126, None)
    # Getting the type of 'osname' (line 91)
    osname_10128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'osname')
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___10129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 9), osname_10128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_10130 = invoke(stypy.reporting.localization.Localization(__file__, 91, 9), getitem___10129, slice_10127)
    
    str_10131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 23), 'str', 'aix')
    # Applying the binary operator '==' (line 91)
    result_eq_10132 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 9), '==', subscript_call_result_10130, str_10131)
    
    # Testing the type of an if condition (line 91)
    if_condition_10133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 9), result_eq_10132)
    # Assigning a type to the variable 'if_condition_10133' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'if_condition_10133', if_condition_10133)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_10134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'str', '%s-%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 92)
    tuple_10135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 92)
    # Adding element type (line 92)
    # Getting the type of 'osname' (line 92)
    osname_10136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'osname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), tuple_10135, osname_10136)
    # Adding element type (line 92)
    # Getting the type of 'version' (line 92)
    version_10137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), tuple_10135, version_10137)
    # Adding element type (line 92)
    # Getting the type of 'release' (line 92)
    release_10138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'release')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), tuple_10135, release_10138)
    
    # Applying the binary operator '%' (line 92)
    result_mod_10139 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '%', str_10134, tuple_10135)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', result_mod_10139)
    # SSA branch for the else part of an if statement (line 91)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_10140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'int')
    slice_10141 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 9), None, int_10140, None)
    # Getting the type of 'osname' (line 93)
    osname_10142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 9), 'osname')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___10143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 9), osname_10142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_10144 = invoke(stypy.reporting.localization.Localization(__file__, 93, 9), getitem___10143, slice_10141)
    
    str_10145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'str', 'cygwin')
    # Applying the binary operator '==' (line 93)
    result_eq_10146 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 9), '==', subscript_call_result_10144, str_10145)
    
    # Testing the type of an if condition (line 93)
    if_condition_10147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 9), result_eq_10146)
    # Assigning a type to the variable 'if_condition_10147' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 9), 'if_condition_10147', if_condition_10147)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 94):
    
    # Assigning a Str to a Name (line 94):
    str_10148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'str', 'cygwin')
    # Assigning a type to the variable 'osname' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'osname', str_10148)
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to compile(...): (line 95)
    # Processing the call arguments (line 95)
    str_10151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'str', '[\\d.]+')
    # Processing the call keyword arguments (line 95)
    kwargs_10152 = {}
    # Getting the type of 're' (line 95)
    re_10149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 're', False)
    # Obtaining the member 'compile' of a type (line 95)
    compile_10150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 17), re_10149, 'compile')
    # Calling compile(args, kwargs) (line 95)
    compile_call_result_10153 = invoke(stypy.reporting.localization.Localization(__file__, 95, 17), compile_10150, *[str_10151], **kwargs_10152)
    
    # Assigning a type to the variable 'rel_re' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'rel_re', compile_call_result_10153)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to match(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'release' (line 96)
    release_10156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'release', False)
    # Processing the call keyword arguments (line 96)
    kwargs_10157 = {}
    # Getting the type of 'rel_re' (line 96)
    rel_re_10154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'rel_re', False)
    # Obtaining the member 'match' of a type (line 96)
    match_10155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), rel_re_10154, 'match')
    # Calling match(args, kwargs) (line 96)
    match_call_result_10158 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), match_10155, *[release_10156], **kwargs_10157)
    
    # Assigning a type to the variable 'm' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'm', match_call_result_10158)
    
    # Getting the type of 'm' (line 97)
    m_10159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'm')
    # Testing the type of an if condition (line 97)
    if_condition_10160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), m_10159)
    # Assigning a type to the variable 'if_condition_10160' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_10160', if_condition_10160)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to group(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_10163 = {}
    # Getting the type of 'm' (line 98)
    m_10161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'm', False)
    # Obtaining the member 'group' of a type (line 98)
    group_10162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 22), m_10161, 'group')
    # Calling group(args, kwargs) (line 98)
    group_call_result_10164 = invoke(stypy.reporting.localization.Localization(__file__, 98, 22), group_10162, *[], **kwargs_10163)
    
    # Assigning a type to the variable 'release' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'release', group_call_result_10164)
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 93)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_10165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'int')
    slice_10166 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 9), None, int_10165, None)
    # Getting the type of 'osname' (line 99)
    osname_10167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'osname')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___10168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 9), osname_10167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_10169 = invoke(stypy.reporting.localization.Localization(__file__, 99, 9), getitem___10168, slice_10166)
    
    str_10170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'str', 'darwin')
    # Applying the binary operator '==' (line 99)
    result_eq_10171 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 9), '==', subscript_call_result_10169, str_10170)
    
    # Testing the type of an if condition (line 99)
    if_condition_10172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 9), result_eq_10171)
    # Assigning a type to the variable 'if_condition_10172' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'if_condition_10172', if_condition_10172)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 100, 8))
    
    # Multiple import statement. import _osx_support (1/2) (line 100)
    import _osx_support

    import_module(stypy.reporting.localization.Localization(__file__, 100, 8), '_osx_support', _osx_support, module_type_store)
    # Multiple import statement. import distutils.sysconfig (2/2) (line 100)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_10173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.sysconfig')

    if (type(import_10173) is not StypyTypeError):

        if (import_10173 != 'pyd_module'):
            __import__(import_10173)
            sys_modules_10174 = sys.modules[import_10173]
            import_module(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.sysconfig', sys_modules_10174.module_type_store, module_type_store)
        else:
            import distutils.sysconfig

            import_module(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.sysconfig', distutils.sysconfig, module_type_store)

    else:
        # Assigning a type to the variable 'distutils.sysconfig' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.sysconfig', import_10173)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    
    # Assigning a Call to a Tuple (line 101):
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_10175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
    
    # Call to get_platform_osx(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to get_config_vars(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_10181 = {}
    # Getting the type of 'distutils' (line 102)
    distutils_10178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'distutils', False)
    # Obtaining the member 'sysconfig' of a type (line 102)
    sysconfig_10179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), distutils_10178, 'sysconfig')
    # Obtaining the member 'get_config_vars' of a type (line 102)
    get_config_vars_10180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), sysconfig_10179, 'get_config_vars')
    # Calling get_config_vars(args, kwargs) (line 102)
    get_config_vars_call_result_10182 = invoke(stypy.reporting.localization.Localization(__file__, 102, 40), get_config_vars_10180, *[], **kwargs_10181)
    
    # Getting the type of 'osname' (line 103)
    osname_10183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'osname', False)
    # Getting the type of 'release' (line 103)
    release_10184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'release', False)
    # Getting the type of 'machine' (line 103)
    machine_10185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 57), 'machine', False)
    # Processing the call keyword arguments (line 101)
    kwargs_10186 = {}
    # Getting the type of '_osx_support' (line 101)
    _osx_support_10176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), '_osx_support', False)
    # Obtaining the member 'get_platform_osx' of a type (line 101)
    get_platform_osx_10177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), _osx_support_10176, 'get_platform_osx')
    # Calling get_platform_osx(args, kwargs) (line 101)
    get_platform_osx_call_result_10187 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), get_platform_osx_10177, *[get_config_vars_call_result_10182, osname_10183, release_10184, machine_10185], **kwargs_10186)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___10188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), get_platform_osx_call_result_10187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_10189 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___10188, int_10175)
    
    # Assigning a type to the variable 'tuple_var_assignment_9885' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_9885', subscript_call_result_10189)
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_10190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
    
    # Call to get_platform_osx(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to get_config_vars(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_10196 = {}
    # Getting the type of 'distutils' (line 102)
    distutils_10193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'distutils', False)
    # Obtaining the member 'sysconfig' of a type (line 102)
    sysconfig_10194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), distutils_10193, 'sysconfig')
    # Obtaining the member 'get_config_vars' of a type (line 102)
    get_config_vars_10195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), sysconfig_10194, 'get_config_vars')
    # Calling get_config_vars(args, kwargs) (line 102)
    get_config_vars_call_result_10197 = invoke(stypy.reporting.localization.Localization(__file__, 102, 40), get_config_vars_10195, *[], **kwargs_10196)
    
    # Getting the type of 'osname' (line 103)
    osname_10198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'osname', False)
    # Getting the type of 'release' (line 103)
    release_10199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'release', False)
    # Getting the type of 'machine' (line 103)
    machine_10200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 57), 'machine', False)
    # Processing the call keyword arguments (line 101)
    kwargs_10201 = {}
    # Getting the type of '_osx_support' (line 101)
    _osx_support_10191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), '_osx_support', False)
    # Obtaining the member 'get_platform_osx' of a type (line 101)
    get_platform_osx_10192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), _osx_support_10191, 'get_platform_osx')
    # Calling get_platform_osx(args, kwargs) (line 101)
    get_platform_osx_call_result_10202 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), get_platform_osx_10192, *[get_config_vars_call_result_10197, osname_10198, release_10199, machine_10200], **kwargs_10201)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___10203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), get_platform_osx_call_result_10202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_10204 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___10203, int_10190)
    
    # Assigning a type to the variable 'tuple_var_assignment_9886' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_9886', subscript_call_result_10204)
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_10205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
    
    # Call to get_platform_osx(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to get_config_vars(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_10211 = {}
    # Getting the type of 'distutils' (line 102)
    distutils_10208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'distutils', False)
    # Obtaining the member 'sysconfig' of a type (line 102)
    sysconfig_10209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), distutils_10208, 'sysconfig')
    # Obtaining the member 'get_config_vars' of a type (line 102)
    get_config_vars_10210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), sysconfig_10209, 'get_config_vars')
    # Calling get_config_vars(args, kwargs) (line 102)
    get_config_vars_call_result_10212 = invoke(stypy.reporting.localization.Localization(__file__, 102, 40), get_config_vars_10210, *[], **kwargs_10211)
    
    # Getting the type of 'osname' (line 103)
    osname_10213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'osname', False)
    # Getting the type of 'release' (line 103)
    release_10214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'release', False)
    # Getting the type of 'machine' (line 103)
    machine_10215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 57), 'machine', False)
    # Processing the call keyword arguments (line 101)
    kwargs_10216 = {}
    # Getting the type of '_osx_support' (line 101)
    _osx_support_10206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), '_osx_support', False)
    # Obtaining the member 'get_platform_osx' of a type (line 101)
    get_platform_osx_10207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), _osx_support_10206, 'get_platform_osx')
    # Calling get_platform_osx(args, kwargs) (line 101)
    get_platform_osx_call_result_10217 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), get_platform_osx_10207, *[get_config_vars_call_result_10212, osname_10213, release_10214, machine_10215], **kwargs_10216)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___10218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), get_platform_osx_call_result_10217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_10219 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___10218, int_10205)
    
    # Assigning a type to the variable 'tuple_var_assignment_9887' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_9887', subscript_call_result_10219)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_9885' (line 101)
    tuple_var_assignment_9885_10220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_9885')
    # Assigning a type to the variable 'osname' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'osname', tuple_var_assignment_9885_10220)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_9886' (line 101)
    tuple_var_assignment_9886_10221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_9886')
    # Assigning a type to the variable 'release' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'release', tuple_var_assignment_9886_10221)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_9887' (line 101)
    tuple_var_assignment_9887_10222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_9887')
    # Assigning a type to the variable 'machine' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'machine', tuple_var_assignment_9887_10222)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    str_10223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 11), 'str', '%s-%s-%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_10224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    # Getting the type of 'osname' (line 105)
    osname_10225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'osname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), tuple_10224, osname_10225)
    # Adding element type (line 105)
    # Getting the type of 'release' (line 105)
    release_10226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'release')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), tuple_10224, release_10226)
    # Adding element type (line 105)
    # Getting the type of 'machine' (line 105)
    machine_10227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'machine')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), tuple_10224, machine_10227)
    
    # Applying the binary operator '%' (line 105)
    result_mod_10228 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), '%', str_10223, tuple_10224)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', result_mod_10228)
    
    # ################# End of 'get_platform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_platform' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_10229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_platform'
    return stypy_return_type_10229

# Assigning a type to the variable 'get_platform' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'get_platform', get_platform)

@norecursion
def convert_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'convert_path'
    module_type_store = module_type_store.open_function_context('convert_path', 110, 0, False)
    
    # Passed parameters checking function
    convert_path.stypy_localization = localization
    convert_path.stypy_type_of_self = None
    convert_path.stypy_type_store = module_type_store
    convert_path.stypy_function_name = 'convert_path'
    convert_path.stypy_param_names_list = ['pathname']
    convert_path.stypy_varargs_param_name = None
    convert_path.stypy_kwargs_param_name = None
    convert_path.stypy_call_defaults = defaults
    convert_path.stypy_call_varargs = varargs
    convert_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convert_path', ['pathname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convert_path', localization, ['pathname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convert_path(...)' code ##################

    str_10230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', "Return 'pathname' as a name that will work on the native filesystem,\n    i.e. split it on '/' and put it back together again using the current\n    directory separator.  Needed because filenames in the setup script are\n    always supplied in Unix style, and have to be converted to the local\n    convention before we can actually use them in the filesystem.  Raises\n    ValueError on non-Unix-ish systems if 'pathname' either starts or\n    ends with a slash.\n    ")
    
    
    # Getting the type of 'os' (line 119)
    os_10231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 7), 'os')
    # Obtaining the member 'sep' of a type (line 119)
    sep_10232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 7), os_10231, 'sep')
    str_10233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 17), 'str', '/')
    # Applying the binary operator '==' (line 119)
    result_eq_10234 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 7), '==', sep_10232, str_10233)
    
    # Testing the type of an if condition (line 119)
    if_condition_10235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 4), result_eq_10234)
    # Assigning a type to the variable 'if_condition_10235' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'if_condition_10235', if_condition_10235)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'pathname' (line 120)
    pathname_10236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'pathname')
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', pathname_10236)
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'pathname' (line 121)
    pathname_10237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'pathname')
    # Applying the 'not' unary operator (line 121)
    result_not__10238 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), 'not', pathname_10237)
    
    # Testing the type of an if condition (line 121)
    if_condition_10239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_not__10238)
    # Assigning a type to the variable 'if_condition_10239' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_10239', if_condition_10239)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'pathname' (line 122)
    pathname_10240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'pathname')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', pathname_10240)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_10241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'int')
    # Getting the type of 'pathname' (line 123)
    pathname_10242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'pathname')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___10243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 7), pathname_10242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_10244 = invoke(stypy.reporting.localization.Localization(__file__, 123, 7), getitem___10243, int_10241)
    
    str_10245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'str', '/')
    # Applying the binary operator '==' (line 123)
    result_eq_10246 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 7), '==', subscript_call_result_10244, str_10245)
    
    # Testing the type of an if condition (line 123)
    if_condition_10247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), result_eq_10246)
    # Assigning a type to the variable 'if_condition_10247' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_10247', if_condition_10247)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ValueError' (line 124)
    ValueError_10248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 124, 8), ValueError_10248, 'raise parameter', BaseException)
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_10249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 16), 'int')
    # Getting the type of 'pathname' (line 125)
    pathname_10250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'pathname')
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___10251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 7), pathname_10250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_10252 = invoke(stypy.reporting.localization.Localization(__file__, 125, 7), getitem___10251, int_10249)
    
    str_10253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'str', '/')
    # Applying the binary operator '==' (line 125)
    result_eq_10254 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 7), '==', subscript_call_result_10252, str_10253)
    
    # Testing the type of an if condition (line 125)
    if_condition_10255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 4), result_eq_10254)
    # Assigning a type to the variable 'if_condition_10255' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'if_condition_10255', if_condition_10255)
    # SSA begins for if statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ValueError' (line 126)
    ValueError_10256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 126, 8), ValueError_10256, 'raise parameter', BaseException)
    # SSA join for if statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to split(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'pathname' (line 128)
    pathname_10259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'pathname', False)
    str_10260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 35), 'str', '/')
    # Processing the call keyword arguments (line 128)
    kwargs_10261 = {}
    # Getting the type of 'string' (line 128)
    string_10257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'string', False)
    # Obtaining the member 'split' of a type (line 128)
    split_10258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), string_10257, 'split')
    # Calling split(args, kwargs) (line 128)
    split_call_result_10262 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), split_10258, *[pathname_10259, str_10260], **kwargs_10261)
    
    # Assigning a type to the variable 'paths' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'paths', split_call_result_10262)
    
    
    str_10263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 10), 'str', '.')
    # Getting the type of 'paths' (line 129)
    paths_10264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'paths')
    # Applying the binary operator 'in' (line 129)
    result_contains_10265 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), 'in', str_10263, paths_10264)
    
    # Testing the type of an if condition (line 129)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 4), result_contains_10265)
    # SSA begins for while statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to remove(...): (line 130)
    # Processing the call arguments (line 130)
    str_10268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'str', '.')
    # Processing the call keyword arguments (line 130)
    kwargs_10269 = {}
    # Getting the type of 'paths' (line 130)
    paths_10266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'paths', False)
    # Obtaining the member 'remove' of a type (line 130)
    remove_10267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), paths_10266, 'remove')
    # Calling remove(args, kwargs) (line 130)
    remove_call_result_10270 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), remove_10267, *[str_10268], **kwargs_10269)
    
    # SSA join for while statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'paths' (line 131)
    paths_10271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'paths')
    # Applying the 'not' unary operator (line 131)
    result_not__10272 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 7), 'not', paths_10271)
    
    # Testing the type of an if condition (line 131)
    if_condition_10273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 4), result_not__10272)
    # Assigning a type to the variable 'if_condition_10273' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'if_condition_10273', if_condition_10273)
    # SSA begins for if statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'os' (line 132)
    os_10274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'os')
    # Obtaining the member 'curdir' of a type (line 132)
    curdir_10275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), os_10274, 'curdir')
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', curdir_10275)
    # SSA join for if statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 133)
    # Getting the type of 'paths' (line 133)
    paths_10279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'paths', False)
    # Processing the call keyword arguments (line 133)
    kwargs_10280 = {}
    # Getting the type of 'os' (line 133)
    os_10276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 133)
    path_10277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 11), os_10276, 'path')
    # Obtaining the member 'join' of a type (line 133)
    join_10278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 11), path_10277, 'join')
    # Calling join(args, kwargs) (line 133)
    join_call_result_10281 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), join_10278, *[paths_10279], **kwargs_10280)
    
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', join_call_result_10281)
    
    # ################# End of 'convert_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convert_path' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_10282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convert_path'
    return stypy_return_type_10282

# Assigning a type to the variable 'convert_path' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'convert_path', convert_path)

@norecursion
def change_root(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'change_root'
    module_type_store = module_type_store.open_function_context('change_root', 138, 0, False)
    
    # Passed parameters checking function
    change_root.stypy_localization = localization
    change_root.stypy_type_of_self = None
    change_root.stypy_type_store = module_type_store
    change_root.stypy_function_name = 'change_root'
    change_root.stypy_param_names_list = ['new_root', 'pathname']
    change_root.stypy_varargs_param_name = None
    change_root.stypy_kwargs_param_name = None
    change_root.stypy_call_defaults = defaults
    change_root.stypy_call_varargs = varargs
    change_root.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'change_root', ['new_root', 'pathname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'change_root', localization, ['new_root', 'pathname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'change_root(...)' code ##################

    str_10283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', 'Return \'pathname\' with \'new_root\' prepended.  If \'pathname\' is\n    relative, this is equivalent to "os.path.join(new_root,pathname)".\n    Otherwise, it requires making \'pathname\' relative and then joining the\n    two, which is tricky on DOS/Windows and Mac OS.\n    ')
    
    
    # Getting the type of 'os' (line 144)
    os_10284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 7), 'os')
    # Obtaining the member 'name' of a type (line 144)
    name_10285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 7), os_10284, 'name')
    str_10286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 18), 'str', 'posix')
    # Applying the binary operator '==' (line 144)
    result_eq_10287 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 7), '==', name_10285, str_10286)
    
    # Testing the type of an if condition (line 144)
    if_condition_10288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 4), result_eq_10287)
    # Assigning a type to the variable 'if_condition_10288' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'if_condition_10288', if_condition_10288)
    # SSA begins for if statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to isabs(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'pathname' (line 145)
    pathname_10292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), 'pathname', False)
    # Processing the call keyword arguments (line 145)
    kwargs_10293 = {}
    # Getting the type of 'os' (line 145)
    os_10289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 145)
    path_10290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), os_10289, 'path')
    # Obtaining the member 'isabs' of a type (line 145)
    isabs_10291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), path_10290, 'isabs')
    # Calling isabs(args, kwargs) (line 145)
    isabs_call_result_10294 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), isabs_10291, *[pathname_10292], **kwargs_10293)
    
    # Applying the 'not' unary operator (line 145)
    result_not__10295 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), 'not', isabs_call_result_10294)
    
    # Testing the type of an if condition (line 145)
    if_condition_10296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_not__10295)
    # Assigning a type to the variable 'if_condition_10296' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_10296', if_condition_10296)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'new_root' (line 146)
    new_root_10300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'new_root', False)
    # Getting the type of 'pathname' (line 146)
    pathname_10301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'pathname', False)
    # Processing the call keyword arguments (line 146)
    kwargs_10302 = {}
    # Getting the type of 'os' (line 146)
    os_10297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 146)
    path_10298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), os_10297, 'path')
    # Obtaining the member 'join' of a type (line 146)
    join_10299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), path_10298, 'join')
    # Calling join(args, kwargs) (line 146)
    join_call_result_10303 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), join_10299, *[new_root_10300, pathname_10301], **kwargs_10302)
    
    # Assigning a type to the variable 'stypy_return_type' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', join_call_result_10303)
    # SSA branch for the else part of an if statement (line 145)
    module_type_store.open_ssa_branch('else')
    
    # Call to join(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'new_root' (line 148)
    new_root_10307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'new_root', False)
    
    # Obtaining the type of the subscript
    int_10308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 51), 'int')
    slice_10309 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 148, 42), int_10308, None, None)
    # Getting the type of 'pathname' (line 148)
    pathname_10310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 42), 'pathname', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___10311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 42), pathname_10310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_10312 = invoke(stypy.reporting.localization.Localization(__file__, 148, 42), getitem___10311, slice_10309)
    
    # Processing the call keyword arguments (line 148)
    kwargs_10313 = {}
    # Getting the type of 'os' (line 148)
    os_10304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 148)
    path_10305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), os_10304, 'path')
    # Obtaining the member 'join' of a type (line 148)
    join_10306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), path_10305, 'join')
    # Calling join(args, kwargs) (line 148)
    join_call_result_10314 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), join_10306, *[new_root_10307, subscript_call_result_10312], **kwargs_10313)
    
    # Assigning a type to the variable 'stypy_return_type' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'stypy_return_type', join_call_result_10314)
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 144)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 150)
    os_10315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 'os')
    # Obtaining the member 'name' of a type (line 150)
    name_10316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 9), os_10315, 'name')
    str_10317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'str', 'nt')
    # Applying the binary operator '==' (line 150)
    result_eq_10318 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 9), '==', name_10316, str_10317)
    
    # Testing the type of an if condition (line 150)
    if_condition_10319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 9), result_eq_10318)
    # Assigning a type to the variable 'if_condition_10319' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 'if_condition_10319', if_condition_10319)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 151):
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_10320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
    
    # Call to splitdrive(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'pathname' (line 151)
    pathname_10324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 43), 'pathname', False)
    # Processing the call keyword arguments (line 151)
    kwargs_10325 = {}
    # Getting the type of 'os' (line 151)
    os_10321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 151)
    path_10322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), os_10321, 'path')
    # Obtaining the member 'splitdrive' of a type (line 151)
    splitdrive_10323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), path_10322, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 151)
    splitdrive_call_result_10326 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), splitdrive_10323, *[pathname_10324], **kwargs_10325)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___10327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), splitdrive_call_result_10326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_10328 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___10327, int_10320)
    
    # Assigning a type to the variable 'tuple_var_assignment_9888' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_9888', subscript_call_result_10328)
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_10329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
    
    # Call to splitdrive(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'pathname' (line 151)
    pathname_10333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 43), 'pathname', False)
    # Processing the call keyword arguments (line 151)
    kwargs_10334 = {}
    # Getting the type of 'os' (line 151)
    os_10330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 151)
    path_10331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), os_10330, 'path')
    # Obtaining the member 'splitdrive' of a type (line 151)
    splitdrive_10332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), path_10331, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 151)
    splitdrive_call_result_10335 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), splitdrive_10332, *[pathname_10333], **kwargs_10334)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___10336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), splitdrive_call_result_10335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_10337 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___10336, int_10329)
    
    # Assigning a type to the variable 'tuple_var_assignment_9889' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_9889', subscript_call_result_10337)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_9888' (line 151)
    tuple_var_assignment_9888_10338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_9888')
    # Assigning a type to the variable 'drive' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 9), 'drive', tuple_var_assignment_9888_10338)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_9889' (line 151)
    tuple_var_assignment_9889_10339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_9889')
    # Assigning a type to the variable 'path' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'path', tuple_var_assignment_9889_10339)
    
    
    
    # Obtaining the type of the subscript
    int_10340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
    # Getting the type of 'path' (line 152)
    path_10341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'path')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___10342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), path_10341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_10343 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), getitem___10342, int_10340)
    
    str_10344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'str', '\\')
    # Applying the binary operator '==' (line 152)
    result_eq_10345 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), '==', subscript_call_result_10343, str_10344)
    
    # Testing the type of an if condition (line 152)
    if_condition_10346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), result_eq_10345)
    # Assigning a type to the variable 'if_condition_10346' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_10346', if_condition_10346)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 153):
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_10347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 24), 'int')
    slice_10348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 19), int_10347, None, None)
    # Getting the type of 'path' (line 153)
    path_10349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'path')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___10350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 19), path_10349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_10351 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), getitem___10350, slice_10348)
    
    # Assigning a type to the variable 'path' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'path', subscript_call_result_10351)
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'new_root' (line 154)
    new_root_10355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'new_root', False)
    # Getting the type of 'path' (line 154)
    path_10356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 38), 'path', False)
    # Processing the call keyword arguments (line 154)
    kwargs_10357 = {}
    # Getting the type of 'os' (line 154)
    os_10352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 154)
    path_10353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), os_10352, 'path')
    # Obtaining the member 'join' of a type (line 154)
    join_10354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), path_10353, 'join')
    # Calling join(args, kwargs) (line 154)
    join_call_result_10358 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), join_10354, *[new_root_10355, path_10356], **kwargs_10357)
    
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', join_call_result_10358)
    # SSA branch for the else part of an if statement (line 150)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 156)
    os_10359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 9), 'os')
    # Obtaining the member 'name' of a type (line 156)
    name_10360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), os_10359, 'name')
    str_10361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'str', 'os2')
    # Applying the binary operator '==' (line 156)
    result_eq_10362 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 9), '==', name_10360, str_10361)
    
    # Testing the type of an if condition (line 156)
    if_condition_10363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 9), result_eq_10362)
    # Assigning a type to the variable 'if_condition_10363' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 9), 'if_condition_10363', if_condition_10363)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 157):
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_10364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
    
    # Call to splitdrive(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'pathname' (line 157)
    pathname_10368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'pathname', False)
    # Processing the call keyword arguments (line 157)
    kwargs_10369 = {}
    # Getting the type of 'os' (line 157)
    os_10365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 157)
    path_10366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), os_10365, 'path')
    # Obtaining the member 'splitdrive' of a type (line 157)
    splitdrive_10367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), path_10366, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 157)
    splitdrive_call_result_10370 = invoke(stypy.reporting.localization.Localization(__file__, 157, 24), splitdrive_10367, *[pathname_10368], **kwargs_10369)
    
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___10371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), splitdrive_call_result_10370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_10372 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___10371, int_10364)
    
    # Assigning a type to the variable 'tuple_var_assignment_9890' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9890', subscript_call_result_10372)
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_10373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
    
    # Call to splitdrive(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'pathname' (line 157)
    pathname_10377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'pathname', False)
    # Processing the call keyword arguments (line 157)
    kwargs_10378 = {}
    # Getting the type of 'os' (line 157)
    os_10374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 157)
    path_10375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), os_10374, 'path')
    # Obtaining the member 'splitdrive' of a type (line 157)
    splitdrive_10376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), path_10375, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 157)
    splitdrive_call_result_10379 = invoke(stypy.reporting.localization.Localization(__file__, 157, 24), splitdrive_10376, *[pathname_10377], **kwargs_10378)
    
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___10380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), splitdrive_call_result_10379, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_10381 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___10380, int_10373)
    
    # Assigning a type to the variable 'tuple_var_assignment_9891' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9891', subscript_call_result_10381)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_var_assignment_9890' (line 157)
    tuple_var_assignment_9890_10382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9890')
    # Assigning a type to the variable 'drive' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 9), 'drive', tuple_var_assignment_9890_10382)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_var_assignment_9891' (line 157)
    tuple_var_assignment_9891_10383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9891')
    # Assigning a type to the variable 'path' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'path', tuple_var_assignment_9891_10383)
    
    
    
    # Obtaining the type of the subscript
    int_10384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 16), 'int')
    # Getting the type of 'path' (line 158)
    path_10385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'path')
    # Obtaining the member '__getitem__' of a type (line 158)
    getitem___10386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), path_10385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 158)
    subscript_call_result_10387 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), getitem___10386, int_10384)
    
    # Getting the type of 'os' (line 158)
    os_10388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'os')
    # Obtaining the member 'sep' of a type (line 158)
    sep_10389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), os_10388, 'sep')
    # Applying the binary operator '==' (line 158)
    result_eq_10390 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), '==', subscript_call_result_10387, sep_10389)
    
    # Testing the type of an if condition (line 158)
    if_condition_10391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_10390)
    # Assigning a type to the variable 'if_condition_10391' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_10391', if_condition_10391)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 159):
    
    # Assigning a Subscript to a Name (line 159):
    
    # Obtaining the type of the subscript
    int_10392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 24), 'int')
    slice_10393 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 19), int_10392, None, None)
    # Getting the type of 'path' (line 159)
    path_10394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'path')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___10395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 19), path_10394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_10396 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), getitem___10395, slice_10393)
    
    # Assigning a type to the variable 'path' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'path', subscript_call_result_10396)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'new_root' (line 160)
    new_root_10400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'new_root', False)
    # Getting the type of 'path' (line 160)
    path_10401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'path', False)
    # Processing the call keyword arguments (line 160)
    kwargs_10402 = {}
    # Getting the type of 'os' (line 160)
    os_10397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 160)
    path_10398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), os_10397, 'path')
    # Obtaining the member 'join' of a type (line 160)
    join_10399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), path_10398, 'join')
    # Calling join(args, kwargs) (line 160)
    join_call_result_10403 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), join_10399, *[new_root_10400, path_10401], **kwargs_10402)
    
    # Assigning a type to the variable 'stypy_return_type' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', join_call_result_10403)
    # SSA branch for the else part of an if statement (line 156)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'DistutilsPlatformError' (line 163)
    DistutilsPlatformError_10404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 14), 'DistutilsPlatformError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 163, 8), DistutilsPlatformError_10404, 'raise parameter', BaseException)
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 144)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'change_root(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'change_root' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_10405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10405)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'change_root'
    return stypy_return_type_10405

# Assigning a type to the variable 'change_root' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'change_root', change_root)

# Assigning a Num to a Name (line 167):

# Assigning a Num to a Name (line 167):
int_10406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 19), 'int')
# Assigning a type to the variable '_environ_checked' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), '_environ_checked', int_10406)

@norecursion
def check_environ(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_environ'
    module_type_store = module_type_store.open_function_context('check_environ', 168, 0, False)
    
    # Passed parameters checking function
    check_environ.stypy_localization = localization
    check_environ.stypy_type_of_self = None
    check_environ.stypy_type_store = module_type_store
    check_environ.stypy_function_name = 'check_environ'
    check_environ.stypy_param_names_list = []
    check_environ.stypy_varargs_param_name = None
    check_environ.stypy_kwargs_param_name = None
    check_environ.stypy_call_defaults = defaults
    check_environ.stypy_call_varargs = varargs
    check_environ.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_environ', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_environ', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_environ(...)' code ##################

    str_10407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, (-1)), 'str', "Ensure that 'os.environ' has all the environment variables we\n    guarantee that users can use in config files, command-line options,\n    etc.  Currently this includes:\n      HOME - user's home directory (Unix only)\n      PLAT - description of the current platform, including hardware\n             and OS (see 'get_platform()')\n    ")
    # Marking variables as global (line 176)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 176, 4), '_environ_checked')
    
    # Getting the type of '_environ_checked' (line 177)
    _environ_checked_10408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), '_environ_checked')
    # Testing the type of an if condition (line 177)
    if_condition_10409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), _environ_checked_10408)
    # Assigning a type to the variable 'if_condition_10409' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_10409', if_condition_10409)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'os' (line 180)
    os_10410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 7), 'os')
    # Obtaining the member 'name' of a type (line 180)
    name_10411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 7), os_10410, 'name')
    str_10412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 18), 'str', 'posix')
    # Applying the binary operator '==' (line 180)
    result_eq_10413 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), '==', name_10411, str_10412)
    
    
    str_10414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 30), 'str', 'HOME')
    # Getting the type of 'os' (line 180)
    os_10415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 44), 'os')
    # Obtaining the member 'environ' of a type (line 180)
    environ_10416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 44), os_10415, 'environ')
    # Applying the binary operator 'notin' (line 180)
    result_contains_10417 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 30), 'notin', str_10414, environ_10416)
    
    # Applying the binary operator 'and' (line 180)
    result_and_keyword_10418 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), 'and', result_eq_10413, result_contains_10417)
    
    # Testing the type of an if condition (line 180)
    if_condition_10419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 4), result_and_keyword_10418)
    # Assigning a type to the variable 'if_condition_10419' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'if_condition_10419', if_condition_10419)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 181, 8))
    
    # 'import pwd' statement (line 181)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_10420 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 181, 8), 'pwd')

    if (type(import_10420) is not StypyTypeError):

        if (import_10420 != 'pyd_module'):
            __import__(import_10420)
            sys_modules_10421 = sys.modules[import_10420]
            import_module(stypy.reporting.localization.Localization(__file__, 181, 8), 'pwd', sys_modules_10421.module_type_store, module_type_store)
        else:
            import pwd

            import_module(stypy.reporting.localization.Localization(__file__, 181, 8), 'pwd', pwd, module_type_store)

    else:
        # Assigning a type to the variable 'pwd' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'pwd', import_10420)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    
    # Assigning a Subscript to a Subscript (line 182):
    
    # Assigning a Subscript to a Subscript (line 182):
    
    # Obtaining the type of the subscript
    int_10422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 55), 'int')
    
    # Call to getpwuid(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to getuid(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_10427 = {}
    # Getting the type of 'os' (line 182)
    os_10425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 42), 'os', False)
    # Obtaining the member 'getuid' of a type (line 182)
    getuid_10426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 42), os_10425, 'getuid')
    # Calling getuid(args, kwargs) (line 182)
    getuid_call_result_10428 = invoke(stypy.reporting.localization.Localization(__file__, 182, 42), getuid_10426, *[], **kwargs_10427)
    
    # Processing the call keyword arguments (line 182)
    kwargs_10429 = {}
    # Getting the type of 'pwd' (line 182)
    pwd_10423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'pwd', False)
    # Obtaining the member 'getpwuid' of a type (line 182)
    getpwuid_10424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 29), pwd_10423, 'getpwuid')
    # Calling getpwuid(args, kwargs) (line 182)
    getpwuid_call_result_10430 = invoke(stypy.reporting.localization.Localization(__file__, 182, 29), getpwuid_10424, *[getuid_call_result_10428], **kwargs_10429)
    
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___10431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 29), getpwuid_call_result_10430, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_10432 = invoke(stypy.reporting.localization.Localization(__file__, 182, 29), getitem___10431, int_10422)
    
    # Getting the type of 'os' (line 182)
    os_10433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'os')
    # Obtaining the member 'environ' of a type (line 182)
    environ_10434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), os_10433, 'environ')
    str_10435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'str', 'HOME')
    # Storing an element on a container (line 182)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), environ_10434, (str_10435, subscript_call_result_10432))
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_10436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 7), 'str', 'PLAT')
    # Getting the type of 'os' (line 184)
    os_10437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'os')
    # Obtaining the member 'environ' of a type (line 184)
    environ_10438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 21), os_10437, 'environ')
    # Applying the binary operator 'notin' (line 184)
    result_contains_10439 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 7), 'notin', str_10436, environ_10438)
    
    # Testing the type of an if condition (line 184)
    if_condition_10440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 4), result_contains_10439)
    # Assigning a type to the variable 'if_condition_10440' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'if_condition_10440', if_condition_10440)
    # SSA begins for if statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 185):
    
    # Assigning a Call to a Subscript (line 185):
    
    # Call to get_platform(...): (line 185)
    # Processing the call keyword arguments (line 185)
    kwargs_10442 = {}
    # Getting the type of 'get_platform' (line 185)
    get_platform_10441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'get_platform', False)
    # Calling get_platform(args, kwargs) (line 185)
    get_platform_call_result_10443 = invoke(stypy.reporting.localization.Localization(__file__, 185, 29), get_platform_10441, *[], **kwargs_10442)
    
    # Getting the type of 'os' (line 185)
    os_10444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'os')
    # Obtaining the member 'environ' of a type (line 185)
    environ_10445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), os_10444, 'environ')
    str_10446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'str', 'PLAT')
    # Storing an element on a container (line 185)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 8), environ_10445, (str_10446, get_platform_call_result_10443))
    # SSA join for if statement (line 184)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 187):
    
    # Assigning a Num to a Name (line 187):
    int_10447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 23), 'int')
    # Assigning a type to the variable '_environ_checked' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), '_environ_checked', int_10447)
    
    # ################# End of 'check_environ(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_environ' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_10448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10448)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_environ'
    return stypy_return_type_10448

# Assigning a type to the variable 'check_environ' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'check_environ', check_environ)

@norecursion
def subst_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'subst_vars'
    module_type_store = module_type_store.open_function_context('subst_vars', 190, 0, False)
    
    # Passed parameters checking function
    subst_vars.stypy_localization = localization
    subst_vars.stypy_type_of_self = None
    subst_vars.stypy_type_store = module_type_store
    subst_vars.stypy_function_name = 'subst_vars'
    subst_vars.stypy_param_names_list = ['s', 'local_vars']
    subst_vars.stypy_varargs_param_name = None
    subst_vars.stypy_kwargs_param_name = None
    subst_vars.stypy_call_defaults = defaults
    subst_vars.stypy_call_varargs = varargs
    subst_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'subst_vars', ['s', 'local_vars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'subst_vars', localization, ['s', 'local_vars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'subst_vars(...)' code ##################

    str_10449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'str', "Perform shell/Perl-style variable substitution on 'string'.  Every\n    occurrence of '$' followed by a name is considered a variable, and\n    variable is substituted by the value found in the 'local_vars'\n    dictionary, or in 'os.environ' if it's not in 'local_vars'.\n    'os.environ' is first checked/augmented to guarantee that it contains\n    certain values: see 'check_environ()'.  Raise ValueError for any\n    variables not found in either 'local_vars' or 'os.environ'.\n    ")
    
    # Call to check_environ(...): (line 199)
    # Processing the call keyword arguments (line 199)
    kwargs_10451 = {}
    # Getting the type of 'check_environ' (line 199)
    check_environ_10450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'check_environ', False)
    # Calling check_environ(args, kwargs) (line 199)
    check_environ_call_result_10452 = invoke(stypy.reporting.localization.Localization(__file__, 199, 4), check_environ_10450, *[], **kwargs_10451)
    

    @norecursion
    def _subst(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'local_vars' (line 200)
        local_vars_10453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'local_vars')
        defaults = [local_vars_10453]
        # Create a new context for function '_subst'
        module_type_store = module_type_store.open_function_context('_subst', 200, 4, False)
        
        # Passed parameters checking function
        _subst.stypy_localization = localization
        _subst.stypy_type_of_self = None
        _subst.stypy_type_store = module_type_store
        _subst.stypy_function_name = '_subst'
        _subst.stypy_param_names_list = ['match', 'local_vars']
        _subst.stypy_varargs_param_name = None
        _subst.stypy_kwargs_param_name = None
        _subst.stypy_call_defaults = defaults
        _subst.stypy_call_varargs = varargs
        _subst.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_subst', ['match', 'local_vars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_subst', localization, ['match', 'local_vars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_subst(...)' code ##################

        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to group(...): (line 201)
        # Processing the call arguments (line 201)
        int_10456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 31), 'int')
        # Processing the call keyword arguments (line 201)
        kwargs_10457 = {}
        # Getting the type of 'match' (line 201)
        match_10454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'match', False)
        # Obtaining the member 'group' of a type (line 201)
        group_10455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 19), match_10454, 'group')
        # Calling group(args, kwargs) (line 201)
        group_call_result_10458 = invoke(stypy.reporting.localization.Localization(__file__, 201, 19), group_10455, *[int_10456], **kwargs_10457)
        
        # Assigning a type to the variable 'var_name' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'var_name', group_call_result_10458)
        
        
        # Getting the type of 'var_name' (line 202)
        var_name_10459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'var_name')
        # Getting the type of 'local_vars' (line 202)
        local_vars_10460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'local_vars')
        # Applying the binary operator 'in' (line 202)
        result_contains_10461 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 11), 'in', var_name_10459, local_vars_10460)
        
        # Testing the type of an if condition (line 202)
        if_condition_10462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_contains_10461)
        # Assigning a type to the variable 'if_condition_10462' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_10462', if_condition_10462)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to str(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        # Getting the type of 'var_name' (line 203)
        var_name_10464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), 'var_name', False)
        # Getting the type of 'local_vars' (line 203)
        local_vars_10465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'local_vars', False)
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___10466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), local_vars_10465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_10467 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), getitem___10466, var_name_10464)
        
        # Processing the call keyword arguments (line 203)
        kwargs_10468 = {}
        # Getting the type of 'str' (line 203)
        str_10463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'str', False)
        # Calling str(args, kwargs) (line 203)
        str_call_result_10469 = invoke(stypy.reporting.localization.Localization(__file__, 203, 19), str_10463, *[subscript_call_result_10467], **kwargs_10468)
        
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'stypy_return_type', str_call_result_10469)
        # SSA branch for the else part of an if statement (line 202)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining the type of the subscript
        # Getting the type of 'var_name' (line 205)
        var_name_10470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'var_name')
        # Getting the type of 'os' (line 205)
        os_10471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'os')
        # Obtaining the member 'environ' of a type (line 205)
        environ_10472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 19), os_10471, 'environ')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___10473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 19), environ_10472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_10474 = invoke(stypy.reporting.localization.Localization(__file__, 205, 19), getitem___10473, var_name_10470)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'stypy_return_type', subscript_call_result_10474)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_subst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_subst' in the type store
        # Getting the type of 'stypy_return_type' (line 200)
        stypy_return_type_10475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_subst'
        return stypy_return_type_10475

    # Assigning a type to the variable '_subst' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), '_subst', _subst)
    
    
    # SSA begins for try-except statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to sub(...): (line 208)
    # Processing the call arguments (line 208)
    str_10478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'str', '\\$([a-zA-Z_][a-zA-Z_0-9]*)')
    # Getting the type of '_subst' (line 208)
    _subst_10479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 53), '_subst', False)
    # Getting the type of 's' (line 208)
    s_10480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 61), 's', False)
    # Processing the call keyword arguments (line 208)
    kwargs_10481 = {}
    # Getting the type of 're' (line 208)
    re_10476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 're', False)
    # Obtaining the member 'sub' of a type (line 208)
    sub_10477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), re_10476, 'sub')
    # Calling sub(args, kwargs) (line 208)
    sub_call_result_10482 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), sub_10477, *[str_10478, _subst_10479, s_10480], **kwargs_10481)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', sub_call_result_10482)
    # SSA branch for the except part of a try statement (line 207)
    # SSA branch for the except 'KeyError' branch of a try statement (line 207)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'KeyError' (line 209)
    KeyError_10483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'KeyError')
    # Assigning a type to the variable 'var' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'var', KeyError_10483)
    # Getting the type of 'ValueError' (line 210)
    ValueError_10484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 210, 8), ValueError_10484, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'subst_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'subst_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_10485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10485)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'subst_vars'
    return stypy_return_type_10485

# Assigning a type to the variable 'subst_vars' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'subst_vars', subst_vars)

@norecursion
def grok_environment_error(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_10486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 40), 'str', 'error: ')
    defaults = [str_10486]
    # Create a new context for function 'grok_environment_error'
    module_type_store = module_type_store.open_function_context('grok_environment_error', 215, 0, False)
    
    # Passed parameters checking function
    grok_environment_error.stypy_localization = localization
    grok_environment_error.stypy_type_of_self = None
    grok_environment_error.stypy_type_store = module_type_store
    grok_environment_error.stypy_function_name = 'grok_environment_error'
    grok_environment_error.stypy_param_names_list = ['exc', 'prefix']
    grok_environment_error.stypy_varargs_param_name = None
    grok_environment_error.stypy_kwargs_param_name = None
    grok_environment_error.stypy_call_defaults = defaults
    grok_environment_error.stypy_call_varargs = varargs
    grok_environment_error.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'grok_environment_error', ['exc', 'prefix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'grok_environment_error', localization, ['exc', 'prefix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'grok_environment_error(...)' code ##################

    # Getting the type of 'prefix' (line 219)
    prefix_10487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'prefix')
    
    # Call to str(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'exc' (line 219)
    exc_10489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'exc', False)
    # Processing the call keyword arguments (line 219)
    kwargs_10490 = {}
    # Getting the type of 'str' (line 219)
    str_10488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'str', False)
    # Calling str(args, kwargs) (line 219)
    str_call_result_10491 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), str_10488, *[exc_10489], **kwargs_10490)
    
    # Applying the binary operator '+' (line 219)
    result_add_10492 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), '+', prefix_10487, str_call_result_10491)
    
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type', result_add_10492)
    
    # ################# End of 'grok_environment_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'grok_environment_error' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_10493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10493)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'grok_environment_error'
    return stypy_return_type_10493

# Assigning a type to the variable 'grok_environment_error' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'grok_environment_error', grok_environment_error)

# Multiple assignment of 3 elements.

# Assigning a Name to a Name (line 223):
# Getting the type of 'None' (line 223)
None_10494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'None')
# Assigning a type to the variable '_dquote_re' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), '_dquote_re', None_10494)

# Assigning a Name to a Name (line 223):
# Getting the type of '_dquote_re' (line 223)
_dquote_re_10495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), '_dquote_re')
# Assigning a type to the variable '_squote_re' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), '_squote_re', _dquote_re_10495)

# Assigning a Name to a Name (line 223):
# Getting the type of '_squote_re' (line 223)
_squote_re_10496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), '_squote_re')
# Assigning a type to the variable '_wordchars_re' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), '_wordchars_re', _squote_re_10496)

@norecursion
def _init_regex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_init_regex'
    module_type_store = module_type_store.open_function_context('_init_regex', 224, 0, False)
    
    # Passed parameters checking function
    _init_regex.stypy_localization = localization
    _init_regex.stypy_type_of_self = None
    _init_regex.stypy_type_store = module_type_store
    _init_regex.stypy_function_name = '_init_regex'
    _init_regex.stypy_param_names_list = []
    _init_regex.stypy_varargs_param_name = None
    _init_regex.stypy_kwargs_param_name = None
    _init_regex.stypy_call_defaults = defaults
    _init_regex.stypy_call_varargs = varargs
    _init_regex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_init_regex', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_init_regex', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_init_regex(...)' code ##################

    # Marking variables as global (line 225)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 225, 4), '_wordchars_re')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 225, 4), '_squote_re')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 225, 4), '_dquote_re')
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to compile(...): (line 226)
    # Processing the call arguments (line 226)
    str_10499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 31), 'str', '[^\\\\\\\'\\"%s ]*')
    # Getting the type of 'string' (line 226)
    string_10500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 50), 'string', False)
    # Obtaining the member 'whitespace' of a type (line 226)
    whitespace_10501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 50), string_10500, 'whitespace')
    # Applying the binary operator '%' (line 226)
    result_mod_10502 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 31), '%', str_10499, whitespace_10501)
    
    # Processing the call keyword arguments (line 226)
    kwargs_10503 = {}
    # Getting the type of 're' (line 226)
    re_10497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 're', False)
    # Obtaining the member 'compile' of a type (line 226)
    compile_10498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), re_10497, 'compile')
    # Calling compile(args, kwargs) (line 226)
    compile_call_result_10504 = invoke(stypy.reporting.localization.Localization(__file__, 226, 20), compile_10498, *[result_mod_10502], **kwargs_10503)
    
    # Assigning a type to the variable '_wordchars_re' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), '_wordchars_re', compile_call_result_10504)
    
    # Assigning a Call to a Name (line 227):
    
    # Assigning a Call to a Name (line 227):
    
    # Call to compile(...): (line 227)
    # Processing the call arguments (line 227)
    str_10507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 28), 'str', "'(?:[^'\\\\]|\\\\.)*'")
    # Processing the call keyword arguments (line 227)
    kwargs_10508 = {}
    # Getting the type of 're' (line 227)
    re_10505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 're', False)
    # Obtaining the member 'compile' of a type (line 227)
    compile_10506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 17), re_10505, 'compile')
    # Calling compile(args, kwargs) (line 227)
    compile_call_result_10509 = invoke(stypy.reporting.localization.Localization(__file__, 227, 17), compile_10506, *[str_10507], **kwargs_10508)
    
    # Assigning a type to the variable '_squote_re' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), '_squote_re', compile_call_result_10509)
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to compile(...): (line 228)
    # Processing the call arguments (line 228)
    str_10512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 28), 'str', '"(?:[^"\\\\]|\\\\.)*"')
    # Processing the call keyword arguments (line 228)
    kwargs_10513 = {}
    # Getting the type of 're' (line 228)
    re_10510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 're', False)
    # Obtaining the member 'compile' of a type (line 228)
    compile_10511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 17), re_10510, 'compile')
    # Calling compile(args, kwargs) (line 228)
    compile_call_result_10514 = invoke(stypy.reporting.localization.Localization(__file__, 228, 17), compile_10511, *[str_10512], **kwargs_10513)
    
    # Assigning a type to the variable '_dquote_re' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), '_dquote_re', compile_call_result_10514)
    
    # ################# End of '_init_regex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_init_regex' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_10515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10515)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_init_regex'
    return stypy_return_type_10515

# Assigning a type to the variable '_init_regex' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), '_init_regex', _init_regex)

@norecursion
def split_quoted(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'split_quoted'
    module_type_store = module_type_store.open_function_context('split_quoted', 230, 0, False)
    
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

    str_10516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, (-1)), 'str', 'Split a string up according to Unix shell-like rules for quotes and\n    backslashes.  In short: words are delimited by spaces, as long as those\n    spaces are not escaped by a backslash, or inside a quoted string.\n    Single and double quotes are equivalent, and the quote characters can\n    be backslash-escaped.  The backslash is stripped from any two-character\n    escape sequence, leaving only the escaped character.  The quote\n    characters are stripped from any quoted string.  Returns a list of\n    words.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 244)
    # Getting the type of '_wordchars_re' (line 244)
    _wordchars_re_10517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 7), '_wordchars_re')
    # Getting the type of 'None' (line 244)
    None_10518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'None')
    
    (may_be_10519, more_types_in_union_10520) = may_be_none(_wordchars_re_10517, None_10518)

    if may_be_10519:

        if more_types_in_union_10520:
            # Runtime conditional SSA (line 244)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _init_regex(...): (line 244)
        # Processing the call keyword arguments (line 244)
        kwargs_10522 = {}
        # Getting the type of '_init_regex' (line 244)
        _init_regex_10521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 30), '_init_regex', False)
        # Calling _init_regex(args, kwargs) (line 244)
        _init_regex_call_result_10523 = invoke(stypy.reporting.localization.Localization(__file__, 244, 30), _init_regex_10521, *[], **kwargs_10522)
        

        if more_types_in_union_10520:
            # SSA join for if statement (line 244)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to strip(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 's' (line 246)
    s_10526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 's', False)
    # Processing the call keyword arguments (line 246)
    kwargs_10527 = {}
    # Getting the type of 'string' (line 246)
    string_10524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'string', False)
    # Obtaining the member 'strip' of a type (line 246)
    strip_10525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), string_10524, 'strip')
    # Calling strip(args, kwargs) (line 246)
    strip_call_result_10528 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), strip_10525, *[s_10526], **kwargs_10527)
    
    # Assigning a type to the variable 's' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 's', strip_call_result_10528)
    
    # Assigning a List to a Name (line 247):
    
    # Assigning a List to a Name (line 247):
    
    # Obtaining an instance of the builtin type 'list' (line 247)
    list_10529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 247)
    
    # Assigning a type to the variable 'words' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'words', list_10529)
    
    # Assigning a Num to a Name (line 248):
    
    # Assigning a Num to a Name (line 248):
    int_10530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 10), 'int')
    # Assigning a type to the variable 'pos' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'pos', int_10530)
    
    # Getting the type of 's' (line 250)
    s_10531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 10), 's')
    # Testing the type of an if condition (line 250)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 4), s_10531)
    # SSA begins for while statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 251):
    
    # Assigning a Call to a Name (line 251):
    
    # Call to match(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 's' (line 251)
    s_10534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 32), 's', False)
    # Getting the type of 'pos' (line 251)
    pos_10535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 35), 'pos', False)
    # Processing the call keyword arguments (line 251)
    kwargs_10536 = {}
    # Getting the type of '_wordchars_re' (line 251)
    _wordchars_re_10532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), '_wordchars_re', False)
    # Obtaining the member 'match' of a type (line 251)
    match_10533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), _wordchars_re_10532, 'match')
    # Calling match(args, kwargs) (line 251)
    match_call_result_10537 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), match_10533, *[s_10534, pos_10535], **kwargs_10536)
    
    # Assigning a type to the variable 'm' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'm', match_call_result_10537)
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to end(...): (line 252)
    # Processing the call keyword arguments (line 252)
    kwargs_10540 = {}
    # Getting the type of 'm' (line 252)
    m_10538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 14), 'm', False)
    # Obtaining the member 'end' of a type (line 252)
    end_10539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 14), m_10538, 'end')
    # Calling end(args, kwargs) (line 252)
    end_call_result_10541 = invoke(stypy.reporting.localization.Localization(__file__, 252, 14), end_10539, *[], **kwargs_10540)
    
    # Assigning a type to the variable 'end' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'end', end_call_result_10541)
    
    
    # Getting the type of 'end' (line 253)
    end_10542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'end')
    
    # Call to len(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 's' (line 253)
    s_10544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 's', False)
    # Processing the call keyword arguments (line 253)
    kwargs_10545 = {}
    # Getting the type of 'len' (line 253)
    len_10543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'len', False)
    # Calling len(args, kwargs) (line 253)
    len_call_result_10546 = invoke(stypy.reporting.localization.Localization(__file__, 253, 18), len_10543, *[s_10544], **kwargs_10545)
    
    # Applying the binary operator '==' (line 253)
    result_eq_10547 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 11), '==', end_10542, len_call_result_10546)
    
    # Testing the type of an if condition (line 253)
    if_condition_10548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), result_eq_10547)
    # Assigning a type to the variable 'if_condition_10548' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_10548', if_condition_10548)
    # SSA begins for if statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 254)
    # Processing the call arguments (line 254)
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 254)
    end_10551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'end', False)
    slice_10552 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 254, 25), None, end_10551, None)
    # Getting the type of 's' (line 254)
    s_10553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 's', False)
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___10554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), s_10553, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_10555 = invoke(stypy.reporting.localization.Localization(__file__, 254, 25), getitem___10554, slice_10552)
    
    # Processing the call keyword arguments (line 254)
    kwargs_10556 = {}
    # Getting the type of 'words' (line 254)
    words_10549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'words', False)
    # Obtaining the member 'append' of a type (line 254)
    append_10550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), words_10549, 'append')
    # Calling append(args, kwargs) (line 254)
    append_call_result_10557 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), append_10550, *[subscript_call_result_10555], **kwargs_10556)
    
    # SSA join for if statement (line 253)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 257)
    end_10558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'end')
    # Getting the type of 's' (line 257)
    s_10559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 's')
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___10560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), s_10559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_10561 = invoke(stypy.reporting.localization.Localization(__file__, 257, 11), getitem___10560, end_10558)
    
    # Getting the type of 'string' (line 257)
    string_10562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 21), 'string')
    # Obtaining the member 'whitespace' of a type (line 257)
    whitespace_10563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 21), string_10562, 'whitespace')
    # Applying the binary operator 'in' (line 257)
    result_contains_10564 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), 'in', subscript_call_result_10561, whitespace_10563)
    
    # Testing the type of an if condition (line 257)
    if_condition_10565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_contains_10564)
    # Assigning a type to the variable 'if_condition_10565' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_10565', if_condition_10565)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 258)
    end_10568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'end', False)
    slice_10569 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 258, 25), None, end_10568, None)
    # Getting the type of 's' (line 258)
    s_10570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 25), 's', False)
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___10571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 25), s_10570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_10572 = invoke(stypy.reporting.localization.Localization(__file__, 258, 25), getitem___10571, slice_10569)
    
    # Processing the call keyword arguments (line 258)
    kwargs_10573 = {}
    # Getting the type of 'words' (line 258)
    words_10566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'words', False)
    # Obtaining the member 'append' of a type (line 258)
    append_10567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), words_10566, 'append')
    # Calling append(args, kwargs) (line 258)
    append_call_result_10574 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), append_10567, *[subscript_call_result_10572], **kwargs_10573)
    
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to lstrip(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 259)
    end_10577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 32), 'end', False)
    slice_10578 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 30), end_10577, None, None)
    # Getting the type of 's' (line 259)
    s_10579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 30), 's', False)
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___10580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 30), s_10579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_10581 = invoke(stypy.reporting.localization.Localization(__file__, 259, 30), getitem___10580, slice_10578)
    
    # Processing the call keyword arguments (line 259)
    kwargs_10582 = {}
    # Getting the type of 'string' (line 259)
    string_10575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'string', False)
    # Obtaining the member 'lstrip' of a type (line 259)
    lstrip_10576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 16), string_10575, 'lstrip')
    # Calling lstrip(args, kwargs) (line 259)
    lstrip_call_result_10583 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), lstrip_10576, *[subscript_call_result_10581], **kwargs_10582)
    
    # Assigning a type to the variable 's' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 's', lstrip_call_result_10583)
    
    # Assigning a Num to a Name (line 260):
    
    # Assigning a Num to a Name (line 260):
    int_10584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 18), 'int')
    # Assigning a type to the variable 'pos' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'pos', int_10584)
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 262)
    end_10585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'end')
    # Getting the type of 's' (line 262)
    s_10586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 's')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___10587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 13), s_10586, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_10588 = invoke(stypy.reporting.localization.Localization(__file__, 262, 13), getitem___10587, end_10585)
    
    str_10589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 23), 'str', '\\')
    # Applying the binary operator '==' (line 262)
    result_eq_10590 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 13), '==', subscript_call_result_10588, str_10589)
    
    # Testing the type of an if condition (line 262)
    if_condition_10591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 13), result_eq_10590)
    # Assigning a type to the variable 'if_condition_10591' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 'if_condition_10591', if_condition_10591)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 264):
    
    # Assigning a BinOp to a Name (line 264):
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 264)
    end_10592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'end')
    slice_10593 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 16), None, end_10592, None)
    # Getting the type of 's' (line 264)
    s_10594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 's')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___10595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), s_10594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_10596 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), getitem___10595, slice_10593)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 264)
    end_10597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 28), 'end')
    int_10598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 32), 'int')
    # Applying the binary operator '+' (line 264)
    result_add_10599 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 28), '+', end_10597, int_10598)
    
    slice_10600 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 26), result_add_10599, None, None)
    # Getting the type of 's' (line 264)
    s_10601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 26), 's')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___10602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 26), s_10601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_10603 = invoke(stypy.reporting.localization.Localization(__file__, 264, 26), getitem___10602, slice_10600)
    
    # Applying the binary operator '+' (line 264)
    result_add_10604 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 16), '+', subscript_call_result_10596, subscript_call_result_10603)
    
    # Assigning a type to the variable 's' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 's', result_add_10604)
    
    # Assigning a BinOp to a Name (line 265):
    
    # Assigning a BinOp to a Name (line 265):
    # Getting the type of 'end' (line 265)
    end_10605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'end')
    int_10606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 22), 'int')
    # Applying the binary operator '+' (line 265)
    result_add_10607 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 18), '+', end_10605, int_10606)
    
    # Assigning a type to the variable 'pos' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'pos', result_add_10607)
    # SSA branch for the else part of an if statement (line 262)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 268)
    end_10608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'end')
    # Getting the type of 's' (line 268)
    s_10609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 's')
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___10610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), s_10609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_10611 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), getitem___10610, end_10608)
    
    str_10612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 25), 'str', "'")
    # Applying the binary operator '==' (line 268)
    result_eq_10613 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 15), '==', subscript_call_result_10611, str_10612)
    
    # Testing the type of an if condition (line 268)
    if_condition_10614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 12), result_eq_10613)
    # Assigning a type to the variable 'if_condition_10614' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'if_condition_10614', if_condition_10614)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to match(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 's' (line 269)
    s_10617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 37), 's', False)
    # Getting the type of 'end' (line 269)
    end_10618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 40), 'end', False)
    # Processing the call keyword arguments (line 269)
    kwargs_10619 = {}
    # Getting the type of '_squote_re' (line 269)
    _squote_re_10615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), '_squote_re', False)
    # Obtaining the member 'match' of a type (line 269)
    match_10616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 20), _squote_re_10615, 'match')
    # Calling match(args, kwargs) (line 269)
    match_call_result_10620 = invoke(stypy.reporting.localization.Localization(__file__, 269, 20), match_10616, *[s_10617, end_10618], **kwargs_10619)
    
    # Assigning a type to the variable 'm' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'm', match_call_result_10620)
    # SSA branch for the else part of an if statement (line 268)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 270)
    end_10621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'end')
    # Getting the type of 's' (line 270)
    s_10622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 17), 's')
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___10623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 17), s_10622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_10624 = invoke(stypy.reporting.localization.Localization(__file__, 270, 17), getitem___10623, end_10621)
    
    str_10625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 27), 'str', '"')
    # Applying the binary operator '==' (line 270)
    result_eq_10626 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 17), '==', subscript_call_result_10624, str_10625)
    
    # Testing the type of an if condition (line 270)
    if_condition_10627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 17), result_eq_10626)
    # Assigning a type to the variable 'if_condition_10627' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 17), 'if_condition_10627', if_condition_10627)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to match(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 's' (line 271)
    s_10630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 37), 's', False)
    # Getting the type of 'end' (line 271)
    end_10631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 40), 'end', False)
    # Processing the call keyword arguments (line 271)
    kwargs_10632 = {}
    # Getting the type of '_dquote_re' (line 271)
    _dquote_re_10628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), '_dquote_re', False)
    # Obtaining the member 'match' of a type (line 271)
    match_10629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 20), _dquote_re_10628, 'match')
    # Calling match(args, kwargs) (line 271)
    match_call_result_10633 = invoke(stypy.reporting.localization.Localization(__file__, 271, 20), match_10629, *[s_10630, end_10631], **kwargs_10632)
    
    # Assigning a type to the variable 'm' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'm', match_call_result_10633)
    # SSA branch for the else part of an if statement (line 270)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'RuntimeError' (line 273)
    RuntimeError_10634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'RuntimeError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 273, 16), RuntimeError_10634, 'raise parameter', BaseException)
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 276)
    # Getting the type of 'm' (line 276)
    m_10635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'm')
    # Getting the type of 'None' (line 276)
    None_10636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'None')
    
    (may_be_10637, more_types_in_union_10638) = may_be_none(m_10635, None_10636)

    if may_be_10637:

        if more_types_in_union_10638:
            # Runtime conditional SSA (line 276)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'ValueError' (line 277)
        ValueError_10639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 277, 16), ValueError_10639, 'raise parameter', BaseException)

        if more_types_in_union_10638:
            # SSA join for if statement (line 276)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 280):
    
    # Assigning a Subscript to a Name (line 280):
    
    # Obtaining the type of the subscript
    int_10640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
    
    # Call to span(...): (line 280)
    # Processing the call keyword arguments (line 280)
    kwargs_10643 = {}
    # Getting the type of 'm' (line 280)
    m_10641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'm', False)
    # Obtaining the member 'span' of a type (line 280)
    span_10642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 25), m_10641, 'span')
    # Calling span(args, kwargs) (line 280)
    span_call_result_10644 = invoke(stypy.reporting.localization.Localization(__file__, 280, 25), span_10642, *[], **kwargs_10643)
    
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___10645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), span_call_result_10644, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_10646 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), getitem___10645, int_10640)
    
    # Assigning a type to the variable 'tuple_var_assignment_9892' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'tuple_var_assignment_9892', subscript_call_result_10646)
    
    # Assigning a Subscript to a Name (line 280):
    
    # Obtaining the type of the subscript
    int_10647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
    
    # Call to span(...): (line 280)
    # Processing the call keyword arguments (line 280)
    kwargs_10650 = {}
    # Getting the type of 'm' (line 280)
    m_10648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'm', False)
    # Obtaining the member 'span' of a type (line 280)
    span_10649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 25), m_10648, 'span')
    # Calling span(args, kwargs) (line 280)
    span_call_result_10651 = invoke(stypy.reporting.localization.Localization(__file__, 280, 25), span_10649, *[], **kwargs_10650)
    
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___10652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), span_call_result_10651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_10653 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), getitem___10652, int_10647)
    
    # Assigning a type to the variable 'tuple_var_assignment_9893' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'tuple_var_assignment_9893', subscript_call_result_10653)
    
    # Assigning a Name to a Name (line 280):
    # Getting the type of 'tuple_var_assignment_9892' (line 280)
    tuple_var_assignment_9892_10654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'tuple_var_assignment_9892')
    # Assigning a type to the variable 'beg' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'beg', tuple_var_assignment_9892_10654)
    
    # Assigning a Name to a Name (line 280):
    # Getting the type of 'tuple_var_assignment_9893' (line 280)
    tuple_var_assignment_9893_10655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'tuple_var_assignment_9893')
    # Assigning a type to the variable 'end' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'end', tuple_var_assignment_9893_10655)
    
    # Assigning a BinOp to a Name (line 281):
    
    # Assigning a BinOp to a Name (line 281):
    
    # Obtaining the type of the subscript
    # Getting the type of 'beg' (line 281)
    beg_10656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'beg')
    slice_10657 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 16), None, beg_10656, None)
    # Getting the type of 's' (line 281)
    s_10658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 's')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___10659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), s_10658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_10660 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), getitem___10659, slice_10657)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'beg' (line 281)
    beg_10661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'beg')
    int_10662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 32), 'int')
    # Applying the binary operator '+' (line 281)
    result_add_10663 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 28), '+', beg_10661, int_10662)
    
    # Getting the type of 'end' (line 281)
    end_10664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 34), 'end')
    int_10665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'int')
    # Applying the binary operator '-' (line 281)
    result_sub_10666 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 34), '-', end_10664, int_10665)
    
    slice_10667 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 26), result_add_10663, result_sub_10666, None)
    # Getting the type of 's' (line 281)
    s_10668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 's')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___10669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 26), s_10668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_10670 = invoke(stypy.reporting.localization.Localization(__file__, 281, 26), getitem___10669, slice_10667)
    
    # Applying the binary operator '+' (line 281)
    result_add_10671 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 16), '+', subscript_call_result_10660, subscript_call_result_10670)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 281)
    end_10672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 45), 'end')
    slice_10673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 43), end_10672, None, None)
    # Getting the type of 's' (line 281)
    s_10674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 43), 's')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___10675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 43), s_10674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_10676 = invoke(stypy.reporting.localization.Localization(__file__, 281, 43), getitem___10675, slice_10673)
    
    # Applying the binary operator '+' (line 281)
    result_add_10677 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 41), '+', result_add_10671, subscript_call_result_10676)
    
    # Assigning a type to the variable 's' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 's', result_add_10677)
    
    # Assigning a BinOp to a Name (line 282):
    
    # Assigning a BinOp to a Name (line 282):
    
    # Call to end(...): (line 282)
    # Processing the call keyword arguments (line 282)
    kwargs_10680 = {}
    # Getting the type of 'm' (line 282)
    m_10678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'm', False)
    # Obtaining the member 'end' of a type (line 282)
    end_10679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 18), m_10678, 'end')
    # Calling end(args, kwargs) (line 282)
    end_call_result_10681 = invoke(stypy.reporting.localization.Localization(__file__, 282, 18), end_10679, *[], **kwargs_10680)
    
    int_10682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 28), 'int')
    # Applying the binary operator '-' (line 282)
    result_sub_10683 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 18), '-', end_call_result_10681, int_10682)
    
    # Assigning a type to the variable 'pos' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'pos', result_sub_10683)
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'pos' (line 284)
    pos_10684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'pos')
    
    # Call to len(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 's' (line 284)
    s_10686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 22), 's', False)
    # Processing the call keyword arguments (line 284)
    kwargs_10687 = {}
    # Getting the type of 'len' (line 284)
    len_10685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'len', False)
    # Calling len(args, kwargs) (line 284)
    len_call_result_10688 = invoke(stypy.reporting.localization.Localization(__file__, 284, 18), len_10685, *[s_10686], **kwargs_10687)
    
    # Applying the binary operator '>=' (line 284)
    result_ge_10689 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), '>=', pos_10684, len_call_result_10688)
    
    # Testing the type of an if condition (line 284)
    if_condition_10690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_ge_10689)
    # Assigning a type to the variable 'if_condition_10690' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_10690', if_condition_10690)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 's' (line 285)
    s_10693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 's', False)
    # Processing the call keyword arguments (line 285)
    kwargs_10694 = {}
    # Getting the type of 'words' (line 285)
    words_10691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'words', False)
    # Obtaining the member 'append' of a type (line 285)
    append_10692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), words_10691, 'append')
    # Calling append(args, kwargs) (line 285)
    append_call_result_10695 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), append_10692, *[s_10693], **kwargs_10694)
    
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 250)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'words' (line 288)
    words_10696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'words')
    # Assigning a type to the variable 'stypy_return_type' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type', words_10696)
    
    # ################# End of 'split_quoted(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_quoted' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_10697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_quoted'
    return stypy_return_type_10697

# Assigning a type to the variable 'split_quoted' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'split_quoted', split_quoted)

@norecursion
def execute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 293)
    None_10698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 29), 'None')
    int_10699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 43), 'int')
    int_10700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 54), 'int')
    defaults = [None_10698, int_10699, int_10700]
    # Create a new context for function 'execute'
    module_type_store = module_type_store.open_function_context('execute', 293, 0, False)
    
    # Passed parameters checking function
    execute.stypy_localization = localization
    execute.stypy_type_of_self = None
    execute.stypy_type_store = module_type_store
    execute.stypy_function_name = 'execute'
    execute.stypy_param_names_list = ['func', 'args', 'msg', 'verbose', 'dry_run']
    execute.stypy_varargs_param_name = None
    execute.stypy_kwargs_param_name = None
    execute.stypy_call_defaults = defaults
    execute.stypy_call_varargs = varargs
    execute.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'execute', ['func', 'args', 'msg', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'execute', localization, ['func', 'args', 'msg', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'execute(...)' code ##################

    str_10701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'str', 'Perform some action that affects the outside world (eg.  by\n    writing to the filesystem).  Such actions are special because they\n    are disabled by the \'dry_run\' flag.  This method takes care of all\n    that bureaucracy for you; all you have to do is supply the\n    function to call and an argument tuple for it (to embody the\n    "external action" being performed), and an optional message to\n    print.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 302)
    # Getting the type of 'msg' (line 302)
    msg_10702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 7), 'msg')
    # Getting the type of 'None' (line 302)
    None_10703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'None')
    
    (may_be_10704, more_types_in_union_10705) = may_be_none(msg_10702, None_10703)

    if may_be_10704:

        if more_types_in_union_10705:
            # Runtime conditional SSA (line 302)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 303):
        
        # Assigning a BinOp to a Name (line 303):
        str_10706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 14), 'str', '%s%r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 303)
        tuple_10707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 303)
        # Adding element type (line 303)
        # Getting the type of 'func' (line 303)
        func_10708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'func')
        # Obtaining the member '__name__' of a type (line 303)
        name___10709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 24), func_10708, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 24), tuple_10707, name___10709)
        # Adding element type (line 303)
        # Getting the type of 'args' (line 303)
        args_10710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 39), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 24), tuple_10707, args_10710)
        
        # Applying the binary operator '%' (line 303)
        result_mod_10711 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 14), '%', str_10706, tuple_10707)
        
        # Assigning a type to the variable 'msg' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'msg', result_mod_10711)
        
        
        
        # Obtaining the type of the subscript
        int_10712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 15), 'int')
        slice_10713 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 11), int_10712, None, None)
        # Getting the type of 'msg' (line 304)
        msg_10714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'msg')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___10715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 11), msg_10714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_10716 = invoke(stypy.reporting.localization.Localization(__file__, 304, 11), getitem___10715, slice_10713)
        
        str_10717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 23), 'str', ',)')
        # Applying the binary operator '==' (line 304)
        result_eq_10718 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 11), '==', subscript_call_result_10716, str_10717)
        
        # Testing the type of an if condition (line 304)
        if_condition_10719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 8), result_eq_10718)
        # Assigning a type to the variable 'if_condition_10719' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'if_condition_10719', if_condition_10719)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 305):
        
        # Assigning a BinOp to a Name (line 305):
        
        # Obtaining the type of the subscript
        int_10720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 22), 'int')
        int_10721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 24), 'int')
        slice_10722 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 305, 18), int_10720, int_10721, None)
        # Getting the type of 'msg' (line 305)
        msg_10723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 18), 'msg')
        # Obtaining the member '__getitem__' of a type (line 305)
        getitem___10724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 18), msg_10723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 305)
        subscript_call_result_10725 = invoke(stypy.reporting.localization.Localization(__file__, 305, 18), getitem___10724, slice_10722)
        
        str_10726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 30), 'str', ')')
        # Applying the binary operator '+' (line 305)
        result_add_10727 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 18), '+', subscript_call_result_10725, str_10726)
        
        # Assigning a type to the variable 'msg' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'msg', result_add_10727)
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_10705:
            # SSA join for if statement (line 302)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to info(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'msg' (line 307)
    msg_10730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'msg', False)
    # Processing the call keyword arguments (line 307)
    kwargs_10731 = {}
    # Getting the type of 'log' (line 307)
    log_10728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 307)
    info_10729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), log_10728, 'info')
    # Calling info(args, kwargs) (line 307)
    info_call_result_10732 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), info_10729, *[msg_10730], **kwargs_10731)
    
    
    
    # Getting the type of 'dry_run' (line 308)
    dry_run_10733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'dry_run')
    # Applying the 'not' unary operator (line 308)
    result_not__10734 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 7), 'not', dry_run_10733)
    
    # Testing the type of an if condition (line 308)
    if_condition_10735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 4), result_not__10734)
    # Assigning a type to the variable 'if_condition_10735' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'if_condition_10735', if_condition_10735)
    # SSA begins for if statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to func(...): (line 309)
    # Getting the type of 'args' (line 309)
    args_10737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 14), 'args', False)
    # Processing the call keyword arguments (line 309)
    kwargs_10738 = {}
    # Getting the type of 'func' (line 309)
    func_10736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'func', False)
    # Calling func(args, kwargs) (line 309)
    func_call_result_10739 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), func_10736, *[args_10737], **kwargs_10738)
    
    # SSA join for if statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'execute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'execute' in the type store
    # Getting the type of 'stypy_return_type' (line 293)
    stypy_return_type_10740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10740)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'execute'
    return stypy_return_type_10740

# Assigning a type to the variable 'execute' (line 293)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'execute', execute)

@norecursion
def strtobool(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'strtobool'
    module_type_store = module_type_store.open_function_context('strtobool', 312, 0, False)
    
    # Passed parameters checking function
    strtobool.stypy_localization = localization
    strtobool.stypy_type_of_self = None
    strtobool.stypy_type_store = module_type_store
    strtobool.stypy_function_name = 'strtobool'
    strtobool.stypy_param_names_list = ['val']
    strtobool.stypy_varargs_param_name = None
    strtobool.stypy_kwargs_param_name = None
    strtobool.stypy_call_defaults = defaults
    strtobool.stypy_call_varargs = varargs
    strtobool.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'strtobool', ['val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'strtobool', localization, ['val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'strtobool(...)' code ##################

    str_10741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, (-1)), 'str', "Convert a string representation of truth to true (1) or false (0).\n\n    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values\n    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if\n    'val' is anything else.\n    ")
    
    # Assigning a Call to a Name (line 319):
    
    # Assigning a Call to a Name (line 319):
    
    # Call to lower(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'val' (line 319)
    val_10744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 23), 'val', False)
    # Processing the call keyword arguments (line 319)
    kwargs_10745 = {}
    # Getting the type of 'string' (line 319)
    string_10742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 10), 'string', False)
    # Obtaining the member 'lower' of a type (line 319)
    lower_10743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 10), string_10742, 'lower')
    # Calling lower(args, kwargs) (line 319)
    lower_call_result_10746 = invoke(stypy.reporting.localization.Localization(__file__, 319, 10), lower_10743, *[val_10744], **kwargs_10745)
    
    # Assigning a type to the variable 'val' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'val', lower_call_result_10746)
    
    
    # Getting the type of 'val' (line 320)
    val_10747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 7), 'val')
    
    # Obtaining an instance of the builtin type 'tuple' (line 320)
    tuple_10748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 320)
    # Adding element type (line 320)
    str_10749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 15), 'str', 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_10748, str_10749)
    # Adding element type (line 320)
    str_10750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 20), 'str', 'yes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_10748, str_10750)
    # Adding element type (line 320)
    str_10751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 27), 'str', 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_10748, str_10751)
    # Adding element type (line 320)
    str_10752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 32), 'str', 'true')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_10748, str_10752)
    # Adding element type (line 320)
    str_10753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 40), 'str', 'on')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_10748, str_10753)
    # Adding element type (line 320)
    str_10754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 46), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), tuple_10748, str_10754)
    
    # Applying the binary operator 'in' (line 320)
    result_contains_10755 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 7), 'in', val_10747, tuple_10748)
    
    # Testing the type of an if condition (line 320)
    if_condition_10756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 4), result_contains_10755)
    # Assigning a type to the variable 'if_condition_10756' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'if_condition_10756', if_condition_10756)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_10757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'stypy_return_type', int_10757)
    # SSA branch for the else part of an if statement (line 320)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'val' (line 322)
    val_10758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 9), 'val')
    
    # Obtaining an instance of the builtin type 'tuple' (line 322)
    tuple_10759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 322)
    # Adding element type (line 322)
    str_10760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 17), 'str', 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 17), tuple_10759, str_10760)
    # Adding element type (line 322)
    str_10761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 22), 'str', 'no')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 17), tuple_10759, str_10761)
    # Adding element type (line 322)
    str_10762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 28), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 17), tuple_10759, str_10762)
    # Adding element type (line 322)
    str_10763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 33), 'str', 'false')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 17), tuple_10759, str_10763)
    # Adding element type (line 322)
    str_10764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 42), 'str', 'off')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 17), tuple_10759, str_10764)
    # Adding element type (line 322)
    str_10765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 49), 'str', '0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 17), tuple_10759, str_10765)
    
    # Applying the binary operator 'in' (line 322)
    result_contains_10766 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 9), 'in', val_10758, tuple_10759)
    
    # Testing the type of an if condition (line 322)
    if_condition_10767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 9), result_contains_10766)
    # Assigning a type to the variable 'if_condition_10767' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 9), 'if_condition_10767', if_condition_10767)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_10768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'stypy_return_type', int_10768)
    # SSA branch for the else part of an if statement (line 322)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'ValueError' (line 325)
    ValueError_10769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 325, 8), ValueError_10769, 'raise parameter', BaseException)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'strtobool(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'strtobool' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_10770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'strtobool'
    return stypy_return_type_10770

# Assigning a type to the variable 'strtobool' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'strtobool', strtobool)

@norecursion
def byte_compile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_10771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 27), 'int')
    int_10772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 36), 'int')
    # Getting the type of 'None' (line 330)
    None_10773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 25), 'None')
    # Getting the type of 'None' (line 330)
    None_10774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 40), 'None')
    int_10775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 26), 'int')
    int_10776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 37), 'int')
    # Getting the type of 'None' (line 332)
    None_10777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'None')
    defaults = [int_10771, int_10772, None_10773, None_10774, int_10775, int_10776, None_10777]
    # Create a new context for function 'byte_compile'
    module_type_store = module_type_store.open_function_context('byte_compile', 328, 0, False)
    
    # Passed parameters checking function
    byte_compile.stypy_localization = localization
    byte_compile.stypy_type_of_self = None
    byte_compile.stypy_type_store = module_type_store
    byte_compile.stypy_function_name = 'byte_compile'
    byte_compile.stypy_param_names_list = ['py_files', 'optimize', 'force', 'prefix', 'base_dir', 'verbose', 'dry_run', 'direct']
    byte_compile.stypy_varargs_param_name = None
    byte_compile.stypy_kwargs_param_name = None
    byte_compile.stypy_call_defaults = defaults
    byte_compile.stypy_call_varargs = varargs
    byte_compile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'byte_compile', ['py_files', 'optimize', 'force', 'prefix', 'base_dir', 'verbose', 'dry_run', 'direct'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'byte_compile', localization, ['py_files', 'optimize', 'force', 'prefix', 'base_dir', 'verbose', 'dry_run', 'direct'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'byte_compile(...)' code ##################

    str_10778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, (-1)), 'str', 'Byte-compile a collection of Python source files to either .pyc\n    or .pyo files in the same directory.  \'py_files\' is a list of files\n    to compile; any files that don\'t end in ".py" are silently skipped.\n    \'optimize\' must be one of the following:\n      0 - don\'t optimize (generate .pyc)\n      1 - normal optimization (like "python -O")\n      2 - extra optimization (like "python -OO")\n    If \'force\' is true, all files are recompiled regardless of\n    timestamps.\n\n    The source filename encoded in each bytecode file defaults to the\n    filenames listed in \'py_files\'; you can modify these with \'prefix\' and\n    \'basedir\'.  \'prefix\' is a string that will be stripped off of each\n    source filename, and \'base_dir\' is a directory name that will be\n    prepended (after \'prefix\' is stripped).  You can supply either or both\n    (or neither) of \'prefix\' and \'base_dir\', as you wish.\n\n    If \'dry_run\' is true, doesn\'t actually do anything that would\n    affect the filesystem.\n\n    Byte-compilation is either done directly in this interpreter process\n    with the standard py_compile module, or indirectly by writing a\n    temporary script and executing it.  Normally, you should let\n    \'byte_compile()\' figure out to use direct compilation or not (see\n    the source for details).  The \'direct\' flag is used by the script\n    generated in indirect mode; unless you know what you\'re doing, leave\n    it set to None.\n    ')
    
    # Getting the type of 'sys' (line 362)
    sys_10779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 7), 'sys')
    # Obtaining the member 'dont_write_bytecode' of a type (line 362)
    dont_write_bytecode_10780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 7), sys_10779, 'dont_write_bytecode')
    # Testing the type of an if condition (line 362)
    if_condition_10781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 4), dont_write_bytecode_10780)
    # Assigning a type to the variable 'if_condition_10781' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'if_condition_10781', if_condition_10781)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsByteCompileError(...): (line 363)
    # Processing the call arguments (line 363)
    str_10783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 40), 'str', 'byte-compiling is disabled.')
    # Processing the call keyword arguments (line 363)
    kwargs_10784 = {}
    # Getting the type of 'DistutilsByteCompileError' (line 363)
    DistutilsByteCompileError_10782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'DistutilsByteCompileError', False)
    # Calling DistutilsByteCompileError(args, kwargs) (line 363)
    DistutilsByteCompileError_call_result_10785 = invoke(stypy.reporting.localization.Localization(__file__, 363, 14), DistutilsByteCompileError_10782, *[str_10783], **kwargs_10784)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 363, 8), DistutilsByteCompileError_call_result_10785, 'raise parameter', BaseException)
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 375)
    # Getting the type of 'direct' (line 375)
    direct_10786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 7), 'direct')
    # Getting the type of 'None' (line 375)
    None_10787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 17), 'None')
    
    (may_be_10788, more_types_in_union_10789) = may_be_none(direct_10786, None_10787)

    if may_be_10788:

        if more_types_in_union_10789:
            # Runtime conditional SSA (line 375)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BoolOp to a Name (line 376):
        
        # Assigning a BoolOp to a Name (line 376):
        
        # Evaluating a boolean operation
        # Getting the type of '__debug__' (line 376)
        debug___10790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), '__debug__')
        
        # Getting the type of 'optimize' (line 376)
        optimize_10791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 32), 'optimize')
        int_10792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 44), 'int')
        # Applying the binary operator '==' (line 376)
        result_eq_10793 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 32), '==', optimize_10791, int_10792)
        
        # Applying the binary operator 'and' (line 376)
        result_and_keyword_10794 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 18), 'and', debug___10790, result_eq_10793)
        
        # Assigning a type to the variable 'direct' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'direct', result_and_keyword_10794)

        if more_types_in_union_10789:
            # SSA join for if statement (line 375)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'direct' (line 380)
    direct_10795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 11), 'direct')
    # Applying the 'not' unary operator (line 380)
    result_not__10796 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 7), 'not', direct_10795)
    
    # Testing the type of an if condition (line 380)
    if_condition_10797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 4), result_not__10796)
    # Assigning a type to the variable 'if_condition_10797' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'if_condition_10797', if_condition_10797)
    # SSA begins for if statement (line 380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 382, 12))
    
    # 'from tempfile import mkstemp' statement (line 382)
    try:
        from tempfile import mkstemp

    except:
        mkstemp = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 382, 12), 'tempfile', None, module_type_store, ['mkstemp'], [mkstemp])
    
    
    # Assigning a Call to a Tuple (line 383):
    
    # Assigning a Subscript to a Name (line 383):
    
    # Obtaining the type of the subscript
    int_10798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 12), 'int')
    
    # Call to mkstemp(...): (line 383)
    # Processing the call arguments (line 383)
    str_10800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 47), 'str', '.py')
    # Processing the call keyword arguments (line 383)
    kwargs_10801 = {}
    # Getting the type of 'mkstemp' (line 383)
    mkstemp_10799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 39), 'mkstemp', False)
    # Calling mkstemp(args, kwargs) (line 383)
    mkstemp_call_result_10802 = invoke(stypy.reporting.localization.Localization(__file__, 383, 39), mkstemp_10799, *[str_10800], **kwargs_10801)
    
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___10803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), mkstemp_call_result_10802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_10804 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), getitem___10803, int_10798)
    
    # Assigning a type to the variable 'tuple_var_assignment_9894' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'tuple_var_assignment_9894', subscript_call_result_10804)
    
    # Assigning a Subscript to a Name (line 383):
    
    # Obtaining the type of the subscript
    int_10805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 12), 'int')
    
    # Call to mkstemp(...): (line 383)
    # Processing the call arguments (line 383)
    str_10807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 47), 'str', '.py')
    # Processing the call keyword arguments (line 383)
    kwargs_10808 = {}
    # Getting the type of 'mkstemp' (line 383)
    mkstemp_10806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 39), 'mkstemp', False)
    # Calling mkstemp(args, kwargs) (line 383)
    mkstemp_call_result_10809 = invoke(stypy.reporting.localization.Localization(__file__, 383, 39), mkstemp_10806, *[str_10807], **kwargs_10808)
    
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___10810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), mkstemp_call_result_10809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_10811 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), getitem___10810, int_10805)
    
    # Assigning a type to the variable 'tuple_var_assignment_9895' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'tuple_var_assignment_9895', subscript_call_result_10811)
    
    # Assigning a Name to a Name (line 383):
    # Getting the type of 'tuple_var_assignment_9894' (line 383)
    tuple_var_assignment_9894_10812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'tuple_var_assignment_9894')
    # Assigning a type to the variable 'script_fd' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'script_fd', tuple_var_assignment_9894_10812)
    
    # Assigning a Name to a Name (line 383):
    # Getting the type of 'tuple_var_assignment_9895' (line 383)
    tuple_var_assignment_9895_10813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'tuple_var_assignment_9895')
    # Assigning a type to the variable 'script_name' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'script_name', tuple_var_assignment_9895_10813)
    # SSA branch for the except part of a try statement (line 381)
    # SSA branch for the except 'ImportError' branch of a try statement (line 381)
    module_type_store.open_ssa_branch('except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 385, 12))
    
    # 'from tempfile import mktemp' statement (line 385)
    try:
        from tempfile import mktemp

    except:
        mktemp = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 385, 12), 'tempfile', None, module_type_store, ['mktemp'], [mktemp])
    
    
    # Assigning a Tuple to a Tuple (line 386):
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'None' (line 386)
    None_10814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 39), 'None')
    # Assigning a type to the variable 'tuple_assignment_9896' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_assignment_9896', None_10814)
    
    # Assigning a Call to a Name (line 386):
    
    # Call to mktemp(...): (line 386)
    # Processing the call arguments (line 386)
    str_10816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 52), 'str', '.py')
    # Processing the call keyword arguments (line 386)
    kwargs_10817 = {}
    # Getting the type of 'mktemp' (line 386)
    mktemp_10815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 45), 'mktemp', False)
    # Calling mktemp(args, kwargs) (line 386)
    mktemp_call_result_10818 = invoke(stypy.reporting.localization.Localization(__file__, 386, 45), mktemp_10815, *[str_10816], **kwargs_10817)
    
    # Assigning a type to the variable 'tuple_assignment_9897' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_assignment_9897', mktemp_call_result_10818)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_assignment_9896' (line 386)
    tuple_assignment_9896_10819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_assignment_9896')
    # Assigning a type to the variable 'script_fd' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'script_fd', tuple_assignment_9896_10819)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_assignment_9897' (line 386)
    tuple_assignment_9897_10820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_assignment_9897')
    # Assigning a type to the variable 'script_name' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'script_name', tuple_assignment_9897_10820)
    # SSA join for try-except statement (line 381)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 387)
    # Processing the call arguments (line 387)
    str_10823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 17), 'str', "writing byte-compilation script '%s'")
    # Getting the type of 'script_name' (line 387)
    script_name_10824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'script_name', False)
    # Processing the call keyword arguments (line 387)
    kwargs_10825 = {}
    # Getting the type of 'log' (line 387)
    log_10821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'log', False)
    # Obtaining the member 'info' of a type (line 387)
    info_10822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), log_10821, 'info')
    # Calling info(args, kwargs) (line 387)
    info_call_result_10826 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), info_10822, *[str_10823, script_name_10824], **kwargs_10825)
    
    
    
    # Getting the type of 'dry_run' (line 388)
    dry_run_10827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'dry_run')
    # Applying the 'not' unary operator (line 388)
    result_not__10828 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 11), 'not', dry_run_10827)
    
    # Testing the type of an if condition (line 388)
    if_condition_10829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 8), result_not__10828)
    # Assigning a type to the variable 'if_condition_10829' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'if_condition_10829', if_condition_10829)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 389)
    # Getting the type of 'script_fd' (line 389)
    script_fd_10830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'script_fd')
    # Getting the type of 'None' (line 389)
    None_10831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'None')
    
    (may_be_10832, more_types_in_union_10833) = may_not_be_none(script_fd_10830, None_10831)

    if may_be_10832:

        if more_types_in_union_10833:
            # Runtime conditional SSA (line 389)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 390):
        
        # Assigning a Call to a Name (line 390):
        
        # Call to fdopen(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'script_fd' (line 390)
        script_fd_10836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 35), 'script_fd', False)
        str_10837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 46), 'str', 'w')
        # Processing the call keyword arguments (line 390)
        kwargs_10838 = {}
        # Getting the type of 'os' (line 390)
        os_10834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 25), 'os', False)
        # Obtaining the member 'fdopen' of a type (line 390)
        fdopen_10835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 25), os_10834, 'fdopen')
        # Calling fdopen(args, kwargs) (line 390)
        fdopen_call_result_10839 = invoke(stypy.reporting.localization.Localization(__file__, 390, 25), fdopen_10835, *[script_fd_10836, str_10837], **kwargs_10838)
        
        # Assigning a type to the variable 'script' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'script', fdopen_call_result_10839)

        if more_types_in_union_10833:
            # Runtime conditional SSA for else branch (line 389)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_10832) or more_types_in_union_10833):
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to open(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'script_name' (line 392)
        script_name_10841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 30), 'script_name', False)
        str_10842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 43), 'str', 'w')
        # Processing the call keyword arguments (line 392)
        kwargs_10843 = {}
        # Getting the type of 'open' (line 392)
        open_10840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'open', False)
        # Calling open(args, kwargs) (line 392)
        open_call_result_10844 = invoke(stypy.reporting.localization.Localization(__file__, 392, 25), open_10840, *[script_name_10841, str_10842], **kwargs_10843)
        
        # Assigning a type to the variable 'script' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'script', open_call_result_10844)

        if (may_be_10832 and more_types_in_union_10833):
            # SSA join for if statement (line 389)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to write(...): (line 394)
    # Processing the call arguments (line 394)
    str_10847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, (-1)), 'str', 'from distutils.util import byte_compile\nfiles = [\n')
    # Processing the call keyword arguments (line 394)
    kwargs_10848 = {}
    # Getting the type of 'script' (line 394)
    script_10845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'script', False)
    # Obtaining the member 'write' of a type (line 394)
    write_10846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), script_10845, 'write')
    # Calling write(args, kwargs) (line 394)
    write_call_result_10849 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), write_10846, *[str_10847], **kwargs_10848)
    
    
    # Call to write(...): (line 413)
    # Processing the call arguments (line 413)
    
    # Call to join(...): (line 413)
    # Processing the call arguments (line 413)
    
    # Call to map(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'repr' (line 413)
    repr_10855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 41), 'repr', False)
    # Getting the type of 'py_files' (line 413)
    py_files_10856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 47), 'py_files', False)
    # Processing the call keyword arguments (line 413)
    kwargs_10857 = {}
    # Getting the type of 'map' (line 413)
    map_10854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 37), 'map', False)
    # Calling map(args, kwargs) (line 413)
    map_call_result_10858 = invoke(stypy.reporting.localization.Localization(__file__, 413, 37), map_10854, *[repr_10855, py_files_10856], **kwargs_10857)
    
    str_10859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 58), 'str', ',\n')
    # Processing the call keyword arguments (line 413)
    kwargs_10860 = {}
    # Getting the type of 'string' (line 413)
    string_10852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 25), 'string', False)
    # Obtaining the member 'join' of a type (line 413)
    join_10853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 25), string_10852, 'join')
    # Calling join(args, kwargs) (line 413)
    join_call_result_10861 = invoke(stypy.reporting.localization.Localization(__file__, 413, 25), join_10853, *[map_call_result_10858, str_10859], **kwargs_10860)
    
    str_10862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 67), 'str', ']\n')
    # Applying the binary operator '+' (line 413)
    result_add_10863 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 25), '+', join_call_result_10861, str_10862)
    
    # Processing the call keyword arguments (line 413)
    kwargs_10864 = {}
    # Getting the type of 'script' (line 413)
    script_10850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'script', False)
    # Obtaining the member 'write' of a type (line 413)
    write_10851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), script_10850, 'write')
    # Calling write(args, kwargs) (line 413)
    write_call_result_10865 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), write_10851, *[result_add_10863], **kwargs_10864)
    
    
    # Call to write(...): (line 414)
    # Processing the call arguments (line 414)
    str_10868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, (-1)), 'str', '\nbyte_compile(files, optimize=%r, force=%r,\n             prefix=%r, base_dir=%r,\n             verbose=%r, dry_run=0,\n             direct=1)\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 419)
    tuple_10869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 7), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 419)
    # Adding element type (line 419)
    # Getting the type of 'optimize' (line 419)
    optimize_10870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'optimize', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 7), tuple_10869, optimize_10870)
    # Adding element type (line 419)
    # Getting the type of 'force' (line 419)
    force_10871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'force', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 7), tuple_10869, force_10871)
    # Adding element type (line 419)
    # Getting the type of 'prefix' (line 419)
    prefix_10872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'prefix', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 7), tuple_10869, prefix_10872)
    # Adding element type (line 419)
    # Getting the type of 'base_dir' (line 419)
    base_dir_10873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 32), 'base_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 7), tuple_10869, base_dir_10873)
    # Adding element type (line 419)
    # Getting the type of 'verbose' (line 419)
    verbose_10874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 42), 'verbose', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 7), tuple_10869, verbose_10874)
    
    # Applying the binary operator '%' (line 419)
    result_mod_10875 = python_operator(stypy.reporting.localization.Localization(__file__, 419, (-1)), '%', str_10868, tuple_10869)
    
    # Processing the call keyword arguments (line 414)
    kwargs_10876 = {}
    # Getting the type of 'script' (line 414)
    script_10866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'script', False)
    # Obtaining the member 'write' of a type (line 414)
    write_10867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), script_10866, 'write')
    # Calling write(args, kwargs) (line 414)
    write_call_result_10877 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), write_10867, *[result_mod_10875], **kwargs_10876)
    
    
    # Call to close(...): (line 421)
    # Processing the call keyword arguments (line 421)
    kwargs_10880 = {}
    # Getting the type of 'script' (line 421)
    script_10878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'script', False)
    # Obtaining the member 'close' of a type (line 421)
    close_10879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), script_10878, 'close')
    # Calling close(args, kwargs) (line 421)
    close_call_result_10881 = invoke(stypy.reporting.localization.Localization(__file__, 421, 12), close_10879, *[], **kwargs_10880)
    
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 423):
    
    # Assigning a List to a Name (line 423):
    
    # Obtaining an instance of the builtin type 'list' (line 423)
    list_10882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'sys' (line 423)
    sys_10883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'sys')
    # Obtaining the member 'executable' of a type (line 423)
    executable_10884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), sys_10883, 'executable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 14), list_10882, executable_10884)
    # Adding element type (line 423)
    # Getting the type of 'script_name' (line 423)
    script_name_10885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'script_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 14), list_10882, script_name_10885)
    
    # Assigning a type to the variable 'cmd' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'cmd', list_10882)
    
    
    # Getting the type of 'optimize' (line 424)
    optimize_10886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'optimize')
    int_10887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 23), 'int')
    # Applying the binary operator '==' (line 424)
    result_eq_10888 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 11), '==', optimize_10886, int_10887)
    
    # Testing the type of an if condition (line 424)
    if_condition_10889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 8), result_eq_10888)
    # Assigning a type to the variable 'if_condition_10889' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'if_condition_10889', if_condition_10889)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to insert(...): (line 425)
    # Processing the call arguments (line 425)
    int_10892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 23), 'int')
    str_10893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 26), 'str', '-O')
    # Processing the call keyword arguments (line 425)
    kwargs_10894 = {}
    # Getting the type of 'cmd' (line 425)
    cmd_10890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'cmd', False)
    # Obtaining the member 'insert' of a type (line 425)
    insert_10891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), cmd_10890, 'insert')
    # Calling insert(args, kwargs) (line 425)
    insert_call_result_10895 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), insert_10891, *[int_10892, str_10893], **kwargs_10894)
    
    # SSA branch for the else part of an if statement (line 424)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'optimize' (line 426)
    optimize_10896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'optimize')
    int_10897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 25), 'int')
    # Applying the binary operator '==' (line 426)
    result_eq_10898 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 13), '==', optimize_10896, int_10897)
    
    # Testing the type of an if condition (line 426)
    if_condition_10899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 13), result_eq_10898)
    # Assigning a type to the variable 'if_condition_10899' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'if_condition_10899', if_condition_10899)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to insert(...): (line 427)
    # Processing the call arguments (line 427)
    int_10902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 23), 'int')
    str_10903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 26), 'str', '-OO')
    # Processing the call keyword arguments (line 427)
    kwargs_10904 = {}
    # Getting the type of 'cmd' (line 427)
    cmd_10900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'cmd', False)
    # Obtaining the member 'insert' of a type (line 427)
    insert_10901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 12), cmd_10900, 'insert')
    # Calling insert(args, kwargs) (line 427)
    insert_call_result_10905 = invoke(stypy.reporting.localization.Localization(__file__, 427, 12), insert_10901, *[int_10902, str_10903], **kwargs_10904)
    
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to spawn(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'cmd' (line 428)
    cmd_10907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 14), 'cmd', False)
    # Processing the call keyword arguments (line 428)
    # Getting the type of 'dry_run' (line 428)
    dry_run_10908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 27), 'dry_run', False)
    keyword_10909 = dry_run_10908
    kwargs_10910 = {'dry_run': keyword_10909}
    # Getting the type of 'spawn' (line 428)
    spawn_10906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'spawn', False)
    # Calling spawn(args, kwargs) (line 428)
    spawn_call_result_10911 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), spawn_10906, *[cmd_10907], **kwargs_10910)
    
    
    # Call to execute(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'os' (line 429)
    os_10913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'os', False)
    # Obtaining the member 'remove' of a type (line 429)
    remove_10914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 16), os_10913, 'remove')
    
    # Obtaining an instance of the builtin type 'tuple' (line 429)
    tuple_10915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 429)
    # Adding element type (line 429)
    # Getting the type of 'script_name' (line 429)
    script_name_10916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 28), 'script_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 28), tuple_10915, script_name_10916)
    
    str_10917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 43), 'str', 'removing %s')
    # Getting the type of 'script_name' (line 429)
    script_name_10918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 59), 'script_name', False)
    # Applying the binary operator '%' (line 429)
    result_mod_10919 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 43), '%', str_10917, script_name_10918)
    
    # Processing the call keyword arguments (line 429)
    # Getting the type of 'dry_run' (line 430)
    dry_run_10920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'dry_run', False)
    keyword_10921 = dry_run_10920
    kwargs_10922 = {'dry_run': keyword_10921}
    # Getting the type of 'execute' (line 429)
    execute_10912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'execute', False)
    # Calling execute(args, kwargs) (line 429)
    execute_call_result_10923 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), execute_10912, *[remove_10914, tuple_10915, result_mod_10919], **kwargs_10922)
    
    # SSA branch for the else part of an if statement (line 380)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 437, 8))
    
    # 'from py_compile import compile' statement (line 437)
    try:
        from py_compile import compile

    except:
        compile = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 437, 8), 'py_compile', None, module_type_store, ['compile'], [compile])
    
    
    # Getting the type of 'py_files' (line 439)
    py_files_10924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'py_files')
    # Testing the type of a for loop iterable (line 439)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 439, 8), py_files_10924)
    # Getting the type of the for loop variable (line 439)
    for_loop_var_10925 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 439, 8), py_files_10924)
    # Assigning a type to the variable 'file' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'file', for_loop_var_10925)
    # SSA begins for a for statement (line 439)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_10926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 20), 'int')
    slice_10927 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 440, 15), int_10926, None, None)
    # Getting the type of 'file' (line 440)
    file_10928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'file')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___10929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 15), file_10928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_10930 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), getitem___10929, slice_10927)
    
    str_10931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 28), 'str', '.py')
    # Applying the binary operator '!=' (line 440)
    result_ne_10932 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 15), '!=', subscript_call_result_10930, str_10931)
    
    # Testing the type of an if condition (line 440)
    if_condition_10933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 12), result_ne_10932)
    # Assigning a type to the variable 'if_condition_10933' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'if_condition_10933', if_condition_10933)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 448):
    
    # Assigning a BinOp to a Name (line 448):
    # Getting the type of 'file' (line 448)
    file_10934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 20), 'file')
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    # Getting the type of '__debug__' (line 448)
    debug___10935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 28), '__debug__')
    str_10936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 42), 'str', 'c')
    # Applying the binary operator 'and' (line 448)
    result_and_keyword_10937 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 28), 'and', debug___10935, str_10936)
    
    str_10938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 49), 'str', 'o')
    # Applying the binary operator 'or' (line 448)
    result_or_keyword_10939 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 28), 'or', result_and_keyword_10937, str_10938)
    
    # Applying the binary operator '+' (line 448)
    result_add_10940 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 20), '+', file_10934, result_or_keyword_10939)
    
    # Assigning a type to the variable 'cfile' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'cfile', result_add_10940)
    
    # Assigning a Name to a Name (line 449):
    
    # Assigning a Name to a Name (line 449):
    # Getting the type of 'file' (line 449)
    file_10941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), 'file')
    # Assigning a type to the variable 'dfile' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'dfile', file_10941)
    
    # Getting the type of 'prefix' (line 450)
    prefix_10942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'prefix')
    # Testing the type of an if condition (line 450)
    if_condition_10943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 12), prefix_10942)
    # Assigning a type to the variable 'if_condition_10943' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'if_condition_10943', if_condition_10943)
    # SSA begins for if statement (line 450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'prefix' (line 451)
    prefix_10945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 29), 'prefix', False)
    # Processing the call keyword arguments (line 451)
    kwargs_10946 = {}
    # Getting the type of 'len' (line 451)
    len_10944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 25), 'len', False)
    # Calling len(args, kwargs) (line 451)
    len_call_result_10947 = invoke(stypy.reporting.localization.Localization(__file__, 451, 25), len_10944, *[prefix_10945], **kwargs_10946)
    
    slice_10948 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 451, 19), None, len_call_result_10947, None)
    # Getting the type of 'file' (line 451)
    file_10949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 19), 'file')
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___10950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 19), file_10949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_10951 = invoke(stypy.reporting.localization.Localization(__file__, 451, 19), getitem___10950, slice_10948)
    
    # Getting the type of 'prefix' (line 451)
    prefix_10952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'prefix')
    # Applying the binary operator '!=' (line 451)
    result_ne_10953 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 19), '!=', subscript_call_result_10951, prefix_10952)
    
    # Testing the type of an if condition (line 451)
    if_condition_10954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 16), result_ne_10953)
    # Assigning a type to the variable 'if_condition_10954' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'if_condition_10954', if_condition_10954)
    # SSA begins for if statement (line 451)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ValueError' (line 452)
    ValueError_10955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 26), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 452, 20), ValueError_10955, 'raise parameter', BaseException)
    # SSA join for if statement (line 451)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 455):
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'prefix' (line 455)
    prefix_10957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 34), 'prefix', False)
    # Processing the call keyword arguments (line 455)
    kwargs_10958 = {}
    # Getting the type of 'len' (line 455)
    len_10956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 30), 'len', False)
    # Calling len(args, kwargs) (line 455)
    len_call_result_10959 = invoke(stypy.reporting.localization.Localization(__file__, 455, 30), len_10956, *[prefix_10957], **kwargs_10958)
    
    slice_10960 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 455, 24), len_call_result_10959, None, None)
    # Getting the type of 'dfile' (line 455)
    dfile_10961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 24), 'dfile')
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___10962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 24), dfile_10961, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_10963 = invoke(stypy.reporting.localization.Localization(__file__, 455, 24), getitem___10962, slice_10960)
    
    # Assigning a type to the variable 'dfile' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'dfile', subscript_call_result_10963)
    # SSA join for if statement (line 450)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'base_dir' (line 456)
    base_dir_10964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'base_dir')
    # Testing the type of an if condition (line 456)
    if_condition_10965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 12), base_dir_10964)
    # Assigning a type to the variable 'if_condition_10965' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'if_condition_10965', if_condition_10965)
    # SSA begins for if statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 457):
    
    # Assigning a Call to a Name (line 457):
    
    # Call to join(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'base_dir' (line 457)
    base_dir_10969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 37), 'base_dir', False)
    # Getting the type of 'dfile' (line 457)
    dfile_10970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 47), 'dfile', False)
    # Processing the call keyword arguments (line 457)
    kwargs_10971 = {}
    # Getting the type of 'os' (line 457)
    os_10966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 457)
    path_10967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 24), os_10966, 'path')
    # Obtaining the member 'join' of a type (line 457)
    join_10968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 24), path_10967, 'join')
    # Calling join(args, kwargs) (line 457)
    join_call_result_10972 = invoke(stypy.reporting.localization.Localization(__file__, 457, 24), join_10968, *[base_dir_10969, dfile_10970], **kwargs_10971)
    
    # Assigning a type to the variable 'dfile' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'dfile', join_call_result_10972)
    # SSA join for if statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to basename(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'cfile' (line 459)
    cfile_10976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 42), 'cfile', False)
    # Processing the call keyword arguments (line 459)
    kwargs_10977 = {}
    # Getting the type of 'os' (line 459)
    os_10973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 459)
    path_10974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), os_10973, 'path')
    # Obtaining the member 'basename' of a type (line 459)
    basename_10975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), path_10974, 'basename')
    # Calling basename(args, kwargs) (line 459)
    basename_call_result_10978 = invoke(stypy.reporting.localization.Localization(__file__, 459, 25), basename_10975, *[cfile_10976], **kwargs_10977)
    
    # Assigning a type to the variable 'cfile_base' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'cfile_base', basename_call_result_10978)
    
    # Getting the type of 'direct' (line 460)
    direct_10979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 15), 'direct')
    # Testing the type of an if condition (line 460)
    if_condition_10980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 12), direct_10979)
    # Assigning a type to the variable 'if_condition_10980' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'if_condition_10980', if_condition_10980)
    # SSA begins for if statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'force' (line 461)
    force_10981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'force')
    
    # Call to newer(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'file' (line 461)
    file_10983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 34), 'file', False)
    # Getting the type of 'cfile' (line 461)
    cfile_10984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 40), 'cfile', False)
    # Processing the call keyword arguments (line 461)
    kwargs_10985 = {}
    # Getting the type of 'newer' (line 461)
    newer_10982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'newer', False)
    # Calling newer(args, kwargs) (line 461)
    newer_call_result_10986 = invoke(stypy.reporting.localization.Localization(__file__, 461, 28), newer_10982, *[file_10983, cfile_10984], **kwargs_10985)
    
    # Applying the binary operator 'or' (line 461)
    result_or_keyword_10987 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 19), 'or', force_10981, newer_call_result_10986)
    
    # Testing the type of an if condition (line 461)
    if_condition_10988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 16), result_or_keyword_10987)
    # Assigning a type to the variable 'if_condition_10988' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'if_condition_10988', if_condition_10988)
    # SSA begins for if statement (line 461)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 462)
    # Processing the call arguments (line 462)
    str_10991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 29), 'str', 'byte-compiling %s to %s')
    # Getting the type of 'file' (line 462)
    file_10992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 56), 'file', False)
    # Getting the type of 'cfile_base' (line 462)
    cfile_base_10993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 62), 'cfile_base', False)
    # Processing the call keyword arguments (line 462)
    kwargs_10994 = {}
    # Getting the type of 'log' (line 462)
    log_10989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'log', False)
    # Obtaining the member 'info' of a type (line 462)
    info_10990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 20), log_10989, 'info')
    # Calling info(args, kwargs) (line 462)
    info_call_result_10995 = invoke(stypy.reporting.localization.Localization(__file__, 462, 20), info_10990, *[str_10991, file_10992, cfile_base_10993], **kwargs_10994)
    
    
    
    # Getting the type of 'dry_run' (line 463)
    dry_run_10996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 27), 'dry_run')
    # Applying the 'not' unary operator (line 463)
    result_not__10997 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 23), 'not', dry_run_10996)
    
    # Testing the type of an if condition (line 463)
    if_condition_10998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 20), result_not__10997)
    # Assigning a type to the variable 'if_condition_10998' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 20), 'if_condition_10998', if_condition_10998)
    # SSA begins for if statement (line 463)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to compile(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'file' (line 464)
    file_11000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 32), 'file', False)
    # Getting the type of 'cfile' (line 464)
    cfile_11001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'cfile', False)
    # Getting the type of 'dfile' (line 464)
    dfile_11002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 45), 'dfile', False)
    # Processing the call keyword arguments (line 464)
    kwargs_11003 = {}
    # Getting the type of 'compile' (line 464)
    compile_10999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'compile', False)
    # Calling compile(args, kwargs) (line 464)
    compile_call_result_11004 = invoke(stypy.reporting.localization.Localization(__file__, 464, 24), compile_10999, *[file_11000, cfile_11001, dfile_11002], **kwargs_11003)
    
    # SSA join for if statement (line 463)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 461)
    module_type_store.open_ssa_branch('else')
    
    # Call to debug(...): (line 466)
    # Processing the call arguments (line 466)
    str_11007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 30), 'str', 'skipping byte-compilation of %s to %s')
    # Getting the type of 'file' (line 467)
    file_11008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 30), 'file', False)
    # Getting the type of 'cfile_base' (line 467)
    cfile_base_11009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 36), 'cfile_base', False)
    # Processing the call keyword arguments (line 466)
    kwargs_11010 = {}
    # Getting the type of 'log' (line 466)
    log_11005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'log', False)
    # Obtaining the member 'debug' of a type (line 466)
    debug_11006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 20), log_11005, 'debug')
    # Calling debug(args, kwargs) (line 466)
    debug_call_result_11011 = invoke(stypy.reporting.localization.Localization(__file__, 466, 20), debug_11006, *[str_11007, file_11008, cfile_base_11009], **kwargs_11010)
    
    # SSA join for if statement (line 461)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 380)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'byte_compile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'byte_compile' in the type store
    # Getting the type of 'stypy_return_type' (line 328)
    stypy_return_type_11012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11012)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'byte_compile'
    return stypy_return_type_11012

# Assigning a type to the variable 'byte_compile' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'byte_compile', byte_compile)

@norecursion
def rfc822_escape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rfc822_escape'
    module_type_store = module_type_store.open_function_context('rfc822_escape', 471, 0, False)
    
    # Passed parameters checking function
    rfc822_escape.stypy_localization = localization
    rfc822_escape.stypy_type_of_self = None
    rfc822_escape.stypy_type_store = module_type_store
    rfc822_escape.stypy_function_name = 'rfc822_escape'
    rfc822_escape.stypy_param_names_list = ['header']
    rfc822_escape.stypy_varargs_param_name = None
    rfc822_escape.stypy_kwargs_param_name = None
    rfc822_escape.stypy_call_defaults = defaults
    rfc822_escape.stypy_call_varargs = varargs
    rfc822_escape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfc822_escape', ['header'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfc822_escape', localization, ['header'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfc822_escape(...)' code ##################

    str_11013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, (-1)), 'str', 'Return a version of the string escaped for inclusion in an\n    RFC-822 header, by ensuring there are 8 spaces space after each newline.\n    ')
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to split(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'header' (line 475)
    header_11016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 25), 'header', False)
    str_11017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 33), 'str', '\n')
    # Processing the call keyword arguments (line 475)
    kwargs_11018 = {}
    # Getting the type of 'string' (line 475)
    string_11014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'string', False)
    # Obtaining the member 'split' of a type (line 475)
    split_11015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), string_11014, 'split')
    # Calling split(args, kwargs) (line 475)
    split_call_result_11019 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), split_11015, *[header_11016, str_11017], **kwargs_11018)
    
    # Assigning a type to the variable 'lines' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'lines', split_call_result_11019)
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to join(...): (line 476)
    # Processing the call arguments (line 476)
    # Getting the type of 'lines' (line 476)
    lines_11022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 25), 'lines', False)
    str_11023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 32), 'str', '\n')
    int_11024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 39), 'int')
    str_11025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 41), 'str', ' ')
    # Applying the binary operator '*' (line 476)
    result_mul_11026 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 39), '*', int_11024, str_11025)
    
    # Applying the binary operator '+' (line 476)
    result_add_11027 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 32), '+', str_11023, result_mul_11026)
    
    # Processing the call keyword arguments (line 476)
    kwargs_11028 = {}
    # Getting the type of 'string' (line 476)
    string_11020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 13), 'string', False)
    # Obtaining the member 'join' of a type (line 476)
    join_11021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 13), string_11020, 'join')
    # Calling join(args, kwargs) (line 476)
    join_call_result_11029 = invoke(stypy.reporting.localization.Localization(__file__, 476, 13), join_11021, *[lines_11022, result_add_11027], **kwargs_11028)
    
    # Assigning a type to the variable 'header' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'header', join_call_result_11029)
    # Getting the type of 'header' (line 477)
    header_11030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'header')
    # Assigning a type to the variable 'stypy_return_type' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type', header_11030)
    
    # ################# End of 'rfc822_escape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfc822_escape' in the type store
    # Getting the type of 'stypy_return_type' (line 471)
    stypy_return_type_11031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11031)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfc822_escape'
    return stypy_return_type_11031

# Assigning a type to the variable 'rfc822_escape' (line 471)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 0), 'rfc822_escape', rfc822_escape)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Provide access to Python's configuration information.  The specific
2: configuration variables available depend heavily on the platform and
3: configuration.  The values may be retrieved using
4: get_config_var(name), and the list of variables is available via
5: get_config_vars().keys().  Additional convenience functions are also
6: available.
7: 
8: Written by:   Fred L. Drake, Jr.
9: Email:        <fdrake@acm.org>
10: '''
11: 
12: __revision__ = "$Id$"
13: 
14: import os
15: import re
16: import string
17: import sys
18: 
19: from distutils.errors import DistutilsPlatformError
20: 
21: # These are needed in a couple of spots, so just compute them once.
22: PREFIX = os.path.normpath(sys.prefix)
23: EXEC_PREFIX = os.path.normpath(sys.exec_prefix)
24: 
25: # Path to the base directory of the project. On Windows the binary may
26: # live in project/PCBuild9.  If we're dealing with an x64 Windows build,
27: # it'll live in project/PCbuild/amd64.
28: project_base = os.path.dirname(os.path.abspath(sys.executable))
29: if os.name == "nt" and "pcbuild" in project_base[-8:].lower():
30:     project_base = os.path.abspath(os.path.join(project_base, os.path.pardir))
31: # PC/VS7.1
32: if os.name == "nt" and "\\pc\\v" in project_base[-10:].lower():
33:     project_base = os.path.abspath(os.path.join(project_base, os.path.pardir,
34:                                                 os.path.pardir))
35: # PC/AMD64
36: if os.name == "nt" and "\\pcbuild\\amd64" in project_base[-14:].lower():
37:     project_base = os.path.abspath(os.path.join(project_base, os.path.pardir,
38:                                                 os.path.pardir))
39: 
40: # set for cross builds
41: if "_PYTHON_PROJECT_BASE" in os.environ:
42:     # this is the build directory, at least for posix
43:     project_base = os.path.normpath(os.environ["_PYTHON_PROJECT_BASE"])
44: 
45: # python_build: (Boolean) if true, we're either building Python or
46: # building an extension with an un-installed Python, so we use
47: # different (hard-wired) directories.
48: # Setup.local is available for Makefile builds including VPATH builds,
49: # Setup.dist is available on Windows
50: def _python_build():
51:     for fn in ("Setup.dist", "Setup.local"):
52:         if os.path.isfile(os.path.join(project_base, "Modules", fn)):
53:             return True
54:     return False
55: python_build = _python_build()
56: 
57: 
58: def get_python_version():
59:     '''Return a string containing the major and minor Python version,
60:     leaving off the patchlevel.  Sample return values could be '1.5'
61:     or '2.2'.
62:     '''
63:     return sys.version[:3]
64: 
65: 
66: def get_python_inc(plat_specific=0, prefix=None):
67:     '''Return the directory containing installed Python header files.
68: 
69:     If 'plat_specific' is false (the default), this is the path to the
70:     non-platform-specific header files, i.e. Python.h and so on;
71:     otherwise, this is the path to platform-specific header files
72:     (namely pyconfig.h).
73: 
74:     If 'prefix' is supplied, use it instead of sys.prefix or
75:     sys.exec_prefix -- i.e., ignore 'plat_specific'.
76:     '''
77:     if prefix is None:
78:         prefix = plat_specific and EXEC_PREFIX or PREFIX
79: 
80:     if os.name == "posix":
81:         if python_build:
82:             buildir = os.path.dirname(sys.executable)
83:             if plat_specific:
84:                 # python.h is located in the buildir
85:                 inc_dir = buildir
86:             else:
87:                 # the source dir is relative to the buildir
88:                 srcdir = os.path.abspath(os.path.join(buildir,
89:                                          get_config_var('srcdir')))
90:                 # Include is located in the srcdir
91:                 inc_dir = os.path.join(srcdir, "Include")
92:             return inc_dir
93:         return os.path.join(prefix, "include", "python" + get_python_version())
94:     elif os.name == "nt":
95:         return os.path.join(prefix, "include")
96:     elif os.name == "os2":
97:         return os.path.join(prefix, "Include")
98:     else:
99:         raise DistutilsPlatformError(
100:             "I don't know where Python installs its C header files "
101:             "on platform '%s'" % os.name)
102: 
103: 
104: def get_python_lib(plat_specific=0, standard_lib=0, prefix=None):
105:     '''Return the directory containing the Python library (standard or
106:     site additions).
107: 
108:     If 'plat_specific' is true, return the directory containing
109:     platform-specific modules, i.e. any module from a non-pure-Python
110:     module distribution; otherwise, return the platform-shared library
111:     directory.  If 'standard_lib' is true, return the directory
112:     containing standard Python library modules; otherwise, return the
113:     directory for site-specific modules.
114: 
115:     If 'prefix' is supplied, use it instead of sys.prefix or
116:     sys.exec_prefix -- i.e., ignore 'plat_specific'.
117:     '''
118:     if prefix is None:
119:         prefix = plat_specific and EXEC_PREFIX or PREFIX
120: 
121:     if os.name == "posix":
122:         libpython = os.path.join(prefix,
123:                                  "lib", "python" + get_python_version())
124:         if standard_lib:
125:             return libpython
126:         else:
127:             return os.path.join(libpython, "site-packages")
128: 
129:     elif os.name == "nt":
130:         if standard_lib:
131:             return os.path.join(prefix, "Lib")
132:         else:
133:             if get_python_version() < "2.2":
134:                 return prefix
135:             else:
136:                 return os.path.join(prefix, "Lib", "site-packages")
137: 
138:     elif os.name == "os2":
139:         if standard_lib:
140:             return os.path.join(prefix, "Lib")
141:         else:
142:             return os.path.join(prefix, "Lib", "site-packages")
143: 
144:     else:
145:         raise DistutilsPlatformError(
146:             "I don't know where Python installs its library "
147:             "on platform '%s'" % os.name)
148: 
149: 
150: 
151: def customize_compiler(compiler):
152:     '''Do any platform-specific customization of a CCompiler instance.
153: 
154:     Mainly needed on Unix, so we can plug in the information that
155:     varies across Unices and is stored in Python's Makefile.
156:     '''
157:     if compiler.compiler_type == "unix":
158:         if sys.platform == "darwin":
159:             # Perform first-time customization of compiler-related
160:             # config vars on OS X now that we know we need a compiler.
161:             # This is primarily to support Pythons from binary
162:             # installers.  The kind and paths to build tools on
163:             # the user system may vary significantly from the system
164:             # that Python itself was built on.  Also the user OS
165:             # version and build tools may not support the same set
166:             # of CPU architectures for universal builds.
167:             global _config_vars
168:             # Use get_config_var() to ensure _config_vars is initialized.
169:             if not get_config_var('CUSTOMIZED_OSX_COMPILER'):
170:                 import _osx_support
171:                 _osx_support.customize_compiler(_config_vars)
172:                 _config_vars['CUSTOMIZED_OSX_COMPILER'] = 'True'
173: 
174:         (cc, cxx, opt, cflags, ccshared, ldshared, so_ext, ar, ar_flags) = \
175:             get_config_vars('CC', 'CXX', 'OPT', 'CFLAGS',
176:                             'CCSHARED', 'LDSHARED', 'SO', 'AR',
177:                             'ARFLAGS')
178: 
179:         if 'CC' in os.environ:
180:             newcc = os.environ['CC']
181:             if (sys.platform == 'darwin'
182:                     and 'LDSHARED' not in os.environ
183:                     and ldshared.startswith(cc)):
184:                 # On OS X, if CC is overridden, use that as the default
185:                 #       command for LDSHARED as well
186:                 ldshared = newcc + ldshared[len(cc):]
187:             cc = newcc
188:         if 'CXX' in os.environ:
189:             cxx = os.environ['CXX']
190:         if 'LDSHARED' in os.environ:
191:             ldshared = os.environ['LDSHARED']
192:         if 'CPP' in os.environ:
193:             cpp = os.environ['CPP']
194:         else:
195:             cpp = cc + " -E"           # not always
196:         if 'LDFLAGS' in os.environ:
197:             ldshared = ldshared + ' ' + os.environ['LDFLAGS']
198:         if 'CFLAGS' in os.environ:
199:             cflags = opt + ' ' + os.environ['CFLAGS']
200:             ldshared = ldshared + ' ' + os.environ['CFLAGS']
201:         if 'CPPFLAGS' in os.environ:
202:             cpp = cpp + ' ' + os.environ['CPPFLAGS']
203:             cflags = cflags + ' ' + os.environ['CPPFLAGS']
204:             ldshared = ldshared + ' ' + os.environ['CPPFLAGS']
205:         if 'AR' in os.environ:
206:             ar = os.environ['AR']
207:         if 'ARFLAGS' in os.environ:
208:             archiver = ar + ' ' + os.environ['ARFLAGS']
209:         else:
210:             archiver = ar + ' ' + ar_flags
211: 
212:         cc_cmd = cc + ' ' + cflags
213:         compiler.set_executables(
214:             preprocessor=cpp,
215:             compiler=cc_cmd,
216:             compiler_so=cc_cmd + ' ' + ccshared,
217:             compiler_cxx=cxx,
218:             linker_so=ldshared,
219:             linker_exe=cc,
220:             archiver=archiver)
221: 
222:         compiler.shared_lib_extension = so_ext
223: 
224: 
225: def get_config_h_filename():
226:     '''Return full pathname of installed pyconfig.h file.'''
227:     if python_build:
228:         if os.name == "nt":
229:             inc_dir = os.path.join(project_base, "PC")
230:         else:
231:             inc_dir = project_base
232:     else:
233:         inc_dir = get_python_inc(plat_specific=1)
234:     if get_python_version() < '2.2':
235:         config_h = 'config.h'
236:     else:
237:         # The name of the config.h file changed in 2.2
238:         config_h = 'pyconfig.h'
239:     return os.path.join(inc_dir, config_h)
240: 
241: 
242: def get_makefile_filename():
243:     '''Return full pathname of installed Makefile from the Python build.'''
244:     if python_build:
245:         return os.path.join(project_base, "Makefile")
246:     lib_dir = get_python_lib(plat_specific=1, standard_lib=1)
247:     return os.path.join(lib_dir, "config", "Makefile")
248: 
249: 
250: def parse_config_h(fp, g=None):
251:     '''Parse a config.h-style file.
252: 
253:     A dictionary containing name/value pairs is returned.  If an
254:     optional dictionary is passed in as the second argument, it is
255:     used instead of a new dictionary.
256:     '''
257:     if g is None:
258:         g = {}
259:     define_rx = re.compile("#define ([A-Z][A-Za-z0-9_]+) (.*)\n")
260:     undef_rx = re.compile("/[*] #undef ([A-Z][A-Za-z0-9_]+) [*]/\n")
261:     #
262:     while 1:
263:         line = fp.readline()
264:         if not line:
265:             break
266:         m = define_rx.match(line)
267:         if m:
268:             n, v = m.group(1, 2)
269:             try: v = int(v)
270:             except ValueError: pass
271:             g[n] = v
272:         else:
273:             m = undef_rx.match(line)
274:             if m:
275:                 g[m.group(1)] = 0
276:     return g
277: 
278: 
279: # Regexes needed for parsing Makefile (and similar syntaxes,
280: # like old-style Setup files).
281: _variable_rx = re.compile("([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
282: _findvar1_rx = re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
283: _findvar2_rx = re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")
284: 
285: def parse_makefile(fn, g=None):
286:     '''Parse a Makefile-style file.
287: 
288:     A dictionary containing name/value pairs is returned.  If an
289:     optional dictionary is passed in as the second argument, it is
290:     used instead of a new dictionary.
291:     '''
292:     from distutils.text_file import TextFile
293:     fp = TextFile(fn, strip_comments=1, skip_blanks=1, join_lines=1)
294: 
295:     if g is None:
296:         g = {}
297:     done = {}
298:     notdone = {}
299: 
300:     while 1:
301:         line = fp.readline()
302:         if line is None:  # eof
303:             break
304:         m = _variable_rx.match(line)
305:         if m:
306:             n, v = m.group(1, 2)
307:             v = v.strip()
308:             # `$$' is a literal `$' in make
309:             tmpv = v.replace('$$', '')
310: 
311:             if "$" in tmpv:
312:                 notdone[n] = v
313:             else:
314:                 try:
315:                     v = int(v)
316:                 except ValueError:
317:                     # insert literal `$'
318:                     done[n] = v.replace('$$', '$')
319:                 else:
320:                     done[n] = v
321: 
322:     # do variable interpolation here
323:     while notdone:
324:         for name in notdone.keys():
325:             value = notdone[name]
326:             m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
327:             if m:
328:                 n = m.group(1)
329:                 found = True
330:                 if n in done:
331:                     item = str(done[n])
332:                 elif n in notdone:
333:                     # get it on a subsequent round
334:                     found = False
335:                 elif n in os.environ:
336:                     # do it like make: fall back to environment
337:                     item = os.environ[n]
338:                 else:
339:                     done[n] = item = ""
340:                 if found:
341:                     after = value[m.end():]
342:                     value = value[:m.start()] + item + after
343:                     if "$" in after:
344:                         notdone[name] = value
345:                     else:
346:                         try: value = int(value)
347:                         except ValueError:
348:                             done[name] = value.strip()
349:                         else:
350:                             done[name] = value
351:                         del notdone[name]
352:             else:
353:                 # bogus variable reference; just drop it since we can't deal
354:                 del notdone[name]
355: 
356:     fp.close()
357: 
358:     # strip spurious spaces
359:     for k, v in done.items():
360:         if isinstance(v, str):
361:             done[k] = v.strip()
362: 
363:     # save the results in the global dictionary
364:     g.update(done)
365:     return g
366: 
367: 
368: def expand_makefile_vars(s, vars):
369:     '''Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in
370:     'string' according to 'vars' (a dictionary mapping variable names to
371:     values).  Variables not present in 'vars' are silently expanded to the
372:     empty string.  The variable values in 'vars' should not contain further
373:     variable expansions; if 'vars' is the output of 'parse_makefile()',
374:     you're fine.  Returns a variable-expanded version of 's'.
375:     '''
376: 
377:     # This algorithm does multiple expansion, so if vars['foo'] contains
378:     # "${bar}", it will expand ${foo} to ${bar}, and then expand
379:     # ${bar}... and so forth.  This is fine as long as 'vars' comes from
380:     # 'parse_makefile()', which takes care of such expansions eagerly,
381:     # according to make's variable expansion semantics.
382: 
383:     while 1:
384:         m = _findvar1_rx.search(s) or _findvar2_rx.search(s)
385:         if m:
386:             (beg, end) = m.span()
387:             s = s[0:beg] + vars.get(m.group(1)) + s[end:]
388:         else:
389:             break
390:     return s
391: 
392: 
393: _config_vars = None
394: 
395: def _init_posix():
396:     '''Initialize the module as appropriate for POSIX systems.'''
397:     # _sysconfigdata is generated at build time, see the sysconfig module
398:     from _sysconfigdata import build_time_vars
399:     global _config_vars
400:     _config_vars = {}
401:     _config_vars.update(build_time_vars)
402: 
403: 
404: def _init_nt():
405:     '''Initialize the module as appropriate for NT'''
406:     g = {}
407:     # set basic install directories
408:     g['LIBDEST'] = get_python_lib(plat_specific=0, standard_lib=1)
409:     g['BINLIBDEST'] = get_python_lib(plat_specific=1, standard_lib=1)
410: 
411:     # XXX hmmm.. a normal install puts include files here
412:     g['INCLUDEPY'] = get_python_inc(plat_specific=0)
413: 
414:     g['SO'] = '.pyd'
415:     g['EXE'] = ".exe"
416:     g['VERSION'] = get_python_version().replace(".", "")
417:     g['BINDIR'] = os.path.dirname(os.path.abspath(sys.executable))
418: 
419:     global _config_vars
420:     _config_vars = g
421: 
422: 
423: def _init_os2():
424:     '''Initialize the module as appropriate for OS/2'''
425:     g = {}
426:     # set basic install directories
427:     g['LIBDEST'] = get_python_lib(plat_specific=0, standard_lib=1)
428:     g['BINLIBDEST'] = get_python_lib(plat_specific=1, standard_lib=1)
429: 
430:     # XXX hmmm.. a normal install puts include files here
431:     g['INCLUDEPY'] = get_python_inc(plat_specific=0)
432: 
433:     g['SO'] = '.pyd'
434:     g['EXE'] = ".exe"
435: 
436:     global _config_vars
437:     _config_vars = g
438: 
439: 
440: def get_config_vars(*args):
441:     '''With no arguments, return a dictionary of all configuration
442:     variables relevant for the current platform.  Generally this includes
443:     everything needed to build extensions and install both pure modules and
444:     extensions.  On Unix, this means every variable defined in Python's
445:     installed Makefile; on Windows and Mac OS it's a much smaller set.
446: 
447:     With arguments, return a list of values that result from looking up
448:     each argument in the configuration variable dictionary.
449:     '''
450:     global _config_vars
451:     if _config_vars is None:
452:         func = globals().get("_init_" + os.name)
453:         if func:
454:             func()
455:         else:
456:             _config_vars = {}
457: 
458:         # Normalized versions of prefix and exec_prefix are handy to have;
459:         # in fact, these are the standard versions used most places in the
460:         # Distutils.
461:         _config_vars['prefix'] = PREFIX
462:         _config_vars['exec_prefix'] = EXEC_PREFIX
463: 
464:         # OS X platforms require special customization to handle
465:         # multi-architecture, multi-os-version installers
466:         if sys.platform == 'darwin':
467:             import _osx_support
468:             _osx_support.customize_config_vars(_config_vars)
469: 
470:     if args:
471:         vals = []
472:         for name in args:
473:             vals.append(_config_vars.get(name))
474:         return vals
475:     else:
476:         return _config_vars
477: 
478: def get_config_var(name):
479:     '''Return the value of a single variable using the dictionary
480:     returned by 'get_config_vars()'.  Equivalent to
481:     get_config_vars().get(name)
482:     '''
483:     return get_config_vars().get(name)
484: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_7255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', "Provide access to Python's configuration information.  The specific\nconfiguration variables available depend heavily on the platform and\nconfiguration.  The values may be retrieved using\nget_config_var(name), and the list of variables is available via\nget_config_vars().keys().  Additional convenience functions are also\navailable.\n\nWritten by:   Fred L. Drake, Jr.\nEmail:        <fdrake@acm.org>\n")

# Assigning a Str to a Name (line 12):

# Assigning a Str to a Name (line 12):
str_7256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__revision__', str_7256)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import os' statement (line 14)
import os

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import re' statement (line 15)
import re

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import string' statement (line 16)
import string

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import sys' statement (line 17)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.errors import DistutilsPlatformError' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_7257 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.errors')

if (type(import_7257) is not StypyTypeError):

    if (import_7257 != 'pyd_module'):
        __import__(import_7257)
        sys_modules_7258 = sys.modules[import_7257]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.errors', sys_modules_7258.module_type_store, module_type_store, ['DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_7258, sys_modules_7258.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError'], [DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.errors', import_7257)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


# Assigning a Call to a Name (line 22):

# Assigning a Call to a Name (line 22):

# Call to normpath(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'sys' (line 22)
sys_7262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'sys', False)
# Obtaining the member 'prefix' of a type (line 22)
prefix_7263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 26), sys_7262, 'prefix')
# Processing the call keyword arguments (line 22)
kwargs_7264 = {}
# Getting the type of 'os' (line 22)
os_7259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'os', False)
# Obtaining the member 'path' of a type (line 22)
path_7260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 9), os_7259, 'path')
# Obtaining the member 'normpath' of a type (line 22)
normpath_7261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 9), path_7260, 'normpath')
# Calling normpath(args, kwargs) (line 22)
normpath_call_result_7265 = invoke(stypy.reporting.localization.Localization(__file__, 22, 9), normpath_7261, *[prefix_7263], **kwargs_7264)

# Assigning a type to the variable 'PREFIX' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'PREFIX', normpath_call_result_7265)

# Assigning a Call to a Name (line 23):

# Assigning a Call to a Name (line 23):

# Call to normpath(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'sys' (line 23)
sys_7269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'sys', False)
# Obtaining the member 'exec_prefix' of a type (line 23)
exec_prefix_7270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 31), sys_7269, 'exec_prefix')
# Processing the call keyword arguments (line 23)
kwargs_7271 = {}
# Getting the type of 'os' (line 23)
os_7266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'os', False)
# Obtaining the member 'path' of a type (line 23)
path_7267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 14), os_7266, 'path')
# Obtaining the member 'normpath' of a type (line 23)
normpath_7268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 14), path_7267, 'normpath')
# Calling normpath(args, kwargs) (line 23)
normpath_call_result_7272 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), normpath_7268, *[exec_prefix_7270], **kwargs_7271)

# Assigning a type to the variable 'EXEC_PREFIX' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'EXEC_PREFIX', normpath_call_result_7272)

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to dirname(...): (line 28)
# Processing the call arguments (line 28)

# Call to abspath(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'sys' (line 28)
sys_7279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 47), 'sys', False)
# Obtaining the member 'executable' of a type (line 28)
executable_7280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 47), sys_7279, 'executable')
# Processing the call keyword arguments (line 28)
kwargs_7281 = {}
# Getting the type of 'os' (line 28)
os_7276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'os', False)
# Obtaining the member 'path' of a type (line 28)
path_7277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), os_7276, 'path')
# Obtaining the member 'abspath' of a type (line 28)
abspath_7278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), path_7277, 'abspath')
# Calling abspath(args, kwargs) (line 28)
abspath_call_result_7282 = invoke(stypy.reporting.localization.Localization(__file__, 28, 31), abspath_7278, *[executable_7280], **kwargs_7281)

# Processing the call keyword arguments (line 28)
kwargs_7283 = {}
# Getting the type of 'os' (line 28)
os_7273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'os', False)
# Obtaining the member 'path' of a type (line 28)
path_7274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), os_7273, 'path')
# Obtaining the member 'dirname' of a type (line 28)
dirname_7275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), path_7274, 'dirname')
# Calling dirname(args, kwargs) (line 28)
dirname_call_result_7284 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), dirname_7275, *[abspath_call_result_7282], **kwargs_7283)

# Assigning a type to the variable 'project_base' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'project_base', dirname_call_result_7284)


# Evaluating a boolean operation

# Getting the type of 'os' (line 29)
os_7285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 3), 'os')
# Obtaining the member 'name' of a type (line 29)
name_7286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 3), os_7285, 'name')
str_7287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'str', 'nt')
# Applying the binary operator '==' (line 29)
result_eq_7288 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 3), '==', name_7286, str_7287)


str_7289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'str', 'pcbuild')

# Call to lower(...): (line 29)
# Processing the call keyword arguments (line 29)
kwargs_7296 = {}

# Obtaining the type of the subscript
int_7290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 49), 'int')
slice_7291 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 29, 36), int_7290, None, None)
# Getting the type of 'project_base' (line 29)
project_base_7292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 36), 'project_base', False)
# Obtaining the member '__getitem__' of a type (line 29)
getitem___7293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 36), project_base_7292, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 29)
subscript_call_result_7294 = invoke(stypy.reporting.localization.Localization(__file__, 29, 36), getitem___7293, slice_7291)

# Obtaining the member 'lower' of a type (line 29)
lower_7295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 36), subscript_call_result_7294, 'lower')
# Calling lower(args, kwargs) (line 29)
lower_call_result_7297 = invoke(stypy.reporting.localization.Localization(__file__, 29, 36), lower_7295, *[], **kwargs_7296)

# Applying the binary operator 'in' (line 29)
result_contains_7298 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 23), 'in', str_7289, lower_call_result_7297)

# Applying the binary operator 'and' (line 29)
result_and_keyword_7299 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 3), 'and', result_eq_7288, result_contains_7298)

# Testing the type of an if condition (line 29)
if_condition_7300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 0), result_and_keyword_7299)
# Assigning a type to the variable 'if_condition_7300' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'if_condition_7300', if_condition_7300)
# SSA begins for if statement (line 29)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to abspath(...): (line 30)
# Processing the call arguments (line 30)

# Call to join(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'project_base' (line 30)
project_base_7307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 48), 'project_base', False)
# Getting the type of 'os' (line 30)
os_7308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 62), 'os', False)
# Obtaining the member 'path' of a type (line 30)
path_7309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 62), os_7308, 'path')
# Obtaining the member 'pardir' of a type (line 30)
pardir_7310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 62), path_7309, 'pardir')
# Processing the call keyword arguments (line 30)
kwargs_7311 = {}
# Getting the type of 'os' (line 30)
os_7304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'os', False)
# Obtaining the member 'path' of a type (line 30)
path_7305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 35), os_7304, 'path')
# Obtaining the member 'join' of a type (line 30)
join_7306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 35), path_7305, 'join')
# Calling join(args, kwargs) (line 30)
join_call_result_7312 = invoke(stypy.reporting.localization.Localization(__file__, 30, 35), join_7306, *[project_base_7307, pardir_7310], **kwargs_7311)

# Processing the call keyword arguments (line 30)
kwargs_7313 = {}
# Getting the type of 'os' (line 30)
os_7301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'os', False)
# Obtaining the member 'path' of a type (line 30)
path_7302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), os_7301, 'path')
# Obtaining the member 'abspath' of a type (line 30)
abspath_7303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), path_7302, 'abspath')
# Calling abspath(args, kwargs) (line 30)
abspath_call_result_7314 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), abspath_7303, *[join_call_result_7312], **kwargs_7313)

# Assigning a type to the variable 'project_base' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'project_base', abspath_call_result_7314)
# SSA join for if statement (line 29)
module_type_store = module_type_store.join_ssa_context()



# Evaluating a boolean operation

# Getting the type of 'os' (line 32)
os_7315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 3), 'os')
# Obtaining the member 'name' of a type (line 32)
name_7316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 3), os_7315, 'name')
str_7317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'str', 'nt')
# Applying the binary operator '==' (line 32)
result_eq_7318 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 3), '==', name_7316, str_7317)


str_7319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', '\\pc\\v')

# Call to lower(...): (line 32)
# Processing the call keyword arguments (line 32)
kwargs_7326 = {}

# Obtaining the type of the subscript
int_7320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 49), 'int')
slice_7321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 32, 36), int_7320, None, None)
# Getting the type of 'project_base' (line 32)
project_base_7322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 36), 'project_base', False)
# Obtaining the member '__getitem__' of a type (line 32)
getitem___7323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 36), project_base_7322, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 32)
subscript_call_result_7324 = invoke(stypy.reporting.localization.Localization(__file__, 32, 36), getitem___7323, slice_7321)

# Obtaining the member 'lower' of a type (line 32)
lower_7325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 36), subscript_call_result_7324, 'lower')
# Calling lower(args, kwargs) (line 32)
lower_call_result_7327 = invoke(stypy.reporting.localization.Localization(__file__, 32, 36), lower_7325, *[], **kwargs_7326)

# Applying the binary operator 'in' (line 32)
result_contains_7328 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), 'in', str_7319, lower_call_result_7327)

# Applying the binary operator 'and' (line 32)
result_and_keyword_7329 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 3), 'and', result_eq_7318, result_contains_7328)

# Testing the type of an if condition (line 32)
if_condition_7330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 0), result_and_keyword_7329)
# Assigning a type to the variable 'if_condition_7330' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'if_condition_7330', if_condition_7330)
# SSA begins for if statement (line 32)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 33):

# Assigning a Call to a Name (line 33):

# Call to abspath(...): (line 33)
# Processing the call arguments (line 33)

# Call to join(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'project_base' (line 33)
project_base_7337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 48), 'project_base', False)
# Getting the type of 'os' (line 33)
os_7338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 62), 'os', False)
# Obtaining the member 'path' of a type (line 33)
path_7339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 62), os_7338, 'path')
# Obtaining the member 'pardir' of a type (line 33)
pardir_7340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 62), path_7339, 'pardir')
# Getting the type of 'os' (line 34)
os_7341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 48), 'os', False)
# Obtaining the member 'path' of a type (line 34)
path_7342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 48), os_7341, 'path')
# Obtaining the member 'pardir' of a type (line 34)
pardir_7343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 48), path_7342, 'pardir')
# Processing the call keyword arguments (line 33)
kwargs_7344 = {}
# Getting the type of 'os' (line 33)
os_7334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), 'os', False)
# Obtaining the member 'path' of a type (line 33)
path_7335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 35), os_7334, 'path')
# Obtaining the member 'join' of a type (line 33)
join_7336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 35), path_7335, 'join')
# Calling join(args, kwargs) (line 33)
join_call_result_7345 = invoke(stypy.reporting.localization.Localization(__file__, 33, 35), join_7336, *[project_base_7337, pardir_7340, pardir_7343], **kwargs_7344)

# Processing the call keyword arguments (line 33)
kwargs_7346 = {}
# Getting the type of 'os' (line 33)
os_7331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'os', False)
# Obtaining the member 'path' of a type (line 33)
path_7332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), os_7331, 'path')
# Obtaining the member 'abspath' of a type (line 33)
abspath_7333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), path_7332, 'abspath')
# Calling abspath(args, kwargs) (line 33)
abspath_call_result_7347 = invoke(stypy.reporting.localization.Localization(__file__, 33, 19), abspath_7333, *[join_call_result_7345], **kwargs_7346)

# Assigning a type to the variable 'project_base' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'project_base', abspath_call_result_7347)
# SSA join for if statement (line 32)
module_type_store = module_type_store.join_ssa_context()



# Evaluating a boolean operation

# Getting the type of 'os' (line 36)
os_7348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 3), 'os')
# Obtaining the member 'name' of a type (line 36)
name_7349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 3), os_7348, 'name')
str_7350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', 'nt')
# Applying the binary operator '==' (line 36)
result_eq_7351 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 3), '==', name_7349, str_7350)


str_7352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', '\\pcbuild\\amd64')

# Call to lower(...): (line 36)
# Processing the call keyword arguments (line 36)
kwargs_7359 = {}

# Obtaining the type of the subscript
int_7353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 58), 'int')
slice_7354 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 36, 45), int_7353, None, None)
# Getting the type of 'project_base' (line 36)
project_base_7355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'project_base', False)
# Obtaining the member '__getitem__' of a type (line 36)
getitem___7356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 45), project_base_7355, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 36)
subscript_call_result_7357 = invoke(stypy.reporting.localization.Localization(__file__, 36, 45), getitem___7356, slice_7354)

# Obtaining the member 'lower' of a type (line 36)
lower_7358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 45), subscript_call_result_7357, 'lower')
# Calling lower(args, kwargs) (line 36)
lower_call_result_7360 = invoke(stypy.reporting.localization.Localization(__file__, 36, 45), lower_7358, *[], **kwargs_7359)

# Applying the binary operator 'in' (line 36)
result_contains_7361 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), 'in', str_7352, lower_call_result_7360)

# Applying the binary operator 'and' (line 36)
result_and_keyword_7362 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 3), 'and', result_eq_7351, result_contains_7361)

# Testing the type of an if condition (line 36)
if_condition_7363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 0), result_and_keyword_7362)
# Assigning a type to the variable 'if_condition_7363' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'if_condition_7363', if_condition_7363)
# SSA begins for if statement (line 36)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 37):

# Assigning a Call to a Name (line 37):

# Call to abspath(...): (line 37)
# Processing the call arguments (line 37)

# Call to join(...): (line 37)
# Processing the call arguments (line 37)
# Getting the type of 'project_base' (line 37)
project_base_7370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'project_base', False)
# Getting the type of 'os' (line 37)
os_7371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 62), 'os', False)
# Obtaining the member 'path' of a type (line 37)
path_7372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 62), os_7371, 'path')
# Obtaining the member 'pardir' of a type (line 37)
pardir_7373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 62), path_7372, 'pardir')
# Getting the type of 'os' (line 38)
os_7374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 48), 'os', False)
# Obtaining the member 'path' of a type (line 38)
path_7375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 48), os_7374, 'path')
# Obtaining the member 'pardir' of a type (line 38)
pardir_7376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 48), path_7375, 'pardir')
# Processing the call keyword arguments (line 37)
kwargs_7377 = {}
# Getting the type of 'os' (line 37)
os_7367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'os', False)
# Obtaining the member 'path' of a type (line 37)
path_7368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 35), os_7367, 'path')
# Obtaining the member 'join' of a type (line 37)
join_7369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 35), path_7368, 'join')
# Calling join(args, kwargs) (line 37)
join_call_result_7378 = invoke(stypy.reporting.localization.Localization(__file__, 37, 35), join_7369, *[project_base_7370, pardir_7373, pardir_7376], **kwargs_7377)

# Processing the call keyword arguments (line 37)
kwargs_7379 = {}
# Getting the type of 'os' (line 37)
os_7364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'os', False)
# Obtaining the member 'path' of a type (line 37)
path_7365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 19), os_7364, 'path')
# Obtaining the member 'abspath' of a type (line 37)
abspath_7366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 19), path_7365, 'abspath')
# Calling abspath(args, kwargs) (line 37)
abspath_call_result_7380 = invoke(stypy.reporting.localization.Localization(__file__, 37, 19), abspath_7366, *[join_call_result_7378], **kwargs_7379)

# Assigning a type to the variable 'project_base' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'project_base', abspath_call_result_7380)
# SSA join for if statement (line 36)
module_type_store = module_type_store.join_ssa_context()



str_7381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 3), 'str', '_PYTHON_PROJECT_BASE')
# Getting the type of 'os' (line 41)
os_7382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'os')
# Obtaining the member 'environ' of a type (line 41)
environ_7383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 29), os_7382, 'environ')
# Applying the binary operator 'in' (line 41)
result_contains_7384 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 3), 'in', str_7381, environ_7383)

# Testing the type of an if condition (line 41)
if_condition_7385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 0), result_contains_7384)
# Assigning a type to the variable 'if_condition_7385' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'if_condition_7385', if_condition_7385)
# SSA begins for if statement (line 41)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 43):

# Assigning a Call to a Name (line 43):

# Call to normpath(...): (line 43)
# Processing the call arguments (line 43)

# Obtaining the type of the subscript
str_7389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'str', '_PYTHON_PROJECT_BASE')
# Getting the type of 'os' (line 43)
os_7390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'os', False)
# Obtaining the member 'environ' of a type (line 43)
environ_7391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 36), os_7390, 'environ')
# Obtaining the member '__getitem__' of a type (line 43)
getitem___7392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 36), environ_7391, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 43)
subscript_call_result_7393 = invoke(stypy.reporting.localization.Localization(__file__, 43, 36), getitem___7392, str_7389)

# Processing the call keyword arguments (line 43)
kwargs_7394 = {}
# Getting the type of 'os' (line 43)
os_7386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'os', False)
# Obtaining the member 'path' of a type (line 43)
path_7387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 19), os_7386, 'path')
# Obtaining the member 'normpath' of a type (line 43)
normpath_7388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 19), path_7387, 'normpath')
# Calling normpath(args, kwargs) (line 43)
normpath_call_result_7395 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), normpath_7388, *[subscript_call_result_7393], **kwargs_7394)

# Assigning a type to the variable 'project_base' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'project_base', normpath_call_result_7395)
# SSA join for if statement (line 41)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _python_build(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_python_build'
    module_type_store = module_type_store.open_function_context('_python_build', 50, 0, False)
    
    # Passed parameters checking function
    _python_build.stypy_localization = localization
    _python_build.stypy_type_of_self = None
    _python_build.stypy_type_store = module_type_store
    _python_build.stypy_function_name = '_python_build'
    _python_build.stypy_param_names_list = []
    _python_build.stypy_varargs_param_name = None
    _python_build.stypy_kwargs_param_name = None
    _python_build.stypy_call_defaults = defaults
    _python_build.stypy_call_varargs = varargs
    _python_build.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_python_build', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_python_build', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_python_build(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 51)
    tuple_7396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 51)
    # Adding element type (line 51)
    str_7397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'str', 'Setup.dist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), tuple_7396, str_7397)
    # Adding element type (line 51)
    str_7398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'str', 'Setup.local')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), tuple_7396, str_7398)
    
    # Testing the type of a for loop iterable (line 51)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 4), tuple_7396)
    # Getting the type of the for loop variable (line 51)
    for_loop_var_7399 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 4), tuple_7396)
    # Assigning a type to the variable 'fn' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'fn', for_loop_var_7399)
    # SSA begins for a for statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isfile(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to join(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'project_base' (line 52)
    project_base_7406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'project_base', False)
    str_7407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 53), 'str', 'Modules')
    # Getting the type of 'fn' (line 52)
    fn_7408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 64), 'fn', False)
    # Processing the call keyword arguments (line 52)
    kwargs_7409 = {}
    # Getting the type of 'os' (line 52)
    os_7403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'os', False)
    # Obtaining the member 'path' of a type (line 52)
    path_7404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 26), os_7403, 'path')
    # Obtaining the member 'join' of a type (line 52)
    join_7405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 26), path_7404, 'join')
    # Calling join(args, kwargs) (line 52)
    join_call_result_7410 = invoke(stypy.reporting.localization.Localization(__file__, 52, 26), join_7405, *[project_base_7406, str_7407, fn_7408], **kwargs_7409)
    
    # Processing the call keyword arguments (line 52)
    kwargs_7411 = {}
    # Getting the type of 'os' (line 52)
    os_7400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 52)
    path_7401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), os_7400, 'path')
    # Obtaining the member 'isfile' of a type (line 52)
    isfile_7402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), path_7401, 'isfile')
    # Calling isfile(args, kwargs) (line 52)
    isfile_call_result_7412 = invoke(stypy.reporting.localization.Localization(__file__, 52, 11), isfile_7402, *[join_call_result_7410], **kwargs_7411)
    
    # Testing the type of an if condition (line 52)
    if_condition_7413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), isfile_call_result_7412)
    # Assigning a type to the variable 'if_condition_7413' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_7413', if_condition_7413)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 53)
    True_7414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'stypy_return_type', True_7414)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 54)
    False_7415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', False_7415)
    
    # ################# End of '_python_build(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_python_build' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_7416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7416)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_python_build'
    return stypy_return_type_7416

# Assigning a type to the variable '_python_build' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '_python_build', _python_build)

# Assigning a Call to a Name (line 55):

# Assigning a Call to a Name (line 55):

# Call to _python_build(...): (line 55)
# Processing the call keyword arguments (line 55)
kwargs_7418 = {}
# Getting the type of '_python_build' (line 55)
_python_build_7417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), '_python_build', False)
# Calling _python_build(args, kwargs) (line 55)
_python_build_call_result_7419 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), _python_build_7417, *[], **kwargs_7418)

# Assigning a type to the variable 'python_build' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'python_build', _python_build_call_result_7419)

@norecursion
def get_python_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_python_version'
    module_type_store = module_type_store.open_function_context('get_python_version', 58, 0, False)
    
    # Passed parameters checking function
    get_python_version.stypy_localization = localization
    get_python_version.stypy_type_of_self = None
    get_python_version.stypy_type_store = module_type_store
    get_python_version.stypy_function_name = 'get_python_version'
    get_python_version.stypy_param_names_list = []
    get_python_version.stypy_varargs_param_name = None
    get_python_version.stypy_kwargs_param_name = None
    get_python_version.stypy_call_defaults = defaults
    get_python_version.stypy_call_varargs = varargs
    get_python_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_python_version', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_python_version', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_python_version(...)' code ##################

    str_7420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', "Return a string containing the major and minor Python version,\n    leaving off the patchlevel.  Sample return values could be '1.5'\n    or '2.2'.\n    ")
    
    # Obtaining the type of the subscript
    int_7421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
    slice_7422 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 11), None, int_7421, None)
    # Getting the type of 'sys' (line 63)
    sys_7423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'sys')
    # Obtaining the member 'version' of a type (line 63)
    version_7424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 11), sys_7423, 'version')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___7425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 11), version_7424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_7426 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), getitem___7425, slice_7422)
    
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', subscript_call_result_7426)
    
    # ################# End of 'get_python_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_python_version' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_7427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_python_version'
    return stypy_return_type_7427

# Assigning a type to the variable 'get_python_version' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'get_python_version', get_python_version)

@norecursion
def get_python_inc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_7428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'int')
    # Getting the type of 'None' (line 66)
    None_7429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 43), 'None')
    defaults = [int_7428, None_7429]
    # Create a new context for function 'get_python_inc'
    module_type_store = module_type_store.open_function_context('get_python_inc', 66, 0, False)
    
    # Passed parameters checking function
    get_python_inc.stypy_localization = localization
    get_python_inc.stypy_type_of_self = None
    get_python_inc.stypy_type_store = module_type_store
    get_python_inc.stypy_function_name = 'get_python_inc'
    get_python_inc.stypy_param_names_list = ['plat_specific', 'prefix']
    get_python_inc.stypy_varargs_param_name = None
    get_python_inc.stypy_kwargs_param_name = None
    get_python_inc.stypy_call_defaults = defaults
    get_python_inc.stypy_call_varargs = varargs
    get_python_inc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_python_inc', ['plat_specific', 'prefix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_python_inc', localization, ['plat_specific', 'prefix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_python_inc(...)' code ##################

    str_7430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', "Return the directory containing installed Python header files.\n\n    If 'plat_specific' is false (the default), this is the path to the\n    non-platform-specific header files, i.e. Python.h and so on;\n    otherwise, this is the path to platform-specific header files\n    (namely pyconfig.h).\n\n    If 'prefix' is supplied, use it instead of sys.prefix or\n    sys.exec_prefix -- i.e., ignore 'plat_specific'.\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    # Getting the type of 'prefix' (line 77)
    prefix_7431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'prefix')
    # Getting the type of 'None' (line 77)
    None_7432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'None')
    
    (may_be_7433, more_types_in_union_7434) = may_be_none(prefix_7431, None_7432)

    if may_be_7433:

        if more_types_in_union_7434:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BoolOp to a Name (line 78):
        
        # Assigning a BoolOp to a Name (line 78):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'plat_specific' (line 78)
        plat_specific_7435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'plat_specific')
        # Getting the type of 'EXEC_PREFIX' (line 78)
        EXEC_PREFIX_7436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'EXEC_PREFIX')
        # Applying the binary operator 'and' (line 78)
        result_and_keyword_7437 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 17), 'and', plat_specific_7435, EXEC_PREFIX_7436)
        
        # Getting the type of 'PREFIX' (line 78)
        PREFIX_7438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 50), 'PREFIX')
        # Applying the binary operator 'or' (line 78)
        result_or_keyword_7439 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 17), 'or', result_and_keyword_7437, PREFIX_7438)
        
        # Assigning a type to the variable 'prefix' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'prefix', result_or_keyword_7439)

        if more_types_in_union_7434:
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'os' (line 80)
    os_7440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'os')
    # Obtaining the member 'name' of a type (line 80)
    name_7441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 7), os_7440, 'name')
    str_7442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'str', 'posix')
    # Applying the binary operator '==' (line 80)
    result_eq_7443 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '==', name_7441, str_7442)
    
    # Testing the type of an if condition (line 80)
    if_condition_7444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_eq_7443)
    # Assigning a type to the variable 'if_condition_7444' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_7444', if_condition_7444)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'python_build' (line 81)
    python_build_7445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'python_build')
    # Testing the type of an if condition (line 81)
    if_condition_7446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), python_build_7445)
    # Assigning a type to the variable 'if_condition_7446' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_7446', if_condition_7446)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to dirname(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'sys' (line 82)
    sys_7450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'sys', False)
    # Obtaining the member 'executable' of a type (line 82)
    executable_7451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), sys_7450, 'executable')
    # Processing the call keyword arguments (line 82)
    kwargs_7452 = {}
    # Getting the type of 'os' (line 82)
    os_7447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'os', False)
    # Obtaining the member 'path' of a type (line 82)
    path_7448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), os_7447, 'path')
    # Obtaining the member 'dirname' of a type (line 82)
    dirname_7449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), path_7448, 'dirname')
    # Calling dirname(args, kwargs) (line 82)
    dirname_call_result_7453 = invoke(stypy.reporting.localization.Localization(__file__, 82, 22), dirname_7449, *[executable_7451], **kwargs_7452)
    
    # Assigning a type to the variable 'buildir' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'buildir', dirname_call_result_7453)
    
    # Getting the type of 'plat_specific' (line 83)
    plat_specific_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'plat_specific')
    # Testing the type of an if condition (line 83)
    if_condition_7455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), plat_specific_7454)
    # Assigning a type to the variable 'if_condition_7455' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_7455', if_condition_7455)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 85):
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'buildir' (line 85)
    buildir_7456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'buildir')
    # Assigning a type to the variable 'inc_dir' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'inc_dir', buildir_7456)
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to abspath(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to join(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'buildir' (line 88)
    buildir_7463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 54), 'buildir', False)
    
    # Call to get_config_var(...): (line 89)
    # Processing the call arguments (line 89)
    str_7465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 56), 'str', 'srcdir')
    # Processing the call keyword arguments (line 89)
    kwargs_7466 = {}
    # Getting the type of 'get_config_var' (line 89)
    get_config_var_7464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'get_config_var', False)
    # Calling get_config_var(args, kwargs) (line 89)
    get_config_var_call_result_7467 = invoke(stypy.reporting.localization.Localization(__file__, 89, 41), get_config_var_7464, *[str_7465], **kwargs_7466)
    
    # Processing the call keyword arguments (line 88)
    kwargs_7468 = {}
    # Getting the type of 'os' (line 88)
    os_7460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 41), 'os', False)
    # Obtaining the member 'path' of a type (line 88)
    path_7461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 41), os_7460, 'path')
    # Obtaining the member 'join' of a type (line 88)
    join_7462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 41), path_7461, 'join')
    # Calling join(args, kwargs) (line 88)
    join_call_result_7469 = invoke(stypy.reporting.localization.Localization(__file__, 88, 41), join_7462, *[buildir_7463, get_config_var_call_result_7467], **kwargs_7468)
    
    # Processing the call keyword arguments (line 88)
    kwargs_7470 = {}
    # Getting the type of 'os' (line 88)
    os_7457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 88)
    path_7458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), os_7457, 'path')
    # Obtaining the member 'abspath' of a type (line 88)
    abspath_7459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), path_7458, 'abspath')
    # Calling abspath(args, kwargs) (line 88)
    abspath_call_result_7471 = invoke(stypy.reporting.localization.Localization(__file__, 88, 25), abspath_7459, *[join_call_result_7469], **kwargs_7470)
    
    # Assigning a type to the variable 'srcdir' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'srcdir', abspath_call_result_7471)
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to join(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'srcdir' (line 91)
    srcdir_7475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 39), 'srcdir', False)
    str_7476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 47), 'str', 'Include')
    # Processing the call keyword arguments (line 91)
    kwargs_7477 = {}
    # Getting the type of 'os' (line 91)
    os_7472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'os', False)
    # Obtaining the member 'path' of a type (line 91)
    path_7473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 26), os_7472, 'path')
    # Obtaining the member 'join' of a type (line 91)
    join_7474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 26), path_7473, 'join')
    # Calling join(args, kwargs) (line 91)
    join_call_result_7478 = invoke(stypy.reporting.localization.Localization(__file__, 91, 26), join_7474, *[srcdir_7475, str_7476], **kwargs_7477)
    
    # Assigning a type to the variable 'inc_dir' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'inc_dir', join_call_result_7478)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'inc_dir' (line 92)
    inc_dir_7479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'inc_dir')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'stypy_return_type', inc_dir_7479)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'prefix' (line 93)
    prefix_7483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'prefix', False)
    str_7484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 36), 'str', 'include')
    str_7485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 47), 'str', 'python')
    
    # Call to get_python_version(...): (line 93)
    # Processing the call keyword arguments (line 93)
    kwargs_7487 = {}
    # Getting the type of 'get_python_version' (line 93)
    get_python_version_7486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 58), 'get_python_version', False)
    # Calling get_python_version(args, kwargs) (line 93)
    get_python_version_call_result_7488 = invoke(stypy.reporting.localization.Localization(__file__, 93, 58), get_python_version_7486, *[], **kwargs_7487)
    
    # Applying the binary operator '+' (line 93)
    result_add_7489 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 47), '+', str_7485, get_python_version_call_result_7488)
    
    # Processing the call keyword arguments (line 93)
    kwargs_7490 = {}
    # Getting the type of 'os' (line 93)
    os_7480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 93)
    path_7481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), os_7480, 'path')
    # Obtaining the member 'join' of a type (line 93)
    join_7482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), path_7481, 'join')
    # Calling join(args, kwargs) (line 93)
    join_call_result_7491 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), join_7482, *[prefix_7483, str_7484, result_add_7489], **kwargs_7490)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', join_call_result_7491)
    # SSA branch for the else part of an if statement (line 80)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 94)
    os_7492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 9), 'os')
    # Obtaining the member 'name' of a type (line 94)
    name_7493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 9), os_7492, 'name')
    str_7494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'str', 'nt')
    # Applying the binary operator '==' (line 94)
    result_eq_7495 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 9), '==', name_7493, str_7494)
    
    # Testing the type of an if condition (line 94)
    if_condition_7496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 9), result_eq_7495)
    # Assigning a type to the variable 'if_condition_7496' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 9), 'if_condition_7496', if_condition_7496)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'prefix' (line 95)
    prefix_7500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'prefix', False)
    str_7501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'str', 'include')
    # Processing the call keyword arguments (line 95)
    kwargs_7502 = {}
    # Getting the type of 'os' (line 95)
    os_7497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 95)
    path_7498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), os_7497, 'path')
    # Obtaining the member 'join' of a type (line 95)
    join_7499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), path_7498, 'join')
    # Calling join(args, kwargs) (line 95)
    join_call_result_7503 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), join_7499, *[prefix_7500, str_7501], **kwargs_7502)
    
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', join_call_result_7503)
    # SSA branch for the else part of an if statement (line 94)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 96)
    os_7504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'os')
    # Obtaining the member 'name' of a type (line 96)
    name_7505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 9), os_7504, 'name')
    str_7506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'str', 'os2')
    # Applying the binary operator '==' (line 96)
    result_eq_7507 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 9), '==', name_7505, str_7506)
    
    # Testing the type of an if condition (line 96)
    if_condition_7508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 9), result_eq_7507)
    # Assigning a type to the variable 'if_condition_7508' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'if_condition_7508', if_condition_7508)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'prefix' (line 97)
    prefix_7512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'prefix', False)
    str_7513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 36), 'str', 'Include')
    # Processing the call keyword arguments (line 97)
    kwargs_7514 = {}
    # Getting the type of 'os' (line 97)
    os_7509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 97)
    path_7510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), os_7509, 'path')
    # Obtaining the member 'join' of a type (line 97)
    join_7511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), path_7510, 'join')
    # Calling join(args, kwargs) (line 97)
    join_call_result_7515 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), join_7511, *[prefix_7512, str_7513], **kwargs_7514)
    
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', join_call_result_7515)
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    # Call to DistutilsPlatformError(...): (line 99)
    # Processing the call arguments (line 99)
    str_7517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'str', "I don't know where Python installs its C header files on platform '%s'")
    # Getting the type of 'os' (line 101)
    os_7518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'os', False)
    # Obtaining the member 'name' of a type (line 101)
    name_7519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 33), os_7518, 'name')
    # Applying the binary operator '%' (line 100)
    result_mod_7520 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), '%', str_7517, name_7519)
    
    # Processing the call keyword arguments (line 99)
    kwargs_7521 = {}
    # Getting the type of 'DistutilsPlatformError' (line 99)
    DistutilsPlatformError_7516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'DistutilsPlatformError', False)
    # Calling DistutilsPlatformError(args, kwargs) (line 99)
    DistutilsPlatformError_call_result_7522 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), DistutilsPlatformError_7516, *[result_mod_7520], **kwargs_7521)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 8), DistutilsPlatformError_call_result_7522, 'raise parameter', BaseException)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_python_inc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_python_inc' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_7523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_python_inc'
    return stypy_return_type_7523

# Assigning a type to the variable 'get_python_inc' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'get_python_inc', get_python_inc)

@norecursion
def get_python_lib(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_7524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'int')
    int_7525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 49), 'int')
    # Getting the type of 'None' (line 104)
    None_7526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 59), 'None')
    defaults = [int_7524, int_7525, None_7526]
    # Create a new context for function 'get_python_lib'
    module_type_store = module_type_store.open_function_context('get_python_lib', 104, 0, False)
    
    # Passed parameters checking function
    get_python_lib.stypy_localization = localization
    get_python_lib.stypy_type_of_self = None
    get_python_lib.stypy_type_store = module_type_store
    get_python_lib.stypy_function_name = 'get_python_lib'
    get_python_lib.stypy_param_names_list = ['plat_specific', 'standard_lib', 'prefix']
    get_python_lib.stypy_varargs_param_name = None
    get_python_lib.stypy_kwargs_param_name = None
    get_python_lib.stypy_call_defaults = defaults
    get_python_lib.stypy_call_varargs = varargs
    get_python_lib.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_python_lib', ['plat_specific', 'standard_lib', 'prefix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_python_lib', localization, ['plat_specific', 'standard_lib', 'prefix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_python_lib(...)' code ##################

    str_7527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', "Return the directory containing the Python library (standard or\n    site additions).\n\n    If 'plat_specific' is true, return the directory containing\n    platform-specific modules, i.e. any module from a non-pure-Python\n    module distribution; otherwise, return the platform-shared library\n    directory.  If 'standard_lib' is true, return the directory\n    containing standard Python library modules; otherwise, return the\n    directory for site-specific modules.\n\n    If 'prefix' is supplied, use it instead of sys.prefix or\n    sys.exec_prefix -- i.e., ignore 'plat_specific'.\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 118)
    # Getting the type of 'prefix' (line 118)
    prefix_7528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'prefix')
    # Getting the type of 'None' (line 118)
    None_7529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'None')
    
    (may_be_7530, more_types_in_union_7531) = may_be_none(prefix_7528, None_7529)

    if may_be_7530:

        if more_types_in_union_7531:
            # Runtime conditional SSA (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BoolOp to a Name (line 119):
        
        # Assigning a BoolOp to a Name (line 119):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'plat_specific' (line 119)
        plat_specific_7532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'plat_specific')
        # Getting the type of 'EXEC_PREFIX' (line 119)
        EXEC_PREFIX_7533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'EXEC_PREFIX')
        # Applying the binary operator 'and' (line 119)
        result_and_keyword_7534 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 17), 'and', plat_specific_7532, EXEC_PREFIX_7533)
        
        # Getting the type of 'PREFIX' (line 119)
        PREFIX_7535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 50), 'PREFIX')
        # Applying the binary operator 'or' (line 119)
        result_or_keyword_7536 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 17), 'or', result_and_keyword_7534, PREFIX_7535)
        
        # Assigning a type to the variable 'prefix' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'prefix', result_or_keyword_7536)

        if more_types_in_union_7531:
            # SSA join for if statement (line 118)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'os' (line 121)
    os_7537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'os')
    # Obtaining the member 'name' of a type (line 121)
    name_7538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 7), os_7537, 'name')
    str_7539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 18), 'str', 'posix')
    # Applying the binary operator '==' (line 121)
    result_eq_7540 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), '==', name_7538, str_7539)
    
    # Testing the type of an if condition (line 121)
    if_condition_7541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_eq_7540)
    # Assigning a type to the variable 'if_condition_7541' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_7541', if_condition_7541)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to join(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'prefix' (line 122)
    prefix_7545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'prefix', False)
    str_7546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 33), 'str', 'lib')
    str_7547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 40), 'str', 'python')
    
    # Call to get_python_version(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_7549 = {}
    # Getting the type of 'get_python_version' (line 123)
    get_python_version_7548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 51), 'get_python_version', False)
    # Calling get_python_version(args, kwargs) (line 123)
    get_python_version_call_result_7550 = invoke(stypy.reporting.localization.Localization(__file__, 123, 51), get_python_version_7548, *[], **kwargs_7549)
    
    # Applying the binary operator '+' (line 123)
    result_add_7551 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 40), '+', str_7547, get_python_version_call_result_7550)
    
    # Processing the call keyword arguments (line 122)
    kwargs_7552 = {}
    # Getting the type of 'os' (line 122)
    os_7542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 122)
    path_7543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 20), os_7542, 'path')
    # Obtaining the member 'join' of a type (line 122)
    join_7544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 20), path_7543, 'join')
    # Calling join(args, kwargs) (line 122)
    join_call_result_7553 = invoke(stypy.reporting.localization.Localization(__file__, 122, 20), join_7544, *[prefix_7545, str_7546, result_add_7551], **kwargs_7552)
    
    # Assigning a type to the variable 'libpython' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'libpython', join_call_result_7553)
    
    # Getting the type of 'standard_lib' (line 124)
    standard_lib_7554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'standard_lib')
    # Testing the type of an if condition (line 124)
    if_condition_7555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), standard_lib_7554)
    # Assigning a type to the variable 'if_condition_7555' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_7555', if_condition_7555)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'libpython' (line 125)
    libpython_7556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'libpython')
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', libpython_7556)
    # SSA branch for the else part of an if statement (line 124)
    module_type_store.open_ssa_branch('else')
    
    # Call to join(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'libpython' (line 127)
    libpython_7560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'libpython', False)
    str_7561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 43), 'str', 'site-packages')
    # Processing the call keyword arguments (line 127)
    kwargs_7562 = {}
    # Getting the type of 'os' (line 127)
    os_7557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 127)
    path_7558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), os_7557, 'path')
    # Obtaining the member 'join' of a type (line 127)
    join_7559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), path_7558, 'join')
    # Calling join(args, kwargs) (line 127)
    join_call_result_7563 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), join_7559, *[libpython_7560, str_7561], **kwargs_7562)
    
    # Assigning a type to the variable 'stypy_return_type' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'stypy_return_type', join_call_result_7563)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 121)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 129)
    os_7564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'os')
    # Obtaining the member 'name' of a type (line 129)
    name_7565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), os_7564, 'name')
    str_7566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'str', 'nt')
    # Applying the binary operator '==' (line 129)
    result_eq_7567 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 9), '==', name_7565, str_7566)
    
    # Testing the type of an if condition (line 129)
    if_condition_7568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 9), result_eq_7567)
    # Assigning a type to the variable 'if_condition_7568' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'if_condition_7568', if_condition_7568)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'standard_lib' (line 130)
    standard_lib_7569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'standard_lib')
    # Testing the type of an if condition (line 130)
    if_condition_7570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), standard_lib_7569)
    # Assigning a type to the variable 'if_condition_7570' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_7570', if_condition_7570)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'prefix' (line 131)
    prefix_7574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 32), 'prefix', False)
    str_7575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 40), 'str', 'Lib')
    # Processing the call keyword arguments (line 131)
    kwargs_7576 = {}
    # Getting the type of 'os' (line 131)
    os_7571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 131)
    path_7572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 19), os_7571, 'path')
    # Obtaining the member 'join' of a type (line 131)
    join_7573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 19), path_7572, 'join')
    # Calling join(args, kwargs) (line 131)
    join_call_result_7577 = invoke(stypy.reporting.localization.Localization(__file__, 131, 19), join_7573, *[prefix_7574, str_7575], **kwargs_7576)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', join_call_result_7577)
    # SSA branch for the else part of an if statement (line 130)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to get_python_version(...): (line 133)
    # Processing the call keyword arguments (line 133)
    kwargs_7579 = {}
    # Getting the type of 'get_python_version' (line 133)
    get_python_version_7578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'get_python_version', False)
    # Calling get_python_version(args, kwargs) (line 133)
    get_python_version_call_result_7580 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), get_python_version_7578, *[], **kwargs_7579)
    
    str_7581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'str', '2.2')
    # Applying the binary operator '<' (line 133)
    result_lt_7582 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), '<', get_python_version_call_result_7580, str_7581)
    
    # Testing the type of an if condition (line 133)
    if_condition_7583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 12), result_lt_7582)
    # Assigning a type to the variable 'if_condition_7583' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'if_condition_7583', if_condition_7583)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'prefix' (line 134)
    prefix_7584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'prefix')
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'stypy_return_type', prefix_7584)
    # SSA branch for the else part of an if statement (line 133)
    module_type_store.open_ssa_branch('else')
    
    # Call to join(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'prefix' (line 136)
    prefix_7588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 36), 'prefix', False)
    str_7589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 44), 'str', 'Lib')
    str_7590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 51), 'str', 'site-packages')
    # Processing the call keyword arguments (line 136)
    kwargs_7591 = {}
    # Getting the type of 'os' (line 136)
    os_7585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 136)
    path_7586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 23), os_7585, 'path')
    # Obtaining the member 'join' of a type (line 136)
    join_7587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 23), path_7586, 'join')
    # Calling join(args, kwargs) (line 136)
    join_call_result_7592 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), join_7587, *[prefix_7588, str_7589, str_7590], **kwargs_7591)
    
    # Assigning a type to the variable 'stypy_return_type' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'stypy_return_type', join_call_result_7592)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 129)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 138)
    os_7593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'os')
    # Obtaining the member 'name' of a type (line 138)
    name_7594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 9), os_7593, 'name')
    str_7595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'str', 'os2')
    # Applying the binary operator '==' (line 138)
    result_eq_7596 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 9), '==', name_7594, str_7595)
    
    # Testing the type of an if condition (line 138)
    if_condition_7597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 9), result_eq_7596)
    # Assigning a type to the variable 'if_condition_7597' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'if_condition_7597', if_condition_7597)
    # SSA begins for if statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'standard_lib' (line 139)
    standard_lib_7598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'standard_lib')
    # Testing the type of an if condition (line 139)
    if_condition_7599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), standard_lib_7598)
    # Assigning a type to the variable 'if_condition_7599' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_7599', if_condition_7599)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'prefix' (line 140)
    prefix_7603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'prefix', False)
    str_7604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 40), 'str', 'Lib')
    # Processing the call keyword arguments (line 140)
    kwargs_7605 = {}
    # Getting the type of 'os' (line 140)
    os_7600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 140)
    path_7601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), os_7600, 'path')
    # Obtaining the member 'join' of a type (line 140)
    join_7602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), path_7601, 'join')
    # Calling join(args, kwargs) (line 140)
    join_call_result_7606 = invoke(stypy.reporting.localization.Localization(__file__, 140, 19), join_7602, *[prefix_7603, str_7604], **kwargs_7605)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'stypy_return_type', join_call_result_7606)
    # SSA branch for the else part of an if statement (line 139)
    module_type_store.open_ssa_branch('else')
    
    # Call to join(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'prefix' (line 142)
    prefix_7610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'prefix', False)
    str_7611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'str', 'Lib')
    str_7612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 47), 'str', 'site-packages')
    # Processing the call keyword arguments (line 142)
    kwargs_7613 = {}
    # Getting the type of 'os' (line 142)
    os_7607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 142)
    path_7608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), os_7607, 'path')
    # Obtaining the member 'join' of a type (line 142)
    join_7609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), path_7608, 'join')
    # Calling join(args, kwargs) (line 142)
    join_call_result_7614 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), join_7609, *[prefix_7610, str_7611, str_7612], **kwargs_7613)
    
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'stypy_return_type', join_call_result_7614)
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 138)
    module_type_store.open_ssa_branch('else')
    
    # Call to DistutilsPlatformError(...): (line 145)
    # Processing the call arguments (line 145)
    str_7616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'str', "I don't know where Python installs its library on platform '%s'")
    # Getting the type of 'os' (line 147)
    os_7617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'os', False)
    # Obtaining the member 'name' of a type (line 147)
    name_7618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 33), os_7617, 'name')
    # Applying the binary operator '%' (line 146)
    result_mod_7619 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), '%', str_7616, name_7618)
    
    # Processing the call keyword arguments (line 145)
    kwargs_7620 = {}
    # Getting the type of 'DistutilsPlatformError' (line 145)
    DistutilsPlatformError_7615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 14), 'DistutilsPlatformError', False)
    # Calling DistutilsPlatformError(args, kwargs) (line 145)
    DistutilsPlatformError_call_result_7621 = invoke(stypy.reporting.localization.Localization(__file__, 145, 14), DistutilsPlatformError_7615, *[result_mod_7619], **kwargs_7620)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 145, 8), DistutilsPlatformError_call_result_7621, 'raise parameter', BaseException)
    # SSA join for if statement (line 138)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_python_lib(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_python_lib' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_7622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_python_lib'
    return stypy_return_type_7622

# Assigning a type to the variable 'get_python_lib' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'get_python_lib', get_python_lib)

@norecursion
def customize_compiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'customize_compiler'
    module_type_store = module_type_store.open_function_context('customize_compiler', 151, 0, False)
    
    # Passed parameters checking function
    customize_compiler.stypy_localization = localization
    customize_compiler.stypy_type_of_self = None
    customize_compiler.stypy_type_store = module_type_store
    customize_compiler.stypy_function_name = 'customize_compiler'
    customize_compiler.stypy_param_names_list = ['compiler']
    customize_compiler.stypy_varargs_param_name = None
    customize_compiler.stypy_kwargs_param_name = None
    customize_compiler.stypy_call_defaults = defaults
    customize_compiler.stypy_call_varargs = varargs
    customize_compiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'customize_compiler', ['compiler'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'customize_compiler', localization, ['compiler'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'customize_compiler(...)' code ##################

    str_7623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', "Do any platform-specific customization of a CCompiler instance.\n\n    Mainly needed on Unix, so we can plug in the information that\n    varies across Unices and is stored in Python's Makefile.\n    ")
    
    
    # Getting the type of 'compiler' (line 157)
    compiler_7624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 7), 'compiler')
    # Obtaining the member 'compiler_type' of a type (line 157)
    compiler_type_7625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 7), compiler_7624, 'compiler_type')
    str_7626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'str', 'unix')
    # Applying the binary operator '==' (line 157)
    result_eq_7627 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 7), '==', compiler_type_7625, str_7626)
    
    # Testing the type of an if condition (line 157)
    if_condition_7628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 4), result_eq_7627)
    # Assigning a type to the variable 'if_condition_7628' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'if_condition_7628', if_condition_7628)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'sys' (line 158)
    sys_7629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'sys')
    # Obtaining the member 'platform' of a type (line 158)
    platform_7630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), sys_7629, 'platform')
    str_7631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 27), 'str', 'darwin')
    # Applying the binary operator '==' (line 158)
    result_eq_7632 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), '==', platform_7630, str_7631)
    
    # Testing the type of an if condition (line 158)
    if_condition_7633 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_7632)
    # Assigning a type to the variable 'if_condition_7633' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_7633', if_condition_7633)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Marking variables as global (line 167)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 167, 12), '_config_vars')
    
    
    
    # Call to get_config_var(...): (line 169)
    # Processing the call arguments (line 169)
    str_7635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 34), 'str', 'CUSTOMIZED_OSX_COMPILER')
    # Processing the call keyword arguments (line 169)
    kwargs_7636 = {}
    # Getting the type of 'get_config_var' (line 169)
    get_config_var_7634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'get_config_var', False)
    # Calling get_config_var(args, kwargs) (line 169)
    get_config_var_call_result_7637 = invoke(stypy.reporting.localization.Localization(__file__, 169, 19), get_config_var_7634, *[str_7635], **kwargs_7636)
    
    # Applying the 'not' unary operator (line 169)
    result_not__7638 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), 'not', get_config_var_call_result_7637)
    
    # Testing the type of an if condition (line 169)
    if_condition_7639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), result_not__7638)
    # Assigning a type to the variable 'if_condition_7639' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_7639', if_condition_7639)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 170, 16))
    
    # 'import _osx_support' statement (line 170)
    import _osx_support

    import_module(stypy.reporting.localization.Localization(__file__, 170, 16), '_osx_support', _osx_support, module_type_store)
    
    
    # Call to customize_compiler(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of '_config_vars' (line 171)
    _config_vars_7642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), '_config_vars', False)
    # Processing the call keyword arguments (line 171)
    kwargs_7643 = {}
    # Getting the type of '_osx_support' (line 171)
    _osx_support_7640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), '_osx_support', False)
    # Obtaining the member 'customize_compiler' of a type (line 171)
    customize_compiler_7641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), _osx_support_7640, 'customize_compiler')
    # Calling customize_compiler(args, kwargs) (line 171)
    customize_compiler_call_result_7644 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), customize_compiler_7641, *[_config_vars_7642], **kwargs_7643)
    
    
    # Assigning a Str to a Subscript (line 172):
    
    # Assigning a Str to a Subscript (line 172):
    str_7645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 58), 'str', 'True')
    # Getting the type of '_config_vars' (line 172)
    _config_vars_7646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), '_config_vars')
    str_7647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'str', 'CUSTOMIZED_OSX_COMPILER')
    # Storing an element on a container (line 172)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 16), _config_vars_7646, (str_7647, str_7645))
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 174):
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7659 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7660 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7649, *[str_7650, str_7651, str_7652, str_7653, str_7654, str_7655, str_7656, str_7657, str_7658], **kwargs_7659)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7662 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7661, int_7648)
    
    # Assigning a type to the variable 'tuple_var_assignment_7240' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7240', subscript_call_result_7662)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7674 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7675 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7664, *[str_7665, str_7666, str_7667, str_7668, str_7669, str_7670, str_7671, str_7672, str_7673], **kwargs_7674)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7677 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7676, int_7663)
    
    # Assigning a type to the variable 'tuple_var_assignment_7241' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7241', subscript_call_result_7677)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7689 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7690 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7679, *[str_7680, str_7681, str_7682, str_7683, str_7684, str_7685, str_7686, str_7687, str_7688], **kwargs_7689)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7692 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7691, int_7678)
    
    # Assigning a type to the variable 'tuple_var_assignment_7242' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7242', subscript_call_result_7692)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7704 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7705 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7694, *[str_7695, str_7696, str_7697, str_7698, str_7699, str_7700, str_7701, str_7702, str_7703], **kwargs_7704)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7707 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7706, int_7693)
    
    # Assigning a type to the variable 'tuple_var_assignment_7243' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7243', subscript_call_result_7707)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7719 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7720 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7709, *[str_7710, str_7711, str_7712, str_7713, str_7714, str_7715, str_7716, str_7717, str_7718], **kwargs_7719)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7722 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7721, int_7708)
    
    # Assigning a type to the variable 'tuple_var_assignment_7244' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7244', subscript_call_result_7722)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7734 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7735 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7724, *[str_7725, str_7726, str_7727, str_7728, str_7729, str_7730, str_7731, str_7732, str_7733], **kwargs_7734)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7735, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7737 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7736, int_7723)
    
    # Assigning a type to the variable 'tuple_var_assignment_7245' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7245', subscript_call_result_7737)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7749 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7750 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7739, *[str_7740, str_7741, str_7742, str_7743, str_7744, str_7745, str_7746, str_7747, str_7748], **kwargs_7749)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7752 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7751, int_7738)
    
    # Assigning a type to the variable 'tuple_var_assignment_7246' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7246', subscript_call_result_7752)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7764 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7765 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7754, *[str_7755, str_7756, str_7757, str_7758, str_7759, str_7760, str_7761, str_7762, str_7763], **kwargs_7764)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7767 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7766, int_7753)
    
    # Assigning a type to the variable 'tuple_var_assignment_7247' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7247', subscript_call_result_7767)
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_7768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
    
    # Call to get_config_vars(...): (line 175)
    # Processing the call arguments (line 175)
    str_7770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'CC')
    str_7771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'str', 'CXX')
    str_7772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', 'OPT')
    str_7773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'str', 'CFLAGS')
    str_7774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'CCSHARED')
    str_7775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'LDSHARED')
    str_7776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'SO')
    str_7777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'AR')
    str_7778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'ARFLAGS')
    # Processing the call keyword arguments (line 175)
    kwargs_7779 = {}
    # Getting the type of 'get_config_vars' (line 175)
    get_config_vars_7769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 175)
    get_config_vars_call_result_7780 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_config_vars_7769, *[str_7770, str_7771, str_7772, str_7773, str_7774, str_7775, str_7776, str_7777, str_7778], **kwargs_7779)
    
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___7781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), get_config_vars_call_result_7780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_7782 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___7781, int_7768)
    
    # Assigning a type to the variable 'tuple_var_assignment_7248' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7248', subscript_call_result_7782)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7240' (line 174)
    tuple_var_assignment_7240_7783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7240')
    # Assigning a type to the variable 'cc' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 9), 'cc', tuple_var_assignment_7240_7783)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7241' (line 174)
    tuple_var_assignment_7241_7784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7241')
    # Assigning a type to the variable 'cxx' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'cxx', tuple_var_assignment_7241_7784)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7242' (line 174)
    tuple_var_assignment_7242_7785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7242')
    # Assigning a type to the variable 'opt' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'opt', tuple_var_assignment_7242_7785)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7243' (line 174)
    tuple_var_assignment_7243_7786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7243')
    # Assigning a type to the variable 'cflags' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'cflags', tuple_var_assignment_7243_7786)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7244' (line 174)
    tuple_var_assignment_7244_7787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7244')
    # Assigning a type to the variable 'ccshared' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'ccshared', tuple_var_assignment_7244_7787)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7245' (line 174)
    tuple_var_assignment_7245_7788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7245')
    # Assigning a type to the variable 'ldshared' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 41), 'ldshared', tuple_var_assignment_7245_7788)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7246' (line 174)
    tuple_var_assignment_7246_7789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7246')
    # Assigning a type to the variable 'so_ext' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 51), 'so_ext', tuple_var_assignment_7246_7789)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7247' (line 174)
    tuple_var_assignment_7247_7790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7247')
    # Assigning a type to the variable 'ar' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 59), 'ar', tuple_var_assignment_7247_7790)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_var_assignment_7248' (line 174)
    tuple_var_assignment_7248_7791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_7248')
    # Assigning a type to the variable 'ar_flags' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 63), 'ar_flags', tuple_var_assignment_7248_7791)
    
    
    str_7792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 11), 'str', 'CC')
    # Getting the type of 'os' (line 179)
    os_7793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'os')
    # Obtaining the member 'environ' of a type (line 179)
    environ_7794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), os_7793, 'environ')
    # Applying the binary operator 'in' (line 179)
    result_contains_7795 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 11), 'in', str_7792, environ_7794)
    
    # Testing the type of an if condition (line 179)
    if_condition_7796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 8), result_contains_7795)
    # Assigning a type to the variable 'if_condition_7796' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'if_condition_7796', if_condition_7796)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 180):
    
    # Assigning a Subscript to a Name (line 180):
    
    # Obtaining the type of the subscript
    str_7797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'str', 'CC')
    # Getting the type of 'os' (line 180)
    os_7798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'os')
    # Obtaining the member 'environ' of a type (line 180)
    environ_7799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), os_7798, 'environ')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___7800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), environ_7799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_7801 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), getitem___7800, str_7797)
    
    # Assigning a type to the variable 'newcc' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'newcc', subscript_call_result_7801)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sys' (line 181)
    sys_7802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'sys')
    # Obtaining the member 'platform' of a type (line 181)
    platform_7803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), sys_7802, 'platform')
    str_7804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 32), 'str', 'darwin')
    # Applying the binary operator '==' (line 181)
    result_eq_7805 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 16), '==', platform_7803, str_7804)
    
    
    str_7806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 24), 'str', 'LDSHARED')
    # Getting the type of 'os' (line 182)
    os_7807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 42), 'os')
    # Obtaining the member 'environ' of a type (line 182)
    environ_7808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 42), os_7807, 'environ')
    # Applying the binary operator 'notin' (line 182)
    result_contains_7809 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 24), 'notin', str_7806, environ_7808)
    
    # Applying the binary operator 'and' (line 181)
    result_and_keyword_7810 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 16), 'and', result_eq_7805, result_contains_7809)
    
    # Call to startswith(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'cc' (line 183)
    cc_7813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 44), 'cc', False)
    # Processing the call keyword arguments (line 183)
    kwargs_7814 = {}
    # Getting the type of 'ldshared' (line 183)
    ldshared_7811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'ldshared', False)
    # Obtaining the member 'startswith' of a type (line 183)
    startswith_7812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 24), ldshared_7811, 'startswith')
    # Calling startswith(args, kwargs) (line 183)
    startswith_call_result_7815 = invoke(stypy.reporting.localization.Localization(__file__, 183, 24), startswith_7812, *[cc_7813], **kwargs_7814)
    
    # Applying the binary operator 'and' (line 181)
    result_and_keyword_7816 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 16), 'and', result_and_keyword_7810, startswith_call_result_7815)
    
    # Testing the type of an if condition (line 181)
    if_condition_7817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 12), result_and_keyword_7816)
    # Assigning a type to the variable 'if_condition_7817' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'if_condition_7817', if_condition_7817)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 186):
    
    # Assigning a BinOp to a Name (line 186):
    # Getting the type of 'newcc' (line 186)
    newcc_7818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'newcc')
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'cc' (line 186)
    cc_7820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 48), 'cc', False)
    # Processing the call keyword arguments (line 186)
    kwargs_7821 = {}
    # Getting the type of 'len' (line 186)
    len_7819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 44), 'len', False)
    # Calling len(args, kwargs) (line 186)
    len_call_result_7822 = invoke(stypy.reporting.localization.Localization(__file__, 186, 44), len_7819, *[cc_7820], **kwargs_7821)
    
    slice_7823 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 186, 35), len_call_result_7822, None, None)
    # Getting the type of 'ldshared' (line 186)
    ldshared_7824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 35), 'ldshared')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___7825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 35), ldshared_7824, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_7826 = invoke(stypy.reporting.localization.Localization(__file__, 186, 35), getitem___7825, slice_7823)
    
    # Applying the binary operator '+' (line 186)
    result_add_7827 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 27), '+', newcc_7818, subscript_call_result_7826)
    
    # Assigning a type to the variable 'ldshared' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'ldshared', result_add_7827)
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 187):
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'newcc' (line 187)
    newcc_7828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'newcc')
    # Assigning a type to the variable 'cc' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'cc', newcc_7828)
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 11), 'str', 'CXX')
    # Getting the type of 'os' (line 188)
    os_7830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'os')
    # Obtaining the member 'environ' of a type (line 188)
    environ_7831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 20), os_7830, 'environ')
    # Applying the binary operator 'in' (line 188)
    result_contains_7832 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), 'in', str_7829, environ_7831)
    
    # Testing the type of an if condition (line 188)
    if_condition_7833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), result_contains_7832)
    # Assigning a type to the variable 'if_condition_7833' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_7833', if_condition_7833)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 189):
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    str_7834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 29), 'str', 'CXX')
    # Getting the type of 'os' (line 189)
    os_7835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'os')
    # Obtaining the member 'environ' of a type (line 189)
    environ_7836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 18), os_7835, 'environ')
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___7837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 18), environ_7836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_7838 = invoke(stypy.reporting.localization.Localization(__file__, 189, 18), getitem___7837, str_7834)
    
    # Assigning a type to the variable 'cxx' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'cxx', subscript_call_result_7838)
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'str', 'LDSHARED')
    # Getting the type of 'os' (line 190)
    os_7840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'os')
    # Obtaining the member 'environ' of a type (line 190)
    environ_7841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 25), os_7840, 'environ')
    # Applying the binary operator 'in' (line 190)
    result_contains_7842 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), 'in', str_7839, environ_7841)
    
    # Testing the type of an if condition (line 190)
    if_condition_7843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_contains_7842)
    # Assigning a type to the variable 'if_condition_7843' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_7843', if_condition_7843)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 191):
    
    # Assigning a Subscript to a Name (line 191):
    
    # Obtaining the type of the subscript
    str_7844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 34), 'str', 'LDSHARED')
    # Getting the type of 'os' (line 191)
    os_7845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'os')
    # Obtaining the member 'environ' of a type (line 191)
    environ_7846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 23), os_7845, 'environ')
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___7847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 23), environ_7846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_7848 = invoke(stypy.reporting.localization.Localization(__file__, 191, 23), getitem___7847, str_7844)
    
    # Assigning a type to the variable 'ldshared' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'ldshared', subscript_call_result_7848)
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 11), 'str', 'CPP')
    # Getting the type of 'os' (line 192)
    os_7850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'os')
    # Obtaining the member 'environ' of a type (line 192)
    environ_7851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), os_7850, 'environ')
    # Applying the binary operator 'in' (line 192)
    result_contains_7852 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 11), 'in', str_7849, environ_7851)
    
    # Testing the type of an if condition (line 192)
    if_condition_7853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), result_contains_7852)
    # Assigning a type to the variable 'if_condition_7853' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_7853', if_condition_7853)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 193):
    
    # Assigning a Subscript to a Name (line 193):
    
    # Obtaining the type of the subscript
    str_7854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 29), 'str', 'CPP')
    # Getting the type of 'os' (line 193)
    os_7855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'os')
    # Obtaining the member 'environ' of a type (line 193)
    environ_7856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 18), os_7855, 'environ')
    # Obtaining the member '__getitem__' of a type (line 193)
    getitem___7857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 18), environ_7856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 193)
    subscript_call_result_7858 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), getitem___7857, str_7854)
    
    # Assigning a type to the variable 'cpp' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'cpp', subscript_call_result_7858)
    # SSA branch for the else part of an if statement (line 192)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 195):
    
    # Assigning a BinOp to a Name (line 195):
    # Getting the type of 'cc' (line 195)
    cc_7859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'cc')
    str_7860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'str', ' -E')
    # Applying the binary operator '+' (line 195)
    result_add_7861 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 18), '+', cc_7859, str_7860)
    
    # Assigning a type to the variable 'cpp' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'cpp', result_add_7861)
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 11), 'str', 'LDFLAGS')
    # Getting the type of 'os' (line 196)
    os_7863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'os')
    # Obtaining the member 'environ' of a type (line 196)
    environ_7864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 24), os_7863, 'environ')
    # Applying the binary operator 'in' (line 196)
    result_contains_7865 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), 'in', str_7862, environ_7864)
    
    # Testing the type of an if condition (line 196)
    if_condition_7866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), result_contains_7865)
    # Assigning a type to the variable 'if_condition_7866' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_7866', if_condition_7866)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 197):
    
    # Assigning a BinOp to a Name (line 197):
    # Getting the type of 'ldshared' (line 197)
    ldshared_7867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 23), 'ldshared')
    str_7868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 34), 'str', ' ')
    # Applying the binary operator '+' (line 197)
    result_add_7869 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 23), '+', ldshared_7867, str_7868)
    
    
    # Obtaining the type of the subscript
    str_7870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 51), 'str', 'LDFLAGS')
    # Getting the type of 'os' (line 197)
    os_7871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 40), 'os')
    # Obtaining the member 'environ' of a type (line 197)
    environ_7872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 40), os_7871, 'environ')
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___7873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 40), environ_7872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_7874 = invoke(stypy.reporting.localization.Localization(__file__, 197, 40), getitem___7873, str_7870)
    
    # Applying the binary operator '+' (line 197)
    result_add_7875 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 38), '+', result_add_7869, subscript_call_result_7874)
    
    # Assigning a type to the variable 'ldshared' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'ldshared', result_add_7875)
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 11), 'str', 'CFLAGS')
    # Getting the type of 'os' (line 198)
    os_7877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'os')
    # Obtaining the member 'environ' of a type (line 198)
    environ_7878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 23), os_7877, 'environ')
    # Applying the binary operator 'in' (line 198)
    result_contains_7879 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), 'in', str_7876, environ_7878)
    
    # Testing the type of an if condition (line 198)
    if_condition_7880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_contains_7879)
    # Assigning a type to the variable 'if_condition_7880' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_7880', if_condition_7880)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 199):
    
    # Assigning a BinOp to a Name (line 199):
    # Getting the type of 'opt' (line 199)
    opt_7881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'opt')
    str_7882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 27), 'str', ' ')
    # Applying the binary operator '+' (line 199)
    result_add_7883 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 21), '+', opt_7881, str_7882)
    
    
    # Obtaining the type of the subscript
    str_7884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 44), 'str', 'CFLAGS')
    # Getting the type of 'os' (line 199)
    os_7885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'os')
    # Obtaining the member 'environ' of a type (line 199)
    environ_7886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 33), os_7885, 'environ')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___7887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 33), environ_7886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_7888 = invoke(stypy.reporting.localization.Localization(__file__, 199, 33), getitem___7887, str_7884)
    
    # Applying the binary operator '+' (line 199)
    result_add_7889 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 31), '+', result_add_7883, subscript_call_result_7888)
    
    # Assigning a type to the variable 'cflags' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'cflags', result_add_7889)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    # Getting the type of 'ldshared' (line 200)
    ldshared_7890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'ldshared')
    str_7891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'str', ' ')
    # Applying the binary operator '+' (line 200)
    result_add_7892 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 23), '+', ldshared_7890, str_7891)
    
    
    # Obtaining the type of the subscript
    str_7893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 51), 'str', 'CFLAGS')
    # Getting the type of 'os' (line 200)
    os_7894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'os')
    # Obtaining the member 'environ' of a type (line 200)
    environ_7895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 40), os_7894, 'environ')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___7896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 40), environ_7895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_7897 = invoke(stypy.reporting.localization.Localization(__file__, 200, 40), getitem___7896, str_7893)
    
    # Applying the binary operator '+' (line 200)
    result_add_7898 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 38), '+', result_add_7892, subscript_call_result_7897)
    
    # Assigning a type to the variable 'ldshared' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'ldshared', result_add_7898)
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 11), 'str', 'CPPFLAGS')
    # Getting the type of 'os' (line 201)
    os_7900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'os')
    # Obtaining the member 'environ' of a type (line 201)
    environ_7901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 25), os_7900, 'environ')
    # Applying the binary operator 'in' (line 201)
    result_contains_7902 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), 'in', str_7899, environ_7901)
    
    # Testing the type of an if condition (line 201)
    if_condition_7903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_contains_7902)
    # Assigning a type to the variable 'if_condition_7903' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_7903', if_condition_7903)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 202):
    
    # Assigning a BinOp to a Name (line 202):
    # Getting the type of 'cpp' (line 202)
    cpp_7904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'cpp')
    str_7905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 24), 'str', ' ')
    # Applying the binary operator '+' (line 202)
    result_add_7906 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 18), '+', cpp_7904, str_7905)
    
    
    # Obtaining the type of the subscript
    str_7907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 41), 'str', 'CPPFLAGS')
    # Getting the type of 'os' (line 202)
    os_7908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'os')
    # Obtaining the member 'environ' of a type (line 202)
    environ_7909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 30), os_7908, 'environ')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___7910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 30), environ_7909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_7911 = invoke(stypy.reporting.localization.Localization(__file__, 202, 30), getitem___7910, str_7907)
    
    # Applying the binary operator '+' (line 202)
    result_add_7912 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '+', result_add_7906, subscript_call_result_7911)
    
    # Assigning a type to the variable 'cpp' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'cpp', result_add_7912)
    
    # Assigning a BinOp to a Name (line 203):
    
    # Assigning a BinOp to a Name (line 203):
    # Getting the type of 'cflags' (line 203)
    cflags_7913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'cflags')
    str_7914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'str', ' ')
    # Applying the binary operator '+' (line 203)
    result_add_7915 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 21), '+', cflags_7913, str_7914)
    
    
    # Obtaining the type of the subscript
    str_7916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 47), 'str', 'CPPFLAGS')
    # Getting the type of 'os' (line 203)
    os_7917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 36), 'os')
    # Obtaining the member 'environ' of a type (line 203)
    environ_7918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 36), os_7917, 'environ')
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___7919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 36), environ_7918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_7920 = invoke(stypy.reporting.localization.Localization(__file__, 203, 36), getitem___7919, str_7916)
    
    # Applying the binary operator '+' (line 203)
    result_add_7921 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 34), '+', result_add_7915, subscript_call_result_7920)
    
    # Assigning a type to the variable 'cflags' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'cflags', result_add_7921)
    
    # Assigning a BinOp to a Name (line 204):
    
    # Assigning a BinOp to a Name (line 204):
    # Getting the type of 'ldshared' (line 204)
    ldshared_7922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'ldshared')
    str_7923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 34), 'str', ' ')
    # Applying the binary operator '+' (line 204)
    result_add_7924 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 23), '+', ldshared_7922, str_7923)
    
    
    # Obtaining the type of the subscript
    str_7925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 51), 'str', 'CPPFLAGS')
    # Getting the type of 'os' (line 204)
    os_7926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 40), 'os')
    # Obtaining the member 'environ' of a type (line 204)
    environ_7927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 40), os_7926, 'environ')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___7928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 40), environ_7927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_7929 = invoke(stypy.reporting.localization.Localization(__file__, 204, 40), getitem___7928, str_7925)
    
    # Applying the binary operator '+' (line 204)
    result_add_7930 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 38), '+', result_add_7924, subscript_call_result_7929)
    
    # Assigning a type to the variable 'ldshared' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'ldshared', result_add_7930)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 11), 'str', 'AR')
    # Getting the type of 'os' (line 205)
    os_7932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'os')
    # Obtaining the member 'environ' of a type (line 205)
    environ_7933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 19), os_7932, 'environ')
    # Applying the binary operator 'in' (line 205)
    result_contains_7934 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'in', str_7931, environ_7933)
    
    # Testing the type of an if condition (line 205)
    if_condition_7935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_contains_7934)
    # Assigning a type to the variable 'if_condition_7935' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_7935', if_condition_7935)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 206):
    
    # Assigning a Subscript to a Name (line 206):
    
    # Obtaining the type of the subscript
    str_7936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 28), 'str', 'AR')
    # Getting the type of 'os' (line 206)
    os_7937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'os')
    # Obtaining the member 'environ' of a type (line 206)
    environ_7938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 17), os_7937, 'environ')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___7939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 17), environ_7938, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_7940 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), getitem___7939, str_7936)
    
    # Assigning a type to the variable 'ar' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'ar', subscript_call_result_7940)
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_7941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 11), 'str', 'ARFLAGS')
    # Getting the type of 'os' (line 207)
    os_7942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'os')
    # Obtaining the member 'environ' of a type (line 207)
    environ_7943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 24), os_7942, 'environ')
    # Applying the binary operator 'in' (line 207)
    result_contains_7944 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), 'in', str_7941, environ_7943)
    
    # Testing the type of an if condition (line 207)
    if_condition_7945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_contains_7944)
    # Assigning a type to the variable 'if_condition_7945' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_7945', if_condition_7945)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 208):
    
    # Assigning a BinOp to a Name (line 208):
    # Getting the type of 'ar' (line 208)
    ar_7946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'ar')
    str_7947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'str', ' ')
    # Applying the binary operator '+' (line 208)
    result_add_7948 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 23), '+', ar_7946, str_7947)
    
    
    # Obtaining the type of the subscript
    str_7949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 45), 'str', 'ARFLAGS')
    # Getting the type of 'os' (line 208)
    os_7950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 34), 'os')
    # Obtaining the member 'environ' of a type (line 208)
    environ_7951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 34), os_7950, 'environ')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___7952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 34), environ_7951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_7953 = invoke(stypy.reporting.localization.Localization(__file__, 208, 34), getitem___7952, str_7949)
    
    # Applying the binary operator '+' (line 208)
    result_add_7954 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 32), '+', result_add_7948, subscript_call_result_7953)
    
    # Assigning a type to the variable 'archiver' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'archiver', result_add_7954)
    # SSA branch for the else part of an if statement (line 207)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 210):
    
    # Assigning a BinOp to a Name (line 210):
    # Getting the type of 'ar' (line 210)
    ar_7955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'ar')
    str_7956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 28), 'str', ' ')
    # Applying the binary operator '+' (line 210)
    result_add_7957 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 23), '+', ar_7955, str_7956)
    
    # Getting the type of 'ar_flags' (line 210)
    ar_flags_7958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'ar_flags')
    # Applying the binary operator '+' (line 210)
    result_add_7959 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 32), '+', result_add_7957, ar_flags_7958)
    
    # Assigning a type to the variable 'archiver' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'archiver', result_add_7959)
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 212):
    
    # Assigning a BinOp to a Name (line 212):
    # Getting the type of 'cc' (line 212)
    cc_7960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'cc')
    str_7961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 22), 'str', ' ')
    # Applying the binary operator '+' (line 212)
    result_add_7962 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 17), '+', cc_7960, str_7961)
    
    # Getting the type of 'cflags' (line 212)
    cflags_7963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'cflags')
    # Applying the binary operator '+' (line 212)
    result_add_7964 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 26), '+', result_add_7962, cflags_7963)
    
    # Assigning a type to the variable 'cc_cmd' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'cc_cmd', result_add_7964)
    
    # Call to set_executables(...): (line 213)
    # Processing the call keyword arguments (line 213)
    # Getting the type of 'cpp' (line 214)
    cpp_7967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'cpp', False)
    keyword_7968 = cpp_7967
    # Getting the type of 'cc_cmd' (line 215)
    cc_cmd_7969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'cc_cmd', False)
    keyword_7970 = cc_cmd_7969
    # Getting the type of 'cc_cmd' (line 216)
    cc_cmd_7971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'cc_cmd', False)
    str_7972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 33), 'str', ' ')
    # Applying the binary operator '+' (line 216)
    result_add_7973 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 24), '+', cc_cmd_7971, str_7972)
    
    # Getting the type of 'ccshared' (line 216)
    ccshared_7974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'ccshared', False)
    # Applying the binary operator '+' (line 216)
    result_add_7975 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 37), '+', result_add_7973, ccshared_7974)
    
    keyword_7976 = result_add_7975
    # Getting the type of 'cxx' (line 217)
    cxx_7977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'cxx', False)
    keyword_7978 = cxx_7977
    # Getting the type of 'ldshared' (line 218)
    ldshared_7979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'ldshared', False)
    keyword_7980 = ldshared_7979
    # Getting the type of 'cc' (line 219)
    cc_7981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'cc', False)
    keyword_7982 = cc_7981
    # Getting the type of 'archiver' (line 220)
    archiver_7983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'archiver', False)
    keyword_7984 = archiver_7983
    kwargs_7985 = {'compiler_cxx': keyword_7978, 'linker_exe': keyword_7982, 'compiler_so': keyword_7976, 'archiver': keyword_7984, 'preprocessor': keyword_7968, 'linker_so': keyword_7980, 'compiler': keyword_7970}
    # Getting the type of 'compiler' (line 213)
    compiler_7965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'compiler', False)
    # Obtaining the member 'set_executables' of a type (line 213)
    set_executables_7966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), compiler_7965, 'set_executables')
    # Calling set_executables(args, kwargs) (line 213)
    set_executables_call_result_7986 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), set_executables_7966, *[], **kwargs_7985)
    
    
    # Assigning a Name to a Attribute (line 222):
    
    # Assigning a Name to a Attribute (line 222):
    # Getting the type of 'so_ext' (line 222)
    so_ext_7987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 40), 'so_ext')
    # Getting the type of 'compiler' (line 222)
    compiler_7988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'compiler')
    # Setting the type of the member 'shared_lib_extension' of a type (line 222)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), compiler_7988, 'shared_lib_extension', so_ext_7987)
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'customize_compiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'customize_compiler' in the type store
    # Getting the type of 'stypy_return_type' (line 151)
    stypy_return_type_7989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7989)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'customize_compiler'
    return stypy_return_type_7989

# Assigning a type to the variable 'customize_compiler' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'customize_compiler', customize_compiler)

@norecursion
def get_config_h_filename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_config_h_filename'
    module_type_store = module_type_store.open_function_context('get_config_h_filename', 225, 0, False)
    
    # Passed parameters checking function
    get_config_h_filename.stypy_localization = localization
    get_config_h_filename.stypy_type_of_self = None
    get_config_h_filename.stypy_type_store = module_type_store
    get_config_h_filename.stypy_function_name = 'get_config_h_filename'
    get_config_h_filename.stypy_param_names_list = []
    get_config_h_filename.stypy_varargs_param_name = None
    get_config_h_filename.stypy_kwargs_param_name = None
    get_config_h_filename.stypy_call_defaults = defaults
    get_config_h_filename.stypy_call_varargs = varargs
    get_config_h_filename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_config_h_filename', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_config_h_filename', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_config_h_filename(...)' code ##################

    str_7990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'str', 'Return full pathname of installed pyconfig.h file.')
    
    # Getting the type of 'python_build' (line 227)
    python_build_7991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'python_build')
    # Testing the type of an if condition (line 227)
    if_condition_7992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), python_build_7991)
    # Assigning a type to the variable 'if_condition_7992' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_7992', if_condition_7992)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'os' (line 228)
    os_7993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'os')
    # Obtaining the member 'name' of a type (line 228)
    name_7994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), os_7993, 'name')
    str_7995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'str', 'nt')
    # Applying the binary operator '==' (line 228)
    result_eq_7996 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '==', name_7994, str_7995)
    
    # Testing the type of an if condition (line 228)
    if_condition_7997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), result_eq_7996)
    # Assigning a type to the variable 'if_condition_7997' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_7997', if_condition_7997)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to join(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'project_base' (line 229)
    project_base_8001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'project_base', False)
    str_8002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 49), 'str', 'PC')
    # Processing the call keyword arguments (line 229)
    kwargs_8003 = {}
    # Getting the type of 'os' (line 229)
    os_7998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 'os', False)
    # Obtaining the member 'path' of a type (line 229)
    path_7999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 22), os_7998, 'path')
    # Obtaining the member 'join' of a type (line 229)
    join_8000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 22), path_7999, 'join')
    # Calling join(args, kwargs) (line 229)
    join_call_result_8004 = invoke(stypy.reporting.localization.Localization(__file__, 229, 22), join_8000, *[project_base_8001, str_8002], **kwargs_8003)
    
    # Assigning a type to the variable 'inc_dir' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'inc_dir', join_call_result_8004)
    # SSA branch for the else part of an if statement (line 228)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 231):
    
    # Assigning a Name to a Name (line 231):
    # Getting the type of 'project_base' (line 231)
    project_base_8005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'project_base')
    # Assigning a type to the variable 'inc_dir' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'inc_dir', project_base_8005)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 227)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 233):
    
    # Assigning a Call to a Name (line 233):
    
    # Call to get_python_inc(...): (line 233)
    # Processing the call keyword arguments (line 233)
    int_8007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 47), 'int')
    keyword_8008 = int_8007
    kwargs_8009 = {'plat_specific': keyword_8008}
    # Getting the type of 'get_python_inc' (line 233)
    get_python_inc_8006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 233)
    get_python_inc_call_result_8010 = invoke(stypy.reporting.localization.Localization(__file__, 233, 18), get_python_inc_8006, *[], **kwargs_8009)
    
    # Assigning a type to the variable 'inc_dir' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'inc_dir', get_python_inc_call_result_8010)
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to get_python_version(...): (line 234)
    # Processing the call keyword arguments (line 234)
    kwargs_8012 = {}
    # Getting the type of 'get_python_version' (line 234)
    get_python_version_8011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 7), 'get_python_version', False)
    # Calling get_python_version(args, kwargs) (line 234)
    get_python_version_call_result_8013 = invoke(stypy.reporting.localization.Localization(__file__, 234, 7), get_python_version_8011, *[], **kwargs_8012)
    
    str_8014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 30), 'str', '2.2')
    # Applying the binary operator '<' (line 234)
    result_lt_8015 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 7), '<', get_python_version_call_result_8013, str_8014)
    
    # Testing the type of an if condition (line 234)
    if_condition_8016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 4), result_lt_8015)
    # Assigning a type to the variable 'if_condition_8016' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'if_condition_8016', if_condition_8016)
    # SSA begins for if statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 235):
    
    # Assigning a Str to a Name (line 235):
    str_8017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 19), 'str', 'config.h')
    # Assigning a type to the variable 'config_h' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'config_h', str_8017)
    # SSA branch for the else part of an if statement (line 234)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 238):
    
    # Assigning a Str to a Name (line 238):
    str_8018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'str', 'pyconfig.h')
    # Assigning a type to the variable 'config_h' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'config_h', str_8018)
    # SSA join for if statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'inc_dir' (line 239)
    inc_dir_8022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'inc_dir', False)
    # Getting the type of 'config_h' (line 239)
    config_h_8023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 33), 'config_h', False)
    # Processing the call keyword arguments (line 239)
    kwargs_8024 = {}
    # Getting the type of 'os' (line 239)
    os_8019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 239)
    path_8020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 11), os_8019, 'path')
    # Obtaining the member 'join' of a type (line 239)
    join_8021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 11), path_8020, 'join')
    # Calling join(args, kwargs) (line 239)
    join_call_result_8025 = invoke(stypy.reporting.localization.Localization(__file__, 239, 11), join_8021, *[inc_dir_8022, config_h_8023], **kwargs_8024)
    
    # Assigning a type to the variable 'stypy_return_type' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type', join_call_result_8025)
    
    # ################# End of 'get_config_h_filename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_config_h_filename' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_8026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_config_h_filename'
    return stypy_return_type_8026

# Assigning a type to the variable 'get_config_h_filename' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'get_config_h_filename', get_config_h_filename)

@norecursion
def get_makefile_filename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_makefile_filename'
    module_type_store = module_type_store.open_function_context('get_makefile_filename', 242, 0, False)
    
    # Passed parameters checking function
    get_makefile_filename.stypy_localization = localization
    get_makefile_filename.stypy_type_of_self = None
    get_makefile_filename.stypy_type_store = module_type_store
    get_makefile_filename.stypy_function_name = 'get_makefile_filename'
    get_makefile_filename.stypy_param_names_list = []
    get_makefile_filename.stypy_varargs_param_name = None
    get_makefile_filename.stypy_kwargs_param_name = None
    get_makefile_filename.stypy_call_defaults = defaults
    get_makefile_filename.stypy_call_varargs = varargs
    get_makefile_filename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_makefile_filename', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_makefile_filename', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_makefile_filename(...)' code ##################

    str_8027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 4), 'str', 'Return full pathname of installed Makefile from the Python build.')
    
    # Getting the type of 'python_build' (line 244)
    python_build_8028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 7), 'python_build')
    # Testing the type of an if condition (line 244)
    if_condition_8029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 4), python_build_8028)
    # Assigning a type to the variable 'if_condition_8029' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'if_condition_8029', if_condition_8029)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'project_base' (line 245)
    project_base_8033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'project_base', False)
    str_8034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 42), 'str', 'Makefile')
    # Processing the call keyword arguments (line 245)
    kwargs_8035 = {}
    # Getting the type of 'os' (line 245)
    os_8030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 245)
    path_8031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), os_8030, 'path')
    # Obtaining the member 'join' of a type (line 245)
    join_8032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), path_8031, 'join')
    # Calling join(args, kwargs) (line 245)
    join_call_result_8036 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), join_8032, *[project_base_8033, str_8034], **kwargs_8035)
    
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', join_call_result_8036)
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to get_python_lib(...): (line 246)
    # Processing the call keyword arguments (line 246)
    int_8038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 43), 'int')
    keyword_8039 = int_8038
    int_8040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 59), 'int')
    keyword_8041 = int_8040
    kwargs_8042 = {'standard_lib': keyword_8041, 'plat_specific': keyword_8039}
    # Getting the type of 'get_python_lib' (line 246)
    get_python_lib_8037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 14), 'get_python_lib', False)
    # Calling get_python_lib(args, kwargs) (line 246)
    get_python_lib_call_result_8043 = invoke(stypy.reporting.localization.Localization(__file__, 246, 14), get_python_lib_8037, *[], **kwargs_8042)
    
    # Assigning a type to the variable 'lib_dir' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'lib_dir', get_python_lib_call_result_8043)
    
    # Call to join(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'lib_dir' (line 247)
    lib_dir_8047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'lib_dir', False)
    str_8048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 33), 'str', 'config')
    str_8049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 43), 'str', 'Makefile')
    # Processing the call keyword arguments (line 247)
    kwargs_8050 = {}
    # Getting the type of 'os' (line 247)
    os_8044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 247)
    path_8045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 11), os_8044, 'path')
    # Obtaining the member 'join' of a type (line 247)
    join_8046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 11), path_8045, 'join')
    # Calling join(args, kwargs) (line 247)
    join_call_result_8051 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), join_8046, *[lib_dir_8047, str_8048, str_8049], **kwargs_8050)
    
    # Assigning a type to the variable 'stypy_return_type' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type', join_call_result_8051)
    
    # ################# End of 'get_makefile_filename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_makefile_filename' in the type store
    # Getting the type of 'stypy_return_type' (line 242)
    stypy_return_type_8052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8052)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_makefile_filename'
    return stypy_return_type_8052

# Assigning a type to the variable 'get_makefile_filename' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'get_makefile_filename', get_makefile_filename)

@norecursion
def parse_config_h(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 250)
    None_8053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'None')
    defaults = [None_8053]
    # Create a new context for function 'parse_config_h'
    module_type_store = module_type_store.open_function_context('parse_config_h', 250, 0, False)
    
    # Passed parameters checking function
    parse_config_h.stypy_localization = localization
    parse_config_h.stypy_type_of_self = None
    parse_config_h.stypy_type_store = module_type_store
    parse_config_h.stypy_function_name = 'parse_config_h'
    parse_config_h.stypy_param_names_list = ['fp', 'g']
    parse_config_h.stypy_varargs_param_name = None
    parse_config_h.stypy_kwargs_param_name = None
    parse_config_h.stypy_call_defaults = defaults
    parse_config_h.stypy_call_varargs = varargs
    parse_config_h.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_config_h', ['fp', 'g'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_config_h', localization, ['fp', 'g'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_config_h(...)' code ##################

    str_8054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, (-1)), 'str', 'Parse a config.h-style file.\n\n    A dictionary containing name/value pairs is returned.  If an\n    optional dictionary is passed in as the second argument, it is\n    used instead of a new dictionary.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 257)
    # Getting the type of 'g' (line 257)
    g_8055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 7), 'g')
    # Getting the type of 'None' (line 257)
    None_8056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'None')
    
    (may_be_8057, more_types_in_union_8058) = may_be_none(g_8055, None_8056)

    if may_be_8057:

        if more_types_in_union_8058:
            # Runtime conditional SSA (line 257)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 258):
        
        # Assigning a Dict to a Name (line 258):
        
        # Obtaining an instance of the builtin type 'dict' (line 258)
        dict_8059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 258)
        
        # Assigning a type to the variable 'g' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'g', dict_8059)

        if more_types_in_union_8058:
            # SSA join for if statement (line 257)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to compile(...): (line 259)
    # Processing the call arguments (line 259)
    str_8062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'str', '#define ([A-Z][A-Za-z0-9_]+) (.*)\n')
    # Processing the call keyword arguments (line 259)
    kwargs_8063 = {}
    # Getting the type of 're' (line 259)
    re_8060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 're', False)
    # Obtaining the member 'compile' of a type (line 259)
    compile_8061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 16), re_8060, 'compile')
    # Calling compile(args, kwargs) (line 259)
    compile_call_result_8064 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), compile_8061, *[str_8062], **kwargs_8063)
    
    # Assigning a type to the variable 'define_rx' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'define_rx', compile_call_result_8064)
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to compile(...): (line 260)
    # Processing the call arguments (line 260)
    str_8067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 26), 'str', '/[*] #undef ([A-Z][A-Za-z0-9_]+) [*]/\n')
    # Processing the call keyword arguments (line 260)
    kwargs_8068 = {}
    # Getting the type of 're' (line 260)
    re_8065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 're', False)
    # Obtaining the member 'compile' of a type (line 260)
    compile_8066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), re_8065, 'compile')
    # Calling compile(args, kwargs) (line 260)
    compile_call_result_8069 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), compile_8066, *[str_8067], **kwargs_8068)
    
    # Assigning a type to the variable 'undef_rx' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'undef_rx', compile_call_result_8069)
    
    int_8070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 10), 'int')
    # Testing the type of an if condition (line 262)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 4), int_8070)
    # SSA begins for while statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 263):
    
    # Assigning a Call to a Name (line 263):
    
    # Call to readline(...): (line 263)
    # Processing the call keyword arguments (line 263)
    kwargs_8073 = {}
    # Getting the type of 'fp' (line 263)
    fp_8071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'fp', False)
    # Obtaining the member 'readline' of a type (line 263)
    readline_8072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), fp_8071, 'readline')
    # Calling readline(args, kwargs) (line 263)
    readline_call_result_8074 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), readline_8072, *[], **kwargs_8073)
    
    # Assigning a type to the variable 'line' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'line', readline_call_result_8074)
    
    
    # Getting the type of 'line' (line 264)
    line_8075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'line')
    # Applying the 'not' unary operator (line 264)
    result_not__8076 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 11), 'not', line_8075)
    
    # Testing the type of an if condition (line 264)
    if_condition_8077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 8), result_not__8076)
    # Assigning a type to the variable 'if_condition_8077' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'if_condition_8077', if_condition_8077)
    # SSA begins for if statement (line 264)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 264)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to match(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'line' (line 266)
    line_8080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'line', False)
    # Processing the call keyword arguments (line 266)
    kwargs_8081 = {}
    # Getting the type of 'define_rx' (line 266)
    define_rx_8078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'define_rx', False)
    # Obtaining the member 'match' of a type (line 266)
    match_8079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), define_rx_8078, 'match')
    # Calling match(args, kwargs) (line 266)
    match_call_result_8082 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), match_8079, *[line_8080], **kwargs_8081)
    
    # Assigning a type to the variable 'm' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'm', match_call_result_8082)
    
    # Getting the type of 'm' (line 267)
    m_8083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'm')
    # Testing the type of an if condition (line 267)
    if_condition_8084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), m_8083)
    # Assigning a type to the variable 'if_condition_8084' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_8084', if_condition_8084)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 268):
    
    # Assigning a Subscript to a Name (line 268):
    
    # Obtaining the type of the subscript
    int_8085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 12), 'int')
    
    # Call to group(...): (line 268)
    # Processing the call arguments (line 268)
    int_8088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 27), 'int')
    int_8089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 30), 'int')
    # Processing the call keyword arguments (line 268)
    kwargs_8090 = {}
    # Getting the type of 'm' (line 268)
    m_8086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 268)
    group_8087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), m_8086, 'group')
    # Calling group(args, kwargs) (line 268)
    group_call_result_8091 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), group_8087, *[int_8088, int_8089], **kwargs_8090)
    
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___8092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), group_call_result_8091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_8093 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), getitem___8092, int_8085)
    
    # Assigning a type to the variable 'tuple_var_assignment_7249' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'tuple_var_assignment_7249', subscript_call_result_8093)
    
    # Assigning a Subscript to a Name (line 268):
    
    # Obtaining the type of the subscript
    int_8094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 12), 'int')
    
    # Call to group(...): (line 268)
    # Processing the call arguments (line 268)
    int_8097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 27), 'int')
    int_8098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 30), 'int')
    # Processing the call keyword arguments (line 268)
    kwargs_8099 = {}
    # Getting the type of 'm' (line 268)
    m_8095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 268)
    group_8096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), m_8095, 'group')
    # Calling group(args, kwargs) (line 268)
    group_call_result_8100 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), group_8096, *[int_8097, int_8098], **kwargs_8099)
    
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___8101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), group_call_result_8100, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_8102 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), getitem___8101, int_8094)
    
    # Assigning a type to the variable 'tuple_var_assignment_7250' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'tuple_var_assignment_7250', subscript_call_result_8102)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'tuple_var_assignment_7249' (line 268)
    tuple_var_assignment_7249_8103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'tuple_var_assignment_7249')
    # Assigning a type to the variable 'n' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'n', tuple_var_assignment_7249_8103)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'tuple_var_assignment_7250' (line 268)
    tuple_var_assignment_7250_8104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'tuple_var_assignment_7250')
    # Assigning a type to the variable 'v' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'v', tuple_var_assignment_7250_8104)
    
    
    # SSA begins for try-except statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to int(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'v' (line 269)
    v_8106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'v', False)
    # Processing the call keyword arguments (line 269)
    kwargs_8107 = {}
    # Getting the type of 'int' (line 269)
    int_8105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'int', False)
    # Calling int(args, kwargs) (line 269)
    int_call_result_8108 = invoke(stypy.reporting.localization.Localization(__file__, 269, 21), int_8105, *[v_8106], **kwargs_8107)
    
    # Assigning a type to the variable 'v' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'v', int_call_result_8108)
    # SSA branch for the except part of a try statement (line 269)
    # SSA branch for the except 'ValueError' branch of a try statement (line 269)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 271):
    
    # Assigning a Name to a Subscript (line 271):
    # Getting the type of 'v' (line 271)
    v_8109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'v')
    # Getting the type of 'g' (line 271)
    g_8110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'g')
    # Getting the type of 'n' (line 271)
    n_8111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'n')
    # Storing an element on a container (line 271)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 12), g_8110, (n_8111, v_8109))
    # SSA branch for the else part of an if statement (line 267)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to match(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'line' (line 273)
    line_8114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 31), 'line', False)
    # Processing the call keyword arguments (line 273)
    kwargs_8115 = {}
    # Getting the type of 'undef_rx' (line 273)
    undef_rx_8112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'undef_rx', False)
    # Obtaining the member 'match' of a type (line 273)
    match_8113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), undef_rx_8112, 'match')
    # Calling match(args, kwargs) (line 273)
    match_call_result_8116 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), match_8113, *[line_8114], **kwargs_8115)
    
    # Assigning a type to the variable 'm' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'm', match_call_result_8116)
    
    # Getting the type of 'm' (line 274)
    m_8117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'm')
    # Testing the type of an if condition (line 274)
    if_condition_8118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 12), m_8117)
    # Assigning a type to the variable 'if_condition_8118' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'if_condition_8118', if_condition_8118)
    # SSA begins for if statement (line 274)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 275):
    
    # Assigning a Num to a Subscript (line 275):
    int_8119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 32), 'int')
    # Getting the type of 'g' (line 275)
    g_8120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'g')
    
    # Call to group(...): (line 275)
    # Processing the call arguments (line 275)
    int_8123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 26), 'int')
    # Processing the call keyword arguments (line 275)
    kwargs_8124 = {}
    # Getting the type of 'm' (line 275)
    m_8121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'm', False)
    # Obtaining the member 'group' of a type (line 275)
    group_8122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 18), m_8121, 'group')
    # Calling group(args, kwargs) (line 275)
    group_call_result_8125 = invoke(stypy.reporting.localization.Localization(__file__, 275, 18), group_8122, *[int_8123], **kwargs_8124)
    
    # Storing an element on a container (line 275)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), g_8120, (group_call_result_8125, int_8119))
    # SSA join for if statement (line 274)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'g' (line 276)
    g_8126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type', g_8126)
    
    # ################# End of 'parse_config_h(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_config_h' in the type store
    # Getting the type of 'stypy_return_type' (line 250)
    stypy_return_type_8127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8127)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_config_h'
    return stypy_return_type_8127

# Assigning a type to the variable 'parse_config_h' (line 250)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'parse_config_h', parse_config_h)

# Assigning a Call to a Name (line 281):

# Assigning a Call to a Name (line 281):

# Call to compile(...): (line 281)
# Processing the call arguments (line 281)
str_8130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 26), 'str', '([a-zA-Z][a-zA-Z0-9_]+)\\s*=\\s*(.*)')
# Processing the call keyword arguments (line 281)
kwargs_8131 = {}
# Getting the type of 're' (line 281)
re_8128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 're', False)
# Obtaining the member 'compile' of a type (line 281)
compile_8129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 15), re_8128, 'compile')
# Calling compile(args, kwargs) (line 281)
compile_call_result_8132 = invoke(stypy.reporting.localization.Localization(__file__, 281, 15), compile_8129, *[str_8130], **kwargs_8131)

# Assigning a type to the variable '_variable_rx' (line 281)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), '_variable_rx', compile_call_result_8132)

# Assigning a Call to a Name (line 282):

# Assigning a Call to a Name (line 282):

# Call to compile(...): (line 282)
# Processing the call arguments (line 282)
str_8135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 26), 'str', '\\$\\(([A-Za-z][A-Za-z0-9_]*)\\)')
# Processing the call keyword arguments (line 282)
kwargs_8136 = {}
# Getting the type of 're' (line 282)
re_8133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 're', False)
# Obtaining the member 'compile' of a type (line 282)
compile_8134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), re_8133, 'compile')
# Calling compile(args, kwargs) (line 282)
compile_call_result_8137 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), compile_8134, *[str_8135], **kwargs_8136)

# Assigning a type to the variable '_findvar1_rx' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), '_findvar1_rx', compile_call_result_8137)

# Assigning a Call to a Name (line 283):

# Assigning a Call to a Name (line 283):

# Call to compile(...): (line 283)
# Processing the call arguments (line 283)
str_8140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 26), 'str', '\\${([A-Za-z][A-Za-z0-9_]*)}')
# Processing the call keyword arguments (line 283)
kwargs_8141 = {}
# Getting the type of 're' (line 283)
re_8138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 're', False)
# Obtaining the member 'compile' of a type (line 283)
compile_8139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 15), re_8138, 'compile')
# Calling compile(args, kwargs) (line 283)
compile_call_result_8142 = invoke(stypy.reporting.localization.Localization(__file__, 283, 15), compile_8139, *[str_8140], **kwargs_8141)

# Assigning a type to the variable '_findvar2_rx' (line 283)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 0), '_findvar2_rx', compile_call_result_8142)

@norecursion
def parse_makefile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 285)
    None_8143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'None')
    defaults = [None_8143]
    # Create a new context for function 'parse_makefile'
    module_type_store = module_type_store.open_function_context('parse_makefile', 285, 0, False)
    
    # Passed parameters checking function
    parse_makefile.stypy_localization = localization
    parse_makefile.stypy_type_of_self = None
    parse_makefile.stypy_type_store = module_type_store
    parse_makefile.stypy_function_name = 'parse_makefile'
    parse_makefile.stypy_param_names_list = ['fn', 'g']
    parse_makefile.stypy_varargs_param_name = None
    parse_makefile.stypy_kwargs_param_name = None
    parse_makefile.stypy_call_defaults = defaults
    parse_makefile.stypy_call_varargs = varargs
    parse_makefile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_makefile', ['fn', 'g'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_makefile', localization, ['fn', 'g'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_makefile(...)' code ##################

    str_8144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'str', 'Parse a Makefile-style file.\n\n    A dictionary containing name/value pairs is returned.  If an\n    optional dictionary is passed in as the second argument, it is\n    used instead of a new dictionary.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 292, 4))
    
    # 'from distutils.text_file import TextFile' statement (line 292)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_8145 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 292, 4), 'distutils.text_file')

    if (type(import_8145) is not StypyTypeError):

        if (import_8145 != 'pyd_module'):
            __import__(import_8145)
            sys_modules_8146 = sys.modules[import_8145]
            import_from_module(stypy.reporting.localization.Localization(__file__, 292, 4), 'distutils.text_file', sys_modules_8146.module_type_store, module_type_store, ['TextFile'])
            nest_module(stypy.reporting.localization.Localization(__file__, 292, 4), __file__, sys_modules_8146, sys_modules_8146.module_type_store, module_type_store)
        else:
            from distutils.text_file import TextFile

            import_from_module(stypy.reporting.localization.Localization(__file__, 292, 4), 'distutils.text_file', None, module_type_store, ['TextFile'], [TextFile])

    else:
        # Assigning a type to the variable 'distutils.text_file' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'distutils.text_file', import_8145)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    
    # Assigning a Call to a Name (line 293):
    
    # Assigning a Call to a Name (line 293):
    
    # Call to TextFile(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'fn' (line 293)
    fn_8148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'fn', False)
    # Processing the call keyword arguments (line 293)
    int_8149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 37), 'int')
    keyword_8150 = int_8149
    int_8151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 52), 'int')
    keyword_8152 = int_8151
    int_8153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 66), 'int')
    keyword_8154 = int_8153
    kwargs_8155 = {'strip_comments': keyword_8150, 'join_lines': keyword_8154, 'skip_blanks': keyword_8152}
    # Getting the type of 'TextFile' (line 293)
    TextFile_8147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 9), 'TextFile', False)
    # Calling TextFile(args, kwargs) (line 293)
    TextFile_call_result_8156 = invoke(stypy.reporting.localization.Localization(__file__, 293, 9), TextFile_8147, *[fn_8148], **kwargs_8155)
    
    # Assigning a type to the variable 'fp' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'fp', TextFile_call_result_8156)
    
    # Type idiom detected: calculating its left and rigth part (line 295)
    # Getting the type of 'g' (line 295)
    g_8157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 7), 'g')
    # Getting the type of 'None' (line 295)
    None_8158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'None')
    
    (may_be_8159, more_types_in_union_8160) = may_be_none(g_8157, None_8158)

    if may_be_8159:

        if more_types_in_union_8160:
            # Runtime conditional SSA (line 295)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 296):
        
        # Assigning a Dict to a Name (line 296):
        
        # Obtaining an instance of the builtin type 'dict' (line 296)
        dict_8161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 296)
        
        # Assigning a type to the variable 'g' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'g', dict_8161)

        if more_types_in_union_8160:
            # SSA join for if statement (line 295)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 297):
    
    # Assigning a Dict to a Name (line 297):
    
    # Obtaining an instance of the builtin type 'dict' (line 297)
    dict_8162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 297)
    
    # Assigning a type to the variable 'done' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'done', dict_8162)
    
    # Assigning a Dict to a Name (line 298):
    
    # Assigning a Dict to a Name (line 298):
    
    # Obtaining an instance of the builtin type 'dict' (line 298)
    dict_8163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 298)
    
    # Assigning a type to the variable 'notdone' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'notdone', dict_8163)
    
    int_8164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 10), 'int')
    # Testing the type of an if condition (line 300)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), int_8164)
    # SSA begins for while statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to readline(...): (line 301)
    # Processing the call keyword arguments (line 301)
    kwargs_8167 = {}
    # Getting the type of 'fp' (line 301)
    fp_8165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'fp', False)
    # Obtaining the member 'readline' of a type (line 301)
    readline_8166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), fp_8165, 'readline')
    # Calling readline(args, kwargs) (line 301)
    readline_call_result_8168 = invoke(stypy.reporting.localization.Localization(__file__, 301, 15), readline_8166, *[], **kwargs_8167)
    
    # Assigning a type to the variable 'line' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'line', readline_call_result_8168)
    
    # Type idiom detected: calculating its left and rigth part (line 302)
    # Getting the type of 'line' (line 302)
    line_8169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'line')
    # Getting the type of 'None' (line 302)
    None_8170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'None')
    
    (may_be_8171, more_types_in_union_8172) = may_be_none(line_8169, None_8170)

    if may_be_8171:

        if more_types_in_union_8172:
            # Runtime conditional SSA (line 302)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_8172:
            # SSA join for if statement (line 302)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to match(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'line' (line 304)
    line_8175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'line', False)
    # Processing the call keyword arguments (line 304)
    kwargs_8176 = {}
    # Getting the type of '_variable_rx' (line 304)
    _variable_rx_8173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), '_variable_rx', False)
    # Obtaining the member 'match' of a type (line 304)
    match_8174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), _variable_rx_8173, 'match')
    # Calling match(args, kwargs) (line 304)
    match_call_result_8177 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), match_8174, *[line_8175], **kwargs_8176)
    
    # Assigning a type to the variable 'm' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'm', match_call_result_8177)
    
    # Getting the type of 'm' (line 305)
    m_8178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'm')
    # Testing the type of an if condition (line 305)
    if_condition_8179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), m_8178)
    # Assigning a type to the variable 'if_condition_8179' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_8179', if_condition_8179)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 306):
    
    # Assigning a Subscript to a Name (line 306):
    
    # Obtaining the type of the subscript
    int_8180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'int')
    
    # Call to group(...): (line 306)
    # Processing the call arguments (line 306)
    int_8183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 27), 'int')
    int_8184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 30), 'int')
    # Processing the call keyword arguments (line 306)
    kwargs_8185 = {}
    # Getting the type of 'm' (line 306)
    m_8181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 306)
    group_8182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), m_8181, 'group')
    # Calling group(args, kwargs) (line 306)
    group_call_result_8186 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), group_8182, *[int_8183, int_8184], **kwargs_8185)
    
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___8187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), group_call_result_8186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_8188 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___8187, int_8180)
    
    # Assigning a type to the variable 'tuple_var_assignment_7251' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_7251', subscript_call_result_8188)
    
    # Assigning a Subscript to a Name (line 306):
    
    # Obtaining the type of the subscript
    int_8189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'int')
    
    # Call to group(...): (line 306)
    # Processing the call arguments (line 306)
    int_8192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 27), 'int')
    int_8193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 30), 'int')
    # Processing the call keyword arguments (line 306)
    kwargs_8194 = {}
    # Getting the type of 'm' (line 306)
    m_8190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 306)
    group_8191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), m_8190, 'group')
    # Calling group(args, kwargs) (line 306)
    group_call_result_8195 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), group_8191, *[int_8192, int_8193], **kwargs_8194)
    
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___8196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), group_call_result_8195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_8197 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___8196, int_8189)
    
    # Assigning a type to the variable 'tuple_var_assignment_7252' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_7252', subscript_call_result_8197)
    
    # Assigning a Name to a Name (line 306):
    # Getting the type of 'tuple_var_assignment_7251' (line 306)
    tuple_var_assignment_7251_8198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_7251')
    # Assigning a type to the variable 'n' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'n', tuple_var_assignment_7251_8198)
    
    # Assigning a Name to a Name (line 306):
    # Getting the type of 'tuple_var_assignment_7252' (line 306)
    tuple_var_assignment_7252_8199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_7252')
    # Assigning a type to the variable 'v' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'v', tuple_var_assignment_7252_8199)
    
    # Assigning a Call to a Name (line 307):
    
    # Assigning a Call to a Name (line 307):
    
    # Call to strip(...): (line 307)
    # Processing the call keyword arguments (line 307)
    kwargs_8202 = {}
    # Getting the type of 'v' (line 307)
    v_8200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'v', False)
    # Obtaining the member 'strip' of a type (line 307)
    strip_8201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 16), v_8200, 'strip')
    # Calling strip(args, kwargs) (line 307)
    strip_call_result_8203 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), strip_8201, *[], **kwargs_8202)
    
    # Assigning a type to the variable 'v' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'v', strip_call_result_8203)
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to replace(...): (line 309)
    # Processing the call arguments (line 309)
    str_8206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'str', '$$')
    str_8207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 35), 'str', '')
    # Processing the call keyword arguments (line 309)
    kwargs_8208 = {}
    # Getting the type of 'v' (line 309)
    v_8204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'v', False)
    # Obtaining the member 'replace' of a type (line 309)
    replace_8205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), v_8204, 'replace')
    # Calling replace(args, kwargs) (line 309)
    replace_call_result_8209 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), replace_8205, *[str_8206, str_8207], **kwargs_8208)
    
    # Assigning a type to the variable 'tmpv' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'tmpv', replace_call_result_8209)
    
    
    str_8210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 15), 'str', '$')
    # Getting the type of 'tmpv' (line 311)
    tmpv_8211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 22), 'tmpv')
    # Applying the binary operator 'in' (line 311)
    result_contains_8212 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 15), 'in', str_8210, tmpv_8211)
    
    # Testing the type of an if condition (line 311)
    if_condition_8213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 12), result_contains_8212)
    # Assigning a type to the variable 'if_condition_8213' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'if_condition_8213', if_condition_8213)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 312):
    
    # Assigning a Name to a Subscript (line 312):
    # Getting the type of 'v' (line 312)
    v_8214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'v')
    # Getting the type of 'notdone' (line 312)
    notdone_8215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'notdone')
    # Getting the type of 'n' (line 312)
    n_8216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'n')
    # Storing an element on a container (line 312)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 16), notdone_8215, (n_8216, v_8214))
    # SSA branch for the else part of an if statement (line 311)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 315):
    
    # Assigning a Call to a Name (line 315):
    
    # Call to int(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'v' (line 315)
    v_8218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 28), 'v', False)
    # Processing the call keyword arguments (line 315)
    kwargs_8219 = {}
    # Getting the type of 'int' (line 315)
    int_8217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'int', False)
    # Calling int(args, kwargs) (line 315)
    int_call_result_8220 = invoke(stypy.reporting.localization.Localization(__file__, 315, 24), int_8217, *[v_8218], **kwargs_8219)
    
    # Assigning a type to the variable 'v' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'v', int_call_result_8220)
    # SSA branch for the except part of a try statement (line 314)
    # SSA branch for the except 'ValueError' branch of a try statement (line 314)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Subscript (line 318):
    
    # Assigning a Call to a Subscript (line 318):
    
    # Call to replace(...): (line 318)
    # Processing the call arguments (line 318)
    str_8223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 40), 'str', '$$')
    str_8224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 46), 'str', '$')
    # Processing the call keyword arguments (line 318)
    kwargs_8225 = {}
    # Getting the type of 'v' (line 318)
    v_8221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'v', False)
    # Obtaining the member 'replace' of a type (line 318)
    replace_8222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 30), v_8221, 'replace')
    # Calling replace(args, kwargs) (line 318)
    replace_call_result_8226 = invoke(stypy.reporting.localization.Localization(__file__, 318, 30), replace_8222, *[str_8223, str_8224], **kwargs_8225)
    
    # Getting the type of 'done' (line 318)
    done_8227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'done')
    # Getting the type of 'n' (line 318)
    n_8228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'n')
    # Storing an element on a container (line 318)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 20), done_8227, (n_8228, replace_call_result_8226))
    # SSA branch for the else branch of a try statement (line 314)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Name to a Subscript (line 320):
    
    # Assigning a Name to a Subscript (line 320):
    # Getting the type of 'v' (line 320)
    v_8229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'v')
    # Getting the type of 'done' (line 320)
    done_8230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'done')
    # Getting the type of 'n' (line 320)
    n_8231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'n')
    # Storing an element on a container (line 320)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), done_8230, (n_8231, v_8229))
    # SSA join for try-except statement (line 314)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'notdone' (line 323)
    notdone_8232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 10), 'notdone')
    # Testing the type of an if condition (line 323)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 4), notdone_8232)
    # SSA begins for while statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Call to keys(...): (line 324)
    # Processing the call keyword arguments (line 324)
    kwargs_8235 = {}
    # Getting the type of 'notdone' (line 324)
    notdone_8233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'notdone', False)
    # Obtaining the member 'keys' of a type (line 324)
    keys_8234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), notdone_8233, 'keys')
    # Calling keys(args, kwargs) (line 324)
    keys_call_result_8236 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), keys_8234, *[], **kwargs_8235)
    
    # Testing the type of a for loop iterable (line 324)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 324, 8), keys_call_result_8236)
    # Getting the type of the for loop variable (line 324)
    for_loop_var_8237 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 324, 8), keys_call_result_8236)
    # Assigning a type to the variable 'name' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'name', for_loop_var_8237)
    # SSA begins for a for statement (line 324)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 325):
    
    # Assigning a Subscript to a Name (line 325):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 325)
    name_8238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 28), 'name')
    # Getting the type of 'notdone' (line 325)
    notdone_8239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'notdone')
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___8240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 20), notdone_8239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_8241 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), getitem___8240, name_8238)
    
    # Assigning a type to the variable 'value' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'value', subscript_call_result_8241)
    
    # Assigning a BoolOp to a Name (line 326):
    
    # Assigning a BoolOp to a Name (line 326):
    
    # Evaluating a boolean operation
    
    # Call to search(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'value' (line 326)
    value_8244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'value', False)
    # Processing the call keyword arguments (line 326)
    kwargs_8245 = {}
    # Getting the type of '_findvar1_rx' (line 326)
    _findvar1_rx_8242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), '_findvar1_rx', False)
    # Obtaining the member 'search' of a type (line 326)
    search_8243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), _findvar1_rx_8242, 'search')
    # Calling search(args, kwargs) (line 326)
    search_call_result_8246 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), search_8243, *[value_8244], **kwargs_8245)
    
    
    # Call to search(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'value' (line 326)
    value_8249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 66), 'value', False)
    # Processing the call keyword arguments (line 326)
    kwargs_8250 = {}
    # Getting the type of '_findvar2_rx' (line 326)
    _findvar2_rx_8247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 46), '_findvar2_rx', False)
    # Obtaining the member 'search' of a type (line 326)
    search_8248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 46), _findvar2_rx_8247, 'search')
    # Calling search(args, kwargs) (line 326)
    search_call_result_8251 = invoke(stypy.reporting.localization.Localization(__file__, 326, 46), search_8248, *[value_8249], **kwargs_8250)
    
    # Applying the binary operator 'or' (line 326)
    result_or_keyword_8252 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 16), 'or', search_call_result_8246, search_call_result_8251)
    
    # Assigning a type to the variable 'm' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'm', result_or_keyword_8252)
    
    # Getting the type of 'm' (line 327)
    m_8253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'm')
    # Testing the type of an if condition (line 327)
    if_condition_8254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 12), m_8253)
    # Assigning a type to the variable 'if_condition_8254' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'if_condition_8254', if_condition_8254)
    # SSA begins for if statement (line 327)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to group(...): (line 328)
    # Processing the call arguments (line 328)
    int_8257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 28), 'int')
    # Processing the call keyword arguments (line 328)
    kwargs_8258 = {}
    # Getting the type of 'm' (line 328)
    m_8255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 20), 'm', False)
    # Obtaining the member 'group' of a type (line 328)
    group_8256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 20), m_8255, 'group')
    # Calling group(args, kwargs) (line 328)
    group_call_result_8259 = invoke(stypy.reporting.localization.Localization(__file__, 328, 20), group_8256, *[int_8257], **kwargs_8258)
    
    # Assigning a type to the variable 'n' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'n', group_call_result_8259)
    
    # Assigning a Name to a Name (line 329):
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'True' (line 329)
    True_8260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 24), 'True')
    # Assigning a type to the variable 'found' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'found', True_8260)
    
    
    # Getting the type of 'n' (line 330)
    n_8261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'n')
    # Getting the type of 'done' (line 330)
    done_8262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'done')
    # Applying the binary operator 'in' (line 330)
    result_contains_8263 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 19), 'in', n_8261, done_8262)
    
    # Testing the type of an if condition (line 330)
    if_condition_8264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 16), result_contains_8263)
    # Assigning a type to the variable 'if_condition_8264' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'if_condition_8264', if_condition_8264)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 331):
    
    # Assigning a Call to a Name (line 331):
    
    # Call to str(...): (line 331)
    # Processing the call arguments (line 331)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 331)
    n_8266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'n', False)
    # Getting the type of 'done' (line 331)
    done_8267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 31), 'done', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___8268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 31), done_8267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_8269 = invoke(stypy.reporting.localization.Localization(__file__, 331, 31), getitem___8268, n_8266)
    
    # Processing the call keyword arguments (line 331)
    kwargs_8270 = {}
    # Getting the type of 'str' (line 331)
    str_8265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'str', False)
    # Calling str(args, kwargs) (line 331)
    str_call_result_8271 = invoke(stypy.reporting.localization.Localization(__file__, 331, 27), str_8265, *[subscript_call_result_8269], **kwargs_8270)
    
    # Assigning a type to the variable 'item' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'item', str_call_result_8271)
    # SSA branch for the else part of an if statement (line 330)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 332)
    n_8272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'n')
    # Getting the type of 'notdone' (line 332)
    notdone_8273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 26), 'notdone')
    # Applying the binary operator 'in' (line 332)
    result_contains_8274 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 21), 'in', n_8272, notdone_8273)
    
    # Testing the type of an if condition (line 332)
    if_condition_8275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 21), result_contains_8274)
    # Assigning a type to the variable 'if_condition_8275' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'if_condition_8275', if_condition_8275)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 334):
    
    # Assigning a Name to a Name (line 334):
    # Getting the type of 'False' (line 334)
    False_8276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'False')
    # Assigning a type to the variable 'found' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'found', False_8276)
    # SSA branch for the else part of an if statement (line 332)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 335)
    n_8277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'n')
    # Getting the type of 'os' (line 335)
    os_8278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 26), 'os')
    # Obtaining the member 'environ' of a type (line 335)
    environ_8279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 26), os_8278, 'environ')
    # Applying the binary operator 'in' (line 335)
    result_contains_8280 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 21), 'in', n_8277, environ_8279)
    
    # Testing the type of an if condition (line 335)
    if_condition_8281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 21), result_contains_8280)
    # Assigning a type to the variable 'if_condition_8281' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'if_condition_8281', if_condition_8281)
    # SSA begins for if statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 337):
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 337)
    n_8282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 38), 'n')
    # Getting the type of 'os' (line 337)
    os_8283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'os')
    # Obtaining the member 'environ' of a type (line 337)
    environ_8284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), os_8283, 'environ')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___8285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), environ_8284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_8286 = invoke(stypy.reporting.localization.Localization(__file__, 337, 27), getitem___8285, n_8282)
    
    # Assigning a type to the variable 'item' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'item', subscript_call_result_8286)
    # SSA branch for the else part of an if statement (line 335)
    module_type_store.open_ssa_branch('else')
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Str to a Name (line 339):
    str_8287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 37), 'str', '')
    # Assigning a type to the variable 'item' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 30), 'item', str_8287)
    
    # Assigning a Name to a Subscript (line 339):
    # Getting the type of 'item' (line 339)
    item_8288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 30), 'item')
    # Getting the type of 'done' (line 339)
    done_8289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 20), 'done')
    # Getting the type of 'n' (line 339)
    n_8290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 25), 'n')
    # Storing an element on a container (line 339)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 20), done_8289, (n_8290, item_8288))
    # SSA join for if statement (line 335)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'found' (line 340)
    found_8291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 19), 'found')
    # Testing the type of an if condition (line 340)
    if_condition_8292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 16), found_8291)
    # Assigning a type to the variable 'if_condition_8292' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'if_condition_8292', if_condition_8292)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 341):
    
    # Assigning a Subscript to a Name (line 341):
    
    # Obtaining the type of the subscript
    
    # Call to end(...): (line 341)
    # Processing the call keyword arguments (line 341)
    kwargs_8295 = {}
    # Getting the type of 'm' (line 341)
    m_8293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'm', False)
    # Obtaining the member 'end' of a type (line 341)
    end_8294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 34), m_8293, 'end')
    # Calling end(args, kwargs) (line 341)
    end_call_result_8296 = invoke(stypy.reporting.localization.Localization(__file__, 341, 34), end_8294, *[], **kwargs_8295)
    
    slice_8297 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 28), end_call_result_8296, None, None)
    # Getting the type of 'value' (line 341)
    value_8298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 28), 'value')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___8299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 28), value_8298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_8300 = invoke(stypy.reporting.localization.Localization(__file__, 341, 28), getitem___8299, slice_8297)
    
    # Assigning a type to the variable 'after' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'after', subscript_call_result_8300)
    
    # Assigning a BinOp to a Name (line 342):
    
    # Assigning a BinOp to a Name (line 342):
    
    # Obtaining the type of the subscript
    
    # Call to start(...): (line 342)
    # Processing the call keyword arguments (line 342)
    kwargs_8303 = {}
    # Getting the type of 'm' (line 342)
    m_8301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'm', False)
    # Obtaining the member 'start' of a type (line 342)
    start_8302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 35), m_8301, 'start')
    # Calling start(args, kwargs) (line 342)
    start_call_result_8304 = invoke(stypy.reporting.localization.Localization(__file__, 342, 35), start_8302, *[], **kwargs_8303)
    
    slice_8305 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 342, 28), None, start_call_result_8304, None)
    # Getting the type of 'value' (line 342)
    value_8306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'value')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___8307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 28), value_8306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_8308 = invoke(stypy.reporting.localization.Localization(__file__, 342, 28), getitem___8307, slice_8305)
    
    # Getting the type of 'item' (line 342)
    item_8309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 48), 'item')
    # Applying the binary operator '+' (line 342)
    result_add_8310 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 28), '+', subscript_call_result_8308, item_8309)
    
    # Getting the type of 'after' (line 342)
    after_8311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 55), 'after')
    # Applying the binary operator '+' (line 342)
    result_add_8312 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 53), '+', result_add_8310, after_8311)
    
    # Assigning a type to the variable 'value' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'value', result_add_8312)
    
    
    str_8313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 23), 'str', '$')
    # Getting the type of 'after' (line 343)
    after_8314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 30), 'after')
    # Applying the binary operator 'in' (line 343)
    result_contains_8315 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 23), 'in', str_8313, after_8314)
    
    # Testing the type of an if condition (line 343)
    if_condition_8316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 20), result_contains_8315)
    # Assigning a type to the variable 'if_condition_8316' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'if_condition_8316', if_condition_8316)
    # SSA begins for if statement (line 343)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 344):
    
    # Assigning a Name to a Subscript (line 344):
    # Getting the type of 'value' (line 344)
    value_8317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 40), 'value')
    # Getting the type of 'notdone' (line 344)
    notdone_8318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 24), 'notdone')
    # Getting the type of 'name' (line 344)
    name_8319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'name')
    # Storing an element on a container (line 344)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 24), notdone_8318, (name_8319, value_8317))
    # SSA branch for the else part of an if statement (line 343)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 346):
    
    # Assigning a Call to a Name (line 346):
    
    # Call to int(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'value' (line 346)
    value_8321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 41), 'value', False)
    # Processing the call keyword arguments (line 346)
    kwargs_8322 = {}
    # Getting the type of 'int' (line 346)
    int_8320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 37), 'int', False)
    # Calling int(args, kwargs) (line 346)
    int_call_result_8323 = invoke(stypy.reporting.localization.Localization(__file__, 346, 37), int_8320, *[value_8321], **kwargs_8322)
    
    # Assigning a type to the variable 'value' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 29), 'value', int_call_result_8323)
    # SSA branch for the except part of a try statement (line 346)
    # SSA branch for the except 'ValueError' branch of a try statement (line 346)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Subscript (line 348):
    
    # Assigning a Call to a Subscript (line 348):
    
    # Call to strip(...): (line 348)
    # Processing the call keyword arguments (line 348)
    kwargs_8326 = {}
    # Getting the type of 'value' (line 348)
    value_8324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'value', False)
    # Obtaining the member 'strip' of a type (line 348)
    strip_8325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 41), value_8324, 'strip')
    # Calling strip(args, kwargs) (line 348)
    strip_call_result_8327 = invoke(stypy.reporting.localization.Localization(__file__, 348, 41), strip_8325, *[], **kwargs_8326)
    
    # Getting the type of 'done' (line 348)
    done_8328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 28), 'done')
    # Getting the type of 'name' (line 348)
    name_8329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 33), 'name')
    # Storing an element on a container (line 348)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 28), done_8328, (name_8329, strip_call_result_8327))
    # SSA branch for the else branch of a try statement (line 346)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Name to a Subscript (line 350):
    
    # Assigning a Name to a Subscript (line 350):
    # Getting the type of 'value' (line 350)
    value_8330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'value')
    # Getting the type of 'done' (line 350)
    done_8331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'done')
    # Getting the type of 'name' (line 350)
    name_8332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'name')
    # Storing an element on a container (line 350)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 28), done_8331, (name_8332, value_8330))
    # SSA join for try-except statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    # Deleting a member
    # Getting the type of 'notdone' (line 351)
    notdone_8333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 28), 'notdone')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 351)
    name_8334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 36), 'name')
    # Getting the type of 'notdone' (line 351)
    notdone_8335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 28), 'notdone')
    # Obtaining the member '__getitem__' of a type (line 351)
    getitem___8336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 28), notdone_8335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 351)
    subscript_call_result_8337 = invoke(stypy.reporting.localization.Localization(__file__, 351, 28), getitem___8336, name_8334)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 24), notdone_8333, subscript_call_result_8337)
    # SSA join for if statement (line 343)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 327)
    module_type_store.open_ssa_branch('else')
    # Deleting a member
    # Getting the type of 'notdone' (line 354)
    notdone_8338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 20), 'notdone')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 354)
    name_8339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 28), 'name')
    # Getting the type of 'notdone' (line 354)
    notdone_8340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 20), 'notdone')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___8341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 20), notdone_8340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_8342 = invoke(stypy.reporting.localization.Localization(__file__, 354, 20), getitem___8341, name_8339)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 16), notdone_8338, subscript_call_result_8342)
    # SSA join for if statement (line 327)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 356)
    # Processing the call keyword arguments (line 356)
    kwargs_8345 = {}
    # Getting the type of 'fp' (line 356)
    fp_8343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'fp', False)
    # Obtaining the member 'close' of a type (line 356)
    close_8344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 4), fp_8343, 'close')
    # Calling close(args, kwargs) (line 356)
    close_call_result_8346 = invoke(stypy.reporting.localization.Localization(__file__, 356, 4), close_8344, *[], **kwargs_8345)
    
    
    
    # Call to items(...): (line 359)
    # Processing the call keyword arguments (line 359)
    kwargs_8349 = {}
    # Getting the type of 'done' (line 359)
    done_8347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'done', False)
    # Obtaining the member 'items' of a type (line 359)
    items_8348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), done_8347, 'items')
    # Calling items(args, kwargs) (line 359)
    items_call_result_8350 = invoke(stypy.reporting.localization.Localization(__file__, 359, 16), items_8348, *[], **kwargs_8349)
    
    # Testing the type of a for loop iterable (line 359)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 359, 4), items_call_result_8350)
    # Getting the type of the for loop variable (line 359)
    for_loop_var_8351 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 359, 4), items_call_result_8350)
    # Assigning a type to the variable 'k' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 4), for_loop_var_8351))
    # Assigning a type to the variable 'v' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 4), for_loop_var_8351))
    # SSA begins for a for statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 360)
    # Getting the type of 'str' (line 360)
    str_8352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'str')
    # Getting the type of 'v' (line 360)
    v_8353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'v')
    
    (may_be_8354, more_types_in_union_8355) = may_be_subtype(str_8352, v_8353)

    if may_be_8354:

        if more_types_in_union_8355:
            # Runtime conditional SSA (line 360)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'v' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'v', remove_not_subtype_from_union(v_8353, str))
        
        # Assigning a Call to a Subscript (line 361):
        
        # Assigning a Call to a Subscript (line 361):
        
        # Call to strip(...): (line 361)
        # Processing the call keyword arguments (line 361)
        kwargs_8358 = {}
        # Getting the type of 'v' (line 361)
        v_8356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'v', False)
        # Obtaining the member 'strip' of a type (line 361)
        strip_8357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 22), v_8356, 'strip')
        # Calling strip(args, kwargs) (line 361)
        strip_call_result_8359 = invoke(stypy.reporting.localization.Localization(__file__, 361, 22), strip_8357, *[], **kwargs_8358)
        
        # Getting the type of 'done' (line 361)
        done_8360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'done')
        # Getting the type of 'k' (line 361)
        k_8361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 17), 'k')
        # Storing an element on a container (line 361)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 12), done_8360, (k_8361, strip_call_result_8359))

        if more_types_in_union_8355:
            # SSA join for if statement (line 360)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to update(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'done' (line 364)
    done_8364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 13), 'done', False)
    # Processing the call keyword arguments (line 364)
    kwargs_8365 = {}
    # Getting the type of 'g' (line 364)
    g_8362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'g', False)
    # Obtaining the member 'update' of a type (line 364)
    update_8363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 4), g_8362, 'update')
    # Calling update(args, kwargs) (line 364)
    update_call_result_8366 = invoke(stypy.reporting.localization.Localization(__file__, 364, 4), update_8363, *[done_8364], **kwargs_8365)
    
    # Getting the type of 'g' (line 365)
    g_8367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type', g_8367)
    
    # ################# End of 'parse_makefile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_makefile' in the type store
    # Getting the type of 'stypy_return_type' (line 285)
    stypy_return_type_8368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8368)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_makefile'
    return stypy_return_type_8368

# Assigning a type to the variable 'parse_makefile' (line 285)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'parse_makefile', parse_makefile)

@norecursion
def expand_makefile_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expand_makefile_vars'
    module_type_store = module_type_store.open_function_context('expand_makefile_vars', 368, 0, False)
    
    # Passed parameters checking function
    expand_makefile_vars.stypy_localization = localization
    expand_makefile_vars.stypy_type_of_self = None
    expand_makefile_vars.stypy_type_store = module_type_store
    expand_makefile_vars.stypy_function_name = 'expand_makefile_vars'
    expand_makefile_vars.stypy_param_names_list = ['s', 'vars']
    expand_makefile_vars.stypy_varargs_param_name = None
    expand_makefile_vars.stypy_kwargs_param_name = None
    expand_makefile_vars.stypy_call_defaults = defaults
    expand_makefile_vars.stypy_call_varargs = varargs
    expand_makefile_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expand_makefile_vars', ['s', 'vars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expand_makefile_vars', localization, ['s', 'vars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expand_makefile_vars(...)' code ##################

    str_8369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'str', 'Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in\n    \'string\' according to \'vars\' (a dictionary mapping variable names to\n    values).  Variables not present in \'vars\' are silently expanded to the\n    empty string.  The variable values in \'vars\' should not contain further\n    variable expansions; if \'vars\' is the output of \'parse_makefile()\',\n    you\'re fine.  Returns a variable-expanded version of \'s\'.\n    ')
    
    int_8370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 10), 'int')
    # Testing the type of an if condition (line 383)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 4), int_8370)
    # SSA begins for while statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BoolOp to a Name (line 384):
    
    # Assigning a BoolOp to a Name (line 384):
    
    # Evaluating a boolean operation
    
    # Call to search(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 's' (line 384)
    s_8373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 32), 's', False)
    # Processing the call keyword arguments (line 384)
    kwargs_8374 = {}
    # Getting the type of '_findvar1_rx' (line 384)
    _findvar1_rx_8371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), '_findvar1_rx', False)
    # Obtaining the member 'search' of a type (line 384)
    search_8372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 12), _findvar1_rx_8371, 'search')
    # Calling search(args, kwargs) (line 384)
    search_call_result_8375 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), search_8372, *[s_8373], **kwargs_8374)
    
    
    # Call to search(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 's' (line 384)
    s_8378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 58), 's', False)
    # Processing the call keyword arguments (line 384)
    kwargs_8379 = {}
    # Getting the type of '_findvar2_rx' (line 384)
    _findvar2_rx_8376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 38), '_findvar2_rx', False)
    # Obtaining the member 'search' of a type (line 384)
    search_8377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 38), _findvar2_rx_8376, 'search')
    # Calling search(args, kwargs) (line 384)
    search_call_result_8380 = invoke(stypy.reporting.localization.Localization(__file__, 384, 38), search_8377, *[s_8378], **kwargs_8379)
    
    # Applying the binary operator 'or' (line 384)
    result_or_keyword_8381 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 12), 'or', search_call_result_8375, search_call_result_8380)
    
    # Assigning a type to the variable 'm' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'm', result_or_keyword_8381)
    
    # Getting the type of 'm' (line 385)
    m_8382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), 'm')
    # Testing the type of an if condition (line 385)
    if_condition_8383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 8), m_8382)
    # Assigning a type to the variable 'if_condition_8383' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'if_condition_8383', if_condition_8383)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 386):
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_8384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 12), 'int')
    
    # Call to span(...): (line 386)
    # Processing the call keyword arguments (line 386)
    kwargs_8387 = {}
    # Getting the type of 'm' (line 386)
    m_8385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'm', False)
    # Obtaining the member 'span' of a type (line 386)
    span_8386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 25), m_8385, 'span')
    # Calling span(args, kwargs) (line 386)
    span_call_result_8388 = invoke(stypy.reporting.localization.Localization(__file__, 386, 25), span_8386, *[], **kwargs_8387)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___8389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), span_call_result_8388, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_8390 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), getitem___8389, int_8384)
    
    # Assigning a type to the variable 'tuple_var_assignment_7253' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_7253', subscript_call_result_8390)
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_8391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 12), 'int')
    
    # Call to span(...): (line 386)
    # Processing the call keyword arguments (line 386)
    kwargs_8394 = {}
    # Getting the type of 'm' (line 386)
    m_8392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'm', False)
    # Obtaining the member 'span' of a type (line 386)
    span_8393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 25), m_8392, 'span')
    # Calling span(args, kwargs) (line 386)
    span_call_result_8395 = invoke(stypy.reporting.localization.Localization(__file__, 386, 25), span_8393, *[], **kwargs_8394)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___8396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), span_call_result_8395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_8397 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), getitem___8396, int_8391)
    
    # Assigning a type to the variable 'tuple_var_assignment_7254' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_7254', subscript_call_result_8397)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_7253' (line 386)
    tuple_var_assignment_7253_8398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_7253')
    # Assigning a type to the variable 'beg' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'beg', tuple_var_assignment_7253_8398)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_7254' (line 386)
    tuple_var_assignment_7254_8399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_7254')
    # Assigning a type to the variable 'end' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 18), 'end', tuple_var_assignment_7254_8399)
    
    # Assigning a BinOp to a Name (line 387):
    
    # Assigning a BinOp to a Name (line 387):
    
    # Obtaining the type of the subscript
    int_8400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 18), 'int')
    # Getting the type of 'beg' (line 387)
    beg_8401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'beg')
    slice_8402 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 387, 16), int_8400, beg_8401, None)
    # Getting the type of 's' (line 387)
    s_8403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 's')
    # Obtaining the member '__getitem__' of a type (line 387)
    getitem___8404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 16), s_8403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 387)
    subscript_call_result_8405 = invoke(stypy.reporting.localization.Localization(__file__, 387, 16), getitem___8404, slice_8402)
    
    
    # Call to get(...): (line 387)
    # Processing the call arguments (line 387)
    
    # Call to group(...): (line 387)
    # Processing the call arguments (line 387)
    int_8410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 44), 'int')
    # Processing the call keyword arguments (line 387)
    kwargs_8411 = {}
    # Getting the type of 'm' (line 387)
    m_8408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 36), 'm', False)
    # Obtaining the member 'group' of a type (line 387)
    group_8409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 36), m_8408, 'group')
    # Calling group(args, kwargs) (line 387)
    group_call_result_8412 = invoke(stypy.reporting.localization.Localization(__file__, 387, 36), group_8409, *[int_8410], **kwargs_8411)
    
    # Processing the call keyword arguments (line 387)
    kwargs_8413 = {}
    # Getting the type of 'vars' (line 387)
    vars_8406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 27), 'vars', False)
    # Obtaining the member 'get' of a type (line 387)
    get_8407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 27), vars_8406, 'get')
    # Calling get(args, kwargs) (line 387)
    get_call_result_8414 = invoke(stypy.reporting.localization.Localization(__file__, 387, 27), get_8407, *[group_call_result_8412], **kwargs_8413)
    
    # Applying the binary operator '+' (line 387)
    result_add_8415 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 16), '+', subscript_call_result_8405, get_call_result_8414)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 387)
    end_8416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 52), 'end')
    slice_8417 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 387, 50), end_8416, None, None)
    # Getting the type of 's' (line 387)
    s_8418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 50), 's')
    # Obtaining the member '__getitem__' of a type (line 387)
    getitem___8419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 50), s_8418, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 387)
    subscript_call_result_8420 = invoke(stypy.reporting.localization.Localization(__file__, 387, 50), getitem___8419, slice_8417)
    
    # Applying the binary operator '+' (line 387)
    result_add_8421 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 48), '+', result_add_8415, subscript_call_result_8420)
    
    # Assigning a type to the variable 's' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 's', result_add_8421)
    # SSA branch for the else part of an if statement (line 385)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 's' (line 390)
    s_8422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type', s_8422)
    
    # ################# End of 'expand_makefile_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expand_makefile_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 368)
    stypy_return_type_8423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8423)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expand_makefile_vars'
    return stypy_return_type_8423

# Assigning a type to the variable 'expand_makefile_vars' (line 368)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'expand_makefile_vars', expand_makefile_vars)

# Assigning a Name to a Name (line 393):

# Assigning a Name to a Name (line 393):
# Getting the type of 'None' (line 393)
None_8424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'None')
# Assigning a type to the variable '_config_vars' (line 393)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 0), '_config_vars', None_8424)

@norecursion
def _init_posix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_init_posix'
    module_type_store = module_type_store.open_function_context('_init_posix', 395, 0, False)
    
    # Passed parameters checking function
    _init_posix.stypy_localization = localization
    _init_posix.stypy_type_of_self = None
    _init_posix.stypy_type_store = module_type_store
    _init_posix.stypy_function_name = '_init_posix'
    _init_posix.stypy_param_names_list = []
    _init_posix.stypy_varargs_param_name = None
    _init_posix.stypy_kwargs_param_name = None
    _init_posix.stypy_call_defaults = defaults
    _init_posix.stypy_call_varargs = varargs
    _init_posix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_init_posix', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_init_posix', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_init_posix(...)' code ##################

    str_8425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 4), 'str', 'Initialize the module as appropriate for POSIX systems.')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 398, 4))
    
    # 'from _sysconfigdata import build_time_vars' statement (line 398)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_8426 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 398, 4), '_sysconfigdata')

    if (type(import_8426) is not StypyTypeError):

        if (import_8426 != 'pyd_module'):
            __import__(import_8426)
            sys_modules_8427 = sys.modules[import_8426]
            import_from_module(stypy.reporting.localization.Localization(__file__, 398, 4), '_sysconfigdata', sys_modules_8427.module_type_store, module_type_store, ['build_time_vars'])
            nest_module(stypy.reporting.localization.Localization(__file__, 398, 4), __file__, sys_modules_8427, sys_modules_8427.module_type_store, module_type_store)
        else:
            from _sysconfigdata import build_time_vars

            import_from_module(stypy.reporting.localization.Localization(__file__, 398, 4), '_sysconfigdata', None, module_type_store, ['build_time_vars'], [build_time_vars])

    else:
        # Assigning a type to the variable '_sysconfigdata' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), '_sysconfigdata', import_8426)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    # Marking variables as global (line 399)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 399, 4), '_config_vars')
    
    # Assigning a Dict to a Name (line 400):
    
    # Assigning a Dict to a Name (line 400):
    
    # Obtaining an instance of the builtin type 'dict' (line 400)
    dict_8428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 400)
    
    # Assigning a type to the variable '_config_vars' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), '_config_vars', dict_8428)
    
    # Call to update(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'build_time_vars' (line 401)
    build_time_vars_8431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'build_time_vars', False)
    # Processing the call keyword arguments (line 401)
    kwargs_8432 = {}
    # Getting the type of '_config_vars' (line 401)
    _config_vars_8429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), '_config_vars', False)
    # Obtaining the member 'update' of a type (line 401)
    update_8430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 4), _config_vars_8429, 'update')
    # Calling update(args, kwargs) (line 401)
    update_call_result_8433 = invoke(stypy.reporting.localization.Localization(__file__, 401, 4), update_8430, *[build_time_vars_8431], **kwargs_8432)
    
    
    # ################# End of '_init_posix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_init_posix' in the type store
    # Getting the type of 'stypy_return_type' (line 395)
    stypy_return_type_8434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8434)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_init_posix'
    return stypy_return_type_8434

# Assigning a type to the variable '_init_posix' (line 395)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 0), '_init_posix', _init_posix)

@norecursion
def _init_nt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_init_nt'
    module_type_store = module_type_store.open_function_context('_init_nt', 404, 0, False)
    
    # Passed parameters checking function
    _init_nt.stypy_localization = localization
    _init_nt.stypy_type_of_self = None
    _init_nt.stypy_type_store = module_type_store
    _init_nt.stypy_function_name = '_init_nt'
    _init_nt.stypy_param_names_list = []
    _init_nt.stypy_varargs_param_name = None
    _init_nt.stypy_kwargs_param_name = None
    _init_nt.stypy_call_defaults = defaults
    _init_nt.stypy_call_varargs = varargs
    _init_nt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_init_nt', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_init_nt', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_init_nt(...)' code ##################

    str_8435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 4), 'str', 'Initialize the module as appropriate for NT')
    
    # Assigning a Dict to a Name (line 406):
    
    # Assigning a Dict to a Name (line 406):
    
    # Obtaining an instance of the builtin type 'dict' (line 406)
    dict_8436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 406)
    
    # Assigning a type to the variable 'g' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'g', dict_8436)
    
    # Assigning a Call to a Subscript (line 408):
    
    # Assigning a Call to a Subscript (line 408):
    
    # Call to get_python_lib(...): (line 408)
    # Processing the call keyword arguments (line 408)
    int_8438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 48), 'int')
    keyword_8439 = int_8438
    int_8440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 64), 'int')
    keyword_8441 = int_8440
    kwargs_8442 = {'standard_lib': keyword_8441, 'plat_specific': keyword_8439}
    # Getting the type of 'get_python_lib' (line 408)
    get_python_lib_8437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'get_python_lib', False)
    # Calling get_python_lib(args, kwargs) (line 408)
    get_python_lib_call_result_8443 = invoke(stypy.reporting.localization.Localization(__file__, 408, 19), get_python_lib_8437, *[], **kwargs_8442)
    
    # Getting the type of 'g' (line 408)
    g_8444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'g')
    str_8445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 6), 'str', 'LIBDEST')
    # Storing an element on a container (line 408)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 4), g_8444, (str_8445, get_python_lib_call_result_8443))
    
    # Assigning a Call to a Subscript (line 409):
    
    # Assigning a Call to a Subscript (line 409):
    
    # Call to get_python_lib(...): (line 409)
    # Processing the call keyword arguments (line 409)
    int_8447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 51), 'int')
    keyword_8448 = int_8447
    int_8449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 67), 'int')
    keyword_8450 = int_8449
    kwargs_8451 = {'standard_lib': keyword_8450, 'plat_specific': keyword_8448}
    # Getting the type of 'get_python_lib' (line 409)
    get_python_lib_8446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'get_python_lib', False)
    # Calling get_python_lib(args, kwargs) (line 409)
    get_python_lib_call_result_8452 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), get_python_lib_8446, *[], **kwargs_8451)
    
    # Getting the type of 'g' (line 409)
    g_8453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'g')
    str_8454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 6), 'str', 'BINLIBDEST')
    # Storing an element on a container (line 409)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 4), g_8453, (str_8454, get_python_lib_call_result_8452))
    
    # Assigning a Call to a Subscript (line 412):
    
    # Assigning a Call to a Subscript (line 412):
    
    # Call to get_python_inc(...): (line 412)
    # Processing the call keyword arguments (line 412)
    int_8456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 50), 'int')
    keyword_8457 = int_8456
    kwargs_8458 = {'plat_specific': keyword_8457}
    # Getting the type of 'get_python_inc' (line 412)
    get_python_inc_8455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 21), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 412)
    get_python_inc_call_result_8459 = invoke(stypy.reporting.localization.Localization(__file__, 412, 21), get_python_inc_8455, *[], **kwargs_8458)
    
    # Getting the type of 'g' (line 412)
    g_8460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'g')
    str_8461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 6), 'str', 'INCLUDEPY')
    # Storing an element on a container (line 412)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 4), g_8460, (str_8461, get_python_inc_call_result_8459))
    
    # Assigning a Str to a Subscript (line 414):
    
    # Assigning a Str to a Subscript (line 414):
    str_8462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 14), 'str', '.pyd')
    # Getting the type of 'g' (line 414)
    g_8463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'g')
    str_8464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 6), 'str', 'SO')
    # Storing an element on a container (line 414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 4), g_8463, (str_8464, str_8462))
    
    # Assigning a Str to a Subscript (line 415):
    
    # Assigning a Str to a Subscript (line 415):
    str_8465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 15), 'str', '.exe')
    # Getting the type of 'g' (line 415)
    g_8466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'g')
    str_8467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 6), 'str', 'EXE')
    # Storing an element on a container (line 415)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 4), g_8466, (str_8467, str_8465))
    
    # Assigning a Call to a Subscript (line 416):
    
    # Assigning a Call to a Subscript (line 416):
    
    # Call to replace(...): (line 416)
    # Processing the call arguments (line 416)
    str_8472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 48), 'str', '.')
    str_8473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 53), 'str', '')
    # Processing the call keyword arguments (line 416)
    kwargs_8474 = {}
    
    # Call to get_python_version(...): (line 416)
    # Processing the call keyword arguments (line 416)
    kwargs_8469 = {}
    # Getting the type of 'get_python_version' (line 416)
    get_python_version_8468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'get_python_version', False)
    # Calling get_python_version(args, kwargs) (line 416)
    get_python_version_call_result_8470 = invoke(stypy.reporting.localization.Localization(__file__, 416, 19), get_python_version_8468, *[], **kwargs_8469)
    
    # Obtaining the member 'replace' of a type (line 416)
    replace_8471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 19), get_python_version_call_result_8470, 'replace')
    # Calling replace(args, kwargs) (line 416)
    replace_call_result_8475 = invoke(stypy.reporting.localization.Localization(__file__, 416, 19), replace_8471, *[str_8472, str_8473], **kwargs_8474)
    
    # Getting the type of 'g' (line 416)
    g_8476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'g')
    str_8477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 6), 'str', 'VERSION')
    # Storing an element on a container (line 416)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 4), g_8476, (str_8477, replace_call_result_8475))
    
    # Assigning a Call to a Subscript (line 417):
    
    # Assigning a Call to a Subscript (line 417):
    
    # Call to dirname(...): (line 417)
    # Processing the call arguments (line 417)
    
    # Call to abspath(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'sys' (line 417)
    sys_8484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 50), 'sys', False)
    # Obtaining the member 'executable' of a type (line 417)
    executable_8485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 50), sys_8484, 'executable')
    # Processing the call keyword arguments (line 417)
    kwargs_8486 = {}
    # Getting the type of 'os' (line 417)
    os_8481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 417)
    path_8482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 34), os_8481, 'path')
    # Obtaining the member 'abspath' of a type (line 417)
    abspath_8483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 34), path_8482, 'abspath')
    # Calling abspath(args, kwargs) (line 417)
    abspath_call_result_8487 = invoke(stypy.reporting.localization.Localization(__file__, 417, 34), abspath_8483, *[executable_8485], **kwargs_8486)
    
    # Processing the call keyword arguments (line 417)
    kwargs_8488 = {}
    # Getting the type of 'os' (line 417)
    os_8478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 417)
    path_8479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 18), os_8478, 'path')
    # Obtaining the member 'dirname' of a type (line 417)
    dirname_8480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 18), path_8479, 'dirname')
    # Calling dirname(args, kwargs) (line 417)
    dirname_call_result_8489 = invoke(stypy.reporting.localization.Localization(__file__, 417, 18), dirname_8480, *[abspath_call_result_8487], **kwargs_8488)
    
    # Getting the type of 'g' (line 417)
    g_8490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'g')
    str_8491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 6), 'str', 'BINDIR')
    # Storing an element on a container (line 417)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 4), g_8490, (str_8491, dirname_call_result_8489))
    # Marking variables as global (line 419)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 419, 4), '_config_vars')
    
    # Assigning a Name to a Name (line 420):
    
    # Assigning a Name to a Name (line 420):
    # Getting the type of 'g' (line 420)
    g_8492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'g')
    # Assigning a type to the variable '_config_vars' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), '_config_vars', g_8492)
    
    # ################# End of '_init_nt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_init_nt' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_8493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8493)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_init_nt'
    return stypy_return_type_8493

# Assigning a type to the variable '_init_nt' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), '_init_nt', _init_nt)

@norecursion
def _init_os2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_init_os2'
    module_type_store = module_type_store.open_function_context('_init_os2', 423, 0, False)
    
    # Passed parameters checking function
    _init_os2.stypy_localization = localization
    _init_os2.stypy_type_of_self = None
    _init_os2.stypy_type_store = module_type_store
    _init_os2.stypy_function_name = '_init_os2'
    _init_os2.stypy_param_names_list = []
    _init_os2.stypy_varargs_param_name = None
    _init_os2.stypy_kwargs_param_name = None
    _init_os2.stypy_call_defaults = defaults
    _init_os2.stypy_call_varargs = varargs
    _init_os2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_init_os2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_init_os2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_init_os2(...)' code ##################

    str_8494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 4), 'str', 'Initialize the module as appropriate for OS/2')
    
    # Assigning a Dict to a Name (line 425):
    
    # Assigning a Dict to a Name (line 425):
    
    # Obtaining an instance of the builtin type 'dict' (line 425)
    dict_8495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 425)
    
    # Assigning a type to the variable 'g' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'g', dict_8495)
    
    # Assigning a Call to a Subscript (line 427):
    
    # Assigning a Call to a Subscript (line 427):
    
    # Call to get_python_lib(...): (line 427)
    # Processing the call keyword arguments (line 427)
    int_8497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 48), 'int')
    keyword_8498 = int_8497
    int_8499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 64), 'int')
    keyword_8500 = int_8499
    kwargs_8501 = {'standard_lib': keyword_8500, 'plat_specific': keyword_8498}
    # Getting the type of 'get_python_lib' (line 427)
    get_python_lib_8496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 19), 'get_python_lib', False)
    # Calling get_python_lib(args, kwargs) (line 427)
    get_python_lib_call_result_8502 = invoke(stypy.reporting.localization.Localization(__file__, 427, 19), get_python_lib_8496, *[], **kwargs_8501)
    
    # Getting the type of 'g' (line 427)
    g_8503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'g')
    str_8504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 6), 'str', 'LIBDEST')
    # Storing an element on a container (line 427)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 4), g_8503, (str_8504, get_python_lib_call_result_8502))
    
    # Assigning a Call to a Subscript (line 428):
    
    # Assigning a Call to a Subscript (line 428):
    
    # Call to get_python_lib(...): (line 428)
    # Processing the call keyword arguments (line 428)
    int_8506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 51), 'int')
    keyword_8507 = int_8506
    int_8508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 67), 'int')
    keyword_8509 = int_8508
    kwargs_8510 = {'standard_lib': keyword_8509, 'plat_specific': keyword_8507}
    # Getting the type of 'get_python_lib' (line 428)
    get_python_lib_8505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 22), 'get_python_lib', False)
    # Calling get_python_lib(args, kwargs) (line 428)
    get_python_lib_call_result_8511 = invoke(stypy.reporting.localization.Localization(__file__, 428, 22), get_python_lib_8505, *[], **kwargs_8510)
    
    # Getting the type of 'g' (line 428)
    g_8512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'g')
    str_8513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 6), 'str', 'BINLIBDEST')
    # Storing an element on a container (line 428)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 4), g_8512, (str_8513, get_python_lib_call_result_8511))
    
    # Assigning a Call to a Subscript (line 431):
    
    # Assigning a Call to a Subscript (line 431):
    
    # Call to get_python_inc(...): (line 431)
    # Processing the call keyword arguments (line 431)
    int_8515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 50), 'int')
    keyword_8516 = int_8515
    kwargs_8517 = {'plat_specific': keyword_8516}
    # Getting the type of 'get_python_inc' (line 431)
    get_python_inc_8514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 21), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 431)
    get_python_inc_call_result_8518 = invoke(stypy.reporting.localization.Localization(__file__, 431, 21), get_python_inc_8514, *[], **kwargs_8517)
    
    # Getting the type of 'g' (line 431)
    g_8519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'g')
    str_8520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 6), 'str', 'INCLUDEPY')
    # Storing an element on a container (line 431)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 4), g_8519, (str_8520, get_python_inc_call_result_8518))
    
    # Assigning a Str to a Subscript (line 433):
    
    # Assigning a Str to a Subscript (line 433):
    str_8521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 14), 'str', '.pyd')
    # Getting the type of 'g' (line 433)
    g_8522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'g')
    str_8523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 6), 'str', 'SO')
    # Storing an element on a container (line 433)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 4), g_8522, (str_8523, str_8521))
    
    # Assigning a Str to a Subscript (line 434):
    
    # Assigning a Str to a Subscript (line 434):
    str_8524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 15), 'str', '.exe')
    # Getting the type of 'g' (line 434)
    g_8525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'g')
    str_8526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 6), 'str', 'EXE')
    # Storing an element on a container (line 434)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 4), g_8525, (str_8526, str_8524))
    # Marking variables as global (line 436)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 436, 4), '_config_vars')
    
    # Assigning a Name to a Name (line 437):
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'g' (line 437)
    g_8527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'g')
    # Assigning a type to the variable '_config_vars' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), '_config_vars', g_8527)
    
    # ################# End of '_init_os2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_init_os2' in the type store
    # Getting the type of 'stypy_return_type' (line 423)
    stypy_return_type_8528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8528)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_init_os2'
    return stypy_return_type_8528

# Assigning a type to the variable '_init_os2' (line 423)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), '_init_os2', _init_os2)

@norecursion
def get_config_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_config_vars'
    module_type_store = module_type_store.open_function_context('get_config_vars', 440, 0, False)
    
    # Passed parameters checking function
    get_config_vars.stypy_localization = localization
    get_config_vars.stypy_type_of_self = None
    get_config_vars.stypy_type_store = module_type_store
    get_config_vars.stypy_function_name = 'get_config_vars'
    get_config_vars.stypy_param_names_list = []
    get_config_vars.stypy_varargs_param_name = 'args'
    get_config_vars.stypy_kwargs_param_name = None
    get_config_vars.stypy_call_defaults = defaults
    get_config_vars.stypy_call_varargs = varargs
    get_config_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_config_vars', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_config_vars', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_config_vars(...)' code ##################

    str_8529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, (-1)), 'str', "With no arguments, return a dictionary of all configuration\n    variables relevant for the current platform.  Generally this includes\n    everything needed to build extensions and install both pure modules and\n    extensions.  On Unix, this means every variable defined in Python's\n    installed Makefile; on Windows and Mac OS it's a much smaller set.\n\n    With arguments, return a list of values that result from looking up\n    each argument in the configuration variable dictionary.\n    ")
    # Marking variables as global (line 450)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 450, 4), '_config_vars')
    
    # Type idiom detected: calculating its left and rigth part (line 451)
    # Getting the type of '_config_vars' (line 451)
    _config_vars_8530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 7), '_config_vars')
    # Getting the type of 'None' (line 451)
    None_8531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'None')
    
    (may_be_8532, more_types_in_union_8533) = may_be_none(_config_vars_8530, None_8531)

    if may_be_8532:

        if more_types_in_union_8533:
            # Runtime conditional SSA (line 451)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 452):
        
        # Assigning a Call to a Name (line 452):
        
        # Call to get(...): (line 452)
        # Processing the call arguments (line 452)
        str_8538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 29), 'str', '_init_')
        # Getting the type of 'os' (line 452)
        os_8539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 40), 'os', False)
        # Obtaining the member 'name' of a type (line 452)
        name_8540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 40), os_8539, 'name')
        # Applying the binary operator '+' (line 452)
        result_add_8541 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 29), '+', str_8538, name_8540)
        
        # Processing the call keyword arguments (line 452)
        kwargs_8542 = {}
        
        # Call to globals(...): (line 452)
        # Processing the call keyword arguments (line 452)
        kwargs_8535 = {}
        # Getting the type of 'globals' (line 452)
        globals_8534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'globals', False)
        # Calling globals(args, kwargs) (line 452)
        globals_call_result_8536 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), globals_8534, *[], **kwargs_8535)
        
        # Obtaining the member 'get' of a type (line 452)
        get_8537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 15), globals_call_result_8536, 'get')
        # Calling get(args, kwargs) (line 452)
        get_call_result_8543 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), get_8537, *[result_add_8541], **kwargs_8542)
        
        # Assigning a type to the variable 'func' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'func', get_call_result_8543)
        
        # Getting the type of 'func' (line 453)
        func_8544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'func')
        # Testing the type of an if condition (line 453)
        if_condition_8545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), func_8544)
        # Assigning a type to the variable 'if_condition_8545' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_8545', if_condition_8545)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to func(...): (line 454)
        # Processing the call keyword arguments (line 454)
        kwargs_8547 = {}
        # Getting the type of 'func' (line 454)
        func_8546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'func', False)
        # Calling func(args, kwargs) (line 454)
        func_call_result_8548 = invoke(stypy.reporting.localization.Localization(__file__, 454, 12), func_8546, *[], **kwargs_8547)
        
        # SSA branch for the else part of an if statement (line 453)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Dict to a Name (line 456):
        
        # Assigning a Dict to a Name (line 456):
        
        # Obtaining an instance of the builtin type 'dict' (line 456)
        dict_8549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 456)
        
        # Assigning a type to the variable '_config_vars' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), '_config_vars', dict_8549)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 461):
        
        # Assigning a Name to a Subscript (line 461):
        # Getting the type of 'PREFIX' (line 461)
        PREFIX_8550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'PREFIX')
        # Getting the type of '_config_vars' (line 461)
        _config_vars_8551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), '_config_vars')
        str_8552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 21), 'str', 'prefix')
        # Storing an element on a container (line 461)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 8), _config_vars_8551, (str_8552, PREFIX_8550))
        
        # Assigning a Name to a Subscript (line 462):
        
        # Assigning a Name to a Subscript (line 462):
        # Getting the type of 'EXEC_PREFIX' (line 462)
        EXEC_PREFIX_8553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'EXEC_PREFIX')
        # Getting the type of '_config_vars' (line 462)
        _config_vars_8554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), '_config_vars')
        str_8555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'str', 'exec_prefix')
        # Storing an element on a container (line 462)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 8), _config_vars_8554, (str_8555, EXEC_PREFIX_8553))
        
        
        # Getting the type of 'sys' (line 466)
        sys_8556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 466)
        platform_8557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 11), sys_8556, 'platform')
        str_8558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 466)
        result_eq_8559 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 11), '==', platform_8557, str_8558)
        
        # Testing the type of an if condition (line 466)
        if_condition_8560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 8), result_eq_8559)
        # Assigning a type to the variable 'if_condition_8560' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'if_condition_8560', if_condition_8560)
        # SSA begins for if statement (line 466)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 467, 12))
        
        # 'import _osx_support' statement (line 467)
        import _osx_support

        import_module(stypy.reporting.localization.Localization(__file__, 467, 12), '_osx_support', _osx_support, module_type_store)
        
        
        # Call to customize_config_vars(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of '_config_vars' (line 468)
        _config_vars_8563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 47), '_config_vars', False)
        # Processing the call keyword arguments (line 468)
        kwargs_8564 = {}
        # Getting the type of '_osx_support' (line 468)
        _osx_support_8561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), '_osx_support', False)
        # Obtaining the member 'customize_config_vars' of a type (line 468)
        customize_config_vars_8562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 12), _osx_support_8561, 'customize_config_vars')
        # Calling customize_config_vars(args, kwargs) (line 468)
        customize_config_vars_call_result_8565 = invoke(stypy.reporting.localization.Localization(__file__, 468, 12), customize_config_vars_8562, *[_config_vars_8563], **kwargs_8564)
        
        # SSA join for if statement (line 466)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_8533:
            # SSA join for if statement (line 451)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'args' (line 470)
    args_8566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 7), 'args')
    # Testing the type of an if condition (line 470)
    if_condition_8567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 4), args_8566)
    # Assigning a type to the variable 'if_condition_8567' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'if_condition_8567', if_condition_8567)
    # SSA begins for if statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 471):
    
    # Assigning a List to a Name (line 471):
    
    # Obtaining an instance of the builtin type 'list' (line 471)
    list_8568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 471)
    
    # Assigning a type to the variable 'vals' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'vals', list_8568)
    
    # Getting the type of 'args' (line 472)
    args_8569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'args')
    # Testing the type of a for loop iterable (line 472)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 472, 8), args_8569)
    # Getting the type of the for loop variable (line 472)
    for_loop_var_8570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 472, 8), args_8569)
    # Assigning a type to the variable 'name' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'name', for_loop_var_8570)
    # SSA begins for a for statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 473)
    # Processing the call arguments (line 473)
    
    # Call to get(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'name' (line 473)
    name_8575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 41), 'name', False)
    # Processing the call keyword arguments (line 473)
    kwargs_8576 = {}
    # Getting the type of '_config_vars' (line 473)
    _config_vars_8573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 24), '_config_vars', False)
    # Obtaining the member 'get' of a type (line 473)
    get_8574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 24), _config_vars_8573, 'get')
    # Calling get(args, kwargs) (line 473)
    get_call_result_8577 = invoke(stypy.reporting.localization.Localization(__file__, 473, 24), get_8574, *[name_8575], **kwargs_8576)
    
    # Processing the call keyword arguments (line 473)
    kwargs_8578 = {}
    # Getting the type of 'vals' (line 473)
    vals_8571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'vals', False)
    # Obtaining the member 'append' of a type (line 473)
    append_8572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 12), vals_8571, 'append')
    # Calling append(args, kwargs) (line 473)
    append_call_result_8579 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), append_8572, *[get_call_result_8577], **kwargs_8578)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'vals' (line 474)
    vals_8580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'vals')
    # Assigning a type to the variable 'stypy_return_type' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'stypy_return_type', vals_8580)
    # SSA branch for the else part of an if statement (line 470)
    module_type_store.open_ssa_branch('else')
    # Getting the type of '_config_vars' (line 476)
    _config_vars_8581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), '_config_vars')
    # Assigning a type to the variable 'stypy_return_type' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', _config_vars_8581)
    # SSA join for if statement (line 470)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_config_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_config_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 440)
    stypy_return_type_8582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8582)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_config_vars'
    return stypy_return_type_8582

# Assigning a type to the variable 'get_config_vars' (line 440)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'get_config_vars', get_config_vars)

@norecursion
def get_config_var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_config_var'
    module_type_store = module_type_store.open_function_context('get_config_var', 478, 0, False)
    
    # Passed parameters checking function
    get_config_var.stypy_localization = localization
    get_config_var.stypy_type_of_self = None
    get_config_var.stypy_type_store = module_type_store
    get_config_var.stypy_function_name = 'get_config_var'
    get_config_var.stypy_param_names_list = ['name']
    get_config_var.stypy_varargs_param_name = None
    get_config_var.stypy_kwargs_param_name = None
    get_config_var.stypy_call_defaults = defaults
    get_config_var.stypy_call_varargs = varargs
    get_config_var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_config_var', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_config_var', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_config_var(...)' code ##################

    str_8583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, (-1)), 'str', "Return the value of a single variable using the dictionary\n    returned by 'get_config_vars()'.  Equivalent to\n    get_config_vars().get(name)\n    ")
    
    # Call to get(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'name' (line 483)
    name_8588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 33), 'name', False)
    # Processing the call keyword arguments (line 483)
    kwargs_8589 = {}
    
    # Call to get_config_vars(...): (line 483)
    # Processing the call keyword arguments (line 483)
    kwargs_8585 = {}
    # Getting the type of 'get_config_vars' (line 483)
    get_config_vars_8584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 11), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 483)
    get_config_vars_call_result_8586 = invoke(stypy.reporting.localization.Localization(__file__, 483, 11), get_config_vars_8584, *[], **kwargs_8585)
    
    # Obtaining the member 'get' of a type (line 483)
    get_8587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 11), get_config_vars_call_result_8586, 'get')
    # Calling get(args, kwargs) (line 483)
    get_call_result_8590 = invoke(stypy.reporting.localization.Localization(__file__, 483, 11), get_8587, *[name_8588], **kwargs_8589)
    
    # Assigning a type to the variable 'stypy_return_type' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'stypy_return_type', get_call_result_8590)
    
    # ################# End of 'get_config_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_config_var' in the type store
    # Getting the type of 'stypy_return_type' (line 478)
    stypy_return_type_8591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8591)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_config_var'
    return stypy_return_type_8591

# Assigning a type to the variable 'get_config_var' (line 478)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 0), 'get_config_var', get_config_var)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

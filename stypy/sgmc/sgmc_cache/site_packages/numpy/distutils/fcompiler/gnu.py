
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import re
4: import os
5: import sys
6: import warnings
7: import platform
8: import tempfile
9: from subprocess import Popen, PIPE, STDOUT
10: 
11: from numpy.distutils.fcompiler import FCompiler
12: from numpy.distutils.exec_command import exec_command
13: from numpy.distutils.misc_util import msvc_runtime_library
14: from numpy.distutils.compat import get_exception
15: 
16: compilers = ['GnuFCompiler', 'Gnu95FCompiler']
17: 
18: TARGET_R = re.compile("Target: ([a-zA-Z0-9_\-]*)")
19: 
20: # XXX: handle cross compilation
21: def is_win64():
22:     return sys.platform == "win32" and platform.architecture()[0] == "64bit"
23: 
24: if is_win64():
25:     #_EXTRAFLAGS = ["-fno-leading-underscore"]
26:     _EXTRAFLAGS = []
27: else:
28:     _EXTRAFLAGS = []
29: 
30: class GnuFCompiler(FCompiler):
31:     compiler_type = 'gnu'
32:     compiler_aliases = ('g77',)
33:     description = 'GNU Fortran 77 compiler'
34: 
35:     def gnu_version_match(self, version_string):
36:         '''Handle the different versions of GNU fortran compilers'''
37:         # Strip warning(s) that may be emitted by gfortran
38:         while version_string.startswith('gfortran: warning'):
39:             version_string = version_string[version_string.find('\n')+1:]
40: 
41:         # Gfortran versions from after 2010 will output a simple string
42:         # (usually "x.y", "x.y.z" or "x.y.z-q") for ``-dumpversion``; older
43:         # gfortrans may still return long version strings (``-dumpversion`` was
44:         # an alias for ``--version``)
45:         if len(version_string) <= 20:
46:             # Try to find a valid version string
47:             m = re.search(r'([0-9.]+)', version_string)
48:             if m:
49:                 # g77 provides a longer version string that starts with GNU
50:                 # Fortran
51:                 if version_string.startswith('GNU Fortran'):
52:                     return ('g77', m.group(1))
53: 
54:                 # gfortran only outputs a version string such as #.#.#, so check
55:                 # if the match is at the start of the string
56:                 elif m.start() == 0:
57:                     return ('gfortran', m.group(1))
58:         else:
59:             # Output probably from --version, try harder:
60:             m = re.search(r'GNU Fortran\s+95.*?([0-9-.]+)', version_string)
61:             if m:
62:                 return ('gfortran', m.group(1))
63:             m = re.search(r'GNU Fortran.*?\-?([0-9-.]+)', version_string)
64:             if m:
65:                 v = m.group(1)
66:                 if v.startswith('0') or v.startswith('2') or v.startswith('3'):
67:                     # the '0' is for early g77's
68:                     return ('g77', v)
69:                 else:
70:                     # at some point in the 4.x series, the ' 95' was dropped
71:                     # from the version string
72:                     return ('gfortran', v)
73: 
74:         # If still nothing, raise an error to make the problem easy to find.
75:         err = 'A valid Fortran version was not found in this string:\n'
76:         raise ValueError(err + version_string)
77: 
78:     def version_match(self, version_string):
79:         v = self.gnu_version_match(version_string)
80:         if not v or v[0] != 'g77':
81:             return None
82:         return v[1]
83: 
84:     possible_executables = ['g77', 'f77']
85:     executables = {
86:         'version_cmd'  : [None, "-dumpversion"],
87:         'compiler_f77' : [None, "-g", "-Wall", "-fno-second-underscore"],
88:         'compiler_f90' : None,  # Use --fcompiler=gnu95 for f90 codes
89:         'compiler_fix' : None,
90:         'linker_so'    : [None, "-g", "-Wall"],
91:         'archiver'     : ["ar", "-cr"],
92:         'ranlib'       : ["ranlib"],
93:         'linker_exe'   : [None, "-g", "-Wall"]
94:         }
95:     module_dir_switch = None
96:     module_include_switch = None
97: 
98:     # Cygwin: f771: warning: -fPIC ignored for target (all code is
99:     # position independent)
100:     if os.name != 'nt' and sys.platform != 'cygwin':
101:         pic_flags = ['-fPIC']
102: 
103:     # use -mno-cygwin for g77 when Python is not Cygwin-Python
104:     if sys.platform == 'win32':
105:         for key in ['version_cmd', 'compiler_f77', 'linker_so', 'linker_exe']:
106:             executables[key].append('-mno-cygwin')
107: 
108:     g2c = 'g2c'
109:     suggested_f90_compiler = 'gnu95'
110: 
111:     def get_flags_linker_so(self):
112:         opt = self.linker_so[1:]
113:         if sys.platform == 'darwin':
114:             target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
115:             # If MACOSX_DEPLOYMENT_TARGET is set, we simply trust the value
116:             # and leave it alone.  But, distutils will complain if the
117:             # environment's value is different from the one in the Python
118:             # Makefile used to build Python.  We let disutils handle this
119:             # error checking.
120:             if not target:
121:                 # If MACOSX_DEPLOYMENT_TARGET is not set in the environment,
122:                 # we try to get it first from the Python Makefile and then we
123:                 # fall back to setting it to 10.3 to maximize the set of
124:                 # versions we can work with.  This is a reasonable default
125:                 # even when using the official Python dist and those derived
126:                 # from it.
127:                 import distutils.sysconfig as sc
128:                 g = {}
129:                 filename = sc.get_makefile_filename()
130:                 sc.parse_makefile(filename, g)
131:                 target = g.get('MACOSX_DEPLOYMENT_TARGET', '10.3')
132:                 os.environ['MACOSX_DEPLOYMENT_TARGET'] = target
133:                 if target == '10.3':
134:                     s = 'Env. variable MACOSX_DEPLOYMENT_TARGET set to 10.3'
135:                     warnings.warn(s)
136: 
137:             opt.extend(['-undefined', 'dynamic_lookup', '-bundle'])
138:         else:
139:             opt.append("-shared")
140:         if sys.platform.startswith('sunos'):
141:             # SunOS often has dynamically loaded symbols defined in the
142:             # static library libg2c.a  The linker doesn't like this.  To
143:             # ignore the problem, use the -mimpure-text flag.  It isn't
144:             # the safest thing, but seems to work. 'man gcc' says:
145:             # ".. Instead of using -mimpure-text, you should compile all
146:             #  source code with -fpic or -fPIC."
147:             opt.append('-mimpure-text')
148:         return opt
149: 
150:     def get_libgcc_dir(self):
151:         status, output = exec_command(self.compiler_f77 +
152:                                       ['-print-libgcc-file-name'],
153:                                       use_tee=0)
154:         if not status:
155:             return os.path.dirname(output)
156:         return None
157: 
158:     def get_library_dirs(self):
159:         opt = []
160:         if sys.platform[:5] != 'linux':
161:             d = self.get_libgcc_dir()
162:             if d:
163:                 # if windows and not cygwin, libg2c lies in a different folder
164:                 if sys.platform == 'win32' and not d.startswith('/usr/lib'):
165:                     d = os.path.normpath(d)
166:                     path = os.path.join(d, "lib%s.a" % self.g2c)
167:                     if not os.path.exists(path):
168:                         root = os.path.join(d, *((os.pardir,)*4))
169:                         d2 = os.path.abspath(os.path.join(root, 'lib'))
170:                         path = os.path.join(d2, "lib%s.a" % self.g2c)
171:                         if os.path.exists(path):
172:                             opt.append(d2)
173:                 opt.append(d)
174:         return opt
175: 
176:     def get_libraries(self):
177:         opt = []
178:         d = self.get_libgcc_dir()
179:         if d is not None:
180:             g2c = self.g2c + '-pic'
181:             f = self.static_lib_format % (g2c, self.static_lib_extension)
182:             if not os.path.isfile(os.path.join(d, f)):
183:                 g2c = self.g2c
184:         else:
185:             g2c = self.g2c
186: 
187:         if g2c is not None:
188:             opt.append(g2c)
189:         c_compiler = self.c_compiler
190:         if sys.platform == 'win32' and c_compiler and \
191:                c_compiler.compiler_type == 'msvc':
192:             # the following code is not needed (read: breaks) when using MinGW
193:             # in case want to link F77 compiled code with MSVC
194:             opt.append('gcc')
195:             runtime_lib = msvc_runtime_library()
196:             if runtime_lib:
197:                 opt.append(runtime_lib)
198:         if sys.platform == 'darwin':
199:             opt.append('cc_dynamic')
200:         return opt
201: 
202:     def get_flags_debug(self):
203:         return ['-g']
204: 
205:     def get_flags_opt(self):
206:         v = self.get_version()
207:         if v and v <= '3.3.3':
208:             # With this compiler version building Fortran BLAS/LAPACK
209:             # with -O3 caused failures in lib.lapack heevr,syevr tests.
210:             opt = ['-O2']
211:         else:
212:             opt = ['-O3']
213:         opt.append('-funroll-loops')
214:         return opt
215: 
216:     def _c_arch_flags(self):
217:         ''' Return detected arch flags from CFLAGS '''
218:         from distutils import sysconfig
219:         try:
220:             cflags = sysconfig.get_config_vars()['CFLAGS']
221:         except KeyError:
222:             return []
223:         arch_re = re.compile(r"-arch\s+(\w+)")
224:         arch_flags = []
225:         for arch in arch_re.findall(cflags):
226:             arch_flags += ['-arch', arch]
227:         return arch_flags
228: 
229:     def get_flags_arch(self):
230:         return []
231: 
232:     def runtime_library_dir_option(self, dir):
233:         return '-Wl,-rpath="%s"' % dir
234: 
235: class Gnu95FCompiler(GnuFCompiler):
236:     compiler_type = 'gnu95'
237:     compiler_aliases = ('gfortran',)
238:     description = 'GNU Fortran 95 compiler'
239: 
240:     def version_match(self, version_string):
241:         v = self.gnu_version_match(version_string)
242:         if not v or v[0] != 'gfortran':
243:             return None
244:         v = v[1]
245:         if v >= '4.':
246:             # gcc-4 series releases do not support -mno-cygwin option
247:             pass
248:         else:
249:             # use -mno-cygwin flag for gfortran when Python is not
250:             # Cygwin-Python
251:             if sys.platform == 'win32':
252:                 for key in ['version_cmd', 'compiler_f77', 'compiler_f90',
253:                             'compiler_fix', 'linker_so', 'linker_exe']:
254:                     self.executables[key].append('-mno-cygwin')
255:         return v
256: 
257:     possible_executables = ['gfortran', 'f95']
258:     executables = {
259:         'version_cmd'  : ["<F90>", "-dumpversion"],
260:         'compiler_f77' : [None, "-Wall", "-g", "-ffixed-form",
261:                           "-fno-second-underscore"] + _EXTRAFLAGS,
262:         'compiler_f90' : [None, "-Wall", "-g",
263:                           "-fno-second-underscore"] + _EXTRAFLAGS,
264:         'compiler_fix' : [None, "-Wall",  "-g","-ffixed-form",
265:                           "-fno-second-underscore"] + _EXTRAFLAGS,
266:         'linker_so'    : ["<F90>", "-Wall", "-g"],
267:         'archiver'     : ["ar", "-cr"],
268:         'ranlib'       : ["ranlib"],
269:         'linker_exe'   : [None, "-Wall"]
270:         }
271: 
272:     module_dir_switch = '-J'
273:     module_include_switch = '-I'
274: 
275:     g2c = 'gfortran'
276: 
277:     def _universal_flags(self, cmd):
278:         '''Return a list of -arch flags for every supported architecture.'''
279:         if not sys.platform == 'darwin':
280:             return []
281:         arch_flags = []
282:         # get arches the C compiler gets.
283:         c_archs = self._c_arch_flags()
284:         if "i386" in c_archs:
285:             c_archs[c_archs.index("i386")] = "i686"
286:         # check the arches the Fortran compiler supports, and compare with
287:         # arch flags from C compiler
288:         for arch in ["ppc", "i686", "x86_64", "ppc64"]:
289:             if _can_target(cmd, arch) and arch in c_archs:
290:                 arch_flags.extend(["-arch", arch])
291:         return arch_flags
292: 
293:     def get_flags(self):
294:         flags = GnuFCompiler.get_flags(self)
295:         arch_flags = self._universal_flags(self.compiler_f90)
296:         if arch_flags:
297:             flags[:0] = arch_flags
298:         return flags
299: 
300:     def get_flags_linker_so(self):
301:         flags = GnuFCompiler.get_flags_linker_so(self)
302:         arch_flags = self._universal_flags(self.linker_so)
303:         if arch_flags:
304:             flags[:0] = arch_flags
305:         return flags
306: 
307:     def get_library_dirs(self):
308:         opt = GnuFCompiler.get_library_dirs(self)
309:         if sys.platform == 'win32':
310:             c_compiler = self.c_compiler
311:             if c_compiler and c_compiler.compiler_type == "msvc":
312:                 target = self.get_target()
313:                 if target:
314:                     d = os.path.normpath(self.get_libgcc_dir())
315:                     root = os.path.join(d, *((os.pardir,)*4))
316:                     path = os.path.join(root, "lib")
317:                     mingwdir = os.path.normpath(path)
318:                     if os.path.exists(os.path.join(mingwdir, "libmingwex.a")):
319:                         opt.append(mingwdir)
320:         return opt
321: 
322:     def get_libraries(self):
323:         opt = GnuFCompiler.get_libraries(self)
324:         if sys.platform == 'darwin':
325:             opt.remove('cc_dynamic')
326:         if sys.platform == 'win32':
327:             c_compiler = self.c_compiler
328:             if c_compiler and c_compiler.compiler_type == "msvc":
329:                 if "gcc" in opt:
330:                     i = opt.index("gcc")
331:                     opt.insert(i+1, "mingwex")
332:                     opt.insert(i+1, "mingw32")
333:             # XXX: fix this mess, does not work for mingw
334:             if is_win64():
335:                 c_compiler = self.c_compiler
336:                 if c_compiler and c_compiler.compiler_type == "msvc":
337:                     return []
338:                 else:
339:                     pass
340:         return opt
341: 
342:     def get_target(self):
343:         status, output = exec_command(self.compiler_f77 +
344:                                       ['-v'],
345:                                       use_tee=0)
346:         if not status:
347:             m = TARGET_R.search(output)
348:             if m:
349:                 return m.group(1)
350:         return ""
351: 
352:     def get_flags_opt(self):
353:         if is_win64():
354:             return ['-O0']
355:         else:
356:             return GnuFCompiler.get_flags_opt(self)
357: 
358: def _can_target(cmd, arch):
359:     '''Return true if the architecture supports the -arch flag'''
360:     newcmd = cmd[:]
361:     fid, filename = tempfile.mkstemp(suffix=".f")
362:     try:
363:         d = os.path.dirname(filename)
364:         output = os.path.splitext(filename)[0] + ".o"
365:         try:
366:             newcmd.extend(["-arch", arch, "-c", filename])
367:             p = Popen(newcmd, stderr=STDOUT, stdout=PIPE, cwd=d)
368:             p.communicate()
369:             return p.returncode == 0
370:         finally:
371:             if os.path.exists(output):
372:                 os.remove(output)
373:     finally:
374:         os.remove(filename)
375:     return False
376: 
377: if __name__ == '__main__':
378:     from distutils import log
379:     log.set_verbosity(2)
380: 
381:     compiler = GnuFCompiler()
382:     compiler.customize()
383:     print(compiler.get_version())
384: 
385:     try:
386:         compiler = Gnu95FCompiler()
387:         compiler.customize()
388:         print(compiler.get_version())
389:     except Exception:
390:         msg = get_exception()
391:         print(msg)
392: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import re' statement (line 3)
import re

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import platform' statement (line 7)
import platform

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'platform', platform, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import tempfile' statement (line 8)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from subprocess import Popen, PIPE, STDOUT' statement (line 9)
from subprocess import Popen, PIPE, STDOUT

import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'subprocess', None, module_type_store, ['Popen', 'PIPE', 'STDOUT'], [Popen, PIPE, STDOUT])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60628 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.fcompiler')

if (type(import_60628) is not StypyTypeError):

    if (import_60628 != 'pyd_module'):
        __import__(import_60628)
        sys_modules_60629 = sys.modules[import_60628]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.fcompiler', sys_modules_60629.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_60629, sys_modules_60629.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.fcompiler', import_60628)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.distutils.exec_command import exec_command' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.exec_command')

if (type(import_60630) is not StypyTypeError):

    if (import_60630 != 'pyd_module'):
        __import__(import_60630)
        sys_modules_60631 = sys.modules[import_60630]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.exec_command', sys_modules_60631.module_type_store, module_type_store, ['exec_command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_60631, sys_modules_60631.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import exec_command

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.exec_command', None, module_type_store, ['exec_command'], [exec_command])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.exec_command', import_60630)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.distutils.misc_util import msvc_runtime_library' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.misc_util')

if (type(import_60632) is not StypyTypeError):

    if (import_60632 != 'pyd_module'):
        __import__(import_60632)
        sys_modules_60633 = sys.modules[import_60632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.misc_util', sys_modules_60633.module_type_store, module_type_store, ['msvc_runtime_library'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_60633, sys_modules_60633.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import msvc_runtime_library

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.misc_util', None, module_type_store, ['msvc_runtime_library'], [msvc_runtime_library])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.distutils.misc_util', import_60632)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60634 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.compat')

if (type(import_60634) is not StypyTypeError):

    if (import_60634 != 'pyd_module'):
        __import__(import_60634)
        sys_modules_60635 = sys.modules[import_60634]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.compat', sys_modules_60635.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_60635, sys_modules_60635.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.compat', import_60634)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):

# Obtaining an instance of the builtin type 'list' (line 16)
list_60636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_60637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'str', 'GnuFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), list_60636, str_60637)
# Adding element type (line 16)
str_60638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 29), 'str', 'Gnu95FCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), list_60636, str_60638)

# Assigning a type to the variable 'compilers' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'compilers', list_60636)

# Assigning a Call to a Name (line 18):

# Assigning a Call to a Name (line 18):

# Call to compile(...): (line 18)
# Processing the call arguments (line 18)
str_60641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'str', 'Target: ([a-zA-Z0-9_\\-]*)')
# Processing the call keyword arguments (line 18)
kwargs_60642 = {}
# Getting the type of 're' (line 18)
re_60639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 're', False)
# Obtaining the member 'compile' of a type (line 18)
compile_60640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), re_60639, 'compile')
# Calling compile(args, kwargs) (line 18)
compile_call_result_60643 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), compile_60640, *[str_60641], **kwargs_60642)

# Assigning a type to the variable 'TARGET_R' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'TARGET_R', compile_call_result_60643)

@norecursion
def is_win64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_win64'
    module_type_store = module_type_store.open_function_context('is_win64', 21, 0, False)
    
    # Passed parameters checking function
    is_win64.stypy_localization = localization
    is_win64.stypy_type_of_self = None
    is_win64.stypy_type_store = module_type_store
    is_win64.stypy_function_name = 'is_win64'
    is_win64.stypy_param_names_list = []
    is_win64.stypy_varargs_param_name = None
    is_win64.stypy_kwargs_param_name = None
    is_win64.stypy_call_defaults = defaults
    is_win64.stypy_call_varargs = varargs
    is_win64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_win64', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_win64', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_win64(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Getting the type of 'sys' (line 22)
    sys_60644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'sys')
    # Obtaining the member 'platform' of a type (line 22)
    platform_60645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), sys_60644, 'platform')
    str_60646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'str', 'win32')
    # Applying the binary operator '==' (line 22)
    result_eq_60647 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '==', platform_60645, str_60646)
    
    
    
    # Obtaining the type of the subscript
    int_60648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 63), 'int')
    
    # Call to architecture(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_60651 = {}
    # Getting the type of 'platform' (line 22)
    platform_60649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'platform', False)
    # Obtaining the member 'architecture' of a type (line 22)
    architecture_60650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 39), platform_60649, 'architecture')
    # Calling architecture(args, kwargs) (line 22)
    architecture_call_result_60652 = invoke(stypy.reporting.localization.Localization(__file__, 22, 39), architecture_60650, *[], **kwargs_60651)
    
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___60653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 39), architecture_call_result_60652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_60654 = invoke(stypy.reporting.localization.Localization(__file__, 22, 39), getitem___60653, int_60648)
    
    str_60655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 69), 'str', '64bit')
    # Applying the binary operator '==' (line 22)
    result_eq_60656 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 39), '==', subscript_call_result_60654, str_60655)
    
    # Applying the binary operator 'and' (line 22)
    result_and_keyword_60657 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), 'and', result_eq_60647, result_eq_60656)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', result_and_keyword_60657)
    
    # ################# End of 'is_win64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_win64' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_60658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_60658)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_win64'
    return stypy_return_type_60658

# Assigning a type to the variable 'is_win64' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'is_win64', is_win64)


# Call to is_win64(...): (line 24)
# Processing the call keyword arguments (line 24)
kwargs_60660 = {}
# Getting the type of 'is_win64' (line 24)
is_win64_60659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 3), 'is_win64', False)
# Calling is_win64(args, kwargs) (line 24)
is_win64_call_result_60661 = invoke(stypy.reporting.localization.Localization(__file__, 24, 3), is_win64_60659, *[], **kwargs_60660)

# Testing the type of an if condition (line 24)
if_condition_60662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 0), is_win64_call_result_60661)
# Assigning a type to the variable 'if_condition_60662' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'if_condition_60662', if_condition_60662)
# SSA begins for if statement (line 24)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_60663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)

# Assigning a type to the variable '_EXTRAFLAGS' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), '_EXTRAFLAGS', list_60663)
# SSA branch for the else part of an if statement (line 24)
module_type_store.open_ssa_branch('else')

# Assigning a List to a Name (line 28):

# Assigning a List to a Name (line 28):

# Obtaining an instance of the builtin type 'list' (line 28)
list_60664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)

# Assigning a type to the variable '_EXTRAFLAGS' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), '_EXTRAFLAGS', list_60664)
# SSA join for if statement (line 24)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'GnuFCompiler' class
# Getting the type of 'FCompiler' (line 30)
FCompiler_60665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'FCompiler')

class GnuFCompiler(FCompiler_60665, ):
    
    # Assigning a Str to a Name (line 31):
    
    # Assigning a Tuple to a Name (line 32):
    
    # Assigning a Str to a Name (line 33):

    @norecursion
    def gnu_version_match(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'gnu_version_match'
        module_type_store = module_type_store.open_function_context('gnu_version_match', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.gnu_version_match')
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_param_names_list', ['version_string'])
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.gnu_version_match.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.gnu_version_match', ['version_string'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'gnu_version_match', localization, ['version_string'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'gnu_version_match(...)' code ##################

        str_60666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'str', 'Handle the different versions of GNU fortran compilers')
        
        
        # Call to startswith(...): (line 38)
        # Processing the call arguments (line 38)
        str_60669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'str', 'gfortran: warning')
        # Processing the call keyword arguments (line 38)
        kwargs_60670 = {}
        # Getting the type of 'version_string' (line 38)
        version_string_60667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'version_string', False)
        # Obtaining the member 'startswith' of a type (line 38)
        startswith_60668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 14), version_string_60667, 'startswith')
        # Calling startswith(args, kwargs) (line 38)
        startswith_call_result_60671 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), startswith_60668, *[str_60669], **kwargs_60670)
        
        # Testing the type of an if condition (line 38)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), startswith_call_result_60671)
        # SSA begins for while statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Subscript to a Name (line 39):
        
        # Assigning a Subscript to a Name (line 39):
        
        # Obtaining the type of the subscript
        
        # Call to find(...): (line 39)
        # Processing the call arguments (line 39)
        str_60674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 64), 'str', '\n')
        # Processing the call keyword arguments (line 39)
        kwargs_60675 = {}
        # Getting the type of 'version_string' (line 39)
        version_string_60672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 44), 'version_string', False)
        # Obtaining the member 'find' of a type (line 39)
        find_60673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 44), version_string_60672, 'find')
        # Calling find(args, kwargs) (line 39)
        find_call_result_60676 = invoke(stypy.reporting.localization.Localization(__file__, 39, 44), find_60673, *[str_60674], **kwargs_60675)
        
        int_60677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 70), 'int')
        # Applying the binary operator '+' (line 39)
        result_add_60678 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 44), '+', find_call_result_60676, int_60677)
        
        slice_60679 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 39, 29), result_add_60678, None, None)
        # Getting the type of 'version_string' (line 39)
        version_string_60680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'version_string')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___60681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), version_string_60680, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_60682 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), getitem___60681, slice_60679)
        
        # Assigning a type to the variable 'version_string' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'version_string', subscript_call_result_60682)
        # SSA join for while statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'version_string' (line 45)
        version_string_60684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'version_string', False)
        # Processing the call keyword arguments (line 45)
        kwargs_60685 = {}
        # Getting the type of 'len' (line 45)
        len_60683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'len', False)
        # Calling len(args, kwargs) (line 45)
        len_call_result_60686 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), len_60683, *[version_string_60684], **kwargs_60685)
        
        int_60687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'int')
        # Applying the binary operator '<=' (line 45)
        result_le_60688 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '<=', len_call_result_60686, int_60687)
        
        # Testing the type of an if condition (line 45)
        if_condition_60689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_le_60688)
        # Assigning a type to the variable 'if_condition_60689' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_60689', if_condition_60689)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to search(...): (line 47)
        # Processing the call arguments (line 47)
        str_60692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'str', '([0-9.]+)')
        # Getting the type of 'version_string' (line 47)
        version_string_60693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'version_string', False)
        # Processing the call keyword arguments (line 47)
        kwargs_60694 = {}
        # Getting the type of 're' (line 47)
        re_60690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 're', False)
        # Obtaining the member 'search' of a type (line 47)
        search_60691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), re_60690, 'search')
        # Calling search(args, kwargs) (line 47)
        search_call_result_60695 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), search_60691, *[str_60692, version_string_60693], **kwargs_60694)
        
        # Assigning a type to the variable 'm' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'm', search_call_result_60695)
        
        # Getting the type of 'm' (line 48)
        m_60696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'm')
        # Testing the type of an if condition (line 48)
        if_condition_60697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 12), m_60696)
        # Assigning a type to the variable 'if_condition_60697' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'if_condition_60697', if_condition_60697)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to startswith(...): (line 51)
        # Processing the call arguments (line 51)
        str_60700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 45), 'str', 'GNU Fortran')
        # Processing the call keyword arguments (line 51)
        kwargs_60701 = {}
        # Getting the type of 'version_string' (line 51)
        version_string_60698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'version_string', False)
        # Obtaining the member 'startswith' of a type (line 51)
        startswith_60699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), version_string_60698, 'startswith')
        # Calling startswith(args, kwargs) (line 51)
        startswith_call_result_60702 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), startswith_60699, *[str_60700], **kwargs_60701)
        
        # Testing the type of an if condition (line 51)
        if_condition_60703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 16), startswith_call_result_60702)
        # Assigning a type to the variable 'if_condition_60703' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'if_condition_60703', if_condition_60703)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_60704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        str_60705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'str', 'g77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), tuple_60704, str_60705)
        # Adding element type (line 52)
        
        # Call to group(...): (line 52)
        # Processing the call arguments (line 52)
        int_60708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'int')
        # Processing the call keyword arguments (line 52)
        kwargs_60709 = {}
        # Getting the type of 'm' (line 52)
        m_60706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'm', False)
        # Obtaining the member 'group' of a type (line 52)
        group_60707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 35), m_60706, 'group')
        # Calling group(args, kwargs) (line 52)
        group_call_result_60710 = invoke(stypy.reporting.localization.Localization(__file__, 52, 35), group_60707, *[int_60708], **kwargs_60709)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), tuple_60704, group_call_result_60710)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'stypy_return_type', tuple_60704)
        # SSA branch for the else part of an if statement (line 51)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to start(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_60713 = {}
        # Getting the type of 'm' (line 56)
        m_60711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'm', False)
        # Obtaining the member 'start' of a type (line 56)
        start_60712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), m_60711, 'start')
        # Calling start(args, kwargs) (line 56)
        start_call_result_60714 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), start_60712, *[], **kwargs_60713)
        
        int_60715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
        # Applying the binary operator '==' (line 56)
        result_eq_60716 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 21), '==', start_call_result_60714, int_60715)
        
        # Testing the type of an if condition (line 56)
        if_condition_60717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 21), result_eq_60716)
        # Assigning a type to the variable 'if_condition_60717' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'if_condition_60717', if_condition_60717)
        # SSA begins for if statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_60718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        str_60719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'str', 'gfortran')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), tuple_60718, str_60719)
        # Adding element type (line 57)
        
        # Call to group(...): (line 57)
        # Processing the call arguments (line 57)
        int_60722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 48), 'int')
        # Processing the call keyword arguments (line 57)
        kwargs_60723 = {}
        # Getting the type of 'm' (line 57)
        m_60720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 40), 'm', False)
        # Obtaining the member 'group' of a type (line 57)
        group_60721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 40), m_60720, 'group')
        # Calling group(args, kwargs) (line 57)
        group_call_result_60724 = invoke(stypy.reporting.localization.Localization(__file__, 57, 40), group_60721, *[int_60722], **kwargs_60723)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), tuple_60718, group_call_result_60724)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'stypy_return_type', tuple_60718)
        # SSA join for if statement (line 56)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 45)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to search(...): (line 60)
        # Processing the call arguments (line 60)
        str_60727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'str', 'GNU Fortran\\s+95.*?([0-9-.]+)')
        # Getting the type of 'version_string' (line 60)
        version_string_60728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 60), 'version_string', False)
        # Processing the call keyword arguments (line 60)
        kwargs_60729 = {}
        # Getting the type of 're' (line 60)
        re_60725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 're', False)
        # Obtaining the member 'search' of a type (line 60)
        search_60726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), re_60725, 'search')
        # Calling search(args, kwargs) (line 60)
        search_call_result_60730 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), search_60726, *[str_60727, version_string_60728], **kwargs_60729)
        
        # Assigning a type to the variable 'm' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'm', search_call_result_60730)
        
        # Getting the type of 'm' (line 61)
        m_60731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'm')
        # Testing the type of an if condition (line 61)
        if_condition_60732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 12), m_60731)
        # Assigning a type to the variable 'if_condition_60732' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'if_condition_60732', if_condition_60732)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_60733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        str_60734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'str', 'gfortran')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 24), tuple_60733, str_60734)
        # Adding element type (line 62)
        
        # Call to group(...): (line 62)
        # Processing the call arguments (line 62)
        int_60737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'int')
        # Processing the call keyword arguments (line 62)
        kwargs_60738 = {}
        # Getting the type of 'm' (line 62)
        m_60735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 36), 'm', False)
        # Obtaining the member 'group' of a type (line 62)
        group_60736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 36), m_60735, 'group')
        # Calling group(args, kwargs) (line 62)
        group_call_result_60739 = invoke(stypy.reporting.localization.Localization(__file__, 62, 36), group_60736, *[int_60737], **kwargs_60738)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 24), tuple_60733, group_call_result_60739)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'stypy_return_type', tuple_60733)
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to search(...): (line 63)
        # Processing the call arguments (line 63)
        str_60742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'str', 'GNU Fortran.*?\\-?([0-9-.]+)')
        # Getting the type of 'version_string' (line 63)
        version_string_60743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 58), 'version_string', False)
        # Processing the call keyword arguments (line 63)
        kwargs_60744 = {}
        # Getting the type of 're' (line 63)
        re_60740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 're', False)
        # Obtaining the member 'search' of a type (line 63)
        search_60741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), re_60740, 'search')
        # Calling search(args, kwargs) (line 63)
        search_call_result_60745 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), search_60741, *[str_60742, version_string_60743], **kwargs_60744)
        
        # Assigning a type to the variable 'm' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'm', search_call_result_60745)
        
        # Getting the type of 'm' (line 64)
        m_60746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'm')
        # Testing the type of an if condition (line 64)
        if_condition_60747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), m_60746)
        # Assigning a type to the variable 'if_condition_60747' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'if_condition_60747', if_condition_60747)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to group(...): (line 65)
        # Processing the call arguments (line 65)
        int_60750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_60751 = {}
        # Getting the type of 'm' (line 65)
        m_60748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'm', False)
        # Obtaining the member 'group' of a type (line 65)
        group_60749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), m_60748, 'group')
        # Calling group(args, kwargs) (line 65)
        group_call_result_60752 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), group_60749, *[int_60750], **kwargs_60751)
        
        # Assigning a type to the variable 'v' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'v', group_call_result_60752)
        
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 66)
        # Processing the call arguments (line 66)
        str_60755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 32), 'str', '0')
        # Processing the call keyword arguments (line 66)
        kwargs_60756 = {}
        # Getting the type of 'v' (line 66)
        v_60753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'v', False)
        # Obtaining the member 'startswith' of a type (line 66)
        startswith_60754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), v_60753, 'startswith')
        # Calling startswith(args, kwargs) (line 66)
        startswith_call_result_60757 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), startswith_60754, *[str_60755], **kwargs_60756)
        
        
        # Call to startswith(...): (line 66)
        # Processing the call arguments (line 66)
        str_60760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 53), 'str', '2')
        # Processing the call keyword arguments (line 66)
        kwargs_60761 = {}
        # Getting the type of 'v' (line 66)
        v_60758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 40), 'v', False)
        # Obtaining the member 'startswith' of a type (line 66)
        startswith_60759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 40), v_60758, 'startswith')
        # Calling startswith(args, kwargs) (line 66)
        startswith_call_result_60762 = invoke(stypy.reporting.localization.Localization(__file__, 66, 40), startswith_60759, *[str_60760], **kwargs_60761)
        
        # Applying the binary operator 'or' (line 66)
        result_or_keyword_60763 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), 'or', startswith_call_result_60757, startswith_call_result_60762)
        
        # Call to startswith(...): (line 66)
        # Processing the call arguments (line 66)
        str_60766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 74), 'str', '3')
        # Processing the call keyword arguments (line 66)
        kwargs_60767 = {}
        # Getting the type of 'v' (line 66)
        v_60764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'v', False)
        # Obtaining the member 'startswith' of a type (line 66)
        startswith_60765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 61), v_60764, 'startswith')
        # Calling startswith(args, kwargs) (line 66)
        startswith_call_result_60768 = invoke(stypy.reporting.localization.Localization(__file__, 66, 61), startswith_60765, *[str_60766], **kwargs_60767)
        
        # Applying the binary operator 'or' (line 66)
        result_or_keyword_60769 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), 'or', result_or_keyword_60763, startswith_call_result_60768)
        
        # Testing the type of an if condition (line 66)
        if_condition_60770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 16), result_or_keyword_60769)
        # Assigning a type to the variable 'if_condition_60770' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'if_condition_60770', if_condition_60770)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_60771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        str_60772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'str', 'g77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), tuple_60771, str_60772)
        # Adding element type (line 68)
        # Getting the type of 'v' (line 68)
        v_60773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), tuple_60771, v_60773)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'stypy_return_type', tuple_60771)
        # SSA branch for the else part of an if statement (line 66)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_60774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        str_60775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'str', 'gfortran')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), tuple_60774, str_60775)
        # Adding element type (line 72)
        # Getting the type of 'v' (line 72)
        v_60776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), tuple_60774, v_60776)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'stypy_return_type', tuple_60774)
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 75):
        
        # Assigning a Str to a Name (line 75):
        str_60777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'str', 'A valid Fortran version was not found in this string:\n')
        # Assigning a type to the variable 'err' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'err', str_60777)
        
        # Call to ValueError(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'err' (line 76)
        err_60779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'err', False)
        # Getting the type of 'version_string' (line 76)
        version_string_60780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'version_string', False)
        # Applying the binary operator '+' (line 76)
        result_add_60781 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 25), '+', err_60779, version_string_60780)
        
        # Processing the call keyword arguments (line 76)
        kwargs_60782 = {}
        # Getting the type of 'ValueError' (line 76)
        ValueError_60778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 76)
        ValueError_call_result_60783 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), ValueError_60778, *[result_add_60781], **kwargs_60782)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 76, 8), ValueError_call_result_60783, 'raise parameter', BaseException)
        
        # ################# End of 'gnu_version_match(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'gnu_version_match' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_60784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'gnu_version_match'
        return stypy_return_type_60784


    @norecursion
    def version_match(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'version_match'
        module_type_store = module_type_store.open_function_context('version_match', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.version_match')
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_param_names_list', ['version_string'])
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.version_match.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.version_match', ['version_string'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'version_match', localization, ['version_string'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'version_match(...)' code ##################

        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to gnu_version_match(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'version_string' (line 79)
        version_string_60787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 35), 'version_string', False)
        # Processing the call keyword arguments (line 79)
        kwargs_60788 = {}
        # Getting the type of 'self' (line 79)
        self_60785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self', False)
        # Obtaining the member 'gnu_version_match' of a type (line 79)
        gnu_version_match_60786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_60785, 'gnu_version_match')
        # Calling gnu_version_match(args, kwargs) (line 79)
        gnu_version_match_call_result_60789 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), gnu_version_match_60786, *[version_string_60787], **kwargs_60788)
        
        # Assigning a type to the variable 'v' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'v', gnu_version_match_call_result_60789)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'v' (line 80)
        v_60790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'v')
        # Applying the 'not' unary operator (line 80)
        result_not__60791 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), 'not', v_60790)
        
        
        
        # Obtaining the type of the subscript
        int_60792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
        # Getting the type of 'v' (line 80)
        v_60793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'v')
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___60794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 20), v_60793, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_60795 = invoke(stypy.reporting.localization.Localization(__file__, 80, 20), getitem___60794, int_60792)
        
        str_60796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'str', 'g77')
        # Applying the binary operator '!=' (line 80)
        result_ne_60797 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 20), '!=', subscript_call_result_60795, str_60796)
        
        # Applying the binary operator 'or' (line 80)
        result_or_keyword_60798 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), 'or', result_not__60791, result_ne_60797)
        
        # Testing the type of an if condition (line 80)
        if_condition_60799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_or_keyword_60798)
        # Assigning a type to the variable 'if_condition_60799' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_60799', if_condition_60799)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 81)
        None_60800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', None_60800)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_60801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 17), 'int')
        # Getting the type of 'v' (line 82)
        v_60802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'v')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___60803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), v_60802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_60804 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), getitem___60803, int_60801)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', subscript_call_result_60804)
        
        # ################# End of 'version_match(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'version_match' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_60805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'version_match'
        return stypy_return_type_60805

    
    # Assigning a List to a Name (line 84):
    
    # Assigning a Dict to a Name (line 85):
    
    # Assigning a Str to a Name (line 108):
    
    # Assigning a Str to a Name (line 109):

    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_flags_linker_so')
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_linker_so', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_linker_so(...)' code ##################

        
        # Assigning a Subscript to a Name (line 112):
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_60806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 29), 'int')
        slice_60807 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 14), int_60806, None, None)
        # Getting the type of 'self' (line 112)
        self_60808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'self')
        # Obtaining the member 'linker_so' of a type (line 112)
        linker_so_60809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 14), self_60808, 'linker_so')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___60810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 14), linker_so_60809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_60811 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), getitem___60810, slice_60807)
        
        # Assigning a type to the variable 'opt' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'opt', subscript_call_result_60811)
        
        
        # Getting the type of 'sys' (line 113)
        sys_60812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 113)
        platform_60813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), sys_60812, 'platform')
        str_60814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 113)
        result_eq_60815 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), '==', platform_60813, str_60814)
        
        # Testing the type of an if condition (line 113)
        if_condition_60816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_eq_60815)
        # Assigning a type to the variable 'if_condition_60816' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_60816', if_condition_60816)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to get(...): (line 114)
        # Processing the call arguments (line 114)
        str_60820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'str', 'MACOSX_DEPLOYMENT_TARGET')
        # Getting the type of 'None' (line 114)
        None_60821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 64), 'None', False)
        # Processing the call keyword arguments (line 114)
        kwargs_60822 = {}
        # Getting the type of 'os' (line 114)
        os_60817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'os', False)
        # Obtaining the member 'environ' of a type (line 114)
        environ_60818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), os_60817, 'environ')
        # Obtaining the member 'get' of a type (line 114)
        get_60819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), environ_60818, 'get')
        # Calling get(args, kwargs) (line 114)
        get_call_result_60823 = invoke(stypy.reporting.localization.Localization(__file__, 114, 21), get_60819, *[str_60820, None_60821], **kwargs_60822)
        
        # Assigning a type to the variable 'target' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'target', get_call_result_60823)
        
        
        # Getting the type of 'target' (line 120)
        target_60824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'target')
        # Applying the 'not' unary operator (line 120)
        result_not__60825 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), 'not', target_60824)
        
        # Testing the type of an if condition (line 120)
        if_condition_60826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 12), result_not__60825)
        # Assigning a type to the variable 'if_condition_60826' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'if_condition_60826', if_condition_60826)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 127, 16))
        
        # 'import distutils.sysconfig' statement (line 127)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        import_60827 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 127, 16), 'distutils.sysconfig')

        if (type(import_60827) is not StypyTypeError):

            if (import_60827 != 'pyd_module'):
                __import__(import_60827)
                sys_modules_60828 = sys.modules[import_60827]
                import_module(stypy.reporting.localization.Localization(__file__, 127, 16), 'sc', sys_modules_60828.module_type_store, module_type_store)
            else:
                import distutils.sysconfig as sc

                import_module(stypy.reporting.localization.Localization(__file__, 127, 16), 'sc', distutils.sysconfig, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.sysconfig' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'distutils.sysconfig', import_60827)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        
        
        # Assigning a Dict to a Name (line 128):
        
        # Assigning a Dict to a Name (line 128):
        
        # Obtaining an instance of the builtin type 'dict' (line 128)
        dict_60829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 128)
        
        # Assigning a type to the variable 'g' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'g', dict_60829)
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to get_makefile_filename(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_60832 = {}
        # Getting the type of 'sc' (line 129)
        sc_60830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'sc', False)
        # Obtaining the member 'get_makefile_filename' of a type (line 129)
        get_makefile_filename_60831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 27), sc_60830, 'get_makefile_filename')
        # Calling get_makefile_filename(args, kwargs) (line 129)
        get_makefile_filename_call_result_60833 = invoke(stypy.reporting.localization.Localization(__file__, 129, 27), get_makefile_filename_60831, *[], **kwargs_60832)
        
        # Assigning a type to the variable 'filename' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'filename', get_makefile_filename_call_result_60833)
        
        # Call to parse_makefile(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'filename' (line 130)
        filename_60836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 'filename', False)
        # Getting the type of 'g' (line 130)
        g_60837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 44), 'g', False)
        # Processing the call keyword arguments (line 130)
        kwargs_60838 = {}
        # Getting the type of 'sc' (line 130)
        sc_60834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'sc', False)
        # Obtaining the member 'parse_makefile' of a type (line 130)
        parse_makefile_60835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), sc_60834, 'parse_makefile')
        # Calling parse_makefile(args, kwargs) (line 130)
        parse_makefile_call_result_60839 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), parse_makefile_60835, *[filename_60836, g_60837], **kwargs_60838)
        
        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to get(...): (line 131)
        # Processing the call arguments (line 131)
        str_60842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 31), 'str', 'MACOSX_DEPLOYMENT_TARGET')
        str_60843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 59), 'str', '10.3')
        # Processing the call keyword arguments (line 131)
        kwargs_60844 = {}
        # Getting the type of 'g' (line 131)
        g_60840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'g', False)
        # Obtaining the member 'get' of a type (line 131)
        get_60841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), g_60840, 'get')
        # Calling get(args, kwargs) (line 131)
        get_call_result_60845 = invoke(stypy.reporting.localization.Localization(__file__, 131, 25), get_60841, *[str_60842, str_60843], **kwargs_60844)
        
        # Assigning a type to the variable 'target' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'target', get_call_result_60845)
        
        # Assigning a Name to a Subscript (line 132):
        
        # Assigning a Name to a Subscript (line 132):
        # Getting the type of 'target' (line 132)
        target_60846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 57), 'target')
        # Getting the type of 'os' (line 132)
        os_60847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'os')
        # Obtaining the member 'environ' of a type (line 132)
        environ_60848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), os_60847, 'environ')
        str_60849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 27), 'str', 'MACOSX_DEPLOYMENT_TARGET')
        # Storing an element on a container (line 132)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), environ_60848, (str_60849, target_60846))
        
        
        # Getting the type of 'target' (line 133)
        target_60850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'target')
        str_60851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 29), 'str', '10.3')
        # Applying the binary operator '==' (line 133)
        result_eq_60852 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '==', target_60850, str_60851)
        
        # Testing the type of an if condition (line 133)
        if_condition_60853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 16), result_eq_60852)
        # Assigning a type to the variable 'if_condition_60853' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'if_condition_60853', if_condition_60853)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 134):
        
        # Assigning a Str to a Name (line 134):
        str_60854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'str', 'Env. variable MACOSX_DEPLOYMENT_TARGET set to 10.3')
        # Assigning a type to the variable 's' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 's', str_60854)
        
        # Call to warn(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 's' (line 135)
        s_60857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 's', False)
        # Processing the call keyword arguments (line 135)
        kwargs_60858 = {}
        # Getting the type of 'warnings' (line 135)
        warnings_60855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 135)
        warn_60856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), warnings_60855, 'warn')
        # Calling warn(args, kwargs) (line 135)
        warn_call_result_60859 = invoke(stypy.reporting.localization.Localization(__file__, 135, 20), warn_60856, *[s_60857], **kwargs_60858)
        
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_60862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        str_60863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 24), 'str', '-undefined')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), list_60862, str_60863)
        # Adding element type (line 137)
        str_60864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 38), 'str', 'dynamic_lookup')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), list_60862, str_60864)
        # Adding element type (line 137)
        str_60865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 56), 'str', '-bundle')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), list_60862, str_60865)
        
        # Processing the call keyword arguments (line 137)
        kwargs_60866 = {}
        # Getting the type of 'opt' (line 137)
        opt_60860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'opt', False)
        # Obtaining the member 'extend' of a type (line 137)
        extend_60861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), opt_60860, 'extend')
        # Calling extend(args, kwargs) (line 137)
        extend_call_result_60867 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), extend_60861, *[list_60862], **kwargs_60866)
        
        # SSA branch for the else part of an if statement (line 113)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 139)
        # Processing the call arguments (line 139)
        str_60870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'str', '-shared')
        # Processing the call keyword arguments (line 139)
        kwargs_60871 = {}
        # Getting the type of 'opt' (line 139)
        opt_60868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 139)
        append_60869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), opt_60868, 'append')
        # Calling append(args, kwargs) (line 139)
        append_call_result_60872 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), append_60869, *[str_60870], **kwargs_60871)
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to startswith(...): (line 140)
        # Processing the call arguments (line 140)
        str_60876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 35), 'str', 'sunos')
        # Processing the call keyword arguments (line 140)
        kwargs_60877 = {}
        # Getting the type of 'sys' (line 140)
        sys_60873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'sys', False)
        # Obtaining the member 'platform' of a type (line 140)
        platform_60874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), sys_60873, 'platform')
        # Obtaining the member 'startswith' of a type (line 140)
        startswith_60875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), platform_60874, 'startswith')
        # Calling startswith(args, kwargs) (line 140)
        startswith_call_result_60878 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), startswith_60875, *[str_60876], **kwargs_60877)
        
        # Testing the type of an if condition (line 140)
        if_condition_60879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), startswith_call_result_60878)
        # Assigning a type to the variable 'if_condition_60879' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_60879', if_condition_60879)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 147)
        # Processing the call arguments (line 147)
        str_60882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'str', '-mimpure-text')
        # Processing the call keyword arguments (line 147)
        kwargs_60883 = {}
        # Getting the type of 'opt' (line 147)
        opt_60880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 147)
        append_60881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), opt_60880, 'append')
        # Calling append(args, kwargs) (line 147)
        append_call_result_60884 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), append_60881, *[str_60882], **kwargs_60883)
        
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 148)
        opt_60885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', opt_60885)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_60886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60886)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_60886


    @norecursion
    def get_libgcc_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libgcc_dir'
        module_type_store = module_type_store.open_function_context('get_libgcc_dir', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_libgcc_dir')
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_libgcc_dir.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_libgcc_dir', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libgcc_dir', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libgcc_dir(...)' code ##################

        
        # Assigning a Call to a Tuple (line 151):
        
        # Assigning a Call to a Name:
        
        # Call to exec_command(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'self' (line 151)
        self_60888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 38), 'self', False)
        # Obtaining the member 'compiler_f77' of a type (line 151)
        compiler_f77_60889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 38), self_60888, 'compiler_f77')
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_60890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        str_60891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 39), 'str', '-print-libgcc-file-name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 38), list_60890, str_60891)
        
        # Applying the binary operator '+' (line 151)
        result_add_60892 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 38), '+', compiler_f77_60889, list_60890)
        
        # Processing the call keyword arguments (line 151)
        int_60893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 46), 'int')
        keyword_60894 = int_60893
        kwargs_60895 = {'use_tee': keyword_60894}
        # Getting the type of 'exec_command' (line 151)
        exec_command_60887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'exec_command', False)
        # Calling exec_command(args, kwargs) (line 151)
        exec_command_call_result_60896 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), exec_command_60887, *[result_add_60892], **kwargs_60895)
        
        # Assigning a type to the variable 'call_assignment_60619' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60619', exec_command_call_result_60896)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_60899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60900 = {}
        # Getting the type of 'call_assignment_60619' (line 151)
        call_assignment_60619_60897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60619', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___60898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), call_assignment_60619_60897, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_60901 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60898, *[int_60899], **kwargs_60900)
        
        # Assigning a type to the variable 'call_assignment_60620' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60620', getitem___call_result_60901)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_60620' (line 151)
        call_assignment_60620_60902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60620')
        # Assigning a type to the variable 'status' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'status', call_assignment_60620_60902)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_60905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60906 = {}
        # Getting the type of 'call_assignment_60619' (line 151)
        call_assignment_60619_60903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60619', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___60904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), call_assignment_60619_60903, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_60907 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60904, *[int_60905], **kwargs_60906)
        
        # Assigning a type to the variable 'call_assignment_60621' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60621', getitem___call_result_60907)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_60621' (line 151)
        call_assignment_60621_60908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'call_assignment_60621')
        # Assigning a type to the variable 'output' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'output', call_assignment_60621_60908)
        
        
        # Getting the type of 'status' (line 154)
        status_60909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'status')
        # Applying the 'not' unary operator (line 154)
        result_not__60910 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), 'not', status_60909)
        
        # Testing the type of an if condition (line 154)
        if_condition_60911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_not__60910)
        # Assigning a type to the variable 'if_condition_60911' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_60911', if_condition_60911)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dirname(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'output' (line 155)
        output_60915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'output', False)
        # Processing the call keyword arguments (line 155)
        kwargs_60916 = {}
        # Getting the type of 'os' (line 155)
        os_60912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 155)
        path_60913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), os_60912, 'path')
        # Obtaining the member 'dirname' of a type (line 155)
        dirname_60914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), path_60913, 'dirname')
        # Calling dirname(args, kwargs) (line 155)
        dirname_call_result_60917 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), dirname_60914, *[output_60915], **kwargs_60916)
        
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'stypy_return_type', dirname_call_result_60917)
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 156)
        None_60918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', None_60918)
        
        # ################# End of 'get_libgcc_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libgcc_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_60919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60919)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libgcc_dir'
        return stypy_return_type_60919


    @norecursion
    def get_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_dirs'
        module_type_store = module_type_store.open_function_context('get_library_dirs', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_library_dirs')
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_library_dirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_library_dirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_library_dirs(...)' code ##################

        
        # Assigning a List to a Name (line 159):
        
        # Assigning a List to a Name (line 159):
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_60920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        
        # Assigning a type to the variable 'opt' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'opt', list_60920)
        
        
        
        # Obtaining the type of the subscript
        int_60921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'int')
        slice_60922 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 11), None, int_60921, None)
        # Getting the type of 'sys' (line 160)
        sys_60923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 160)
        platform_60924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), sys_60923, 'platform')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___60925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), platform_60924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_60926 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), getitem___60925, slice_60922)
        
        str_60927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 31), 'str', 'linux')
        # Applying the binary operator '!=' (line 160)
        result_ne_60928 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '!=', subscript_call_result_60926, str_60927)
        
        # Testing the type of an if condition (line 160)
        if_condition_60929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_ne_60928)
        # Assigning a type to the variable 'if_condition_60929' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_60929', if_condition_60929)
        # SSA begins for if statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to get_libgcc_dir(...): (line 161)
        # Processing the call keyword arguments (line 161)
        kwargs_60932 = {}
        # Getting the type of 'self' (line 161)
        self_60930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'self', False)
        # Obtaining the member 'get_libgcc_dir' of a type (line 161)
        get_libgcc_dir_60931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), self_60930, 'get_libgcc_dir')
        # Calling get_libgcc_dir(args, kwargs) (line 161)
        get_libgcc_dir_call_result_60933 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), get_libgcc_dir_60931, *[], **kwargs_60932)
        
        # Assigning a type to the variable 'd' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'd', get_libgcc_dir_call_result_60933)
        
        # Getting the type of 'd' (line 162)
        d_60934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'd')
        # Testing the type of an if condition (line 162)
        if_condition_60935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 12), d_60934)
        # Assigning a type to the variable 'if_condition_60935' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'if_condition_60935', if_condition_60935)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sys' (line 164)
        sys_60936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'sys')
        # Obtaining the member 'platform' of a type (line 164)
        platform_60937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), sys_60936, 'platform')
        str_60938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'str', 'win32')
        # Applying the binary operator '==' (line 164)
        result_eq_60939 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 19), '==', platform_60937, str_60938)
        
        
        
        # Call to startswith(...): (line 164)
        # Processing the call arguments (line 164)
        str_60942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 64), 'str', '/usr/lib')
        # Processing the call keyword arguments (line 164)
        kwargs_60943 = {}
        # Getting the type of 'd' (line 164)
        d_60940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 51), 'd', False)
        # Obtaining the member 'startswith' of a type (line 164)
        startswith_60941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 51), d_60940, 'startswith')
        # Calling startswith(args, kwargs) (line 164)
        startswith_call_result_60944 = invoke(stypy.reporting.localization.Localization(__file__, 164, 51), startswith_60941, *[str_60942], **kwargs_60943)
        
        # Applying the 'not' unary operator (line 164)
        result_not__60945 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 47), 'not', startswith_call_result_60944)
        
        # Applying the binary operator 'and' (line 164)
        result_and_keyword_60946 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 19), 'and', result_eq_60939, result_not__60945)
        
        # Testing the type of an if condition (line 164)
        if_condition_60947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 16), result_and_keyword_60946)
        # Assigning a type to the variable 'if_condition_60947' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'if_condition_60947', if_condition_60947)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 165):
        
        # Assigning a Call to a Name (line 165):
        
        # Call to normpath(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'd' (line 165)
        d_60951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 41), 'd', False)
        # Processing the call keyword arguments (line 165)
        kwargs_60952 = {}
        # Getting the type of 'os' (line 165)
        os_60948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 165)
        path_60949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 24), os_60948, 'path')
        # Obtaining the member 'normpath' of a type (line 165)
        normpath_60950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 24), path_60949, 'normpath')
        # Calling normpath(args, kwargs) (line 165)
        normpath_call_result_60953 = invoke(stypy.reporting.localization.Localization(__file__, 165, 24), normpath_60950, *[d_60951], **kwargs_60952)
        
        # Assigning a type to the variable 'd' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'd', normpath_call_result_60953)
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to join(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'd' (line 166)
        d_60957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 40), 'd', False)
        str_60958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 43), 'str', 'lib%s.a')
        # Getting the type of 'self' (line 166)
        self_60959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 55), 'self', False)
        # Obtaining the member 'g2c' of a type (line 166)
        g2c_60960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 55), self_60959, 'g2c')
        # Applying the binary operator '%' (line 166)
        result_mod_60961 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 43), '%', str_60958, g2c_60960)
        
        # Processing the call keyword arguments (line 166)
        kwargs_60962 = {}
        # Getting the type of 'os' (line 166)
        os_60954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 166)
        path_60955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 27), os_60954, 'path')
        # Obtaining the member 'join' of a type (line 166)
        join_60956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 27), path_60955, 'join')
        # Calling join(args, kwargs) (line 166)
        join_call_result_60963 = invoke(stypy.reporting.localization.Localization(__file__, 166, 27), join_60956, *[d_60957, result_mod_60961], **kwargs_60962)
        
        # Assigning a type to the variable 'path' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'path', join_call_result_60963)
        
        
        
        # Call to exists(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'path' (line 167)
        path_60967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 42), 'path', False)
        # Processing the call keyword arguments (line 167)
        kwargs_60968 = {}
        # Getting the type of 'os' (line 167)
        os_60964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 167)
        path_60965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), os_60964, 'path')
        # Obtaining the member 'exists' of a type (line 167)
        exists_60966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), path_60965, 'exists')
        # Calling exists(args, kwargs) (line 167)
        exists_call_result_60969 = invoke(stypy.reporting.localization.Localization(__file__, 167, 27), exists_60966, *[path_60967], **kwargs_60968)
        
        # Applying the 'not' unary operator (line 167)
        result_not__60970 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 23), 'not', exists_call_result_60969)
        
        # Testing the type of an if condition (line 167)
        if_condition_60971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 20), result_not__60970)
        # Assigning a type to the variable 'if_condition_60971' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'if_condition_60971', if_condition_60971)
        # SSA begins for if statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to join(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'd' (line 168)
        d_60975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'd', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_60976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        # Getting the type of 'os' (line 168)
        os_60977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 50), 'os', False)
        # Obtaining the member 'pardir' of a type (line 168)
        pardir_60978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 50), os_60977, 'pardir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 50), tuple_60976, pardir_60978)
        
        int_60979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 62), 'int')
        # Applying the binary operator '*' (line 168)
        result_mul_60980 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 49), '*', tuple_60976, int_60979)
        
        # Processing the call keyword arguments (line 168)
        kwargs_60981 = {}
        # Getting the type of 'os' (line 168)
        os_60972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 168)
        path_60973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 31), os_60972, 'path')
        # Obtaining the member 'join' of a type (line 168)
        join_60974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 31), path_60973, 'join')
        # Calling join(args, kwargs) (line 168)
        join_call_result_60982 = invoke(stypy.reporting.localization.Localization(__file__, 168, 31), join_60974, *[d_60975, result_mul_60980], **kwargs_60981)
        
        # Assigning a type to the variable 'root' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'root', join_call_result_60982)
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to abspath(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Call to join(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'root' (line 169)
        root_60989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 58), 'root', False)
        str_60990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 64), 'str', 'lib')
        # Processing the call keyword arguments (line 169)
        kwargs_60991 = {}
        # Getting the type of 'os' (line 169)
        os_60986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 45), 'os', False)
        # Obtaining the member 'path' of a type (line 169)
        path_60987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 45), os_60986, 'path')
        # Obtaining the member 'join' of a type (line 169)
        join_60988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 45), path_60987, 'join')
        # Calling join(args, kwargs) (line 169)
        join_call_result_60992 = invoke(stypy.reporting.localization.Localization(__file__, 169, 45), join_60988, *[root_60989, str_60990], **kwargs_60991)
        
        # Processing the call keyword arguments (line 169)
        kwargs_60993 = {}
        # Getting the type of 'os' (line 169)
        os_60983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 169)
        path_60984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 29), os_60983, 'path')
        # Obtaining the member 'abspath' of a type (line 169)
        abspath_60985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 29), path_60984, 'abspath')
        # Calling abspath(args, kwargs) (line 169)
        abspath_call_result_60994 = invoke(stypy.reporting.localization.Localization(__file__, 169, 29), abspath_60985, *[join_call_result_60992], **kwargs_60993)
        
        # Assigning a type to the variable 'd2' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'd2', abspath_call_result_60994)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to join(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'd2' (line 170)
        d2_60998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 44), 'd2', False)
        str_60999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 48), 'str', 'lib%s.a')
        # Getting the type of 'self' (line 170)
        self_61000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 60), 'self', False)
        # Obtaining the member 'g2c' of a type (line 170)
        g2c_61001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 60), self_61000, 'g2c')
        # Applying the binary operator '%' (line 170)
        result_mod_61002 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 48), '%', str_60999, g2c_61001)
        
        # Processing the call keyword arguments (line 170)
        kwargs_61003 = {}
        # Getting the type of 'os' (line 170)
        os_60995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 170)
        path_60996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 31), os_60995, 'path')
        # Obtaining the member 'join' of a type (line 170)
        join_60997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 31), path_60996, 'join')
        # Calling join(args, kwargs) (line 170)
        join_call_result_61004 = invoke(stypy.reporting.localization.Localization(__file__, 170, 31), join_60997, *[d2_60998, result_mod_61002], **kwargs_61003)
        
        # Assigning a type to the variable 'path' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'path', join_call_result_61004)
        
        
        # Call to exists(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'path' (line 171)
        path_61008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'path', False)
        # Processing the call keyword arguments (line 171)
        kwargs_61009 = {}
        # Getting the type of 'os' (line 171)
        os_61005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 171)
        path_61006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 27), os_61005, 'path')
        # Obtaining the member 'exists' of a type (line 171)
        exists_61007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 27), path_61006, 'exists')
        # Calling exists(args, kwargs) (line 171)
        exists_call_result_61010 = invoke(stypy.reporting.localization.Localization(__file__, 171, 27), exists_61007, *[path_61008], **kwargs_61009)
        
        # Testing the type of an if condition (line 171)
        if_condition_61011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 24), exists_call_result_61010)
        # Assigning a type to the variable 'if_condition_61011' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'if_condition_61011', if_condition_61011)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'd2' (line 172)
        d2_61014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 39), 'd2', False)
        # Processing the call keyword arguments (line 172)
        kwargs_61015 = {}
        # Getting the type of 'opt' (line 172)
        opt_61012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'opt', False)
        # Obtaining the member 'append' of a type (line 172)
        append_61013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 28), opt_61012, 'append')
        # Calling append(args, kwargs) (line 172)
        append_call_result_61016 = invoke(stypy.reporting.localization.Localization(__file__, 172, 28), append_61013, *[d2_61014], **kwargs_61015)
        
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'd' (line 173)
        d_61019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'd', False)
        # Processing the call keyword arguments (line 173)
        kwargs_61020 = {}
        # Getting the type of 'opt' (line 173)
        opt_61017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'opt', False)
        # Obtaining the member 'append' of a type (line 173)
        append_61018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), opt_61017, 'append')
        # Calling append(args, kwargs) (line 173)
        append_call_result_61021 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), append_61018, *[d_61019], **kwargs_61020)
        
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 174)
        opt_61022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', opt_61022)
        
        # ################# End of 'get_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_61023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61023)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_dirs'
        return stypy_return_type_61023


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_libraries')
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libraries(...)' code ##################

        
        # Assigning a List to a Name (line 177):
        
        # Assigning a List to a Name (line 177):
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_61024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        
        # Assigning a type to the variable 'opt' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'opt', list_61024)
        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to get_libgcc_dir(...): (line 178)
        # Processing the call keyword arguments (line 178)
        kwargs_61027 = {}
        # Getting the type of 'self' (line 178)
        self_61025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self', False)
        # Obtaining the member 'get_libgcc_dir' of a type (line 178)
        get_libgcc_dir_61026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_61025, 'get_libgcc_dir')
        # Calling get_libgcc_dir(args, kwargs) (line 178)
        get_libgcc_dir_call_result_61028 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), get_libgcc_dir_61026, *[], **kwargs_61027)
        
        # Assigning a type to the variable 'd' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'd', get_libgcc_dir_call_result_61028)
        
        # Type idiom detected: calculating its left and rigth part (line 179)
        # Getting the type of 'd' (line 179)
        d_61029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'd')
        # Getting the type of 'None' (line 179)
        None_61030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'None')
        
        (may_be_61031, more_types_in_union_61032) = may_not_be_none(d_61029, None_61030)

        if may_be_61031:

            if more_types_in_union_61032:
                # Runtime conditional SSA (line 179)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 180):
            
            # Assigning a BinOp to a Name (line 180):
            # Getting the type of 'self' (line 180)
            self_61033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'self')
            # Obtaining the member 'g2c' of a type (line 180)
            g2c_61034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), self_61033, 'g2c')
            str_61035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 29), 'str', '-pic')
            # Applying the binary operator '+' (line 180)
            result_add_61036 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 18), '+', g2c_61034, str_61035)
            
            # Assigning a type to the variable 'g2c' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'g2c', result_add_61036)
            
            # Assigning a BinOp to a Name (line 181):
            
            # Assigning a BinOp to a Name (line 181):
            # Getting the type of 'self' (line 181)
            self_61037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'self')
            # Obtaining the member 'static_lib_format' of a type (line 181)
            static_lib_format_61038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), self_61037, 'static_lib_format')
            
            # Obtaining an instance of the builtin type 'tuple' (line 181)
            tuple_61039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 181)
            # Adding element type (line 181)
            # Getting the type of 'g2c' (line 181)
            g2c_61040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'g2c')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 42), tuple_61039, g2c_61040)
            # Adding element type (line 181)
            # Getting the type of 'self' (line 181)
            self_61041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 47), 'self')
            # Obtaining the member 'static_lib_extension' of a type (line 181)
            static_lib_extension_61042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 47), self_61041, 'static_lib_extension')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 42), tuple_61039, static_lib_extension_61042)
            
            # Applying the binary operator '%' (line 181)
            result_mod_61043 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 16), '%', static_lib_format_61038, tuple_61039)
            
            # Assigning a type to the variable 'f' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'f', result_mod_61043)
            
            
            
            # Call to isfile(...): (line 182)
            # Processing the call arguments (line 182)
            
            # Call to join(...): (line 182)
            # Processing the call arguments (line 182)
            # Getting the type of 'd' (line 182)
            d_61050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 47), 'd', False)
            # Getting the type of 'f' (line 182)
            f_61051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 50), 'f', False)
            # Processing the call keyword arguments (line 182)
            kwargs_61052 = {}
            # Getting the type of 'os' (line 182)
            os_61047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'os', False)
            # Obtaining the member 'path' of a type (line 182)
            path_61048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), os_61047, 'path')
            # Obtaining the member 'join' of a type (line 182)
            join_61049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), path_61048, 'join')
            # Calling join(args, kwargs) (line 182)
            join_call_result_61053 = invoke(stypy.reporting.localization.Localization(__file__, 182, 34), join_61049, *[d_61050, f_61051], **kwargs_61052)
            
            # Processing the call keyword arguments (line 182)
            kwargs_61054 = {}
            # Getting the type of 'os' (line 182)
            os_61044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'os', False)
            # Obtaining the member 'path' of a type (line 182)
            path_61045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 19), os_61044, 'path')
            # Obtaining the member 'isfile' of a type (line 182)
            isfile_61046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 19), path_61045, 'isfile')
            # Calling isfile(args, kwargs) (line 182)
            isfile_call_result_61055 = invoke(stypy.reporting.localization.Localization(__file__, 182, 19), isfile_61046, *[join_call_result_61053], **kwargs_61054)
            
            # Applying the 'not' unary operator (line 182)
            result_not__61056 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 15), 'not', isfile_call_result_61055)
            
            # Testing the type of an if condition (line 182)
            if_condition_61057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), result_not__61056)
            # Assigning a type to the variable 'if_condition_61057' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_61057', if_condition_61057)
            # SSA begins for if statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 183):
            
            # Assigning a Attribute to a Name (line 183):
            # Getting the type of 'self' (line 183)
            self_61058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'self')
            # Obtaining the member 'g2c' of a type (line 183)
            g2c_61059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 22), self_61058, 'g2c')
            # Assigning a type to the variable 'g2c' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'g2c', g2c_61059)
            # SSA join for if statement (line 182)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_61032:
                # Runtime conditional SSA for else branch (line 179)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_61031) or more_types_in_union_61032):
            
            # Assigning a Attribute to a Name (line 185):
            
            # Assigning a Attribute to a Name (line 185):
            # Getting the type of 'self' (line 185)
            self_61060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'self')
            # Obtaining the member 'g2c' of a type (line 185)
            g2c_61061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 18), self_61060, 'g2c')
            # Assigning a type to the variable 'g2c' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'g2c', g2c_61061)

            if (may_be_61031 and more_types_in_union_61032):
                # SSA join for if statement (line 179)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 187)
        # Getting the type of 'g2c' (line 187)
        g2c_61062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'g2c')
        # Getting the type of 'None' (line 187)
        None_61063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'None')
        
        (may_be_61064, more_types_in_union_61065) = may_not_be_none(g2c_61062, None_61063)

        if may_be_61064:

            if more_types_in_union_61065:
                # Runtime conditional SSA (line 187)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 'g2c' (line 188)
            g2c_61068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'g2c', False)
            # Processing the call keyword arguments (line 188)
            kwargs_61069 = {}
            # Getting the type of 'opt' (line 188)
            opt_61066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'opt', False)
            # Obtaining the member 'append' of a type (line 188)
            append_61067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), opt_61066, 'append')
            # Calling append(args, kwargs) (line 188)
            append_call_result_61070 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), append_61067, *[g2c_61068], **kwargs_61069)
            

            if more_types_in_union_61065:
                # SSA join for if statement (line 187)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 189):
        
        # Assigning a Attribute to a Name (line 189):
        # Getting the type of 'self' (line 189)
        self_61071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'self')
        # Obtaining the member 'c_compiler' of a type (line 189)
        c_compiler_61072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 21), self_61071, 'c_compiler')
        # Assigning a type to the variable 'c_compiler' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'c_compiler', c_compiler_61072)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sys' (line 190)
        sys_61073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 190)
        platform_61074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), sys_61073, 'platform')
        str_61075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 190)
        result_eq_61076 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '==', platform_61074, str_61075)
        
        # Getting the type of 'c_compiler' (line 190)
        c_compiler_61077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 39), 'c_compiler')
        # Applying the binary operator 'and' (line 190)
        result_and_keyword_61078 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), 'and', result_eq_61076, c_compiler_61077)
        
        # Getting the type of 'c_compiler' (line 191)
        c_compiler_61079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'c_compiler')
        # Obtaining the member 'compiler_type' of a type (line 191)
        compiler_type_61080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), c_compiler_61079, 'compiler_type')
        str_61081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 43), 'str', 'msvc')
        # Applying the binary operator '==' (line 191)
        result_eq_61082 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '==', compiler_type_61080, str_61081)
        
        # Applying the binary operator 'and' (line 190)
        result_and_keyword_61083 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), 'and', result_and_keyword_61078, result_eq_61082)
        
        # Testing the type of an if condition (line 190)
        if_condition_61084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_and_keyword_61083)
        # Assigning a type to the variable 'if_condition_61084' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_61084', if_condition_61084)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 194)
        # Processing the call arguments (line 194)
        str_61087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 23), 'str', 'gcc')
        # Processing the call keyword arguments (line 194)
        kwargs_61088 = {}
        # Getting the type of 'opt' (line 194)
        opt_61085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 194)
        append_61086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), opt_61085, 'append')
        # Calling append(args, kwargs) (line 194)
        append_call_result_61089 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), append_61086, *[str_61087], **kwargs_61088)
        
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to msvc_runtime_library(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_61091 = {}
        # Getting the type of 'msvc_runtime_library' (line 195)
        msvc_runtime_library_61090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'msvc_runtime_library', False)
        # Calling msvc_runtime_library(args, kwargs) (line 195)
        msvc_runtime_library_call_result_61092 = invoke(stypy.reporting.localization.Localization(__file__, 195, 26), msvc_runtime_library_61090, *[], **kwargs_61091)
        
        # Assigning a type to the variable 'runtime_lib' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'runtime_lib', msvc_runtime_library_call_result_61092)
        
        # Getting the type of 'runtime_lib' (line 196)
        runtime_lib_61093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'runtime_lib')
        # Testing the type of an if condition (line 196)
        if_condition_61094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 12), runtime_lib_61093)
        # Assigning a type to the variable 'if_condition_61094' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'if_condition_61094', if_condition_61094)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'runtime_lib' (line 197)
        runtime_lib_61097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'runtime_lib', False)
        # Processing the call keyword arguments (line 197)
        kwargs_61098 = {}
        # Getting the type of 'opt' (line 197)
        opt_61095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'opt', False)
        # Obtaining the member 'append' of a type (line 197)
        append_61096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), opt_61095, 'append')
        # Calling append(args, kwargs) (line 197)
        append_call_result_61099 = invoke(stypy.reporting.localization.Localization(__file__, 197, 16), append_61096, *[runtime_lib_61097], **kwargs_61098)
        
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 198)
        sys_61100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 198)
        platform_61101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), sys_61100, 'platform')
        str_61102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 198)
        result_eq_61103 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '==', platform_61101, str_61102)
        
        # Testing the type of an if condition (line 198)
        if_condition_61104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_eq_61103)
        # Assigning a type to the variable 'if_condition_61104' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_61104', if_condition_61104)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 199)
        # Processing the call arguments (line 199)
        str_61107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'str', 'cc_dynamic')
        # Processing the call keyword arguments (line 199)
        kwargs_61108 = {}
        # Getting the type of 'opt' (line 199)
        opt_61105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 199)
        append_61106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), opt_61105, 'append')
        # Calling append(args, kwargs) (line 199)
        append_call_result_61109 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), append_61106, *[str_61107], **kwargs_61108)
        
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 200)
        opt_61110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', opt_61110)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_61111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_61111


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_flags_debug')
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_debug', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_debug(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_61112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        str_61113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 15), list_61112, str_61113)
        
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', list_61112)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_61114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_61114


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_flags_opt')
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to get_version(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_61117 = {}
        # Getting the type of 'self' (line 206)
        self_61115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self', False)
        # Obtaining the member 'get_version' of a type (line 206)
        get_version_61116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), self_61115, 'get_version')
        # Calling get_version(args, kwargs) (line 206)
        get_version_call_result_61118 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), get_version_61116, *[], **kwargs_61117)
        
        # Assigning a type to the variable 'v' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'v', get_version_call_result_61118)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'v' (line 207)
        v_61119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'v')
        
        # Getting the type of 'v' (line 207)
        v_61120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 17), 'v')
        str_61121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 22), 'str', '3.3.3')
        # Applying the binary operator '<=' (line 207)
        result_le_61122 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 17), '<=', v_61120, str_61121)
        
        # Applying the binary operator 'and' (line 207)
        result_and_keyword_61123 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), 'and', v_61119, result_le_61122)
        
        # Testing the type of an if condition (line 207)
        if_condition_61124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_and_keyword_61123)
        # Assigning a type to the variable 'if_condition_61124' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_61124', if_condition_61124)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 210):
        
        # Assigning a List to a Name (line 210):
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_61125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        str_61126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 19), 'str', '-O2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 18), list_61125, str_61126)
        
        # Assigning a type to the variable 'opt' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'opt', list_61125)
        # SSA branch for the else part of an if statement (line 207)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 212):
        
        # Assigning a List to a Name (line 212):
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_61127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        # Adding element type (line 212)
        str_61128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 19), 'str', '-O3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 18), list_61127, str_61128)
        
        # Assigning a type to the variable 'opt' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'opt', list_61127)
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 213)
        # Processing the call arguments (line 213)
        str_61131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 19), 'str', '-funroll-loops')
        # Processing the call keyword arguments (line 213)
        kwargs_61132 = {}
        # Getting the type of 'opt' (line 213)
        opt_61129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'opt', False)
        # Obtaining the member 'append' of a type (line 213)
        append_61130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), opt_61129, 'append')
        # Calling append(args, kwargs) (line 213)
        append_call_result_61133 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), append_61130, *[str_61131], **kwargs_61132)
        
        # Getting the type of 'opt' (line 214)
        opt_61134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', opt_61134)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_61135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_61135


    @norecursion
    def _c_arch_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_c_arch_flags'
        module_type_store = module_type_store.open_function_context('_c_arch_flags', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler._c_arch_flags')
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler._c_arch_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler._c_arch_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_c_arch_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_c_arch_flags(...)' code ##################

        str_61136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'str', ' Return detected arch flags from CFLAGS ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 218, 8))
        
        # 'from distutils import sysconfig' statement (line 218)
        from distutils import sysconfig

        import_from_module(stypy.reporting.localization.Localization(__file__, 218, 8), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])
        
        
        
        # SSA begins for try-except statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 220):
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        str_61137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 49), 'str', 'CFLAGS')
        
        # Call to get_config_vars(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_61140 = {}
        # Getting the type of 'sysconfig' (line 220)
        sysconfig_61138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'sysconfig', False)
        # Obtaining the member 'get_config_vars' of a type (line 220)
        get_config_vars_61139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), sysconfig_61138, 'get_config_vars')
        # Calling get_config_vars(args, kwargs) (line 220)
        get_config_vars_call_result_61141 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), get_config_vars_61139, *[], **kwargs_61140)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___61142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), get_config_vars_call_result_61141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_61143 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), getitem___61142, str_61137)
        
        # Assigning a type to the variable 'cflags' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'cflags', subscript_call_result_61143)
        # SSA branch for the except part of a try statement (line 219)
        # SSA branch for the except 'KeyError' branch of a try statement (line 219)
        module_type_store.open_ssa_branch('except')
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_61144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'stypy_return_type', list_61144)
        # SSA join for try-except statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to compile(...): (line 223)
        # Processing the call arguments (line 223)
        str_61147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 29), 'str', '-arch\\s+(\\w+)')
        # Processing the call keyword arguments (line 223)
        kwargs_61148 = {}
        # Getting the type of 're' (line 223)
        re_61145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 're', False)
        # Obtaining the member 'compile' of a type (line 223)
        compile_61146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 18), re_61145, 'compile')
        # Calling compile(args, kwargs) (line 223)
        compile_call_result_61149 = invoke(stypy.reporting.localization.Localization(__file__, 223, 18), compile_61146, *[str_61147], **kwargs_61148)
        
        # Assigning a type to the variable 'arch_re' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'arch_re', compile_call_result_61149)
        
        # Assigning a List to a Name (line 224):
        
        # Assigning a List to a Name (line 224):
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_61150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        
        # Assigning a type to the variable 'arch_flags' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'arch_flags', list_61150)
        
        
        # Call to findall(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'cflags' (line 225)
        cflags_61153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 36), 'cflags', False)
        # Processing the call keyword arguments (line 225)
        kwargs_61154 = {}
        # Getting the type of 'arch_re' (line 225)
        arch_re_61151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'arch_re', False)
        # Obtaining the member 'findall' of a type (line 225)
        findall_61152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 20), arch_re_61151, 'findall')
        # Calling findall(args, kwargs) (line 225)
        findall_call_result_61155 = invoke(stypy.reporting.localization.Localization(__file__, 225, 20), findall_61152, *[cflags_61153], **kwargs_61154)
        
        # Testing the type of a for loop iterable (line 225)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 225, 8), findall_call_result_61155)
        # Getting the type of the for loop variable (line 225)
        for_loop_var_61156 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 225, 8), findall_call_result_61155)
        # Assigning a type to the variable 'arch' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'arch', for_loop_var_61156)
        # SSA begins for a for statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'arch_flags' (line 226)
        arch_flags_61157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'arch_flags')
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_61158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        str_61159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 27), 'str', '-arch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 26), list_61158, str_61159)
        # Adding element type (line 226)
        # Getting the type of 'arch' (line 226)
        arch_61160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'arch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 26), list_61158, arch_61160)
        
        # Applying the binary operator '+=' (line 226)
        result_iadd_61161 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 12), '+=', arch_flags_61157, list_61158)
        # Assigning a type to the variable 'arch_flags' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'arch_flags', result_iadd_61161)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'arch_flags' (line 227)
        arch_flags_61162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'arch_flags')
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'stypy_return_type', arch_flags_61162)
        
        # ################# End of '_c_arch_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_c_arch_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_61163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_c_arch_flags'
        return stypy_return_type_61163


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.get_flags_arch')
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_61164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', list_61164)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_61165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_61165


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'GnuFCompiler.runtime_library_dir_option')
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GnuFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        str_61166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 15), 'str', '-Wl,-rpath="%s"')
        # Getting the type of 'dir' (line 233)
        dir_61167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 35), 'dir')
        # Applying the binary operator '%' (line 233)
        result_mod_61168 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '%', str_61166, dir_61167)
        
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', result_mod_61168)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_61169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_61169


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GnuFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'GnuFCompiler' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'GnuFCompiler', GnuFCompiler)

# Assigning a Str to a Name (line 31):
str_61170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'str', 'gnu')
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61171, 'compiler_type', str_61170)

# Assigning a Tuple to a Name (line 32):

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_61172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
str_61173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'str', 'g77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), tuple_61172, str_61173)

# Getting the type of 'GnuFCompiler'
GnuFCompiler_61174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61174, 'compiler_aliases', tuple_61172)

# Assigning a Str to a Name (line 33):
str_61175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'str', 'GNU Fortran 77 compiler')
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61176, 'description', str_61175)

# Assigning a List to a Name (line 84):

# Obtaining an instance of the builtin type 'list' (line 84)
list_61177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 84)
# Adding element type (line 84)
str_61178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 28), 'str', 'g77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 27), list_61177, str_61178)
# Adding element type (line 84)
str_61179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 35), 'str', 'f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 27), list_61177, str_61179)

# Getting the type of 'GnuFCompiler'
GnuFCompiler_61180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61180, 'possible_executables', list_61177)

# Assigning a Dict to a Name (line 85):

# Obtaining an instance of the builtin type 'dict' (line 85)
dict_61181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 85)
# Adding element type (key, value) (line 85)
str_61182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 86)
list_61183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 86)
# Adding element type (line 86)
# Getting the type of 'None' (line 86)
None_61184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), list_61183, None_61184)
# Adding element type (line 86)
str_61185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 32), 'str', '-dumpversion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), list_61183, str_61185)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61182, list_61183))
# Adding element type (key, value) (line 85)
str_61186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 87)
list_61187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 87)
# Adding element type (line 87)
# Getting the type of 'None' (line 87)
None_61188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 25), list_61187, None_61188)
# Adding element type (line 87)
str_61189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 25), list_61187, str_61189)
# Adding element type (line 87)
str_61190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 38), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 25), list_61187, str_61190)
# Adding element type (line 87)
str_61191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 47), 'str', '-fno-second-underscore')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 25), list_61187, str_61191)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61186, list_61187))
# Adding element type (key, value) (line 85)
str_61192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'str', 'compiler_f90')
# Getting the type of 'None' (line 88)
None_61193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61192, None_61193))
# Adding element type (key, value) (line 85)
str_61194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'str', 'compiler_fix')
# Getting the type of 'None' (line 89)
None_61195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61194, None_61195))
# Adding element type (key, value) (line 85)
str_61196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 90)
list_61197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 90)
# Adding element type (line 90)
# Getting the type of 'None' (line 90)
None_61198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 25), list_61197, None_61198)
# Adding element type (line 90)
str_61199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 25), list_61197, str_61199)
# Adding element type (line 90)
str_61200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 38), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 25), list_61197, str_61200)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61196, list_61197))
# Adding element type (key, value) (line 85)
str_61201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 91)
list_61202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 91)
# Adding element type (line 91)
str_61203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 25), list_61202, str_61203)
# Adding element type (line 91)
str_61204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 25), list_61202, str_61204)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61201, list_61202))
# Adding element type (key, value) (line 85)
str_61205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 92)
list_61206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 92)
# Adding element type (line 92)
str_61207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_61206, str_61207)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61205, list_61206))
# Adding element type (key, value) (line 85)
str_61208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'str', 'linker_exe')

# Obtaining an instance of the builtin type 'list' (line 93)
list_61209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 93)
# Adding element type (line 93)
# Getting the type of 'None' (line 93)
None_61210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), list_61209, None_61210)
# Adding element type (line 93)
str_61211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), list_61209, str_61211)
# Adding element type (line 93)
str_61212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 38), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), list_61209, str_61212)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), dict_61181, (str_61208, list_61209))

# Getting the type of 'GnuFCompiler'
GnuFCompiler_61213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61213, 'executables', dict_61181)

# Assigning a Name to a Name (line 95):
# Getting the type of 'None' (line 95)
None_61214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'None')
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61215, 'module_dir_switch', None_61214)

# Assigning a Name to a Name (line 96):
# Getting the type of 'None' (line 96)
None_61216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'None')
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61217, 'module_include_switch', None_61216)

# Assigning a Name to a Name (line 96):


# Evaluating a boolean operation

# Getting the type of 'os' (line 100)
os_61218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'os')
# Obtaining the member 'name' of a type (line 100)
name_61219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 7), os_61218, 'name')
str_61220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 18), 'str', 'nt')
# Applying the binary operator '!=' (line 100)
result_ne_61221 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), '!=', name_61219, str_61220)


# Getting the type of 'sys' (line 100)
sys_61222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'sys')
# Obtaining the member 'platform' of a type (line 100)
platform_61223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), sys_61222, 'platform')
str_61224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 43), 'str', 'cygwin')
# Applying the binary operator '!=' (line 100)
result_ne_61225 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 27), '!=', platform_61223, str_61224)

# Applying the binary operator 'and' (line 100)
result_and_keyword_61226 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'and', result_ne_61221, result_ne_61225)

# Testing the type of an if condition (line 100)
if_condition_61227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_and_keyword_61226)
# Assigning a type to the variable 'if_condition_61227' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_61227', if_condition_61227)
# SSA begins for if statement (line 100)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a List to a Name (line 101):

# Assigning a List to a Name (line 101):

# Obtaining an instance of the builtin type 'list' (line 101)
list_61228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)
# Adding element type (line 101)
str_61229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'str', '-fPIC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), list_61228, str_61229)

# Assigning a type to the variable 'pic_flags' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'pic_flags', list_61228)
# SSA join for if statement (line 100)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 95):


# Getting the type of 'sys' (line 104)
sys_61230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'sys')
# Obtaining the member 'platform' of a type (line 104)
platform_61231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 7), sys_61230, 'platform')
str_61232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'str', 'win32')
# Applying the binary operator '==' (line 104)
result_eq_61233 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), '==', platform_61231, str_61232)

# Testing the type of an if condition (line 104)
if_condition_61234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), result_eq_61233)
# Assigning a type to the variable 'if_condition_61234' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_61234', if_condition_61234)
# SSA begins for if statement (line 104)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# Obtaining an instance of the builtin type 'list' (line 105)
list_61235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 105)
# Adding element type (line 105)
str_61236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'str', 'version_cmd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_61235, str_61236)
# Adding element type (line 105)
str_61237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 35), 'str', 'compiler_f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_61235, str_61237)
# Adding element type (line 105)
str_61238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 51), 'str', 'linker_so')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_61235, str_61238)
# Adding element type (line 105)
str_61239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 64), 'str', 'linker_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_61235, str_61239)

# Testing the type of a for loop iterable (line 105)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 8), list_61235)
# Getting the type of the for loop variable (line 105)
for_loop_var_61240 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 8), list_61235)
# Assigning a type to the variable 'key' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'key', for_loop_var_61240)
# SSA begins for a for statement (line 105)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to append(...): (line 106)
# Processing the call arguments (line 106)
str_61247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'str', '-mno-cygwin')
# Processing the call keyword arguments (line 106)
kwargs_61248 = {}

# Obtaining the type of the subscript
# Getting the type of 'key' (line 106)
key_61241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'key', False)
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler', False)
# Obtaining the member 'executables' of a type
executables_61243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61242, 'executables')
# Obtaining the member '__getitem__' of a type (line 106)
getitem___61244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), executables_61243, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 106)
subscript_call_result_61245 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), getitem___61244, key_61241)

# Obtaining the member 'append' of a type (line 106)
append_61246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), subscript_call_result_61245, 'append')
# Calling append(args, kwargs) (line 106)
append_call_result_61249 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), append_61246, *[str_61247], **kwargs_61248)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 104)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 108):
str_61250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 10), 'str', 'g2c')
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'g2c' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61251, 'g2c', str_61250)

# Assigning a Str to a Name (line 109):
str_61252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'str', 'gnu95')
# Getting the type of 'GnuFCompiler'
GnuFCompiler_61253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GnuFCompiler')
# Setting the type of the member 'suggested_f90_compiler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GnuFCompiler_61253, 'suggested_f90_compiler', str_61252)
# Declaration of the 'Gnu95FCompiler' class
# Getting the type of 'GnuFCompiler' (line 235)
GnuFCompiler_61254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 21), 'GnuFCompiler')

class Gnu95FCompiler(GnuFCompiler_61254, ):
    
    # Assigning a Str to a Name (line 236):
    
    # Assigning a Tuple to a Name (line 237):
    
    # Assigning a Str to a Name (line 238):

    @norecursion
    def version_match(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'version_match'
        module_type_store = module_type_store.open_function_context('version_match', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.version_match')
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_param_names_list', ['version_string'])
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.version_match.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.version_match', ['version_string'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'version_match', localization, ['version_string'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'version_match(...)' code ##################

        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to gnu_version_match(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'version_string' (line 241)
        version_string_61257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 35), 'version_string', False)
        # Processing the call keyword arguments (line 241)
        kwargs_61258 = {}
        # Getting the type of 'self' (line 241)
        self_61255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
        # Obtaining the member 'gnu_version_match' of a type (line 241)
        gnu_version_match_61256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_61255, 'gnu_version_match')
        # Calling gnu_version_match(args, kwargs) (line 241)
        gnu_version_match_call_result_61259 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), gnu_version_match_61256, *[version_string_61257], **kwargs_61258)
        
        # Assigning a type to the variable 'v' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'v', gnu_version_match_call_result_61259)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'v' (line 242)
        v_61260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'v')
        # Applying the 'not' unary operator (line 242)
        result_not__61261 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), 'not', v_61260)
        
        
        
        # Obtaining the type of the subscript
        int_61262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 22), 'int')
        # Getting the type of 'v' (line 242)
        v_61263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'v')
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___61264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), v_61263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_61265 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), getitem___61264, int_61262)
        
        str_61266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 28), 'str', 'gfortran')
        # Applying the binary operator '!=' (line 242)
        result_ne_61267 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 20), '!=', subscript_call_result_61265, str_61266)
        
        # Applying the binary operator 'or' (line 242)
        result_or_keyword_61268 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), 'or', result_not__61261, result_ne_61267)
        
        # Testing the type of an if condition (line 242)
        if_condition_61269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_or_keyword_61268)
        # Assigning a type to the variable 'if_condition_61269' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_61269', if_condition_61269)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 243)
        None_61270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'stypy_return_type', None_61270)
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 244):
        
        # Assigning a Subscript to a Name (line 244):
        
        # Obtaining the type of the subscript
        int_61271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 14), 'int')
        # Getting the type of 'v' (line 244)
        v_61272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'v')
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___61273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), v_61272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_61274 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), getitem___61273, int_61271)
        
        # Assigning a type to the variable 'v' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'v', subscript_call_result_61274)
        
        
        # Getting the type of 'v' (line 245)
        v_61275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'v')
        str_61276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'str', '4.')
        # Applying the binary operator '>=' (line 245)
        result_ge_61277 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 11), '>=', v_61275, str_61276)
        
        # Testing the type of an if condition (line 245)
        if_condition_61278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 8), result_ge_61277)
        # Assigning a type to the variable 'if_condition_61278' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'if_condition_61278', if_condition_61278)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 245)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sys' (line 251)
        sys_61279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 251)
        platform_61280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), sys_61279, 'platform')
        str_61281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'str', 'win32')
        # Applying the binary operator '==' (line 251)
        result_eq_61282 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 15), '==', platform_61280, str_61281)
        
        # Testing the type of an if condition (line 251)
        if_condition_61283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 12), result_eq_61282)
        # Assigning a type to the variable 'if_condition_61283' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'if_condition_61283', if_condition_61283)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_61284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        str_61285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 28), 'str', 'version_cmd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), list_61284, str_61285)
        # Adding element type (line 252)
        str_61286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 43), 'str', 'compiler_f77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), list_61284, str_61286)
        # Adding element type (line 252)
        str_61287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 59), 'str', 'compiler_f90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), list_61284, str_61287)
        # Adding element type (line 252)
        str_61288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 28), 'str', 'compiler_fix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), list_61284, str_61288)
        # Adding element type (line 252)
        str_61289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 44), 'str', 'linker_so')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), list_61284, str_61289)
        # Adding element type (line 252)
        str_61290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 57), 'str', 'linker_exe')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), list_61284, str_61290)
        
        # Testing the type of a for loop iterable (line 252)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 252, 16), list_61284)
        # Getting the type of the for loop variable (line 252)
        for_loop_var_61291 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 252, 16), list_61284)
        # Assigning a type to the variable 'key' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'key', for_loop_var_61291)
        # SSA begins for a for statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 254)
        # Processing the call arguments (line 254)
        str_61298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 49), 'str', '-mno-cygwin')
        # Processing the call keyword arguments (line 254)
        kwargs_61299 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 254)
        key_61292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 37), 'key', False)
        # Getting the type of 'self' (line 254)
        self_61293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'self', False)
        # Obtaining the member 'executables' of a type (line 254)
        executables_61294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), self_61293, 'executables')
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___61295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), executables_61294, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_61296 = invoke(stypy.reporting.localization.Localization(__file__, 254, 20), getitem___61295, key_61292)
        
        # Obtaining the member 'append' of a type (line 254)
        append_61297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), subscript_call_result_61296, 'append')
        # Calling append(args, kwargs) (line 254)
        append_call_result_61300 = invoke(stypy.reporting.localization.Localization(__file__, 254, 20), append_61297, *[str_61298], **kwargs_61299)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'v' (line 255)
        v_61301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type', v_61301)
        
        # ################# End of 'version_match(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'version_match' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_61302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'version_match'
        return stypy_return_type_61302

    
    # Assigning a List to a Name (line 257):
    
    # Assigning a Dict to a Name (line 258):
    
    # Assigning a Str to a Name (line 272):
    
    # Assigning a Str to a Name (line 273):
    
    # Assigning a Str to a Name (line 275):

    @norecursion
    def _universal_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_universal_flags'
        module_type_store = module_type_store.open_function_context('_universal_flags', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler._universal_flags')
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_param_names_list', ['cmd'])
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler._universal_flags.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler._universal_flags', ['cmd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_universal_flags', localization, ['cmd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_universal_flags(...)' code ##################

        str_61303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'str', 'Return a list of -arch flags for every supported architecture.')
        
        
        
        # Getting the type of 'sys' (line 279)
        sys_61304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 279)
        platform_61305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 15), sys_61304, 'platform')
        str_61306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 31), 'str', 'darwin')
        # Applying the binary operator '==' (line 279)
        result_eq_61307 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 15), '==', platform_61305, str_61306)
        
        # Applying the 'not' unary operator (line 279)
        result_not__61308 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'not', result_eq_61307)
        
        # Testing the type of an if condition (line 279)
        if_condition_61309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_not__61308)
        # Assigning a type to the variable 'if_condition_61309' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_61309', if_condition_61309)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_61310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'stypy_return_type', list_61310)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 281):
        
        # Assigning a List to a Name (line 281):
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_61311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        
        # Assigning a type to the variable 'arch_flags' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'arch_flags', list_61311)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to _c_arch_flags(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_61314 = {}
        # Getting the type of 'self' (line 283)
        self_61312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'self', False)
        # Obtaining the member '_c_arch_flags' of a type (line 283)
        _c_arch_flags_61313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 18), self_61312, '_c_arch_flags')
        # Calling _c_arch_flags(args, kwargs) (line 283)
        _c_arch_flags_call_result_61315 = invoke(stypy.reporting.localization.Localization(__file__, 283, 18), _c_arch_flags_61313, *[], **kwargs_61314)
        
        # Assigning a type to the variable 'c_archs' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'c_archs', _c_arch_flags_call_result_61315)
        
        
        str_61316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 11), 'str', 'i386')
        # Getting the type of 'c_archs' (line 284)
        c_archs_61317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'c_archs')
        # Applying the binary operator 'in' (line 284)
        result_contains_61318 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'in', str_61316, c_archs_61317)
        
        # Testing the type of an if condition (line 284)
        if_condition_61319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_contains_61318)
        # Assigning a type to the variable 'if_condition_61319' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_61319', if_condition_61319)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Subscript (line 285):
        
        # Assigning a Str to a Subscript (line 285):
        str_61320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 45), 'str', 'i686')
        # Getting the type of 'c_archs' (line 285)
        c_archs_61321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'c_archs')
        
        # Call to index(...): (line 285)
        # Processing the call arguments (line 285)
        str_61324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 34), 'str', 'i386')
        # Processing the call keyword arguments (line 285)
        kwargs_61325 = {}
        # Getting the type of 'c_archs' (line 285)
        c_archs_61322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'c_archs', False)
        # Obtaining the member 'index' of a type (line 285)
        index_61323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 20), c_archs_61322, 'index')
        # Calling index(args, kwargs) (line 285)
        index_call_result_61326 = invoke(stypy.reporting.localization.Localization(__file__, 285, 20), index_61323, *[str_61324], **kwargs_61325)
        
        # Storing an element on a container (line 285)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 12), c_archs_61321, (index_call_result_61326, str_61320))
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_61327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        str_61328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'str', 'ppc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 20), list_61327, str_61328)
        # Adding element type (line 288)
        str_61329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'str', 'i686')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 20), list_61327, str_61329)
        # Adding element type (line 288)
        str_61330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 36), 'str', 'x86_64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 20), list_61327, str_61330)
        # Adding element type (line 288)
        str_61331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 46), 'str', 'ppc64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 20), list_61327, str_61331)
        
        # Testing the type of a for loop iterable (line 288)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 288, 8), list_61327)
        # Getting the type of the for loop variable (line 288)
        for_loop_var_61332 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 288, 8), list_61327)
        # Assigning a type to the variable 'arch' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'arch', for_loop_var_61332)
        # SSA begins for a for statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to _can_target(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'cmd' (line 289)
        cmd_61334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'cmd', False)
        # Getting the type of 'arch' (line 289)
        arch_61335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'arch', False)
        # Processing the call keyword arguments (line 289)
        kwargs_61336 = {}
        # Getting the type of '_can_target' (line 289)
        _can_target_61333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), '_can_target', False)
        # Calling _can_target(args, kwargs) (line 289)
        _can_target_call_result_61337 = invoke(stypy.reporting.localization.Localization(__file__, 289, 15), _can_target_61333, *[cmd_61334, arch_61335], **kwargs_61336)
        
        
        # Getting the type of 'arch' (line 289)
        arch_61338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 42), 'arch')
        # Getting the type of 'c_archs' (line 289)
        c_archs_61339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 50), 'c_archs')
        # Applying the binary operator 'in' (line 289)
        result_contains_61340 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 42), 'in', arch_61338, c_archs_61339)
        
        # Applying the binary operator 'and' (line 289)
        result_and_keyword_61341 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), 'and', _can_target_call_result_61337, result_contains_61340)
        
        # Testing the type of an if condition (line 289)
        if_condition_61342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 12), result_and_keyword_61341)
        # Assigning a type to the variable 'if_condition_61342' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'if_condition_61342', if_condition_61342)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_61345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        str_61346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 35), 'str', '-arch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 34), list_61345, str_61346)
        # Adding element type (line 290)
        # Getting the type of 'arch' (line 290)
        arch_61347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'arch', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 34), list_61345, arch_61347)
        
        # Processing the call keyword arguments (line 290)
        kwargs_61348 = {}
        # Getting the type of 'arch_flags' (line 290)
        arch_flags_61343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'arch_flags', False)
        # Obtaining the member 'extend' of a type (line 290)
        extend_61344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), arch_flags_61343, 'extend')
        # Calling extend(args, kwargs) (line 290)
        extend_call_result_61349 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), extend_61344, *[list_61345], **kwargs_61348)
        
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'arch_flags' (line 291)
        arch_flags_61350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'arch_flags')
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', arch_flags_61350)
        
        # ################# End of '_universal_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_universal_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_61351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_universal_flags'
        return stypy_return_type_61351


    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.get_flags')
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags(...)' code ##################

        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to get_flags(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'self' (line 294)
        self_61354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'self', False)
        # Processing the call keyword arguments (line 294)
        kwargs_61355 = {}
        # Getting the type of 'GnuFCompiler' (line 294)
        GnuFCompiler_61352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'GnuFCompiler', False)
        # Obtaining the member 'get_flags' of a type (line 294)
        get_flags_61353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), GnuFCompiler_61352, 'get_flags')
        # Calling get_flags(args, kwargs) (line 294)
        get_flags_call_result_61356 = invoke(stypy.reporting.localization.Localization(__file__, 294, 16), get_flags_61353, *[self_61354], **kwargs_61355)
        
        # Assigning a type to the variable 'flags' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'flags', get_flags_call_result_61356)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to _universal_flags(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'self' (line 295)
        self_61359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 43), 'self', False)
        # Obtaining the member 'compiler_f90' of a type (line 295)
        compiler_f90_61360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 43), self_61359, 'compiler_f90')
        # Processing the call keyword arguments (line 295)
        kwargs_61361 = {}
        # Getting the type of 'self' (line 295)
        self_61357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'self', False)
        # Obtaining the member '_universal_flags' of a type (line 295)
        _universal_flags_61358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 21), self_61357, '_universal_flags')
        # Calling _universal_flags(args, kwargs) (line 295)
        _universal_flags_call_result_61362 = invoke(stypy.reporting.localization.Localization(__file__, 295, 21), _universal_flags_61358, *[compiler_f90_61360], **kwargs_61361)
        
        # Assigning a type to the variable 'arch_flags' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'arch_flags', _universal_flags_call_result_61362)
        
        # Getting the type of 'arch_flags' (line 296)
        arch_flags_61363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'arch_flags')
        # Testing the type of an if condition (line 296)
        if_condition_61364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), arch_flags_61363)
        # Assigning a type to the variable 'if_condition_61364' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'if_condition_61364', if_condition_61364)
        # SSA begins for if statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 297):
        
        # Assigning a Name to a Subscript (line 297):
        # Getting the type of 'arch_flags' (line 297)
        arch_flags_61365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'arch_flags')
        # Getting the type of 'flags' (line 297)
        flags_61366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'flags')
        int_61367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 19), 'int')
        slice_61368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 297, 12), None, int_61367, None)
        # Storing an element on a container (line 297)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 12), flags_61366, (slice_61368, arch_flags_61365))
        # SSA join for if statement (line 296)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'flags' (line 298)
        flags_61369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'flags')
        # Assigning a type to the variable 'stypy_return_type' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type', flags_61369)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_61370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61370)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_61370


    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 300, 4, False)
        # Assigning a type to the variable 'self' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.get_flags_linker_so')
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_linker_so', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_linker_so(...)' code ##################

        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to get_flags_linker_so(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'self' (line 301)
        self_61373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 49), 'self', False)
        # Processing the call keyword arguments (line 301)
        kwargs_61374 = {}
        # Getting the type of 'GnuFCompiler' (line 301)
        GnuFCompiler_61371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'GnuFCompiler', False)
        # Obtaining the member 'get_flags_linker_so' of a type (line 301)
        get_flags_linker_so_61372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), GnuFCompiler_61371, 'get_flags_linker_so')
        # Calling get_flags_linker_so(args, kwargs) (line 301)
        get_flags_linker_so_call_result_61375 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), get_flags_linker_so_61372, *[self_61373], **kwargs_61374)
        
        # Assigning a type to the variable 'flags' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'flags', get_flags_linker_so_call_result_61375)
        
        # Assigning a Call to a Name (line 302):
        
        # Assigning a Call to a Name (line 302):
        
        # Call to _universal_flags(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'self' (line 302)
        self_61378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 43), 'self', False)
        # Obtaining the member 'linker_so' of a type (line 302)
        linker_so_61379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 43), self_61378, 'linker_so')
        # Processing the call keyword arguments (line 302)
        kwargs_61380 = {}
        # Getting the type of 'self' (line 302)
        self_61376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'self', False)
        # Obtaining the member '_universal_flags' of a type (line 302)
        _universal_flags_61377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 21), self_61376, '_universal_flags')
        # Calling _universal_flags(args, kwargs) (line 302)
        _universal_flags_call_result_61381 = invoke(stypy.reporting.localization.Localization(__file__, 302, 21), _universal_flags_61377, *[linker_so_61379], **kwargs_61380)
        
        # Assigning a type to the variable 'arch_flags' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'arch_flags', _universal_flags_call_result_61381)
        
        # Getting the type of 'arch_flags' (line 303)
        arch_flags_61382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'arch_flags')
        # Testing the type of an if condition (line 303)
        if_condition_61383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 8), arch_flags_61382)
        # Assigning a type to the variable 'if_condition_61383' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'if_condition_61383', if_condition_61383)
        # SSA begins for if statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 304):
        
        # Assigning a Name to a Subscript (line 304):
        # Getting the type of 'arch_flags' (line 304)
        arch_flags_61384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'arch_flags')
        # Getting the type of 'flags' (line 304)
        flags_61385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'flags')
        int_61386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'int')
        slice_61387 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 12), None, int_61386, None)
        # Storing an element on a container (line 304)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), flags_61385, (slice_61387, arch_flags_61384))
        # SSA join for if statement (line 303)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'flags' (line 305)
        flags_61388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'flags')
        # Assigning a type to the variable 'stypy_return_type' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'stypy_return_type', flags_61388)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 300)
        stypy_return_type_61389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_61389


    @norecursion
    def get_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_dirs'
        module_type_store = module_type_store.open_function_context('get_library_dirs', 307, 4, False)
        # Assigning a type to the variable 'self' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.get_library_dirs')
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.get_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.get_library_dirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_library_dirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_library_dirs(...)' code ##################

        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to get_library_dirs(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'self' (line 308)
        self_61392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 44), 'self', False)
        # Processing the call keyword arguments (line 308)
        kwargs_61393 = {}
        # Getting the type of 'GnuFCompiler' (line 308)
        GnuFCompiler_61390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), 'GnuFCompiler', False)
        # Obtaining the member 'get_library_dirs' of a type (line 308)
        get_library_dirs_61391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 14), GnuFCompiler_61390, 'get_library_dirs')
        # Calling get_library_dirs(args, kwargs) (line 308)
        get_library_dirs_call_result_61394 = invoke(stypy.reporting.localization.Localization(__file__, 308, 14), get_library_dirs_61391, *[self_61392], **kwargs_61393)
        
        # Assigning a type to the variable 'opt' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'opt', get_library_dirs_call_result_61394)
        
        
        # Getting the type of 'sys' (line 309)
        sys_61395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 309)
        platform_61396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 11), sys_61395, 'platform')
        str_61397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 309)
        result_eq_61398 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 11), '==', platform_61396, str_61397)
        
        # Testing the type of an if condition (line 309)
        if_condition_61399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 8), result_eq_61398)
        # Assigning a type to the variable 'if_condition_61399' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'if_condition_61399', if_condition_61399)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 310):
        
        # Assigning a Attribute to a Name (line 310):
        # Getting the type of 'self' (line 310)
        self_61400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'self')
        # Obtaining the member 'c_compiler' of a type (line 310)
        c_compiler_61401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 25), self_61400, 'c_compiler')
        # Assigning a type to the variable 'c_compiler' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'c_compiler', c_compiler_61401)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'c_compiler' (line 311)
        c_compiler_61402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'c_compiler')
        
        # Getting the type of 'c_compiler' (line 311)
        c_compiler_61403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 30), 'c_compiler')
        # Obtaining the member 'compiler_type' of a type (line 311)
        compiler_type_61404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 30), c_compiler_61403, 'compiler_type')
        str_61405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 58), 'str', 'msvc')
        # Applying the binary operator '==' (line 311)
        result_eq_61406 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 30), '==', compiler_type_61404, str_61405)
        
        # Applying the binary operator 'and' (line 311)
        result_and_keyword_61407 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 15), 'and', c_compiler_61402, result_eq_61406)
        
        # Testing the type of an if condition (line 311)
        if_condition_61408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 12), result_and_keyword_61407)
        # Assigning a type to the variable 'if_condition_61408' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'if_condition_61408', if_condition_61408)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to get_target(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_61411 = {}
        # Getting the type of 'self' (line 312)
        self_61409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'self', False)
        # Obtaining the member 'get_target' of a type (line 312)
        get_target_61410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 25), self_61409, 'get_target')
        # Calling get_target(args, kwargs) (line 312)
        get_target_call_result_61412 = invoke(stypy.reporting.localization.Localization(__file__, 312, 25), get_target_61410, *[], **kwargs_61411)
        
        # Assigning a type to the variable 'target' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'target', get_target_call_result_61412)
        
        # Getting the type of 'target' (line 313)
        target_61413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'target')
        # Testing the type of an if condition (line 313)
        if_condition_61414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 16), target_61413)
        # Assigning a type to the variable 'if_condition_61414' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'if_condition_61414', if_condition_61414)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to normpath(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Call to get_libgcc_dir(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_61420 = {}
        # Getting the type of 'self' (line 314)
        self_61418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 41), 'self', False)
        # Obtaining the member 'get_libgcc_dir' of a type (line 314)
        get_libgcc_dir_61419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 41), self_61418, 'get_libgcc_dir')
        # Calling get_libgcc_dir(args, kwargs) (line 314)
        get_libgcc_dir_call_result_61421 = invoke(stypy.reporting.localization.Localization(__file__, 314, 41), get_libgcc_dir_61419, *[], **kwargs_61420)
        
        # Processing the call keyword arguments (line 314)
        kwargs_61422 = {}
        # Getting the type of 'os' (line 314)
        os_61415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 314)
        path_61416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 24), os_61415, 'path')
        # Obtaining the member 'normpath' of a type (line 314)
        normpath_61417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 24), path_61416, 'normpath')
        # Calling normpath(args, kwargs) (line 314)
        normpath_call_result_61423 = invoke(stypy.reporting.localization.Localization(__file__, 314, 24), normpath_61417, *[get_libgcc_dir_call_result_61421], **kwargs_61422)
        
        # Assigning a type to the variable 'd' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'd', normpath_call_result_61423)
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to join(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'd' (line 315)
        d_61427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 40), 'd', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 315)
        tuple_61428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 315)
        # Adding element type (line 315)
        # Getting the type of 'os' (line 315)
        os_61429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 46), 'os', False)
        # Obtaining the member 'pardir' of a type (line 315)
        pardir_61430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 46), os_61429, 'pardir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 46), tuple_61428, pardir_61430)
        
        int_61431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 58), 'int')
        # Applying the binary operator '*' (line 315)
        result_mul_61432 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 45), '*', tuple_61428, int_61431)
        
        # Processing the call keyword arguments (line 315)
        kwargs_61433 = {}
        # Getting the type of 'os' (line 315)
        os_61424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 315)
        path_61425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 27), os_61424, 'path')
        # Obtaining the member 'join' of a type (line 315)
        join_61426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 27), path_61425, 'join')
        # Calling join(args, kwargs) (line 315)
        join_call_result_61434 = invoke(stypy.reporting.localization.Localization(__file__, 315, 27), join_61426, *[d_61427, result_mul_61432], **kwargs_61433)
        
        # Assigning a type to the variable 'root' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'root', join_call_result_61434)
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to join(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'root' (line 316)
        root_61438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 40), 'root', False)
        str_61439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 46), 'str', 'lib')
        # Processing the call keyword arguments (line 316)
        kwargs_61440 = {}
        # Getting the type of 'os' (line 316)
        os_61435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 316)
        path_61436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 27), os_61435, 'path')
        # Obtaining the member 'join' of a type (line 316)
        join_61437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 27), path_61436, 'join')
        # Calling join(args, kwargs) (line 316)
        join_call_result_61441 = invoke(stypy.reporting.localization.Localization(__file__, 316, 27), join_61437, *[root_61438, str_61439], **kwargs_61440)
        
        # Assigning a type to the variable 'path' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'path', join_call_result_61441)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to normpath(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'path' (line 317)
        path_61445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 48), 'path', False)
        # Processing the call keyword arguments (line 317)
        kwargs_61446 = {}
        # Getting the type of 'os' (line 317)
        os_61442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 317)
        path_61443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 31), os_61442, 'path')
        # Obtaining the member 'normpath' of a type (line 317)
        normpath_61444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 31), path_61443, 'normpath')
        # Calling normpath(args, kwargs) (line 317)
        normpath_call_result_61447 = invoke(stypy.reporting.localization.Localization(__file__, 317, 31), normpath_61444, *[path_61445], **kwargs_61446)
        
        # Assigning a type to the variable 'mingwdir' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'mingwdir', normpath_call_result_61447)
        
        
        # Call to exists(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Call to join(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'mingwdir' (line 318)
        mingwdir_61454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 51), 'mingwdir', False)
        str_61455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 61), 'str', 'libmingwex.a')
        # Processing the call keyword arguments (line 318)
        kwargs_61456 = {}
        # Getting the type of 'os' (line 318)
        os_61451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 38), 'os', False)
        # Obtaining the member 'path' of a type (line 318)
        path_61452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 38), os_61451, 'path')
        # Obtaining the member 'join' of a type (line 318)
        join_61453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 38), path_61452, 'join')
        # Calling join(args, kwargs) (line 318)
        join_call_result_61457 = invoke(stypy.reporting.localization.Localization(__file__, 318, 38), join_61453, *[mingwdir_61454, str_61455], **kwargs_61456)
        
        # Processing the call keyword arguments (line 318)
        kwargs_61458 = {}
        # Getting the type of 'os' (line 318)
        os_61448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 318)
        path_61449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), os_61448, 'path')
        # Obtaining the member 'exists' of a type (line 318)
        exists_61450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), path_61449, 'exists')
        # Calling exists(args, kwargs) (line 318)
        exists_call_result_61459 = invoke(stypy.reporting.localization.Localization(__file__, 318, 23), exists_61450, *[join_call_result_61457], **kwargs_61458)
        
        # Testing the type of an if condition (line 318)
        if_condition_61460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 20), exists_call_result_61459)
        # Assigning a type to the variable 'if_condition_61460' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'if_condition_61460', if_condition_61460)
        # SSA begins for if statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'mingwdir' (line 319)
        mingwdir_61463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 35), 'mingwdir', False)
        # Processing the call keyword arguments (line 319)
        kwargs_61464 = {}
        # Getting the type of 'opt' (line 319)
        opt_61461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'opt', False)
        # Obtaining the member 'append' of a type (line 319)
        append_61462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 24), opt_61461, 'append')
        # Calling append(args, kwargs) (line 319)
        append_call_result_61465 = invoke(stypy.reporting.localization.Localization(__file__, 319, 24), append_61462, *[mingwdir_61463], **kwargs_61464)
        
        # SSA join for if statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 320)
        opt_61466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', opt_61466)
        
        # ################# End of 'get_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 307)
        stypy_return_type_61467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_dirs'
        return stypy_return_type_61467


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.get_libraries')
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libraries(...)' code ##################

        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to get_libraries(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'self' (line 323)
        self_61470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 41), 'self', False)
        # Processing the call keyword arguments (line 323)
        kwargs_61471 = {}
        # Getting the type of 'GnuFCompiler' (line 323)
        GnuFCompiler_61468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'GnuFCompiler', False)
        # Obtaining the member 'get_libraries' of a type (line 323)
        get_libraries_61469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 14), GnuFCompiler_61468, 'get_libraries')
        # Calling get_libraries(args, kwargs) (line 323)
        get_libraries_call_result_61472 = invoke(stypy.reporting.localization.Localization(__file__, 323, 14), get_libraries_61469, *[self_61470], **kwargs_61471)
        
        # Assigning a type to the variable 'opt' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'opt', get_libraries_call_result_61472)
        
        
        # Getting the type of 'sys' (line 324)
        sys_61473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 324)
        platform_61474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 11), sys_61473, 'platform')
        str_61475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 324)
        result_eq_61476 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 11), '==', platform_61474, str_61475)
        
        # Testing the type of an if condition (line 324)
        if_condition_61477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), result_eq_61476)
        # Assigning a type to the variable 'if_condition_61477' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'if_condition_61477', if_condition_61477)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 325)
        # Processing the call arguments (line 325)
        str_61480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'str', 'cc_dynamic')
        # Processing the call keyword arguments (line 325)
        kwargs_61481 = {}
        # Getting the type of 'opt' (line 325)
        opt_61478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'opt', False)
        # Obtaining the member 'remove' of a type (line 325)
        remove_61479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), opt_61478, 'remove')
        # Calling remove(args, kwargs) (line 325)
        remove_call_result_61482 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), remove_61479, *[str_61480], **kwargs_61481)
        
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 326)
        sys_61483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 326)
        platform_61484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 11), sys_61483, 'platform')
        str_61485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 326)
        result_eq_61486 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 11), '==', platform_61484, str_61485)
        
        # Testing the type of an if condition (line 326)
        if_condition_61487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 8), result_eq_61486)
        # Assigning a type to the variable 'if_condition_61487' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'if_condition_61487', if_condition_61487)
        # SSA begins for if statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 327):
        
        # Assigning a Attribute to a Name (line 327):
        # Getting the type of 'self' (line 327)
        self_61488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 25), 'self')
        # Obtaining the member 'c_compiler' of a type (line 327)
        c_compiler_61489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 25), self_61488, 'c_compiler')
        # Assigning a type to the variable 'c_compiler' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'c_compiler', c_compiler_61489)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'c_compiler' (line 328)
        c_compiler_61490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'c_compiler')
        
        # Getting the type of 'c_compiler' (line 328)
        c_compiler_61491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 30), 'c_compiler')
        # Obtaining the member 'compiler_type' of a type (line 328)
        compiler_type_61492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 30), c_compiler_61491, 'compiler_type')
        str_61493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 58), 'str', 'msvc')
        # Applying the binary operator '==' (line 328)
        result_eq_61494 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 30), '==', compiler_type_61492, str_61493)
        
        # Applying the binary operator 'and' (line 328)
        result_and_keyword_61495 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 15), 'and', c_compiler_61490, result_eq_61494)
        
        # Testing the type of an if condition (line 328)
        if_condition_61496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 12), result_and_keyword_61495)
        # Assigning a type to the variable 'if_condition_61496' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'if_condition_61496', if_condition_61496)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        str_61497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'str', 'gcc')
        # Getting the type of 'opt' (line 329)
        opt_61498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'opt')
        # Applying the binary operator 'in' (line 329)
        result_contains_61499 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 19), 'in', str_61497, opt_61498)
        
        # Testing the type of an if condition (line 329)
        if_condition_61500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 16), result_contains_61499)
        # Assigning a type to the variable 'if_condition_61500' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'if_condition_61500', if_condition_61500)
        # SSA begins for if statement (line 329)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 330):
        
        # Assigning a Call to a Name (line 330):
        
        # Call to index(...): (line 330)
        # Processing the call arguments (line 330)
        str_61503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 34), 'str', 'gcc')
        # Processing the call keyword arguments (line 330)
        kwargs_61504 = {}
        # Getting the type of 'opt' (line 330)
        opt_61501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'opt', False)
        # Obtaining the member 'index' of a type (line 330)
        index_61502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 24), opt_61501, 'index')
        # Calling index(args, kwargs) (line 330)
        index_call_result_61505 = invoke(stypy.reporting.localization.Localization(__file__, 330, 24), index_61502, *[str_61503], **kwargs_61504)
        
        # Assigning a type to the variable 'i' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'i', index_call_result_61505)
        
        # Call to insert(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'i' (line 331)
        i_61508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 31), 'i', False)
        int_61509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 33), 'int')
        # Applying the binary operator '+' (line 331)
        result_add_61510 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 31), '+', i_61508, int_61509)
        
        str_61511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 36), 'str', 'mingwex')
        # Processing the call keyword arguments (line 331)
        kwargs_61512 = {}
        # Getting the type of 'opt' (line 331)
        opt_61506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'opt', False)
        # Obtaining the member 'insert' of a type (line 331)
        insert_61507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 20), opt_61506, 'insert')
        # Calling insert(args, kwargs) (line 331)
        insert_call_result_61513 = invoke(stypy.reporting.localization.Localization(__file__, 331, 20), insert_61507, *[result_add_61510, str_61511], **kwargs_61512)
        
        
        # Call to insert(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'i' (line 332)
        i_61516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 31), 'i', False)
        int_61517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'int')
        # Applying the binary operator '+' (line 332)
        result_add_61518 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 31), '+', i_61516, int_61517)
        
        str_61519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 36), 'str', 'mingw32')
        # Processing the call keyword arguments (line 332)
        kwargs_61520 = {}
        # Getting the type of 'opt' (line 332)
        opt_61514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 20), 'opt', False)
        # Obtaining the member 'insert' of a type (line 332)
        insert_61515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 20), opt_61514, 'insert')
        # Calling insert(args, kwargs) (line 332)
        insert_call_result_61521 = invoke(stypy.reporting.localization.Localization(__file__, 332, 20), insert_61515, *[result_add_61518, str_61519], **kwargs_61520)
        
        # SSA join for if statement (line 329)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_win64(...): (line 334)
        # Processing the call keyword arguments (line 334)
        kwargs_61523 = {}
        # Getting the type of 'is_win64' (line 334)
        is_win64_61522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'is_win64', False)
        # Calling is_win64(args, kwargs) (line 334)
        is_win64_call_result_61524 = invoke(stypy.reporting.localization.Localization(__file__, 334, 15), is_win64_61522, *[], **kwargs_61523)
        
        # Testing the type of an if condition (line 334)
        if_condition_61525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 12), is_win64_call_result_61524)
        # Assigning a type to the variable 'if_condition_61525' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'if_condition_61525', if_condition_61525)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 335):
        
        # Assigning a Attribute to a Name (line 335):
        # Getting the type of 'self' (line 335)
        self_61526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 29), 'self')
        # Obtaining the member 'c_compiler' of a type (line 335)
        c_compiler_61527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 29), self_61526, 'c_compiler')
        # Assigning a type to the variable 'c_compiler' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'c_compiler', c_compiler_61527)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'c_compiler' (line 336)
        c_compiler_61528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'c_compiler')
        
        # Getting the type of 'c_compiler' (line 336)
        c_compiler_61529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 34), 'c_compiler')
        # Obtaining the member 'compiler_type' of a type (line 336)
        compiler_type_61530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 34), c_compiler_61529, 'compiler_type')
        str_61531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 62), 'str', 'msvc')
        # Applying the binary operator '==' (line 336)
        result_eq_61532 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 34), '==', compiler_type_61530, str_61531)
        
        # Applying the binary operator 'and' (line 336)
        result_and_keyword_61533 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 19), 'and', c_compiler_61528, result_eq_61532)
        
        # Testing the type of an if condition (line 336)
        if_condition_61534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 16), result_and_keyword_61533)
        # Assigning a type to the variable 'if_condition_61534' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'if_condition_61534', if_condition_61534)
        # SSA begins for if statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_61535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        
        # Assigning a type to the variable 'stypy_return_type' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'stypy_return_type', list_61535)
        # SSA branch for the else part of an if statement (line 336)
        module_type_store.open_ssa_branch('else')
        pass
        # SSA join for if statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 340)
        opt_61536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'stypy_return_type', opt_61536)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_61537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_61537


    @norecursion
    def get_target(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_target'
        module_type_store = module_type_store.open_function_context('get_target', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.get_target')
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_param_names_list', [])
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.get_target.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.get_target', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_target', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_target(...)' code ##################

        
        # Assigning a Call to a Tuple (line 343):
        
        # Assigning a Call to a Name:
        
        # Call to exec_command(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_61539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 38), 'self', False)
        # Obtaining the member 'compiler_f77' of a type (line 343)
        compiler_f77_61540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 38), self_61539, 'compiler_f77')
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_61541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        str_61542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 39), 'str', '-v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 38), list_61541, str_61542)
        
        # Applying the binary operator '+' (line 343)
        result_add_61543 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 38), '+', compiler_f77_61540, list_61541)
        
        # Processing the call keyword arguments (line 343)
        int_61544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 46), 'int')
        keyword_61545 = int_61544
        kwargs_61546 = {'use_tee': keyword_61545}
        # Getting the type of 'exec_command' (line 343)
        exec_command_61538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'exec_command', False)
        # Calling exec_command(args, kwargs) (line 343)
        exec_command_call_result_61547 = invoke(stypy.reporting.localization.Localization(__file__, 343, 25), exec_command_61538, *[result_add_61543], **kwargs_61546)
        
        # Assigning a type to the variable 'call_assignment_60622' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60622', exec_command_call_result_61547)
        
        # Assigning a Call to a Name (line 343):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61551 = {}
        # Getting the type of 'call_assignment_60622' (line 343)
        call_assignment_60622_61548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60622', False)
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___61549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), call_assignment_60622_61548, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61552 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61549, *[int_61550], **kwargs_61551)
        
        # Assigning a type to the variable 'call_assignment_60623' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60623', getitem___call_result_61552)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'call_assignment_60623' (line 343)
        call_assignment_60623_61553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60623')
        # Assigning a type to the variable 'status' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'status', call_assignment_60623_61553)
        
        # Assigning a Call to a Name (line 343):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61557 = {}
        # Getting the type of 'call_assignment_60622' (line 343)
        call_assignment_60622_61554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60622', False)
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___61555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), call_assignment_60622_61554, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61558 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61555, *[int_61556], **kwargs_61557)
        
        # Assigning a type to the variable 'call_assignment_60624' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60624', getitem___call_result_61558)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'call_assignment_60624' (line 343)
        call_assignment_60624_61559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_60624')
        # Assigning a type to the variable 'output' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'output', call_assignment_60624_61559)
        
        
        # Getting the type of 'status' (line 346)
        status_61560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'status')
        # Applying the 'not' unary operator (line 346)
        result_not__61561 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 11), 'not', status_61560)
        
        # Testing the type of an if condition (line 346)
        if_condition_61562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 8), result_not__61561)
        # Assigning a type to the variable 'if_condition_61562' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'if_condition_61562', if_condition_61562)
        # SSA begins for if statement (line 346)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to search(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'output' (line 347)
        output_61565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 32), 'output', False)
        # Processing the call keyword arguments (line 347)
        kwargs_61566 = {}
        # Getting the type of 'TARGET_R' (line 347)
        TARGET_R_61563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'TARGET_R', False)
        # Obtaining the member 'search' of a type (line 347)
        search_61564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), TARGET_R_61563, 'search')
        # Calling search(args, kwargs) (line 347)
        search_call_result_61567 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), search_61564, *[output_61565], **kwargs_61566)
        
        # Assigning a type to the variable 'm' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'm', search_call_result_61567)
        
        # Getting the type of 'm' (line 348)
        m_61568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 15), 'm')
        # Testing the type of an if condition (line 348)
        if_condition_61569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 12), m_61568)
        # Assigning a type to the variable 'if_condition_61569' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'if_condition_61569', if_condition_61569)
        # SSA begins for if statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to group(...): (line 349)
        # Processing the call arguments (line 349)
        int_61572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 31), 'int')
        # Processing the call keyword arguments (line 349)
        kwargs_61573 = {}
        # Getting the type of 'm' (line 349)
        m_61570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'm', False)
        # Obtaining the member 'group' of a type (line 349)
        group_61571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 23), m_61570, 'group')
        # Calling group(args, kwargs) (line 349)
        group_call_result_61574 = invoke(stypy.reporting.localization.Localization(__file__, 349, 23), group_61571, *[int_61572], **kwargs_61573)
        
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'stypy_return_type', group_call_result_61574)
        # SSA join for if statement (line 348)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 346)
        module_type_store = module_type_store.join_ssa_context()
        
        str_61575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 15), 'str', '')
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', str_61575)
        
        # ################# End of 'get_target(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_target' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_61576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_target'
        return stypy_return_type_61576


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'Gnu95FCompiler.get_flags_opt')
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gnu95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        
        
        # Call to is_win64(...): (line 353)
        # Processing the call keyword arguments (line 353)
        kwargs_61578 = {}
        # Getting the type of 'is_win64' (line 353)
        is_win64_61577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'is_win64', False)
        # Calling is_win64(args, kwargs) (line 353)
        is_win64_call_result_61579 = invoke(stypy.reporting.localization.Localization(__file__, 353, 11), is_win64_61577, *[], **kwargs_61578)
        
        # Testing the type of an if condition (line 353)
        if_condition_61580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 8), is_win64_call_result_61579)
        # Assigning a type to the variable 'if_condition_61580' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'if_condition_61580', if_condition_61580)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 354)
        list_61581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 354)
        # Adding element type (line 354)
        str_61582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 20), 'str', '-O0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_61581, str_61582)
        
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'stypy_return_type', list_61581)
        # SSA branch for the else part of an if statement (line 353)
        module_type_store.open_ssa_branch('else')
        
        # Call to get_flags_opt(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_61585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 46), 'self', False)
        # Processing the call keyword arguments (line 356)
        kwargs_61586 = {}
        # Getting the type of 'GnuFCompiler' (line 356)
        GnuFCompiler_61583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'GnuFCompiler', False)
        # Obtaining the member 'get_flags_opt' of a type (line 356)
        get_flags_opt_61584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 19), GnuFCompiler_61583, 'get_flags_opt')
        # Calling get_flags_opt(args, kwargs) (line 356)
        get_flags_opt_call_result_61587 = invoke(stypy.reporting.localization.Localization(__file__, 356, 19), get_flags_opt_61584, *[self_61585], **kwargs_61586)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'stypy_return_type', get_flags_opt_call_result_61587)
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_61588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_61588


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 235, 0, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gnu95FCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Gnu95FCompiler' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'Gnu95FCompiler', Gnu95FCompiler)

# Assigning a Str to a Name (line 236):
str_61589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 20), 'str', 'gnu95')
# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61590, 'compiler_type', str_61589)

# Assigning a Tuple to a Name (line 237):

# Obtaining an instance of the builtin type 'tuple' (line 237)
tuple_61591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 237)
# Adding element type (line 237)
str_61592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 24), 'str', 'gfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 24), tuple_61591, str_61592)

# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61593, 'compiler_aliases', tuple_61591)

# Assigning a Str to a Name (line 238):
str_61594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 18), 'str', 'GNU Fortran 95 compiler')
# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61595, 'description', str_61594)

# Assigning a List to a Name (line 257):

# Obtaining an instance of the builtin type 'list' (line 257)
list_61596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 257)
# Adding element type (line 257)
str_61597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'str', 'gfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 27), list_61596, str_61597)
# Adding element type (line 257)
str_61598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 40), 'str', 'f95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 27), list_61596, str_61598)

# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61599, 'possible_executables', list_61596)

# Assigning a Dict to a Name (line 258):

# Obtaining an instance of the builtin type 'dict' (line 258)
dict_61600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 258)
# Adding element type (key, value) (line 258)
str_61601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 259)
list_61602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 259)
# Adding element type (line 259)
str_61603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), list_61602, str_61603)
# Adding element type (line 259)
str_61604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 35), 'str', '-dumpversion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), list_61602, str_61604)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61601, list_61602))
# Adding element type (key, value) (line 258)
str_61605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 260)
list_61606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 260)
# Adding element type (line 260)
# Getting the type of 'None' (line 260)
None_61607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), list_61606, None_61607)
# Adding element type (line 260)
str_61608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), list_61606, str_61608)
# Adding element type (line 260)
str_61609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 41), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), list_61606, str_61609)
# Adding element type (line 260)
str_61610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 47), 'str', '-ffixed-form')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), list_61606, str_61610)
# Adding element type (line 260)
str_61611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 26), 'str', '-fno-second-underscore')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), list_61606, str_61611)

# Getting the type of '_EXTRAFLAGS' (line 261)
_EXTRAFLAGS_61612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 54), '_EXTRAFLAGS')
# Applying the binary operator '+' (line 260)
result_add_61613 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 25), '+', list_61606, _EXTRAFLAGS_61612)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61605, result_add_61613))
# Adding element type (key, value) (line 258)
str_61614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 262)
list_61615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 262)
# Adding element type (line 262)
# Getting the type of 'None' (line 262)
None_61616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), list_61615, None_61616)
# Adding element type (line 262)
str_61617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), list_61615, str_61617)
# Adding element type (line 262)
str_61618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 41), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), list_61615, str_61618)
# Adding element type (line 262)
str_61619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 26), 'str', '-fno-second-underscore')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), list_61615, str_61619)

# Getting the type of '_EXTRAFLAGS' (line 263)
_EXTRAFLAGS_61620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 54), '_EXTRAFLAGS')
# Applying the binary operator '+' (line 262)
result_add_61621 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 25), '+', list_61615, _EXTRAFLAGS_61620)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61614, result_add_61621))
# Adding element type (key, value) (line 258)
str_61622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 264)
list_61623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 264)
# Adding element type (line 264)
# Getting the type of 'None' (line 264)
None_61624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 25), list_61623, None_61624)
# Adding element type (line 264)
str_61625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 32), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 25), list_61623, str_61625)
# Adding element type (line 264)
str_61626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 42), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 25), list_61623, str_61626)
# Adding element type (line 264)
str_61627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 47), 'str', '-ffixed-form')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 25), list_61623, str_61627)
# Adding element type (line 264)
str_61628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 26), 'str', '-fno-second-underscore')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 25), list_61623, str_61628)

# Getting the type of '_EXTRAFLAGS' (line 265)
_EXTRAFLAGS_61629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 54), '_EXTRAFLAGS')
# Applying the binary operator '+' (line 264)
result_add_61630 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 25), '+', list_61623, _EXTRAFLAGS_61629)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61622, result_add_61630))
# Adding element type (key, value) (line 258)
str_61631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 266)
list_61632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 266)
# Adding element type (line 266)
str_61633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 25), list_61632, str_61633)
# Adding element type (line 266)
str_61634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 35), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 25), list_61632, str_61634)
# Adding element type (line 266)
str_61635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 44), 'str', '-g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 25), list_61632, str_61635)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61631, list_61632))
# Adding element type (key, value) (line 258)
str_61636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 267)
list_61637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 267)
# Adding element type (line 267)
str_61638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 25), list_61637, str_61638)
# Adding element type (line 267)
str_61639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 25), list_61637, str_61639)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61636, list_61637))
# Adding element type (key, value) (line 258)
str_61640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 268)
list_61641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 268)
# Adding element type (line 268)
str_61642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 25), list_61641, str_61642)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61640, list_61641))
# Adding element type (key, value) (line 258)
str_61643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 8), 'str', 'linker_exe')

# Obtaining an instance of the builtin type 'list' (line 269)
list_61644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 269)
# Adding element type (line 269)
# Getting the type of 'None' (line 269)
None_61645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 25), list_61644, None_61645)
# Adding element type (line 269)
str_61646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 32), 'str', '-Wall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 25), list_61644, str_61646)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), dict_61600, (str_61643, list_61644))

# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61647, 'executables', dict_61600)

# Assigning a Str to a Name (line 272):
str_61648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 24), 'str', '-J')
# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61649, 'module_dir_switch', str_61648)

# Assigning a Str to a Name (line 273):
str_61650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'str', '-I')
# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61651, 'module_include_switch', str_61650)

# Assigning a Str to a Name (line 275):
str_61652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 10), 'str', 'gfortran')
# Getting the type of 'Gnu95FCompiler'
Gnu95FCompiler_61653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gnu95FCompiler')
# Setting the type of the member 'g2c' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gnu95FCompiler_61653, 'g2c', str_61652)

@norecursion
def _can_target(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_can_target'
    module_type_store = module_type_store.open_function_context('_can_target', 358, 0, False)
    
    # Passed parameters checking function
    _can_target.stypy_localization = localization
    _can_target.stypy_type_of_self = None
    _can_target.stypy_type_store = module_type_store
    _can_target.stypy_function_name = '_can_target'
    _can_target.stypy_param_names_list = ['cmd', 'arch']
    _can_target.stypy_varargs_param_name = None
    _can_target.stypy_kwargs_param_name = None
    _can_target.stypy_call_defaults = defaults
    _can_target.stypy_call_varargs = varargs
    _can_target.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_can_target', ['cmd', 'arch'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_can_target', localization, ['cmd', 'arch'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_can_target(...)' code ##################

    str_61654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 4), 'str', 'Return true if the architecture supports the -arch flag')
    
    # Assigning a Subscript to a Name (line 360):
    
    # Assigning a Subscript to a Name (line 360):
    
    # Obtaining the type of the subscript
    slice_61655 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 360, 13), None, None, None)
    # Getting the type of 'cmd' (line 360)
    cmd_61656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 13), 'cmd')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___61657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 13), cmd_61656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_61658 = invoke(stypy.reporting.localization.Localization(__file__, 360, 13), getitem___61657, slice_61655)
    
    # Assigning a type to the variable 'newcmd' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'newcmd', subscript_call_result_61658)
    
    # Assigning a Call to a Tuple (line 361):
    
    # Assigning a Call to a Name:
    
    # Call to mkstemp(...): (line 361)
    # Processing the call keyword arguments (line 361)
    str_61661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 44), 'str', '.f')
    keyword_61662 = str_61661
    kwargs_61663 = {'suffix': keyword_61662}
    # Getting the type of 'tempfile' (line 361)
    tempfile_61659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 361)
    mkstemp_61660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 20), tempfile_61659, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 361)
    mkstemp_call_result_61664 = invoke(stypy.reporting.localization.Localization(__file__, 361, 20), mkstemp_61660, *[], **kwargs_61663)
    
    # Assigning a type to the variable 'call_assignment_60625' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60625', mkstemp_call_result_61664)
    
    # Assigning a Call to a Name (line 361):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_61667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 4), 'int')
    # Processing the call keyword arguments
    kwargs_61668 = {}
    # Getting the type of 'call_assignment_60625' (line 361)
    call_assignment_60625_61665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60625', False)
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___61666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), call_assignment_60625_61665, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_61669 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61666, *[int_61667], **kwargs_61668)
    
    # Assigning a type to the variable 'call_assignment_60626' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60626', getitem___call_result_61669)
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'call_assignment_60626' (line 361)
    call_assignment_60626_61670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60626')
    # Assigning a type to the variable 'fid' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'fid', call_assignment_60626_61670)
    
    # Assigning a Call to a Name (line 361):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_61673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 4), 'int')
    # Processing the call keyword arguments
    kwargs_61674 = {}
    # Getting the type of 'call_assignment_60625' (line 361)
    call_assignment_60625_61671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60625', False)
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___61672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), call_assignment_60625_61671, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_61675 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61672, *[int_61673], **kwargs_61674)
    
    # Assigning a type to the variable 'call_assignment_60627' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60627', getitem___call_result_61675)
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'call_assignment_60627' (line 361)
    call_assignment_60627_61676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'call_assignment_60627')
    # Assigning a type to the variable 'filename' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 9), 'filename', call_assignment_60627_61676)
    
    # Try-finally block (line 362)
    
    # Assigning a Call to a Name (line 363):
    
    # Assigning a Call to a Name (line 363):
    
    # Call to dirname(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'filename' (line 363)
    filename_61680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 28), 'filename', False)
    # Processing the call keyword arguments (line 363)
    kwargs_61681 = {}
    # Getting the type of 'os' (line 363)
    os_61677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 363)
    path_61678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), os_61677, 'path')
    # Obtaining the member 'dirname' of a type (line 363)
    dirname_61679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), path_61678, 'dirname')
    # Calling dirname(args, kwargs) (line 363)
    dirname_call_result_61682 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), dirname_61679, *[filename_61680], **kwargs_61681)
    
    # Assigning a type to the variable 'd' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'd', dirname_call_result_61682)
    
    # Assigning a BinOp to a Name (line 364):
    
    # Assigning a BinOp to a Name (line 364):
    
    # Obtaining the type of the subscript
    int_61683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 44), 'int')
    
    # Call to splitext(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'filename' (line 364)
    filename_61687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 34), 'filename', False)
    # Processing the call keyword arguments (line 364)
    kwargs_61688 = {}
    # Getting the type of 'os' (line 364)
    os_61684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 364)
    path_61685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 17), os_61684, 'path')
    # Obtaining the member 'splitext' of a type (line 364)
    splitext_61686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 17), path_61685, 'splitext')
    # Calling splitext(args, kwargs) (line 364)
    splitext_call_result_61689 = invoke(stypy.reporting.localization.Localization(__file__, 364, 17), splitext_61686, *[filename_61687], **kwargs_61688)
    
    # Obtaining the member '__getitem__' of a type (line 364)
    getitem___61690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 17), splitext_call_result_61689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 364)
    subscript_call_result_61691 = invoke(stypy.reporting.localization.Localization(__file__, 364, 17), getitem___61690, int_61683)
    
    str_61692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'str', '.o')
    # Applying the binary operator '+' (line 364)
    result_add_61693 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 17), '+', subscript_call_result_61691, str_61692)
    
    # Assigning a type to the variable 'output' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'output', result_add_61693)
    
    # Try-finally block (line 365)
    
    # Call to extend(...): (line 366)
    # Processing the call arguments (line 366)
    
    # Obtaining an instance of the builtin type 'list' (line 366)
    list_61696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 366)
    # Adding element type (line 366)
    str_61697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 27), 'str', '-arch')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 26), list_61696, str_61697)
    # Adding element type (line 366)
    # Getting the type of 'arch' (line 366)
    arch_61698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 36), 'arch', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 26), list_61696, arch_61698)
    # Adding element type (line 366)
    str_61699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 42), 'str', '-c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 26), list_61696, str_61699)
    # Adding element type (line 366)
    # Getting the type of 'filename' (line 366)
    filename_61700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 48), 'filename', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 26), list_61696, filename_61700)
    
    # Processing the call keyword arguments (line 366)
    kwargs_61701 = {}
    # Getting the type of 'newcmd' (line 366)
    newcmd_61694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'newcmd', False)
    # Obtaining the member 'extend' of a type (line 366)
    extend_61695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), newcmd_61694, 'extend')
    # Calling extend(args, kwargs) (line 366)
    extend_call_result_61702 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), extend_61695, *[list_61696], **kwargs_61701)
    
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to Popen(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'newcmd' (line 367)
    newcmd_61704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'newcmd', False)
    # Processing the call keyword arguments (line 367)
    # Getting the type of 'STDOUT' (line 367)
    STDOUT_61705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 37), 'STDOUT', False)
    keyword_61706 = STDOUT_61705
    # Getting the type of 'PIPE' (line 367)
    PIPE_61707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 52), 'PIPE', False)
    keyword_61708 = PIPE_61707
    # Getting the type of 'd' (line 367)
    d_61709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 62), 'd', False)
    keyword_61710 = d_61709
    kwargs_61711 = {'cwd': keyword_61710, 'stderr': keyword_61706, 'stdout': keyword_61708}
    # Getting the type of 'Popen' (line 367)
    Popen_61703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'Popen', False)
    # Calling Popen(args, kwargs) (line 367)
    Popen_call_result_61712 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), Popen_61703, *[newcmd_61704], **kwargs_61711)
    
    # Assigning a type to the variable 'p' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'p', Popen_call_result_61712)
    
    # Call to communicate(...): (line 368)
    # Processing the call keyword arguments (line 368)
    kwargs_61715 = {}
    # Getting the type of 'p' (line 368)
    p_61713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'p', False)
    # Obtaining the member 'communicate' of a type (line 368)
    communicate_61714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 12), p_61713, 'communicate')
    # Calling communicate(args, kwargs) (line 368)
    communicate_call_result_61716 = invoke(stypy.reporting.localization.Localization(__file__, 368, 12), communicate_61714, *[], **kwargs_61715)
    
    
    # Getting the type of 'p' (line 369)
    p_61717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'p')
    # Obtaining the member 'returncode' of a type (line 369)
    returncode_61718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 19), p_61717, 'returncode')
    int_61719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 35), 'int')
    # Applying the binary operator '==' (line 369)
    result_eq_61720 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 19), '==', returncode_61718, int_61719)
    
    # Assigning a type to the variable 'stypy_return_type' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'stypy_return_type', result_eq_61720)
    
    # finally branch of the try-finally block (line 365)
    
    
    # Call to exists(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'output' (line 371)
    output_61724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 30), 'output', False)
    # Processing the call keyword arguments (line 371)
    kwargs_61725 = {}
    # Getting the type of 'os' (line 371)
    os_61721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 371)
    path_61722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), os_61721, 'path')
    # Obtaining the member 'exists' of a type (line 371)
    exists_61723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), path_61722, 'exists')
    # Calling exists(args, kwargs) (line 371)
    exists_call_result_61726 = invoke(stypy.reporting.localization.Localization(__file__, 371, 15), exists_61723, *[output_61724], **kwargs_61725)
    
    # Testing the type of an if condition (line 371)
    if_condition_61727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 12), exists_call_result_61726)
    # Assigning a type to the variable 'if_condition_61727' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'if_condition_61727', if_condition_61727)
    # SSA begins for if statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to remove(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'output' (line 372)
    output_61730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 'output', False)
    # Processing the call keyword arguments (line 372)
    kwargs_61731 = {}
    # Getting the type of 'os' (line 372)
    os_61728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'os', False)
    # Obtaining the member 'remove' of a type (line 372)
    remove_61729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), os_61728, 'remove')
    # Calling remove(args, kwargs) (line 372)
    remove_call_result_61732 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), remove_61729, *[output_61730], **kwargs_61731)
    
    # SSA join for if statement (line 371)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # finally branch of the try-finally block (line 362)
    
    # Call to remove(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'filename' (line 374)
    filename_61735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 18), 'filename', False)
    # Processing the call keyword arguments (line 374)
    kwargs_61736 = {}
    # Getting the type of 'os' (line 374)
    os_61733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'os', False)
    # Obtaining the member 'remove' of a type (line 374)
    remove_61734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), os_61733, 'remove')
    # Calling remove(args, kwargs) (line 374)
    remove_call_result_61737 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), remove_61734, *[filename_61735], **kwargs_61736)
    
    
    # Getting the type of 'False' (line 375)
    False_61738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type', False_61738)
    
    # ################# End of '_can_target(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_can_target' in the type store
    # Getting the type of 'stypy_return_type' (line 358)
    stypy_return_type_61739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_61739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_can_target'
    return stypy_return_type_61739

# Assigning a type to the variable '_can_target' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), '_can_target', _can_target)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 378, 4))
    
    # 'from distutils import log' statement (line 378)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 378, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 379)
    # Processing the call arguments (line 379)
    int_61742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 22), 'int')
    # Processing the call keyword arguments (line 379)
    kwargs_61743 = {}
    # Getting the type of 'log' (line 379)
    log_61740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 379)
    set_verbosity_61741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 4), log_61740, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 379)
    set_verbosity_call_result_61744 = invoke(stypy.reporting.localization.Localization(__file__, 379, 4), set_verbosity_61741, *[int_61742], **kwargs_61743)
    
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to GnuFCompiler(...): (line 381)
    # Processing the call keyword arguments (line 381)
    kwargs_61746 = {}
    # Getting the type of 'GnuFCompiler' (line 381)
    GnuFCompiler_61745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'GnuFCompiler', False)
    # Calling GnuFCompiler(args, kwargs) (line 381)
    GnuFCompiler_call_result_61747 = invoke(stypy.reporting.localization.Localization(__file__, 381, 15), GnuFCompiler_61745, *[], **kwargs_61746)
    
    # Assigning a type to the variable 'compiler' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'compiler', GnuFCompiler_call_result_61747)
    
    # Call to customize(...): (line 382)
    # Processing the call keyword arguments (line 382)
    kwargs_61750 = {}
    # Getting the type of 'compiler' (line 382)
    compiler_61748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 382)
    customize_61749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 4), compiler_61748, 'customize')
    # Calling customize(args, kwargs) (line 382)
    customize_call_result_61751 = invoke(stypy.reporting.localization.Localization(__file__, 382, 4), customize_61749, *[], **kwargs_61750)
    
    
    # Call to print(...): (line 383)
    # Processing the call arguments (line 383)
    
    # Call to get_version(...): (line 383)
    # Processing the call keyword arguments (line 383)
    kwargs_61755 = {}
    # Getting the type of 'compiler' (line 383)
    compiler_61753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 383)
    get_version_61754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 10), compiler_61753, 'get_version')
    # Calling get_version(args, kwargs) (line 383)
    get_version_call_result_61756 = invoke(stypy.reporting.localization.Localization(__file__, 383, 10), get_version_61754, *[], **kwargs_61755)
    
    # Processing the call keyword arguments (line 383)
    kwargs_61757 = {}
    # Getting the type of 'print' (line 383)
    print_61752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'print', False)
    # Calling print(args, kwargs) (line 383)
    print_call_result_61758 = invoke(stypy.reporting.localization.Localization(__file__, 383, 4), print_61752, *[get_version_call_result_61756], **kwargs_61757)
    
    
    
    # SSA begins for try-except statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to Gnu95FCompiler(...): (line 386)
    # Processing the call keyword arguments (line 386)
    kwargs_61760 = {}
    # Getting the type of 'Gnu95FCompiler' (line 386)
    Gnu95FCompiler_61759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'Gnu95FCompiler', False)
    # Calling Gnu95FCompiler(args, kwargs) (line 386)
    Gnu95FCompiler_call_result_61761 = invoke(stypy.reporting.localization.Localization(__file__, 386, 19), Gnu95FCompiler_61759, *[], **kwargs_61760)
    
    # Assigning a type to the variable 'compiler' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'compiler', Gnu95FCompiler_call_result_61761)
    
    # Call to customize(...): (line 387)
    # Processing the call keyword arguments (line 387)
    kwargs_61764 = {}
    # Getting the type of 'compiler' (line 387)
    compiler_61762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 387)
    customize_61763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), compiler_61762, 'customize')
    # Calling customize(args, kwargs) (line 387)
    customize_call_result_61765 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), customize_61763, *[], **kwargs_61764)
    
    
    # Call to print(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Call to get_version(...): (line 388)
    # Processing the call keyword arguments (line 388)
    kwargs_61769 = {}
    # Getting the type of 'compiler' (line 388)
    compiler_61767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 14), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 388)
    get_version_61768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 14), compiler_61767, 'get_version')
    # Calling get_version(args, kwargs) (line 388)
    get_version_call_result_61770 = invoke(stypy.reporting.localization.Localization(__file__, 388, 14), get_version_61768, *[], **kwargs_61769)
    
    # Processing the call keyword arguments (line 388)
    kwargs_61771 = {}
    # Getting the type of 'print' (line 388)
    print_61766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'print', False)
    # Calling print(args, kwargs) (line 388)
    print_call_result_61772 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), print_61766, *[get_version_call_result_61770], **kwargs_61771)
    
    # SSA branch for the except part of a try statement (line 385)
    # SSA branch for the except 'Exception' branch of a try statement (line 385)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to get_exception(...): (line 390)
    # Processing the call keyword arguments (line 390)
    kwargs_61774 = {}
    # Getting the type of 'get_exception' (line 390)
    get_exception_61773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 14), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 390)
    get_exception_call_result_61775 = invoke(stypy.reporting.localization.Localization(__file__, 390, 14), get_exception_61773, *[], **kwargs_61774)
    
    # Assigning a type to the variable 'msg' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'msg', get_exception_call_result_61775)
    
    # Call to print(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'msg' (line 391)
    msg_61777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 14), 'msg', False)
    # Processing the call keyword arguments (line 391)
    kwargs_61778 = {}
    # Getting the type of 'print' (line 391)
    print_61776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'print', False)
    # Calling print(args, kwargs) (line 391)
    print_call_result_61779 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), print_61776, *[msg_61777], **kwargs_61778)
    
    # SSA join for try-except statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

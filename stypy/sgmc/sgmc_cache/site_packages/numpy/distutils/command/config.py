
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Added Fortran compiler support to config. Currently useful only for
2: # try_compile call. try_run works but is untested for most of Fortran
3: # compilers (they must define linker_exe first).
4: # Pearu Peterson
5: from __future__ import division, absolute_import, print_function
6: 
7: import os, signal
8: import warnings
9: import sys
10: 
11: from distutils.command.config import config as old_config
12: from distutils.command.config import LANG_EXT
13: from distutils import log
14: from distutils.file_util import copy_file
15: from distutils.ccompiler import CompileError, LinkError
16: import distutils
17: from numpy.distutils.exec_command import exec_command
18: from numpy.distutils.mingw32ccompiler import generate_manifest
19: from numpy.distutils.command.autodist import (check_gcc_function_attribute,
20:                                               check_gcc_variable_attribute,
21:                                               check_inline,
22:                                               check_restrict,
23:                                               check_compiler_gcc4)
24: from numpy.distutils.compat import get_exception
25: 
26: LANG_EXT['f77'] = '.f'
27: LANG_EXT['f90'] = '.f90'
28: 
29: class config(old_config):
30:     old_config.user_options += [
31:         ('fcompiler=', None, "specify the Fortran compiler type"),
32:         ]
33: 
34:     def initialize_options(self):
35:         self.fcompiler = None
36:         old_config.initialize_options(self)
37: 
38:     def _check_compiler (self):
39:         old_config._check_compiler(self)
40:         from numpy.distutils.fcompiler import FCompiler, new_fcompiler
41: 
42:         if sys.platform == 'win32' and (self.compiler.compiler_type in
43:                                         ('msvc', 'intelw', 'intelemw')):
44:             # XXX: hack to circumvent a python 2.6 bug with msvc9compiler:
45:             # initialize call query_vcvarsall, which throws an IOError, and
46:             # causes an error along the way without much information. We try to
47:             # catch it here, hoping it is early enough, and print an helpful
48:             # message instead of Error: None.
49:             if not self.compiler.initialized:
50:                 try:
51:                     self.compiler.initialize()
52:                 except IOError:
53:                     e = get_exception()
54:                     msg = '''\
55: Could not initialize compiler instance: do you have Visual Studio
56: installed?  If you are trying to build with MinGW, please use "python setup.py
57: build -c mingw32" instead.  If you have Visual Studio installed, check it is
58: correctly installed, and the right version (VS 2008 for python 2.6, 2.7 and 3.2,
59: VS 2010 for >= 3.3).
60: 
61: Original exception was: %s, and the Compiler class was %s
62: ============================================================================''' \
63:                         % (e, self.compiler.__class__.__name__)
64:                     print ('''\
65: ============================================================================''')
66:                     raise distutils.errors.DistutilsPlatformError(msg)
67: 
68:             # After MSVC is initialized, add an explicit /MANIFEST to linker
69:             # flags.  See issues gh-4245 and gh-4101 for details.  Also
70:             # relevant are issues 4431 and 16296 on the Python bug tracker.
71:             from distutils import msvc9compiler
72:             if msvc9compiler.get_build_version() >= 10:
73:                 for ldflags in [self.compiler.ldflags_shared,
74:                                 self.compiler.ldflags_shared_debug]:
75:                     if '/MANIFEST' not in ldflags:
76:                         ldflags.append('/MANIFEST')
77: 
78:         if not isinstance(self.fcompiler, FCompiler):
79:             self.fcompiler = new_fcompiler(compiler=self.fcompiler,
80:                                            dry_run=self.dry_run, force=1,
81:                                            c_compiler=self.compiler)
82:             if self.fcompiler is not None:
83:                 self.fcompiler.customize(self.distribution)
84:                 if self.fcompiler.get_version():
85:                     self.fcompiler.customize_cmd(self)
86:                     self.fcompiler.show_customization()
87: 
88:     def _wrap_method(self, mth, lang, args):
89:         from distutils.ccompiler import CompileError
90:         from distutils.errors import DistutilsExecError
91:         save_compiler = self.compiler
92:         if lang in ['f77', 'f90']:
93:             self.compiler = self.fcompiler
94:         try:
95:             ret = mth(*((self,)+args))
96:         except (DistutilsExecError, CompileError):
97:             msg = str(get_exception())
98:             self.compiler = save_compiler
99:             raise CompileError
100:         self.compiler = save_compiler
101:         return ret
102: 
103:     def _compile (self, body, headers, include_dirs, lang):
104:         return self._wrap_method(old_config._compile, lang,
105:                                  (body, headers, include_dirs, lang))
106: 
107:     def _link (self, body,
108:                headers, include_dirs,
109:                libraries, library_dirs, lang):
110:         if self.compiler.compiler_type=='msvc':
111:             libraries = (libraries or [])[:]
112:             library_dirs = (library_dirs or [])[:]
113:             if lang in ['f77', 'f90']:
114:                 lang = 'c' # always use system linker when using MSVC compiler
115:                 if self.fcompiler:
116:                     for d in self.fcompiler.library_dirs or []:
117:                         # correct path when compiling in Cygwin but with
118:                         # normal Win Python
119:                         if d.startswith('/usr/lib'):
120:                             s, o = exec_command(['cygpath', '-w', d],
121:                                                use_tee=False)
122:                             if not s: d = o
123:                         library_dirs.append(d)
124:                     for libname in self.fcompiler.libraries or []:
125:                         if libname not in libraries:
126:                             libraries.append(libname)
127:             for libname in libraries:
128:                 if libname.startswith('msvc'): continue
129:                 fileexists = False
130:                 for libdir in library_dirs or []:
131:                     libfile = os.path.join(libdir, '%s.lib' % (libname))
132:                     if os.path.isfile(libfile):
133:                         fileexists = True
134:                         break
135:                 if fileexists: continue
136:                 # make g77-compiled static libs available to MSVC
137:                 fileexists = False
138:                 for libdir in library_dirs:
139:                     libfile = os.path.join(libdir, 'lib%s.a' % (libname))
140:                     if os.path.isfile(libfile):
141:                         # copy libname.a file to name.lib so that MSVC linker
142:                         # can find it
143:                         libfile2 = os.path.join(libdir, '%s.lib' % (libname))
144:                         copy_file(libfile, libfile2)
145:                         self.temp_files.append(libfile2)
146:                         fileexists = True
147:                         break
148:                 if fileexists: continue
149:                 log.warn('could not find library %r in directories %s' \
150:                          % (libname, library_dirs))
151:         elif self.compiler.compiler_type == 'mingw32':
152:             generate_manifest(self)
153:         return self._wrap_method(old_config._link, lang,
154:                                  (body, headers, include_dirs,
155:                                   libraries, library_dirs, lang))
156: 
157:     def check_header(self, header, include_dirs=None, library_dirs=None, lang='c'):
158:         self._check_compiler()
159:         return self.try_compile(
160:                 "/* we need a dummy line to make distutils happy */",
161:                 [header], include_dirs)
162: 
163:     def check_decl(self, symbol,
164:                    headers=None, include_dirs=None):
165:         self._check_compiler()
166:         body = '''
167: int main(void)
168: {
169: #ifndef %s
170:     (void) %s;
171: #endif
172:     ;
173:     return 0;
174: }''' % (symbol, symbol)
175: 
176:         return self.try_compile(body, headers, include_dirs)
177: 
178:     def check_macro_true(self, symbol,
179:                          headers=None, include_dirs=None):
180:         self._check_compiler()
181:         body = '''
182: int main(void)
183: {
184: #if %s
185: #else
186: #error false or undefined macro
187: #endif
188:     ;
189:     return 0;
190: }''' % (symbol,)
191: 
192:         return self.try_compile(body, headers, include_dirs)
193: 
194:     def check_type(self, type_name, headers=None, include_dirs=None,
195:             library_dirs=None):
196:         '''Check type availability. Return True if the type can be compiled,
197:         False otherwise'''
198:         self._check_compiler()
199: 
200:         # First check the type can be compiled
201:         body = r'''
202: int main(void) {
203:   if ((%(name)s *) 0)
204:     return 0;
205:   if (sizeof (%(name)s))
206:     return 0;
207: }
208: ''' % {'name': type_name}
209: 
210:         st = False
211:         try:
212:             try:
213:                 self._compile(body % {'type': type_name},
214:                         headers, include_dirs, 'c')
215:                 st = True
216:             except distutils.errors.CompileError:
217:                 st = False
218:         finally:
219:             self._clean()
220: 
221:         return st
222: 
223:     def check_type_size(self, type_name, headers=None, include_dirs=None, library_dirs=None, expected=None):
224:         '''Check size of a given type.'''
225:         self._check_compiler()
226: 
227:         # First check the type can be compiled
228:         body = r'''
229: typedef %(type)s npy_check_sizeof_type;
230: int main (void)
231: {
232:     static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];
233:     test_array [0] = 0
234: 
235:     ;
236:     return 0;
237: }
238: '''
239:         self._compile(body % {'type': type_name},
240:                 headers, include_dirs, 'c')
241:         self._clean()
242: 
243:         if expected:
244:             body = r'''
245: typedef %(type)s npy_check_sizeof_type;
246: int main (void)
247: {
248:     static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)s)];
249:     test_array [0] = 0
250: 
251:     ;
252:     return 0;
253: }
254: '''
255:             for size in expected:
256:                 try:
257:                     self._compile(body % {'type': type_name, 'size': size},
258:                             headers, include_dirs, 'c')
259:                     self._clean()
260:                     return size
261:                 except CompileError:
262:                     pass
263: 
264:         # this fails to *compile* if size > sizeof(type)
265:         body = r'''
266: typedef %(type)s npy_check_sizeof_type;
267: int main (void)
268: {
269:     static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)s)];
270:     test_array [0] = 0
271: 
272:     ;
273:     return 0;
274: }
275: '''
276: 
277:         # The principle is simple: we first find low and high bounds of size
278:         # for the type, where low/high are looked up on a log scale. Then, we
279:         # do a binary search to find the exact size between low and high
280:         low = 0
281:         mid = 0
282:         while True:
283:             try:
284:                 self._compile(body % {'type': type_name, 'size': mid},
285:                         headers, include_dirs, 'c')
286:                 self._clean()
287:                 break
288:             except CompileError:
289:                 #log.info("failure to test for bound %d" % mid)
290:                 low = mid + 1
291:                 mid = 2 * mid + 1
292: 
293:         high = mid
294:         # Binary search:
295:         while low != high:
296:             mid = (high - low) // 2 + low
297:             try:
298:                 self._compile(body % {'type': type_name, 'size': mid},
299:                         headers, include_dirs, 'c')
300:                 self._clean()
301:                 high = mid
302:             except CompileError:
303:                 low = mid + 1
304:         return low
305: 
306:     def check_func(self, func,
307:                    headers=None, include_dirs=None,
308:                    libraries=None, library_dirs=None,
309:                    decl=False, call=False, call_args=None):
310:         # clean up distutils's config a bit: add void to main(), and
311:         # return a value.
312:         self._check_compiler()
313:         body = []
314:         if decl:
315:             if type(decl) == str:
316:                 body.append(decl)
317:             else:
318:                 body.append("int %s (void);" % func)
319:         # Handle MSVC intrinsics: force MS compiler to make a function call.
320:         # Useful to test for some functions when built with optimization on, to
321:         # avoid build error because the intrinsic and our 'fake' test
322:         # declaration do not match.
323:         body.append("#ifdef _MSC_VER")
324:         body.append("#pragma function(%s)" % func)
325:         body.append("#endif")
326:         body.append("int main (void) {")
327:         if call:
328:             if call_args is None:
329:                 call_args = ''
330:             body.append("  %s(%s);" % (func, call_args))
331:         else:
332:             body.append("  %s;" % func)
333:         body.append("  return 0;")
334:         body.append("}")
335:         body = '\n'.join(body) + "\n"
336: 
337:         return self.try_link(body, headers, include_dirs,
338:                              libraries, library_dirs)
339: 
340:     def check_funcs_once(self, funcs,
341:                    headers=None, include_dirs=None,
342:                    libraries=None, library_dirs=None,
343:                    decl=False, call=False, call_args=None):
344:         '''Check a list of functions at once.
345: 
346:         This is useful to speed up things, since all the functions in the funcs
347:         list will be put in one compilation unit.
348: 
349:         Arguments
350:         ---------
351:         funcs : seq
352:             list of functions to test
353:         include_dirs : seq
354:             list of header paths
355:         libraries : seq
356:             list of libraries to link the code snippet to
357:         libraru_dirs : seq
358:             list of library paths
359:         decl : dict
360:             for every (key, value), the declaration in the value will be
361:             used for function in key. If a function is not in the
362:             dictionay, no declaration will be used.
363:         call : dict
364:             for every item (f, value), if the value is True, a call will be
365:             done to the function f.
366:         '''
367:         self._check_compiler()
368:         body = []
369:         if decl:
370:             for f, v in decl.items():
371:                 if v:
372:                     body.append("int %s (void);" % f)
373: 
374:         # Handle MS intrinsics. See check_func for more info.
375:         body.append("#ifdef _MSC_VER")
376:         for func in funcs:
377:             body.append("#pragma function(%s)" % func)
378:         body.append("#endif")
379: 
380:         body.append("int main (void) {")
381:         if call:
382:             for f in funcs:
383:                 if f in call and call[f]:
384:                     if not (call_args and f in call_args and call_args[f]):
385:                         args = ''
386:                     else:
387:                         args = call_args[f]
388:                     body.append("  %s(%s);" % (f, args))
389:                 else:
390:                     body.append("  %s;" % f)
391:         else:
392:             for f in funcs:
393:                 body.append("  %s;" % f)
394:         body.append("  return 0;")
395:         body.append("}")
396:         body = '\n'.join(body) + "\n"
397: 
398:         return self.try_link(body, headers, include_dirs,
399:                              libraries, library_dirs)
400: 
401:     def check_inline(self):
402:         '''Return the inline keyword recognized by the compiler, empty string
403:         otherwise.'''
404:         return check_inline(self)
405: 
406:     def check_restrict(self):
407:         '''Return the restrict keyword recognized by the compiler, empty string
408:         otherwise.'''
409:         return check_restrict(self)
410: 
411:     def check_compiler_gcc4(self):
412:         '''Return True if the C compiler is gcc >= 4.'''
413:         return check_compiler_gcc4(self)
414: 
415:     def check_gcc_function_attribute(self, attribute, name):
416:         return check_gcc_function_attribute(self, attribute, name)
417: 
418:     def check_gcc_variable_attribute(self, attribute):
419:         return check_gcc_variable_attribute(self, attribute)
420: 
421: 
422: class GrabStdout(object):
423: 
424:     def __init__(self):
425:         self.sys_stdout = sys.stdout
426:         self.data = ''
427:         sys.stdout = self
428: 
429:     def write (self, data):
430:         self.sys_stdout.write(data)
431:         self.data += data
432: 
433:     def flush (self):
434:         self.sys_stdout.flush()
435: 
436:     def restore(self):
437:         sys.stdout = self.sys_stdout
438: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# Multiple import statement. import os (1/2) (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)
# Multiple import statement. import signal (2/2) (line 7)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'signal', signal, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.command.config import old_config' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58054 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.config')

if (type(import_58054) is not StypyTypeError):

    if (import_58054 != 'pyd_module'):
        __import__(import_58054)
        sys_modules_58055 = sys.modules[import_58054]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.config', sys_modules_58055.module_type_store, module_type_store, ['config'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_58055, sys_modules_58055.module_type_store, module_type_store)
    else:
        from distutils.command.config import config as old_config

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.config', None, module_type_store, ['config'], [old_config])

else:
    # Assigning a type to the variable 'distutils.command.config' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.config', import_58054)

# Adding an alias
module_type_store.add_alias('old_config', 'config')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.command.config import LANG_EXT' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58056 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.config')

if (type(import_58056) is not StypyTypeError):

    if (import_58056 != 'pyd_module'):
        __import__(import_58056)
        sys_modules_58057 = sys.modules[import_58056]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.config', sys_modules_58057.module_type_store, module_type_store, ['LANG_EXT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_58057, sys_modules_58057.module_type_store, module_type_store)
    else:
        from distutils.command.config import LANG_EXT

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.config', None, module_type_store, ['LANG_EXT'], [LANG_EXT])

else:
    # Assigning a type to the variable 'distutils.command.config' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.config', import_58056)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils import log' statement (line 13)
from distutils import log

import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.file_util import copy_file' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util')

if (type(import_58058) is not StypyTypeError):

    if (import_58058 != 'pyd_module'):
        __import__(import_58058)
        sys_modules_58059 = sys.modules[import_58058]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', sys_modules_58059.module_type_store, module_type_store, ['copy_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_58059, sys_modules_58059.module_type_store, module_type_store)
    else:
        from distutils.file_util import copy_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', None, module_type_store, ['copy_file'], [copy_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', import_58058)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.ccompiler import CompileError, LinkError' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58060 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.ccompiler')

if (type(import_58060) is not StypyTypeError):

    if (import_58060 != 'pyd_module'):
        __import__(import_58060)
        sys_modules_58061 = sys.modules[import_58060]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.ccompiler', sys_modules_58061.module_type_store, module_type_store, ['CompileError', 'LinkError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_58061, sys_modules_58061.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import CompileError, LinkError

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.ccompiler', None, module_type_store, ['CompileError', 'LinkError'], [CompileError, LinkError])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.ccompiler', import_58060)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import distutils' statement (line 16)
import distutils

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils', distutils, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from numpy.distutils.exec_command import exec_command' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58062 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command')

if (type(import_58062) is not StypyTypeError):

    if (import_58062 != 'pyd_module'):
        __import__(import_58062)
        sys_modules_58063 = sys.modules[import_58062]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', sys_modules_58063.module_type_store, module_type_store, ['exec_command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_58063, sys_modules_58063.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import exec_command

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', None, module_type_store, ['exec_command'], [exec_command])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', import_58062)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.distutils.mingw32ccompiler import generate_manifest' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58064 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.mingw32ccompiler')

if (type(import_58064) is not StypyTypeError):

    if (import_58064 != 'pyd_module'):
        __import__(import_58064)
        sys_modules_58065 = sys.modules[import_58064]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.mingw32ccompiler', sys_modules_58065.module_type_store, module_type_store, ['generate_manifest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_58065, sys_modules_58065.module_type_store, module_type_store)
    else:
        from numpy.distutils.mingw32ccompiler import generate_manifest

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.mingw32ccompiler', None, module_type_store, ['generate_manifest'], [generate_manifest])

else:
    # Assigning a type to the variable 'numpy.distutils.mingw32ccompiler' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.mingw32ccompiler', import_58064)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy.distutils.command.autodist import check_gcc_function_attribute, check_gcc_variable_attribute, check_inline, check_restrict, check_compiler_gcc4' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.command.autodist')

if (type(import_58066) is not StypyTypeError):

    if (import_58066 != 'pyd_module'):
        __import__(import_58066)
        sys_modules_58067 = sys.modules[import_58066]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.command.autodist', sys_modules_58067.module_type_store, module_type_store, ['check_gcc_function_attribute', 'check_gcc_variable_attribute', 'check_inline', 'check_restrict', 'check_compiler_gcc4'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_58067, sys_modules_58067.module_type_store, module_type_store)
    else:
        from numpy.distutils.command.autodist import check_gcc_function_attribute, check_gcc_variable_attribute, check_inline, check_restrict, check_compiler_gcc4

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.command.autodist', None, module_type_store, ['check_gcc_function_attribute', 'check_gcc_variable_attribute', 'check_inline', 'check_restrict', 'check_compiler_gcc4'], [check_gcc_function_attribute, check_gcc_variable_attribute, check_inline, check_restrict, check_compiler_gcc4])

else:
    # Assigning a type to the variable 'numpy.distutils.command.autodist' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.command.autodist', import_58066)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.compat')

if (type(import_58068) is not StypyTypeError):

    if (import_58068 != 'pyd_module'):
        __import__(import_58068)
        sys_modules_58069 = sys.modules[import_58068]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.compat', sys_modules_58069.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_58069, sys_modules_58069.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.compat', import_58068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


# Assigning a Str to a Subscript (line 26):

# Assigning a Str to a Subscript (line 26):
str_58070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'str', '.f')
# Getting the type of 'LANG_EXT' (line 26)
LANG_EXT_58071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'LANG_EXT')
str_58072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'str', 'f77')
# Storing an element on a container (line 26)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 0), LANG_EXT_58071, (str_58072, str_58070))

# Assigning a Str to a Subscript (line 27):

# Assigning a Str to a Subscript (line 27):
str_58073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'str', '.f90')
# Getting the type of 'LANG_EXT' (line 27)
LANG_EXT_58074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'LANG_EXT')
str_58075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'str', 'f90')
# Storing an element on a container (line 27)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 0), LANG_EXT_58074, (str_58075, str_58073))
# Declaration of the 'config' class
# Getting the type of 'old_config' (line 29)
old_config_58076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'old_config')

class config(old_config_58076, ):
    
    # Getting the type of 'old_config' (line 30)
    old_config_58077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'old_config')
    # Obtaining the member 'user_options' of a type (line 30)
    user_options_58078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), old_config_58077, 'user_options')
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_58079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_58080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    str_58081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'fcompiler=')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_58080, str_58081)
    # Adding element type (line 31)
    # Getting the type of 'None' (line 31)
    None_58082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_58080, None_58082)
    # Adding element type (line 31)
    str_58083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'str', 'specify the Fortran compiler type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_58080, str_58083)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 31), list_58079, tuple_58080)
    
    # Applying the binary operator '+=' (line 30)
    result_iadd_58084 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 4), '+=', user_options_58078, list_58079)
    # Getting the type of 'old_config' (line 30)
    old_config_58085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'old_config')
    # Setting the type of the member 'user_options' of a type (line 30)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), old_config_58085, 'user_options', result_iadd_58084)
    

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        config.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.initialize_options.__dict__.__setitem__('stypy_function_name', 'config.initialize_options')
        config.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        config.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'None' (line 35)
        None_58086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'None')
        # Getting the type of 'self' (line 35)
        self_58087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'fcompiler' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_58087, 'fcompiler', None_58086)
        
        # Call to initialize_options(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_58090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'self', False)
        # Processing the call keyword arguments (line 36)
        kwargs_58091 = {}
        # Getting the type of 'old_config' (line 36)
        old_config_58088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'old_config', False)
        # Obtaining the member 'initialize_options' of a type (line 36)
        initialize_options_58089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), old_config_58088, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 36)
        initialize_options_call_result_58092 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), initialize_options_58089, *[self_58090], **kwargs_58091)
        
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_58093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58093)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_58093


    @norecursion
    def _check_compiler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_compiler'
        module_type_store = module_type_store.open_function_context('_check_compiler', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._check_compiler.__dict__.__setitem__('stypy_localization', localization)
        config._check_compiler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._check_compiler.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._check_compiler.__dict__.__setitem__('stypy_function_name', 'config._check_compiler')
        config._check_compiler.__dict__.__setitem__('stypy_param_names_list', [])
        config._check_compiler.__dict__.__setitem__('stypy_varargs_param_name', None)
        config._check_compiler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._check_compiler.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._check_compiler.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._check_compiler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._check_compiler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._check_compiler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_compiler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_compiler(...)' code ##################

        
        # Call to _check_compiler(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_58096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 35), 'self', False)
        # Processing the call keyword arguments (line 39)
        kwargs_58097 = {}
        # Getting the type of 'old_config' (line 39)
        old_config_58094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'old_config', False)
        # Obtaining the member '_check_compiler' of a type (line 39)
        _check_compiler_58095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), old_config_58094, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 39)
        _check_compiler_call_result_58098 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), _check_compiler_58095, *[self_58096], **kwargs_58097)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 8))
        
        # 'from numpy.distutils.fcompiler import FCompiler, new_fcompiler' statement (line 40)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_58099 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.distutils.fcompiler')

        if (type(import_58099) is not StypyTypeError):

            if (import_58099 != 'pyd_module'):
                __import__(import_58099)
                sys_modules_58100 = sys.modules[import_58099]
                import_from_module(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.distutils.fcompiler', sys_modules_58100.module_type_store, module_type_store, ['FCompiler', 'new_fcompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 40, 8), __file__, sys_modules_58100, sys_modules_58100.module_type_store, module_type_store)
            else:
                from numpy.distutils.fcompiler import FCompiler, new_fcompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler', 'new_fcompiler'], [FCompiler, new_fcompiler])

        else:
            # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.distutils.fcompiler', import_58099)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sys' (line 42)
        sys_58101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 42)
        platform_58102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), sys_58101, 'platform')
        str_58103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 42)
        result_eq_58104 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), '==', platform_58102, str_58103)
        
        
        # Getting the type of 'self' (line 42)
        self_58105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'self')
        # Obtaining the member 'compiler' of a type (line 42)
        compiler_58106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 40), self_58105, 'compiler')
        # Obtaining the member 'compiler_type' of a type (line 42)
        compiler_type_58107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 40), compiler_58106, 'compiler_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_58108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        str_58109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'str', 'msvc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 41), tuple_58108, str_58109)
        # Adding element type (line 43)
        str_58110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 49), 'str', 'intelw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 41), tuple_58108, str_58110)
        # Adding element type (line 43)
        str_58111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 59), 'str', 'intelemw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 41), tuple_58108, str_58111)
        
        # Applying the binary operator 'in' (line 42)
        result_contains_58112 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 40), 'in', compiler_type_58107, tuple_58108)
        
        # Applying the binary operator 'and' (line 42)
        result_and_keyword_58113 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), 'and', result_eq_58104, result_contains_58112)
        
        # Testing the type of an if condition (line 42)
        if_condition_58114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_and_keyword_58113)
        # Assigning a type to the variable 'if_condition_58114' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_58114', if_condition_58114)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 49)
        self_58115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'self')
        # Obtaining the member 'compiler' of a type (line 49)
        compiler_58116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), self_58115, 'compiler')
        # Obtaining the member 'initialized' of a type (line 49)
        initialized_58117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), compiler_58116, 'initialized')
        # Applying the 'not' unary operator (line 49)
        result_not__58118 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), 'not', initialized_58117)
        
        # Testing the type of an if condition (line 49)
        if_condition_58119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 12), result_not__58118)
        # Assigning a type to the variable 'if_condition_58119' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'if_condition_58119', if_condition_58119)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to initialize(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_58123 = {}
        # Getting the type of 'self' (line 51)
        self_58120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'self', False)
        # Obtaining the member 'compiler' of a type (line 51)
        compiler_58121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), self_58120, 'compiler')
        # Obtaining the member 'initialize' of a type (line 51)
        initialize_58122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), compiler_58121, 'initialize')
        # Calling initialize(args, kwargs) (line 51)
        initialize_call_result_58124 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), initialize_58122, *[], **kwargs_58123)
        
        # SSA branch for the except part of a try statement (line 50)
        # SSA branch for the except 'IOError' branch of a try statement (line 50)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to get_exception(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_58126 = {}
        # Getting the type of 'get_exception' (line 53)
        get_exception_58125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'get_exception', False)
        # Calling get_exception(args, kwargs) (line 53)
        get_exception_call_result_58127 = invoke(stypy.reporting.localization.Localization(__file__, 53, 24), get_exception_58125, *[], **kwargs_58126)
        
        # Assigning a type to the variable 'e' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'e', get_exception_call_result_58127)
        
        # Assigning a BinOp to a Name (line 54):
        
        # Assigning a BinOp to a Name (line 54):
        str_58128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', 'Could not initialize compiler instance: do you have Visual Studio\ninstalled?  If you are trying to build with MinGW, please use "python setup.py\nbuild -c mingw32" instead.  If you have Visual Studio installed, check it is\ncorrectly installed, and the right version (VS 2008 for python 2.6, 2.7 and 3.2,\nVS 2010 for >= 3.3).\n\nOriginal exception was: %s, and the Compiler class was %s\n============================================================================')
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_58129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'e' (line 63)
        e_58130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'e')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 27), tuple_58129, e_58130)
        # Adding element type (line 63)
        # Getting the type of 'self' (line 63)
        self_58131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'self')
        # Obtaining the member 'compiler' of a type (line 63)
        compiler_58132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), self_58131, 'compiler')
        # Obtaining the member '__class__' of a type (line 63)
        class___58133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), compiler_58132, '__class__')
        # Obtaining the member '__name__' of a type (line 63)
        name___58134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), class___58133, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 27), tuple_58129, name___58134)
        
        # Applying the binary operator '%' (line 62)
        result_mod_58135 = python_operator(stypy.reporting.localization.Localization(__file__, 62, (-1)), '%', str_58128, tuple_58129)
        
        # Assigning a type to the variable 'msg' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'msg', result_mod_58135)
        
        # Call to print(...): (line 64)
        # Processing the call arguments (line 64)
        str_58137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '============================================================================')
        # Processing the call keyword arguments (line 64)
        kwargs_58138 = {}
        # Getting the type of 'print' (line 64)
        print_58136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'print', False)
        # Calling print(args, kwargs) (line 64)
        print_call_result_58139 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), print_58136, *[str_58137], **kwargs_58138)
        
        
        # Call to DistutilsPlatformError(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'msg' (line 66)
        msg_58143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 66), 'msg', False)
        # Processing the call keyword arguments (line 66)
        kwargs_58144 = {}
        # Getting the type of 'distutils' (line 66)
        distutils_58140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'distutils', False)
        # Obtaining the member 'errors' of a type (line 66)
        errors_58141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 26), distutils_58140, 'errors')
        # Obtaining the member 'DistutilsPlatformError' of a type (line 66)
        DistutilsPlatformError_58142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 26), errors_58141, 'DistutilsPlatformError')
        # Calling DistutilsPlatformError(args, kwargs) (line 66)
        DistutilsPlatformError_call_result_58145 = invoke(stypy.reporting.localization.Localization(__file__, 66, 26), DistutilsPlatformError_58142, *[msg_58143], **kwargs_58144)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 66, 20), DistutilsPlatformError_call_result_58145, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 71, 12))
        
        # 'from distutils import msvc9compiler' statement (line 71)
        from distutils import msvc9compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 71, 12), 'distutils', None, module_type_store, ['msvc9compiler'], [msvc9compiler])
        
        
        
        
        # Call to get_build_version(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_58148 = {}
        # Getting the type of 'msvc9compiler' (line 72)
        msvc9compiler_58146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'msvc9compiler', False)
        # Obtaining the member 'get_build_version' of a type (line 72)
        get_build_version_58147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), msvc9compiler_58146, 'get_build_version')
        # Calling get_build_version(args, kwargs) (line 72)
        get_build_version_call_result_58149 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), get_build_version_58147, *[], **kwargs_58148)
        
        int_58150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 52), 'int')
        # Applying the binary operator '>=' (line 72)
        result_ge_58151 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 15), '>=', get_build_version_call_result_58149, int_58150)
        
        # Testing the type of an if condition (line 72)
        if_condition_58152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 12), result_ge_58151)
        # Assigning a type to the variable 'if_condition_58152' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'if_condition_58152', if_condition_58152)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_58153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        # Getting the type of 'self' (line 73)
        self_58154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'self')
        # Obtaining the member 'compiler' of a type (line 73)
        compiler_58155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 32), self_58154, 'compiler')
        # Obtaining the member 'ldflags_shared' of a type (line 73)
        ldflags_shared_58156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 32), compiler_58155, 'ldflags_shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 31), list_58153, ldflags_shared_58156)
        # Adding element type (line 73)
        # Getting the type of 'self' (line 74)
        self_58157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'self')
        # Obtaining the member 'compiler' of a type (line 74)
        compiler_58158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 32), self_58157, 'compiler')
        # Obtaining the member 'ldflags_shared_debug' of a type (line 74)
        ldflags_shared_debug_58159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 32), compiler_58158, 'ldflags_shared_debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 31), list_58153, ldflags_shared_debug_58159)
        
        # Testing the type of a for loop iterable (line 73)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 16), list_58153)
        # Getting the type of the for loop variable (line 73)
        for_loop_var_58160 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 16), list_58153)
        # Assigning a type to the variable 'ldflags' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'ldflags', for_loop_var_58160)
        # SSA begins for a for statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        str_58161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'str', '/MANIFEST')
        # Getting the type of 'ldflags' (line 75)
        ldflags_58162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'ldflags')
        # Applying the binary operator 'notin' (line 75)
        result_contains_58163 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 23), 'notin', str_58161, ldflags_58162)
        
        # Testing the type of an if condition (line 75)
        if_condition_58164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 20), result_contains_58163)
        # Assigning a type to the variable 'if_condition_58164' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'if_condition_58164', if_condition_58164)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 76)
        # Processing the call arguments (line 76)
        str_58167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'str', '/MANIFEST')
        # Processing the call keyword arguments (line 76)
        kwargs_58168 = {}
        # Getting the type of 'ldflags' (line 76)
        ldflags_58165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'ldflags', False)
        # Obtaining the member 'append' of a type (line 76)
        append_58166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), ldflags_58165, 'append')
        # Calling append(args, kwargs) (line 76)
        append_call_result_58169 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), append_58166, *[str_58167], **kwargs_58168)
        
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isinstance(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'self' (line 78)
        self_58171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 78)
        fcompiler_58172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 26), self_58171, 'fcompiler')
        # Getting the type of 'FCompiler' (line 78)
        FCompiler_58173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 42), 'FCompiler', False)
        # Processing the call keyword arguments (line 78)
        kwargs_58174 = {}
        # Getting the type of 'isinstance' (line 78)
        isinstance_58170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 78)
        isinstance_call_result_58175 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), isinstance_58170, *[fcompiler_58172, FCompiler_58173], **kwargs_58174)
        
        # Applying the 'not' unary operator (line 78)
        result_not__58176 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'not', isinstance_call_result_58175)
        
        # Testing the type of an if condition (line 78)
        if_condition_58177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_not__58176)
        # Assigning a type to the variable 'if_condition_58177' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_58177', if_condition_58177)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 79):
        
        # Assigning a Call to a Attribute (line 79):
        
        # Call to new_fcompiler(...): (line 79)
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_58179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 52), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 79)
        fcompiler_58180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 52), self_58179, 'fcompiler')
        keyword_58181 = fcompiler_58180
        # Getting the type of 'self' (line 80)
        self_58182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 51), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 80)
        dry_run_58183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 51), self_58182, 'dry_run')
        keyword_58184 = dry_run_58183
        int_58185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 71), 'int')
        keyword_58186 = int_58185
        # Getting the type of 'self' (line 81)
        self_58187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 54), 'self', False)
        # Obtaining the member 'compiler' of a type (line 81)
        compiler_58188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 54), self_58187, 'compiler')
        keyword_58189 = compiler_58188
        kwargs_58190 = {'c_compiler': keyword_58189, 'force': keyword_58186, 'dry_run': keyword_58184, 'compiler': keyword_58181}
        # Getting the type of 'new_fcompiler' (line 79)
        new_fcompiler_58178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'new_fcompiler', False)
        # Calling new_fcompiler(args, kwargs) (line 79)
        new_fcompiler_call_result_58191 = invoke(stypy.reporting.localization.Localization(__file__, 79, 29), new_fcompiler_58178, *[], **kwargs_58190)
        
        # Getting the type of 'self' (line 79)
        self_58192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self')
        # Setting the type of the member 'fcompiler' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_58192, 'fcompiler', new_fcompiler_call_result_58191)
        
        
        # Getting the type of 'self' (line 82)
        self_58193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self')
        # Obtaining the member 'fcompiler' of a type (line 82)
        fcompiler_58194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_58193, 'fcompiler')
        # Getting the type of 'None' (line 82)
        None_58195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'None')
        # Applying the binary operator 'isnot' (line 82)
        result_is_not_58196 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), 'isnot', fcompiler_58194, None_58195)
        
        # Testing the type of an if condition (line 82)
        if_condition_58197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), result_is_not_58196)
        # Assigning a type to the variable 'if_condition_58197' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_58197', if_condition_58197)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to customize(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_58201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'self', False)
        # Obtaining the member 'distribution' of a type (line 83)
        distribution_58202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 41), self_58201, 'distribution')
        # Processing the call keyword arguments (line 83)
        kwargs_58203 = {}
        # Getting the type of 'self' (line 83)
        self_58198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 83)
        fcompiler_58199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), self_58198, 'fcompiler')
        # Obtaining the member 'customize' of a type (line 83)
        customize_58200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), fcompiler_58199, 'customize')
        # Calling customize(args, kwargs) (line 83)
        customize_call_result_58204 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), customize_58200, *[distribution_58202], **kwargs_58203)
        
        
        
        # Call to get_version(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_58208 = {}
        # Getting the type of 'self' (line 84)
        self_58205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 84)
        fcompiler_58206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), self_58205, 'fcompiler')
        # Obtaining the member 'get_version' of a type (line 84)
        get_version_58207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), fcompiler_58206, 'get_version')
        # Calling get_version(args, kwargs) (line 84)
        get_version_call_result_58209 = invoke(stypy.reporting.localization.Localization(__file__, 84, 19), get_version_58207, *[], **kwargs_58208)
        
        # Testing the type of an if condition (line 84)
        if_condition_58210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 16), get_version_call_result_58209)
        # Assigning a type to the variable 'if_condition_58210' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'if_condition_58210', if_condition_58210)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to customize_cmd(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_58214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'self', False)
        # Processing the call keyword arguments (line 85)
        kwargs_58215 = {}
        # Getting the type of 'self' (line 85)
        self_58211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 85)
        fcompiler_58212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), self_58211, 'fcompiler')
        # Obtaining the member 'customize_cmd' of a type (line 85)
        customize_cmd_58213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), fcompiler_58212, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 85)
        customize_cmd_call_result_58216 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), customize_cmd_58213, *[self_58214], **kwargs_58215)
        
        
        # Call to show_customization(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_58220 = {}
        # Getting the type of 'self' (line 86)
        self_58217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 86)
        fcompiler_58218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), self_58217, 'fcompiler')
        # Obtaining the member 'show_customization' of a type (line 86)
        show_customization_58219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), fcompiler_58218, 'show_customization')
        # Calling show_customization(args, kwargs) (line 86)
        show_customization_call_result_58221 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), show_customization_58219, *[], **kwargs_58220)
        
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_compiler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_compiler' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_58222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_compiler'
        return stypy_return_type_58222


    @norecursion
    def _wrap_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_wrap_method'
        module_type_store = module_type_store.open_function_context('_wrap_method', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._wrap_method.__dict__.__setitem__('stypy_localization', localization)
        config._wrap_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._wrap_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._wrap_method.__dict__.__setitem__('stypy_function_name', 'config._wrap_method')
        config._wrap_method.__dict__.__setitem__('stypy_param_names_list', ['mth', 'lang', 'args'])
        config._wrap_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        config._wrap_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._wrap_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._wrap_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._wrap_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._wrap_method.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._wrap_method', ['mth', 'lang', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_wrap_method', localization, ['mth', 'lang', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_wrap_method(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 89, 8))
        
        # 'from distutils.ccompiler import CompileError' statement (line 89)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_58223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 89, 8), 'distutils.ccompiler')

        if (type(import_58223) is not StypyTypeError):

            if (import_58223 != 'pyd_module'):
                __import__(import_58223)
                sys_modules_58224 = sys.modules[import_58223]
                import_from_module(stypy.reporting.localization.Localization(__file__, 89, 8), 'distutils.ccompiler', sys_modules_58224.module_type_store, module_type_store, ['CompileError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 89, 8), __file__, sys_modules_58224, sys_modules_58224.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import CompileError

                import_from_module(stypy.reporting.localization.Localization(__file__, 89, 8), 'distutils.ccompiler', None, module_type_store, ['CompileError'], [CompileError])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'distutils.ccompiler', import_58223)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 90, 8))
        
        # 'from distutils.errors import DistutilsExecError' statement (line 90)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_58225 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 90, 8), 'distutils.errors')

        if (type(import_58225) is not StypyTypeError):

            if (import_58225 != 'pyd_module'):
                __import__(import_58225)
                sys_modules_58226 = sys.modules[import_58225]
                import_from_module(stypy.reporting.localization.Localization(__file__, 90, 8), 'distutils.errors', sys_modules_58226.module_type_store, module_type_store, ['DistutilsExecError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 90, 8), __file__, sys_modules_58226, sys_modules_58226.module_type_store, module_type_store)
            else:
                from distutils.errors import DistutilsExecError

                import_from_module(stypy.reporting.localization.Localization(__file__, 90, 8), 'distutils.errors', None, module_type_store, ['DistutilsExecError'], [DistutilsExecError])

        else:
            # Assigning a type to the variable 'distutils.errors' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'distutils.errors', import_58225)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a Attribute to a Name (line 91):
        
        # Assigning a Attribute to a Name (line 91):
        # Getting the type of 'self' (line 91)
        self_58227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'self')
        # Obtaining the member 'compiler' of a type (line 91)
        compiler_58228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), self_58227, 'compiler')
        # Assigning a type to the variable 'save_compiler' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'save_compiler', compiler_58228)
        
        
        # Getting the type of 'lang' (line 92)
        lang_58229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'lang')
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_58230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        str_58231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'str', 'f77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 19), list_58230, str_58231)
        # Adding element type (line 92)
        str_58232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 27), 'str', 'f90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 19), list_58230, str_58232)
        
        # Applying the binary operator 'in' (line 92)
        result_contains_58233 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), 'in', lang_58229, list_58230)
        
        # Testing the type of an if condition (line 92)
        if_condition_58234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_contains_58233)
        # Assigning a type to the variable 'if_condition_58234' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_58234', if_condition_58234)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 93):
        
        # Assigning a Attribute to a Attribute (line 93):
        # Getting the type of 'self' (line 93)
        self_58235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'self')
        # Obtaining the member 'fcompiler' of a type (line 93)
        fcompiler_58236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), self_58235, 'fcompiler')
        # Getting the type of 'self' (line 93)
        self_58237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
        # Setting the type of the member 'compiler' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_58237, 'compiler', fcompiler_58236)
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to mth(...): (line 95)
        
        # Obtaining an instance of the builtin type 'tuple' (line 95)
        tuple_58239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 95)
        # Adding element type (line 95)
        # Getting the type of 'self' (line 95)
        self_58240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'self', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 25), tuple_58239, self_58240)
        
        # Getting the type of 'args' (line 95)
        args_58241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'args', False)
        # Applying the binary operator '+' (line 95)
        result_add_58242 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 24), '+', tuple_58239, args_58241)
        
        # Processing the call keyword arguments (line 95)
        kwargs_58243 = {}
        # Getting the type of 'mth' (line 95)
        mth_58238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'mth', False)
        # Calling mth(args, kwargs) (line 95)
        mth_call_result_58244 = invoke(stypy.reporting.localization.Localization(__file__, 95, 18), mth_58238, *[result_add_58242], **kwargs_58243)
        
        # Assigning a type to the variable 'ret' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'ret', mth_call_result_58244)
        # SSA branch for the except part of a try statement (line 94)
        # SSA branch for the except 'Tuple' branch of a try statement (line 94)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to str(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to get_exception(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_58247 = {}
        # Getting the type of 'get_exception' (line 97)
        get_exception_58246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'get_exception', False)
        # Calling get_exception(args, kwargs) (line 97)
        get_exception_call_result_58248 = invoke(stypy.reporting.localization.Localization(__file__, 97, 22), get_exception_58246, *[], **kwargs_58247)
        
        # Processing the call keyword arguments (line 97)
        kwargs_58249 = {}
        # Getting the type of 'str' (line 97)
        str_58245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'str', False)
        # Calling str(args, kwargs) (line 97)
        str_call_result_58250 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), str_58245, *[get_exception_call_result_58248], **kwargs_58249)
        
        # Assigning a type to the variable 'msg' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'msg', str_call_result_58250)
        
        # Assigning a Name to a Attribute (line 98):
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'save_compiler' (line 98)
        save_compiler_58251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'save_compiler')
        # Getting the type of 'self' (line 98)
        self_58252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Setting the type of the member 'compiler' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_58252, 'compiler', save_compiler_58251)
        # Getting the type of 'CompileError' (line 99)
        CompileError_58253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 12), CompileError_58253, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 100):
        
        # Assigning a Name to a Attribute (line 100):
        # Getting the type of 'save_compiler' (line 100)
        save_compiler_58254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'save_compiler')
        # Getting the type of 'self' (line 100)
        self_58255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_58255, 'compiler', save_compiler_58254)
        # Getting the type of 'ret' (line 101)
        ret_58256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', ret_58256)
        
        # ################# End of '_wrap_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_wrap_method' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_58257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_wrap_method'
        return stypy_return_type_58257


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._compile.__dict__.__setitem__('stypy_localization', localization)
        config._compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._compile.__dict__.__setitem__('stypy_function_name', 'config._compile')
        config._compile.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'lang'])
        config._compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        config._compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._compile.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._compile', ['body', 'headers', 'include_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compile', localization, ['body', 'headers', 'include_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compile(...)' code ##################

        
        # Call to _wrap_method(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'old_config' (line 104)
        old_config_58260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'old_config', False)
        # Obtaining the member '_compile' of a type (line 104)
        _compile_58261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 33), old_config_58260, '_compile')
        # Getting the type of 'lang' (line 104)
        lang_58262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 54), 'lang', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_58263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        # Getting the type of 'body' (line 105)
        body_58264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'body', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), tuple_58263, body_58264)
        # Adding element type (line 105)
        # Getting the type of 'headers' (line 105)
        headers_58265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'headers', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), tuple_58263, headers_58265)
        # Adding element type (line 105)
        # Getting the type of 'include_dirs' (line 105)
        include_dirs_58266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'include_dirs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), tuple_58263, include_dirs_58266)
        # Adding element type (line 105)
        # Getting the type of 'lang' (line 105)
        lang_58267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 63), 'lang', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), tuple_58263, lang_58267)
        
        # Processing the call keyword arguments (line 104)
        kwargs_58268 = {}
        # Getting the type of 'self' (line 104)
        self_58258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'self', False)
        # Obtaining the member '_wrap_method' of a type (line 104)
        _wrap_method_58259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), self_58258, '_wrap_method')
        # Calling _wrap_method(args, kwargs) (line 104)
        _wrap_method_call_result_58269 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), _wrap_method_58259, *[_compile_58261, lang_58262, tuple_58263], **kwargs_58268)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', _wrap_method_call_result_58269)
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_58270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_58270


    @norecursion
    def _link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_link'
        module_type_store = module_type_store.open_function_context('_link', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._link.__dict__.__setitem__('stypy_localization', localization)
        config._link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._link.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._link.__dict__.__setitem__('stypy_function_name', 'config._link')
        config._link.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'])
        config._link.__dict__.__setitem__('stypy_varargs_param_name', None)
        config._link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._link.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._link.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._link.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._link', ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_link', localization, ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_link(...)' code ##################

        
        
        # Getting the type of 'self' (line 110)
        self_58271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'self')
        # Obtaining the member 'compiler' of a type (line 110)
        compiler_58272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), self_58271, 'compiler')
        # Obtaining the member 'compiler_type' of a type (line 110)
        compiler_type_58273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), compiler_58272, 'compiler_type')
        str_58274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 40), 'str', 'msvc')
        # Applying the binary operator '==' (line 110)
        result_eq_58275 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), '==', compiler_type_58273, str_58274)
        
        # Testing the type of an if condition (line 110)
        if_condition_58276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_eq_58275)
        # Assigning a type to the variable 'if_condition_58276' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_58276', if_condition_58276)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 111):
        
        # Assigning a Subscript to a Name (line 111):
        
        # Obtaining the type of the subscript
        slice_58277 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 25), None, None, None)
        
        # Evaluating a boolean operation
        # Getting the type of 'libraries' (line 111)
        libraries_58278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_58279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        
        # Applying the binary operator 'or' (line 111)
        result_or_keyword_58280 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 25), 'or', libraries_58278, list_58279)
        
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___58281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), result_or_keyword_58280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_58282 = invoke(stypy.reporting.localization.Localization(__file__, 111, 25), getitem___58281, slice_58277)
        
        # Assigning a type to the variable 'libraries' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'libraries', subscript_call_result_58282)
        
        # Assigning a Subscript to a Name (line 112):
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        slice_58283 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 28), None, None, None)
        
        # Evaluating a boolean operation
        # Getting the type of 'library_dirs' (line 112)
        library_dirs_58284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_58285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        
        # Applying the binary operator 'or' (line 112)
        result_or_keyword_58286 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 28), 'or', library_dirs_58284, list_58285)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___58287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), result_or_keyword_58286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_58288 = invoke(stypy.reporting.localization.Localization(__file__, 112, 28), getitem___58287, slice_58283)
        
        # Assigning a type to the variable 'library_dirs' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'library_dirs', subscript_call_result_58288)
        
        
        # Getting the type of 'lang' (line 113)
        lang_58289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'lang')
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_58290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_58291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 24), 'str', 'f77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 23), list_58290, str_58291)
        # Adding element type (line 113)
        str_58292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'str', 'f90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 23), list_58290, str_58292)
        
        # Applying the binary operator 'in' (line 113)
        result_contains_58293 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'in', lang_58289, list_58290)
        
        # Testing the type of an if condition (line 113)
        if_condition_58294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_contains_58293)
        # Assigning a type to the variable 'if_condition_58294' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_58294', if_condition_58294)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 114):
        
        # Assigning a Str to a Name (line 114):
        str_58295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 23), 'str', 'c')
        # Assigning a type to the variable 'lang' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'lang', str_58295)
        
        # Getting the type of 'self' (line 115)
        self_58296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'self')
        # Obtaining the member 'fcompiler' of a type (line 115)
        fcompiler_58297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), self_58296, 'fcompiler')
        # Testing the type of an if condition (line 115)
        if_condition_58298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 16), fcompiler_58297)
        # Assigning a type to the variable 'if_condition_58298' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'if_condition_58298', if_condition_58298)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 116)
        self_58299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'self')
        # Obtaining the member 'fcompiler' of a type (line 116)
        fcompiler_58300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 29), self_58299, 'fcompiler')
        # Obtaining the member 'library_dirs' of a type (line 116)
        library_dirs_58301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 29), fcompiler_58300, 'library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_58302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        
        # Applying the binary operator 'or' (line 116)
        result_or_keyword_58303 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 29), 'or', library_dirs_58301, list_58302)
        
        # Testing the type of a for loop iterable (line 116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 20), result_or_keyword_58303)
        # Getting the type of the for loop variable (line 116)
        for_loop_var_58304 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 20), result_or_keyword_58303)
        # Assigning a type to the variable 'd' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'd', for_loop_var_58304)
        # SSA begins for a for statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 119)
        # Processing the call arguments (line 119)
        str_58307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'str', '/usr/lib')
        # Processing the call keyword arguments (line 119)
        kwargs_58308 = {}
        # Getting the type of 'd' (line 119)
        d_58305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'd', False)
        # Obtaining the member 'startswith' of a type (line 119)
        startswith_58306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 27), d_58305, 'startswith')
        # Calling startswith(args, kwargs) (line 119)
        startswith_call_result_58309 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), startswith_58306, *[str_58307], **kwargs_58308)
        
        # Testing the type of an if condition (line 119)
        if_condition_58310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 24), startswith_call_result_58309)
        # Assigning a type to the variable 'if_condition_58310' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'if_condition_58310', if_condition_58310)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 120):
        
        # Assigning a Call to a Name:
        
        # Call to exec_command(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_58312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        str_58313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 49), 'str', 'cygpath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 48), list_58312, str_58313)
        # Adding element type (line 120)
        str_58314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 60), 'str', '-w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 48), list_58312, str_58314)
        # Adding element type (line 120)
        # Getting the type of 'd' (line 120)
        d_58315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 66), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 48), list_58312, d_58315)
        
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'False' (line 121)
        False_58316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'False', False)
        keyword_58317 = False_58316
        kwargs_58318 = {'use_tee': keyword_58317}
        # Getting the type of 'exec_command' (line 120)
        exec_command_58311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'exec_command', False)
        # Calling exec_command(args, kwargs) (line 120)
        exec_command_call_result_58319 = invoke(stypy.reporting.localization.Localization(__file__, 120, 35), exec_command_58311, *[list_58312], **kwargs_58318)
        
        # Assigning a type to the variable 'call_assignment_58051' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58051', exec_command_call_result_58319)
        
        # Assigning a Call to a Name (line 120):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_58322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 28), 'int')
        # Processing the call keyword arguments
        kwargs_58323 = {}
        # Getting the type of 'call_assignment_58051' (line 120)
        call_assignment_58051_58320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58051', False)
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___58321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 28), call_assignment_58051_58320, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_58324 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___58321, *[int_58322], **kwargs_58323)
        
        # Assigning a type to the variable 'call_assignment_58052' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58052', getitem___call_result_58324)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'call_assignment_58052' (line 120)
        call_assignment_58052_58325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58052')
        # Assigning a type to the variable 's' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 's', call_assignment_58052_58325)
        
        # Assigning a Call to a Name (line 120):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_58328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 28), 'int')
        # Processing the call keyword arguments
        kwargs_58329 = {}
        # Getting the type of 'call_assignment_58051' (line 120)
        call_assignment_58051_58326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58051', False)
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___58327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 28), call_assignment_58051_58326, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_58330 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___58327, *[int_58328], **kwargs_58329)
        
        # Assigning a type to the variable 'call_assignment_58053' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58053', getitem___call_result_58330)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'call_assignment_58053' (line 120)
        call_assignment_58053_58331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'call_assignment_58053')
        # Assigning a type to the variable 'o' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'o', call_assignment_58053_58331)
        
        
        # Getting the type of 's' (line 122)
        s_58332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 's')
        # Applying the 'not' unary operator (line 122)
        result_not__58333 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 31), 'not', s_58332)
        
        # Testing the type of an if condition (line 122)
        if_condition_58334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 28), result_not__58333)
        # Assigning a type to the variable 'if_condition_58334' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'if_condition_58334', if_condition_58334)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 122):
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'o' (line 122)
        o_58335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 42), 'o')
        # Assigning a type to the variable 'd' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'd', o_58335)
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'd' (line 123)
        d_58338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'd', False)
        # Processing the call keyword arguments (line 123)
        kwargs_58339 = {}
        # Getting the type of 'library_dirs' (line 123)
        library_dirs_58336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'library_dirs', False)
        # Obtaining the member 'append' of a type (line 123)
        append_58337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), library_dirs_58336, 'append')
        # Calling append(args, kwargs) (line 123)
        append_call_result_58340 = invoke(stypy.reporting.localization.Localization(__file__, 123, 24), append_58337, *[d_58338], **kwargs_58339)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 124)
        self_58341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'self')
        # Obtaining the member 'fcompiler' of a type (line 124)
        fcompiler_58342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 35), self_58341, 'fcompiler')
        # Obtaining the member 'libraries' of a type (line 124)
        libraries_58343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 35), fcompiler_58342, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_58344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        
        # Applying the binary operator 'or' (line 124)
        result_or_keyword_58345 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 35), 'or', libraries_58343, list_58344)
        
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 20), result_or_keyword_58345)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_58346 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 20), result_or_keyword_58345)
        # Assigning a type to the variable 'libname' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'libname', for_loop_var_58346)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'libname' (line 125)
        libname_58347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'libname')
        # Getting the type of 'libraries' (line 125)
        libraries_58348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'libraries')
        # Applying the binary operator 'notin' (line 125)
        result_contains_58349 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 27), 'notin', libname_58347, libraries_58348)
        
        # Testing the type of an if condition (line 125)
        if_condition_58350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 24), result_contains_58349)
        # Assigning a type to the variable 'if_condition_58350' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'if_condition_58350', if_condition_58350)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'libname' (line 126)
        libname_58353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 45), 'libname', False)
        # Processing the call keyword arguments (line 126)
        kwargs_58354 = {}
        # Getting the type of 'libraries' (line 126)
        libraries_58351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'libraries', False)
        # Obtaining the member 'append' of a type (line 126)
        append_58352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 28), libraries_58351, 'append')
        # Calling append(args, kwargs) (line 126)
        append_call_result_58355 = invoke(stypy.reporting.localization.Localization(__file__, 126, 28), append_58352, *[libname_58353], **kwargs_58354)
        
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'libraries' (line 127)
        libraries_58356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'libraries')
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 12), libraries_58356)
        # Getting the type of the for loop variable (line 127)
        for_loop_var_58357 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 12), libraries_58356)
        # Assigning a type to the variable 'libname' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'libname', for_loop_var_58357)
        # SSA begins for a for statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 128)
        # Processing the call arguments (line 128)
        str_58360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 38), 'str', 'msvc')
        # Processing the call keyword arguments (line 128)
        kwargs_58361 = {}
        # Getting the type of 'libname' (line 128)
        libname_58358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'libname', False)
        # Obtaining the member 'startswith' of a type (line 128)
        startswith_58359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), libname_58358, 'startswith')
        # Calling startswith(args, kwargs) (line 128)
        startswith_call_result_58362 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), startswith_58359, *[str_58360], **kwargs_58361)
        
        # Testing the type of an if condition (line 128)
        if_condition_58363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 16), startswith_call_result_58362)
        # Assigning a type to the variable 'if_condition_58363' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'if_condition_58363', if_condition_58363)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 129):
        
        # Assigning a Name to a Name (line 129):
        # Getting the type of 'False' (line 129)
        False_58364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 29), 'False')
        # Assigning a type to the variable 'fileexists' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'fileexists', False_58364)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'library_dirs' (line 130)
        library_dirs_58365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_58366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        
        # Applying the binary operator 'or' (line 130)
        result_or_keyword_58367 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 30), 'or', library_dirs_58365, list_58366)
        
        # Testing the type of a for loop iterable (line 130)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 130, 16), result_or_keyword_58367)
        # Getting the type of the for loop variable (line 130)
        for_loop_var_58368 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 130, 16), result_or_keyword_58367)
        # Assigning a type to the variable 'libdir' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'libdir', for_loop_var_58368)
        # SSA begins for a for statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to join(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'libdir' (line 131)
        libdir_58372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'libdir', False)
        str_58373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 51), 'str', '%s.lib')
        # Getting the type of 'libname' (line 131)
        libname_58374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 63), 'libname', False)
        # Applying the binary operator '%' (line 131)
        result_mod_58375 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 51), '%', str_58373, libname_58374)
        
        # Processing the call keyword arguments (line 131)
        kwargs_58376 = {}
        # Getting the type of 'os' (line 131)
        os_58369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 131)
        path_58370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), os_58369, 'path')
        # Obtaining the member 'join' of a type (line 131)
        join_58371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), path_58370, 'join')
        # Calling join(args, kwargs) (line 131)
        join_call_result_58377 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), join_58371, *[libdir_58372, result_mod_58375], **kwargs_58376)
        
        # Assigning a type to the variable 'libfile' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'libfile', join_call_result_58377)
        
        
        # Call to isfile(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'libfile' (line 132)
        libfile_58381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'libfile', False)
        # Processing the call keyword arguments (line 132)
        kwargs_58382 = {}
        # Getting the type of 'os' (line 132)
        os_58378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 132)
        path_58379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), os_58378, 'path')
        # Obtaining the member 'isfile' of a type (line 132)
        isfile_58380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), path_58379, 'isfile')
        # Calling isfile(args, kwargs) (line 132)
        isfile_call_result_58383 = invoke(stypy.reporting.localization.Localization(__file__, 132, 23), isfile_58380, *[libfile_58381], **kwargs_58382)
        
        # Testing the type of an if condition (line 132)
        if_condition_58384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 20), isfile_call_result_58383)
        # Assigning a type to the variable 'if_condition_58384' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'if_condition_58384', if_condition_58384)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 133):
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'True' (line 133)
        True_58385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'True')
        # Assigning a type to the variable 'fileexists' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'fileexists', True_58385)
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'fileexists' (line 135)
        fileexists_58386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'fileexists')
        # Testing the type of an if condition (line 135)
        if_condition_58387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 16), fileexists_58386)
        # Assigning a type to the variable 'if_condition_58387' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'if_condition_58387', if_condition_58387)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 137):
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'False' (line 137)
        False_58388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'False')
        # Assigning a type to the variable 'fileexists' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'fileexists', False_58388)
        
        # Getting the type of 'library_dirs' (line 138)
        library_dirs_58389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'library_dirs')
        # Testing the type of a for loop iterable (line 138)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 16), library_dirs_58389)
        # Getting the type of the for loop variable (line 138)
        for_loop_var_58390 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 16), library_dirs_58389)
        # Assigning a type to the variable 'libdir' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'libdir', for_loop_var_58390)
        # SSA begins for a for statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to join(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'libdir' (line 139)
        libdir_58394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'libdir', False)
        str_58395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 51), 'str', 'lib%s.a')
        # Getting the type of 'libname' (line 139)
        libname_58396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 64), 'libname', False)
        # Applying the binary operator '%' (line 139)
        result_mod_58397 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 51), '%', str_58395, libname_58396)
        
        # Processing the call keyword arguments (line 139)
        kwargs_58398 = {}
        # Getting the type of 'os' (line 139)
        os_58391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 139)
        path_58392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), os_58391, 'path')
        # Obtaining the member 'join' of a type (line 139)
        join_58393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), path_58392, 'join')
        # Calling join(args, kwargs) (line 139)
        join_call_result_58399 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), join_58393, *[libdir_58394, result_mod_58397], **kwargs_58398)
        
        # Assigning a type to the variable 'libfile' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'libfile', join_call_result_58399)
        
        
        # Call to isfile(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'libfile' (line 140)
        libfile_58403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'libfile', False)
        # Processing the call keyword arguments (line 140)
        kwargs_58404 = {}
        # Getting the type of 'os' (line 140)
        os_58400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 140)
        path_58401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), os_58400, 'path')
        # Obtaining the member 'isfile' of a type (line 140)
        isfile_58402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), path_58401, 'isfile')
        # Calling isfile(args, kwargs) (line 140)
        isfile_call_result_58405 = invoke(stypy.reporting.localization.Localization(__file__, 140, 23), isfile_58402, *[libfile_58403], **kwargs_58404)
        
        # Testing the type of an if condition (line 140)
        if_condition_58406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 20), isfile_call_result_58405)
        # Assigning a type to the variable 'if_condition_58406' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'if_condition_58406', if_condition_58406)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 143):
        
        # Assigning a Call to a Name (line 143):
        
        # Call to join(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'libdir' (line 143)
        libdir_58410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 48), 'libdir', False)
        str_58411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 56), 'str', '%s.lib')
        # Getting the type of 'libname' (line 143)
        libname_58412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 68), 'libname', False)
        # Applying the binary operator '%' (line 143)
        result_mod_58413 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 56), '%', str_58411, libname_58412)
        
        # Processing the call keyword arguments (line 143)
        kwargs_58414 = {}
        # Getting the type of 'os' (line 143)
        os_58407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 143)
        path_58408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 35), os_58407, 'path')
        # Obtaining the member 'join' of a type (line 143)
        join_58409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 35), path_58408, 'join')
        # Calling join(args, kwargs) (line 143)
        join_call_result_58415 = invoke(stypy.reporting.localization.Localization(__file__, 143, 35), join_58409, *[libdir_58410, result_mod_58413], **kwargs_58414)
        
        # Assigning a type to the variable 'libfile2' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'libfile2', join_call_result_58415)
        
        # Call to copy_file(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'libfile' (line 144)
        libfile_58417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'libfile', False)
        # Getting the type of 'libfile2' (line 144)
        libfile2_58418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 43), 'libfile2', False)
        # Processing the call keyword arguments (line 144)
        kwargs_58419 = {}
        # Getting the type of 'copy_file' (line 144)
        copy_file_58416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 144)
        copy_file_call_result_58420 = invoke(stypy.reporting.localization.Localization(__file__, 144, 24), copy_file_58416, *[libfile_58417, libfile2_58418], **kwargs_58419)
        
        
        # Call to append(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'libfile2' (line 145)
        libfile2_58424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 47), 'libfile2', False)
        # Processing the call keyword arguments (line 145)
        kwargs_58425 = {}
        # Getting the type of 'self' (line 145)
        self_58421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'self', False)
        # Obtaining the member 'temp_files' of a type (line 145)
        temp_files_58422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 24), self_58421, 'temp_files')
        # Obtaining the member 'append' of a type (line 145)
        append_58423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 24), temp_files_58422, 'append')
        # Calling append(args, kwargs) (line 145)
        append_call_result_58426 = invoke(stypy.reporting.localization.Localization(__file__, 145, 24), append_58423, *[libfile2_58424], **kwargs_58425)
        
        
        # Assigning a Name to a Name (line 146):
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'True' (line 146)
        True_58427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'True')
        # Assigning a type to the variable 'fileexists' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'fileexists', True_58427)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'fileexists' (line 148)
        fileexists_58428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'fileexists')
        # Testing the type of an if condition (line 148)
        if_condition_58429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 16), fileexists_58428)
        # Assigning a type to the variable 'if_condition_58429' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'if_condition_58429', if_condition_58429)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to warn(...): (line 149)
        # Processing the call arguments (line 149)
        str_58432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'str', 'could not find library %r in directories %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_58433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        # Getting the type of 'libname' (line 150)
        libname_58434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 28), 'libname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 28), tuple_58433, libname_58434)
        # Adding element type (line 150)
        # Getting the type of 'library_dirs' (line 150)
        library_dirs_58435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'library_dirs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 28), tuple_58433, library_dirs_58435)
        
        # Applying the binary operator '%' (line 149)
        result_mod_58436 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 25), '%', str_58432, tuple_58433)
        
        # Processing the call keyword arguments (line 149)
        kwargs_58437 = {}
        # Getting the type of 'log' (line 149)
        log_58430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 149)
        warn_58431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), log_58430, 'warn')
        # Calling warn(args, kwargs) (line 149)
        warn_call_result_58438 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), warn_58431, *[result_mod_58436], **kwargs_58437)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 110)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 151)
        self_58439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'self')
        # Obtaining the member 'compiler' of a type (line 151)
        compiler_58440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), self_58439, 'compiler')
        # Obtaining the member 'compiler_type' of a type (line 151)
        compiler_type_58441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), compiler_58440, 'compiler_type')
        str_58442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 44), 'str', 'mingw32')
        # Applying the binary operator '==' (line 151)
        result_eq_58443 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 13), '==', compiler_type_58441, str_58442)
        
        # Testing the type of an if condition (line 151)
        if_condition_58444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 13), result_eq_58443)
        # Assigning a type to the variable 'if_condition_58444' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'if_condition_58444', if_condition_58444)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to generate_manifest(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'self' (line 152)
        self_58446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'self', False)
        # Processing the call keyword arguments (line 152)
        kwargs_58447 = {}
        # Getting the type of 'generate_manifest' (line 152)
        generate_manifest_58445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'generate_manifest', False)
        # Calling generate_manifest(args, kwargs) (line 152)
        generate_manifest_call_result_58448 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), generate_manifest_58445, *[self_58446], **kwargs_58447)
        
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _wrap_method(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'old_config' (line 153)
        old_config_58451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'old_config', False)
        # Obtaining the member '_link' of a type (line 153)
        _link_58452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), old_config_58451, '_link')
        # Getting the type of 'lang' (line 153)
        lang_58453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'lang', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_58454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        # Getting the type of 'body' (line 154)
        body_58455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'body', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 34), tuple_58454, body_58455)
        # Adding element type (line 154)
        # Getting the type of 'headers' (line 154)
        headers_58456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 40), 'headers', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 34), tuple_58454, headers_58456)
        # Adding element type (line 154)
        # Getting the type of 'include_dirs' (line 154)
        include_dirs_58457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'include_dirs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 34), tuple_58454, include_dirs_58457)
        # Adding element type (line 154)
        # Getting the type of 'libraries' (line 155)
        libraries_58458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'libraries', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 34), tuple_58454, libraries_58458)
        # Adding element type (line 154)
        # Getting the type of 'library_dirs' (line 155)
        library_dirs_58459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 45), 'library_dirs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 34), tuple_58454, library_dirs_58459)
        # Adding element type (line 154)
        # Getting the type of 'lang' (line 155)
        lang_58460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 59), 'lang', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 34), tuple_58454, lang_58460)
        
        # Processing the call keyword arguments (line 153)
        kwargs_58461 = {}
        # Getting the type of 'self' (line 153)
        self_58449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'self', False)
        # Obtaining the member '_wrap_method' of a type (line 153)
        _wrap_method_58450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), self_58449, '_wrap_method')
        # Calling _wrap_method(args, kwargs) (line 153)
        _wrap_method_call_result_58462 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), _wrap_method_58450, *[_link_58452, lang_58453, tuple_58454], **kwargs_58461)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', _wrap_method_call_result_58462)
        
        # ################# End of '_link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_link' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_58463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_link'
        return stypy_return_type_58463


    @norecursion
    def check_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 157)
        None_58464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 48), 'None')
        # Getting the type of 'None' (line 157)
        None_58465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 67), 'None')
        str_58466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 78), 'str', 'c')
        defaults = [None_58464, None_58465, str_58466]
        # Create a new context for function 'check_header'
        module_type_store = module_type_store.open_function_context('check_header', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_header.__dict__.__setitem__('stypy_localization', localization)
        config.check_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_header.__dict__.__setitem__('stypy_function_name', 'config.check_header')
        config.check_header.__dict__.__setitem__('stypy_param_names_list', ['header', 'include_dirs', 'library_dirs', 'lang'])
        config.check_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_header.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_header', ['header', 'include_dirs', 'library_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_header', localization, ['header', 'include_dirs', 'library_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_header(...)' code ##################

        
        # Call to _check_compiler(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_58469 = {}
        # Getting the type of 'self' (line 158)
        self_58467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 158)
        _check_compiler_58468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_58467, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 158)
        _check_compiler_call_result_58470 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), _check_compiler_58468, *[], **kwargs_58469)
        
        
        # Call to try_compile(...): (line 159)
        # Processing the call arguments (line 159)
        str_58473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 16), 'str', '/* we need a dummy line to make distutils happy */')
        
        # Obtaining an instance of the builtin type 'list' (line 161)
        list_58474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'header' (line 161)
        header_58475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'header', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 16), list_58474, header_58475)
        
        # Getting the type of 'include_dirs' (line 161)
        include_dirs_58476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'include_dirs', False)
        # Processing the call keyword arguments (line 159)
        kwargs_58477 = {}
        # Getting the type of 'self' (line 159)
        self_58471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'self', False)
        # Obtaining the member 'try_compile' of a type (line 159)
        try_compile_58472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), self_58471, 'try_compile')
        # Calling try_compile(args, kwargs) (line 159)
        try_compile_call_result_58478 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), try_compile_58472, *[str_58473, list_58474, include_dirs_58476], **kwargs_58477)
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', try_compile_call_result_58478)
        
        # ################# End of 'check_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_header' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_58479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_header'
        return stypy_return_type_58479


    @norecursion
    def check_decl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 164)
        None_58480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'None')
        # Getting the type of 'None' (line 164)
        None_58481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 46), 'None')
        defaults = [None_58480, None_58481]
        # Create a new context for function 'check_decl'
        module_type_store = module_type_store.open_function_context('check_decl', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_decl.__dict__.__setitem__('stypy_localization', localization)
        config.check_decl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_decl.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_decl.__dict__.__setitem__('stypy_function_name', 'config.check_decl')
        config.check_decl.__dict__.__setitem__('stypy_param_names_list', ['symbol', 'headers', 'include_dirs'])
        config.check_decl.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_decl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_decl.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_decl.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_decl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_decl.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_decl', ['symbol', 'headers', 'include_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_decl', localization, ['symbol', 'headers', 'include_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_decl(...)' code ##################

        
        # Call to _check_compiler(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_58484 = {}
        # Getting the type of 'self' (line 165)
        self_58482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 165)
        _check_compiler_58483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_58482, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 165)
        _check_compiler_call_result_58485 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), _check_compiler_58483, *[], **kwargs_58484)
        
        
        # Assigning a BinOp to a Name (line 166):
        
        # Assigning a BinOp to a Name (line 166):
        str_58486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'str', '\nint main(void)\n{\n#ifndef %s\n    (void) %s;\n#endif\n    ;\n    return 0;\n}')
        
        # Obtaining an instance of the builtin type 'tuple' (line 174)
        tuple_58487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 174)
        # Adding element type (line 174)
        # Getting the type of 'symbol' (line 174)
        symbol_58488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'symbol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 8), tuple_58487, symbol_58488)
        # Adding element type (line 174)
        # Getting the type of 'symbol' (line 174)
        symbol_58489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'symbol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 8), tuple_58487, symbol_58489)
        
        # Applying the binary operator '%' (line 174)
        result_mod_58490 = python_operator(stypy.reporting.localization.Localization(__file__, 174, (-1)), '%', str_58486, tuple_58487)
        
        # Assigning a type to the variable 'body' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'body', result_mod_58490)
        
        # Call to try_compile(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'body' (line 176)
        body_58493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'body', False)
        # Getting the type of 'headers' (line 176)
        headers_58494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 38), 'headers', False)
        # Getting the type of 'include_dirs' (line 176)
        include_dirs_58495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 47), 'include_dirs', False)
        # Processing the call keyword arguments (line 176)
        kwargs_58496 = {}
        # Getting the type of 'self' (line 176)
        self_58491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'self', False)
        # Obtaining the member 'try_compile' of a type (line 176)
        try_compile_58492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), self_58491, 'try_compile')
        # Calling try_compile(args, kwargs) (line 176)
        try_compile_call_result_58497 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), try_compile_58492, *[body_58493, headers_58494, include_dirs_58495], **kwargs_58496)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', try_compile_call_result_58497)
        
        # ################# End of 'check_decl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_decl' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_58498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58498)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_decl'
        return stypy_return_type_58498


    @norecursion
    def check_macro_true(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 179)
        None_58499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 33), 'None')
        # Getting the type of 'None' (line 179)
        None_58500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 52), 'None')
        defaults = [None_58499, None_58500]
        # Create a new context for function 'check_macro_true'
        module_type_store = module_type_store.open_function_context('check_macro_true', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_macro_true.__dict__.__setitem__('stypy_localization', localization)
        config.check_macro_true.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_macro_true.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_macro_true.__dict__.__setitem__('stypy_function_name', 'config.check_macro_true')
        config.check_macro_true.__dict__.__setitem__('stypy_param_names_list', ['symbol', 'headers', 'include_dirs'])
        config.check_macro_true.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_macro_true.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_macro_true.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_macro_true.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_macro_true.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_macro_true.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_macro_true', ['symbol', 'headers', 'include_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_macro_true', localization, ['symbol', 'headers', 'include_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_macro_true(...)' code ##################

        
        # Call to _check_compiler(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_58503 = {}
        # Getting the type of 'self' (line 180)
        self_58501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 180)
        _check_compiler_58502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_58501, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 180)
        _check_compiler_call_result_58504 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), _check_compiler_58502, *[], **kwargs_58503)
        
        
        # Assigning a BinOp to a Name (line 181):
        
        # Assigning a BinOp to a Name (line 181):
        str_58505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\nint main(void)\n{\n#if %s\n#else\n#error false or undefined macro\n#endif\n    ;\n    return 0;\n}')
        
        # Obtaining an instance of the builtin type 'tuple' (line 190)
        tuple_58506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 190)
        # Adding element type (line 190)
        # Getting the type of 'symbol' (line 190)
        symbol_58507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'symbol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), tuple_58506, symbol_58507)
        
        # Applying the binary operator '%' (line 190)
        result_mod_58508 = python_operator(stypy.reporting.localization.Localization(__file__, 190, (-1)), '%', str_58505, tuple_58506)
        
        # Assigning a type to the variable 'body' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'body', result_mod_58508)
        
        # Call to try_compile(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'body' (line 192)
        body_58511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'body', False)
        # Getting the type of 'headers' (line 192)
        headers_58512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 38), 'headers', False)
        # Getting the type of 'include_dirs' (line 192)
        include_dirs_58513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 47), 'include_dirs', False)
        # Processing the call keyword arguments (line 192)
        kwargs_58514 = {}
        # Getting the type of 'self' (line 192)
        self_58509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'self', False)
        # Obtaining the member 'try_compile' of a type (line 192)
        try_compile_58510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 15), self_58509, 'try_compile')
        # Calling try_compile(args, kwargs) (line 192)
        try_compile_call_result_58515 = invoke(stypy.reporting.localization.Localization(__file__, 192, 15), try_compile_58510, *[body_58511, headers_58512, include_dirs_58513], **kwargs_58514)
        
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'stypy_return_type', try_compile_call_result_58515)
        
        # ################# End of 'check_macro_true(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_macro_true' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_58516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58516)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_macro_true'
        return stypy_return_type_58516


    @norecursion
    def check_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 194)
        None_58517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 44), 'None')
        # Getting the type of 'None' (line 194)
        None_58518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 63), 'None')
        # Getting the type of 'None' (line 195)
        None_58519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 'None')
        defaults = [None_58517, None_58518, None_58519]
        # Create a new context for function 'check_type'
        module_type_store = module_type_store.open_function_context('check_type', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_type.__dict__.__setitem__('stypy_localization', localization)
        config.check_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_type.__dict__.__setitem__('stypy_function_name', 'config.check_type')
        config.check_type.__dict__.__setitem__('stypy_param_names_list', ['type_name', 'headers', 'include_dirs', 'library_dirs'])
        config.check_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_type.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_type', ['type_name', 'headers', 'include_dirs', 'library_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_type', localization, ['type_name', 'headers', 'include_dirs', 'library_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_type(...)' code ##################

        str_58520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', 'Check type availability. Return True if the type can be compiled,\n        False otherwise')
        
        # Call to _check_compiler(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_58523 = {}
        # Getting the type of 'self' (line 198)
        self_58521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 198)
        _check_compiler_58522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_58521, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 198)
        _check_compiler_call_result_58524 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), _check_compiler_58522, *[], **kwargs_58523)
        
        
        # Assigning a BinOp to a Name (line 201):
        
        # Assigning a BinOp to a Name (line 201):
        str_58525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, (-1)), 'str', '\nint main(void) {\n  if ((%(name)s *) 0)\n    return 0;\n  if (sizeof (%(name)s))\n    return 0;\n}\n')
        
        # Obtaining an instance of the builtin type 'dict' (line 208)
        dict_58526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 6), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 208)
        # Adding element type (key, value) (line 208)
        str_58527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 7), 'str', 'name')
        # Getting the type of 'type_name' (line 208)
        type_name_58528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'type_name')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 6), dict_58526, (str_58527, type_name_58528))
        
        # Applying the binary operator '%' (line 208)
        result_mod_58529 = python_operator(stypy.reporting.localization.Localization(__file__, 208, (-1)), '%', str_58525, dict_58526)
        
        # Assigning a type to the variable 'body' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'body', result_mod_58529)
        
        # Assigning a Name to a Name (line 210):
        
        # Assigning a Name to a Name (line 210):
        # Getting the type of 'False' (line 210)
        False_58530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'False')
        # Assigning a type to the variable 'st' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'st', False_58530)
        
        # Try-finally block (line 211)
        
        
        # SSA begins for try-except statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _compile(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'body' (line 213)
        body_58533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'body', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 213)
        dict_58534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 37), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 213)
        # Adding element type (key, value) (line 213)
        str_58535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 38), 'str', 'type')
        # Getting the type of 'type_name' (line 213)
        type_name_58536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 46), 'type_name', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 37), dict_58534, (str_58535, type_name_58536))
        
        # Applying the binary operator '%' (line 213)
        result_mod_58537 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 30), '%', body_58533, dict_58534)
        
        # Getting the type of 'headers' (line 214)
        headers_58538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'headers', False)
        # Getting the type of 'include_dirs' (line 214)
        include_dirs_58539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'include_dirs', False)
        str_58540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 47), 'str', 'c')
        # Processing the call keyword arguments (line 213)
        kwargs_58541 = {}
        # Getting the type of 'self' (line 213)
        self_58531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'self', False)
        # Obtaining the member '_compile' of a type (line 213)
        _compile_58532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), self_58531, '_compile')
        # Calling _compile(args, kwargs) (line 213)
        _compile_call_result_58542 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), _compile_58532, *[result_mod_58537, headers_58538, include_dirs_58539, str_58540], **kwargs_58541)
        
        
        # Assigning a Name to a Name (line 215):
        
        # Assigning a Name to a Name (line 215):
        # Getting the type of 'True' (line 215)
        True_58543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'True')
        # Assigning a type to the variable 'st' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'st', True_58543)
        # SSA branch for the except part of a try statement (line 212)
        # SSA branch for the except 'Attribute' branch of a try statement (line 212)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 217):
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'False' (line 217)
        False_58544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'False')
        # Assigning a type to the variable 'st' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'st', False_58544)
        # SSA join for try-except statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 211)
        
        # Call to _clean(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_58547 = {}
        # Getting the type of 'self' (line 219)
        self_58545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'self', False)
        # Obtaining the member '_clean' of a type (line 219)
        _clean_58546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), self_58545, '_clean')
        # Calling _clean(args, kwargs) (line 219)
        _clean_call_result_58548 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), _clean_58546, *[], **kwargs_58547)
        
        
        # Getting the type of 'st' (line 221)
        st_58549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'st')
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', st_58549)
        
        # ################# End of 'check_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_type' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_58550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_type'
        return stypy_return_type_58550


    @norecursion
    def check_type_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 223)
        None_58551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'None')
        # Getting the type of 'None' (line 223)
        None_58552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 68), 'None')
        # Getting the type of 'None' (line 223)
        None_58553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 87), 'None')
        # Getting the type of 'None' (line 223)
        None_58554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 102), 'None')
        defaults = [None_58551, None_58552, None_58553, None_58554]
        # Create a new context for function 'check_type_size'
        module_type_store = module_type_store.open_function_context('check_type_size', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_type_size.__dict__.__setitem__('stypy_localization', localization)
        config.check_type_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_type_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_type_size.__dict__.__setitem__('stypy_function_name', 'config.check_type_size')
        config.check_type_size.__dict__.__setitem__('stypy_param_names_list', ['type_name', 'headers', 'include_dirs', 'library_dirs', 'expected'])
        config.check_type_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_type_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_type_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_type_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_type_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_type_size.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_type_size', ['type_name', 'headers', 'include_dirs', 'library_dirs', 'expected'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_type_size', localization, ['type_name', 'headers', 'include_dirs', 'library_dirs', 'expected'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_type_size(...)' code ##################

        str_58555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'str', 'Check size of a given type.')
        
        # Call to _check_compiler(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_58558 = {}
        # Getting the type of 'self' (line 225)
        self_58556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 225)
        _check_compiler_58557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_58556, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 225)
        _check_compiler_call_result_58559 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), _check_compiler_58557, *[], **kwargs_58558)
        
        
        # Assigning a Str to a Name (line 228):
        
        # Assigning a Str to a Name (line 228):
        str_58560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'str', '\ntypedef %(type)s npy_check_sizeof_type;\nint main (void)\n{\n    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];\n    test_array [0] = 0\n\n    ;\n    return 0;\n}\n')
        # Assigning a type to the variable 'body' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'body', str_58560)
        
        # Call to _compile(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'body' (line 239)
        body_58563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'body', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 239)
        dict_58564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 239)
        # Adding element type (key, value) (line 239)
        str_58565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'str', 'type')
        # Getting the type of 'type_name' (line 239)
        type_name_58566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 38), 'type_name', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 29), dict_58564, (str_58565, type_name_58566))
        
        # Applying the binary operator '%' (line 239)
        result_mod_58567 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 22), '%', body_58563, dict_58564)
        
        # Getting the type of 'headers' (line 240)
        headers_58568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'headers', False)
        # Getting the type of 'include_dirs' (line 240)
        include_dirs_58569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'include_dirs', False)
        str_58570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 39), 'str', 'c')
        # Processing the call keyword arguments (line 239)
        kwargs_58571 = {}
        # Getting the type of 'self' (line 239)
        self_58561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member '_compile' of a type (line 239)
        _compile_58562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_58561, '_compile')
        # Calling _compile(args, kwargs) (line 239)
        _compile_call_result_58572 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), _compile_58562, *[result_mod_58567, headers_58568, include_dirs_58569, str_58570], **kwargs_58571)
        
        
        # Call to _clean(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_58575 = {}
        # Getting the type of 'self' (line 241)
        self_58573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self', False)
        # Obtaining the member '_clean' of a type (line 241)
        _clean_58574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_58573, '_clean')
        # Calling _clean(args, kwargs) (line 241)
        _clean_call_result_58576 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), _clean_58574, *[], **kwargs_58575)
        
        
        # Getting the type of 'expected' (line 243)
        expected_58577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'expected')
        # Testing the type of an if condition (line 243)
        if_condition_58578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), expected_58577)
        # Assigning a type to the variable 'if_condition_58578' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_58578', if_condition_58578)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 244):
        
        # Assigning a Str to a Name (line 244):
        str_58579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, (-1)), 'str', '\ntypedef %(type)s npy_check_sizeof_type;\nint main (void)\n{\n    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)s)];\n    test_array [0] = 0\n\n    ;\n    return 0;\n}\n')
        # Assigning a type to the variable 'body' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'body', str_58579)
        
        # Getting the type of 'expected' (line 255)
        expected_58580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'expected')
        # Testing the type of a for loop iterable (line 255)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 255, 12), expected_58580)
        # Getting the type of the for loop variable (line 255)
        for_loop_var_58581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 255, 12), expected_58580)
        # Assigning a type to the variable 'size' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'size', for_loop_var_58581)
        # SSA begins for a for statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _compile(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'body' (line 257)
        body_58584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'body', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 257)
        dict_58585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 41), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 257)
        # Adding element type (key, value) (line 257)
        str_58586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 42), 'str', 'type')
        # Getting the type of 'type_name' (line 257)
        type_name_58587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 50), 'type_name', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 41), dict_58585, (str_58586, type_name_58587))
        # Adding element type (key, value) (line 257)
        str_58588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 61), 'str', 'size')
        # Getting the type of 'size' (line 257)
        size_58589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 69), 'size', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 41), dict_58585, (str_58588, size_58589))
        
        # Applying the binary operator '%' (line 257)
        result_mod_58590 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 34), '%', body_58584, dict_58585)
        
        # Getting the type of 'headers' (line 258)
        headers_58591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'headers', False)
        # Getting the type of 'include_dirs' (line 258)
        include_dirs_58592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'include_dirs', False)
        str_58593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 51), 'str', 'c')
        # Processing the call keyword arguments (line 257)
        kwargs_58594 = {}
        # Getting the type of 'self' (line 257)
        self_58582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'self', False)
        # Obtaining the member '_compile' of a type (line 257)
        _compile_58583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 20), self_58582, '_compile')
        # Calling _compile(args, kwargs) (line 257)
        _compile_call_result_58595 = invoke(stypy.reporting.localization.Localization(__file__, 257, 20), _compile_58583, *[result_mod_58590, headers_58591, include_dirs_58592, str_58593], **kwargs_58594)
        
        
        # Call to _clean(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_58598 = {}
        # Getting the type of 'self' (line 259)
        self_58596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'self', False)
        # Obtaining the member '_clean' of a type (line 259)
        _clean_58597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 20), self_58596, '_clean')
        # Calling _clean(args, kwargs) (line 259)
        _clean_call_result_58599 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), _clean_58597, *[], **kwargs_58598)
        
        # Getting the type of 'size' (line 260)
        size_58600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 'size')
        # Assigning a type to the variable 'stypy_return_type' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'stypy_return_type', size_58600)
        # SSA branch for the except part of a try statement (line 256)
        # SSA branch for the except 'CompileError' branch of a try statement (line 256)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 265):
        
        # Assigning a Str to a Name (line 265):
        str_58601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'str', '\ntypedef %(type)s npy_check_sizeof_type;\nint main (void)\n{\n    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)s)];\n    test_array [0] = 0\n\n    ;\n    return 0;\n}\n')
        # Assigning a type to the variable 'body' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'body', str_58601)
        
        # Assigning a Num to a Name (line 280):
        
        # Assigning a Num to a Name (line 280):
        int_58602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 14), 'int')
        # Assigning a type to the variable 'low' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'low', int_58602)
        
        # Assigning a Num to a Name (line 281):
        
        # Assigning a Num to a Name (line 281):
        int_58603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 14), 'int')
        # Assigning a type to the variable 'mid' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'mid', int_58603)
        
        # Getting the type of 'True' (line 282)
        True_58604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'True')
        # Testing the type of an if condition (line 282)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 8), True_58604)
        # SSA begins for while statement (line 282)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # SSA begins for try-except statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _compile(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'body' (line 284)
        body_58607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 30), 'body', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 284)
        dict_58608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 37), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 284)
        # Adding element type (key, value) (line 284)
        str_58609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 38), 'str', 'type')
        # Getting the type of 'type_name' (line 284)
        type_name_58610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 46), 'type_name', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 37), dict_58608, (str_58609, type_name_58610))
        # Adding element type (key, value) (line 284)
        str_58611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 57), 'str', 'size')
        # Getting the type of 'mid' (line 284)
        mid_58612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 65), 'mid', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 37), dict_58608, (str_58611, mid_58612))
        
        # Applying the binary operator '%' (line 284)
        result_mod_58613 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 30), '%', body_58607, dict_58608)
        
        # Getting the type of 'headers' (line 285)
        headers_58614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'headers', False)
        # Getting the type of 'include_dirs' (line 285)
        include_dirs_58615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'include_dirs', False)
        str_58616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 47), 'str', 'c')
        # Processing the call keyword arguments (line 284)
        kwargs_58617 = {}
        # Getting the type of 'self' (line 284)
        self_58605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'self', False)
        # Obtaining the member '_compile' of a type (line 284)
        _compile_58606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), self_58605, '_compile')
        # Calling _compile(args, kwargs) (line 284)
        _compile_call_result_58618 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), _compile_58606, *[result_mod_58613, headers_58614, include_dirs_58615, str_58616], **kwargs_58617)
        
        
        # Call to _clean(...): (line 286)
        # Processing the call keyword arguments (line 286)
        kwargs_58621 = {}
        # Getting the type of 'self' (line 286)
        self_58619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'self', False)
        # Obtaining the member '_clean' of a type (line 286)
        _clean_58620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 16), self_58619, '_clean')
        # Calling _clean(args, kwargs) (line 286)
        _clean_call_result_58622 = invoke(stypy.reporting.localization.Localization(__file__, 286, 16), _clean_58620, *[], **kwargs_58621)
        
        # SSA branch for the except part of a try statement (line 283)
        # SSA branch for the except 'CompileError' branch of a try statement (line 283)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BinOp to a Name (line 290):
        
        # Assigning a BinOp to a Name (line 290):
        # Getting the type of 'mid' (line 290)
        mid_58623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'mid')
        int_58624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 28), 'int')
        # Applying the binary operator '+' (line 290)
        result_add_58625 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 22), '+', mid_58623, int_58624)
        
        # Assigning a type to the variable 'low' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'low', result_add_58625)
        
        # Assigning a BinOp to a Name (line 291):
        
        # Assigning a BinOp to a Name (line 291):
        int_58626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 22), 'int')
        # Getting the type of 'mid' (line 291)
        mid_58627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 26), 'mid')
        # Applying the binary operator '*' (line 291)
        result_mul_58628 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 22), '*', int_58626, mid_58627)
        
        int_58629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 32), 'int')
        # Applying the binary operator '+' (line 291)
        result_add_58630 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 22), '+', result_mul_58628, int_58629)
        
        # Assigning a type to the variable 'mid' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'mid', result_add_58630)
        # SSA join for try-except statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 282)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 293):
        
        # Assigning a Name to a Name (line 293):
        # Getting the type of 'mid' (line 293)
        mid_58631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'mid')
        # Assigning a type to the variable 'high' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'high', mid_58631)
        
        
        # Getting the type of 'low' (line 295)
        low_58632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'low')
        # Getting the type of 'high' (line 295)
        high_58633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'high')
        # Applying the binary operator '!=' (line 295)
        result_ne_58634 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 14), '!=', low_58632, high_58633)
        
        # Testing the type of an if condition (line 295)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 8), result_ne_58634)
        # SSA begins for while statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 296):
        
        # Assigning a BinOp to a Name (line 296):
        # Getting the type of 'high' (line 296)
        high_58635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'high')
        # Getting the type of 'low' (line 296)
        low_58636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'low')
        # Applying the binary operator '-' (line 296)
        result_sub_58637 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 19), '-', high_58635, low_58636)
        
        int_58638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 34), 'int')
        # Applying the binary operator '//' (line 296)
        result_floordiv_58639 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 18), '//', result_sub_58637, int_58638)
        
        # Getting the type of 'low' (line 296)
        low_58640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'low')
        # Applying the binary operator '+' (line 296)
        result_add_58641 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 18), '+', result_floordiv_58639, low_58640)
        
        # Assigning a type to the variable 'mid' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'mid', result_add_58641)
        
        
        # SSA begins for try-except statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _compile(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'body' (line 298)
        body_58644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'body', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 298)
        dict_58645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 37), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 298)
        # Adding element type (key, value) (line 298)
        str_58646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 38), 'str', 'type')
        # Getting the type of 'type_name' (line 298)
        type_name_58647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 46), 'type_name', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 37), dict_58645, (str_58646, type_name_58647))
        # Adding element type (key, value) (line 298)
        str_58648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 57), 'str', 'size')
        # Getting the type of 'mid' (line 298)
        mid_58649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 65), 'mid', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 37), dict_58645, (str_58648, mid_58649))
        
        # Applying the binary operator '%' (line 298)
        result_mod_58650 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 30), '%', body_58644, dict_58645)
        
        # Getting the type of 'headers' (line 299)
        headers_58651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'headers', False)
        # Getting the type of 'include_dirs' (line 299)
        include_dirs_58652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'include_dirs', False)
        str_58653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 47), 'str', 'c')
        # Processing the call keyword arguments (line 298)
        kwargs_58654 = {}
        # Getting the type of 'self' (line 298)
        self_58642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'self', False)
        # Obtaining the member '_compile' of a type (line 298)
        _compile_58643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 16), self_58642, '_compile')
        # Calling _compile(args, kwargs) (line 298)
        _compile_call_result_58655 = invoke(stypy.reporting.localization.Localization(__file__, 298, 16), _compile_58643, *[result_mod_58650, headers_58651, include_dirs_58652, str_58653], **kwargs_58654)
        
        
        # Call to _clean(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_58658 = {}
        # Getting the type of 'self' (line 300)
        self_58656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'self', False)
        # Obtaining the member '_clean' of a type (line 300)
        _clean_58657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 16), self_58656, '_clean')
        # Calling _clean(args, kwargs) (line 300)
        _clean_call_result_58659 = invoke(stypy.reporting.localization.Localization(__file__, 300, 16), _clean_58657, *[], **kwargs_58658)
        
        
        # Assigning a Name to a Name (line 301):
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'mid' (line 301)
        mid_58660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'mid')
        # Assigning a type to the variable 'high' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'high', mid_58660)
        # SSA branch for the except part of a try statement (line 297)
        # SSA branch for the except 'CompileError' branch of a try statement (line 297)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BinOp to a Name (line 303):
        
        # Assigning a BinOp to a Name (line 303):
        # Getting the type of 'mid' (line 303)
        mid_58661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'mid')
        int_58662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 28), 'int')
        # Applying the binary operator '+' (line 303)
        result_add_58663 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 22), '+', mid_58661, int_58662)
        
        # Assigning a type to the variable 'low' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'low', result_add_58663)
        # SSA join for try-except statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'low' (line 304)
        low_58664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'low')
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', low_58664)
        
        # ################# End of 'check_type_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_type_size' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_58665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_type_size'
        return stypy_return_type_58665


    @norecursion
    def check_func(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 307)
        None_58666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 'None')
        # Getting the type of 'None' (line 307)
        None_58667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 46), 'None')
        # Getting the type of 'None' (line 308)
        None_58668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'None')
        # Getting the type of 'None' (line 308)
        None_58669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 48), 'None')
        # Getting the type of 'False' (line 309)
        False_58670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'False')
        # Getting the type of 'False' (line 309)
        False_58671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'False')
        # Getting the type of 'None' (line 309)
        None_58672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 53), 'None')
        defaults = [None_58666, None_58667, None_58668, None_58669, False_58670, False_58671, None_58672]
        # Create a new context for function 'check_func'
        module_type_store = module_type_store.open_function_context('check_func', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_func.__dict__.__setitem__('stypy_localization', localization)
        config.check_func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_func.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_func.__dict__.__setitem__('stypy_function_name', 'config.check_func')
        config.check_func.__dict__.__setitem__('stypy_param_names_list', ['func', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call', 'call_args'])
        config.check_func.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_func.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_func.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_func.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_func.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_func', ['func', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call', 'call_args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_func', localization, ['func', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call', 'call_args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_func(...)' code ##################

        
        # Call to _check_compiler(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_58675 = {}
        # Getting the type of 'self' (line 312)
        self_58673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 312)
        _check_compiler_58674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_58673, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 312)
        _check_compiler_call_result_58676 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), _check_compiler_58674, *[], **kwargs_58675)
        
        
        # Assigning a List to a Name (line 313):
        
        # Assigning a List to a Name (line 313):
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_58677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        
        # Assigning a type to the variable 'body' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'body', list_58677)
        
        # Getting the type of 'decl' (line 314)
        decl_58678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'decl')
        # Testing the type of an if condition (line 314)
        if_condition_58679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), decl_58678)
        # Assigning a type to the variable 'if_condition_58679' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_58679', if_condition_58679)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 315)
        # Getting the type of 'decl' (line 315)
        decl_58680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'decl')
        # Getting the type of 'str' (line 315)
        str_58681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 29), 'str')
        
        (may_be_58682, more_types_in_union_58683) = may_be_type(decl_58680, str_58681)

        if may_be_58682:

            if more_types_in_union_58683:
                # Runtime conditional SSA (line 315)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'decl' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'decl', str_58681())
            
            # Call to append(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 'decl' (line 316)
            decl_58686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 28), 'decl', False)
            # Processing the call keyword arguments (line 316)
            kwargs_58687 = {}
            # Getting the type of 'body' (line 316)
            body_58684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'body', False)
            # Obtaining the member 'append' of a type (line 316)
            append_58685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 16), body_58684, 'append')
            # Calling append(args, kwargs) (line 316)
            append_call_result_58688 = invoke(stypy.reporting.localization.Localization(__file__, 316, 16), append_58685, *[decl_58686], **kwargs_58687)
            

            if more_types_in_union_58683:
                # Runtime conditional SSA for else branch (line 315)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_58682) or more_types_in_union_58683):
            # Getting the type of 'decl' (line 315)
            decl_58689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'decl')
            # Assigning a type to the variable 'decl' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'decl', remove_type_from_union(decl_58689, str_58681))
            
            # Call to append(...): (line 318)
            # Processing the call arguments (line 318)
            str_58692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 28), 'str', 'int %s (void);')
            # Getting the type of 'func' (line 318)
            func_58693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 47), 'func', False)
            # Applying the binary operator '%' (line 318)
            result_mod_58694 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 28), '%', str_58692, func_58693)
            
            # Processing the call keyword arguments (line 318)
            kwargs_58695 = {}
            # Getting the type of 'body' (line 318)
            body_58690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'body', False)
            # Obtaining the member 'append' of a type (line 318)
            append_58691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), body_58690, 'append')
            # Calling append(args, kwargs) (line 318)
            append_call_result_58696 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), append_58691, *[result_mod_58694], **kwargs_58695)
            

            if (may_be_58682 and more_types_in_union_58683):
                # SSA join for if statement (line 315)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 323)
        # Processing the call arguments (line 323)
        str_58699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'str', '#ifdef _MSC_VER')
        # Processing the call keyword arguments (line 323)
        kwargs_58700 = {}
        # Getting the type of 'body' (line 323)
        body_58697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 323)
        append_58698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), body_58697, 'append')
        # Calling append(args, kwargs) (line 323)
        append_call_result_58701 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), append_58698, *[str_58699], **kwargs_58700)
        
        
        # Call to append(...): (line 324)
        # Processing the call arguments (line 324)
        str_58704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 20), 'str', '#pragma function(%s)')
        # Getting the type of 'func' (line 324)
        func_58705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 45), 'func', False)
        # Applying the binary operator '%' (line 324)
        result_mod_58706 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 20), '%', str_58704, func_58705)
        
        # Processing the call keyword arguments (line 324)
        kwargs_58707 = {}
        # Getting the type of 'body' (line 324)
        body_58702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 324)
        append_58703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), body_58702, 'append')
        # Calling append(args, kwargs) (line 324)
        append_call_result_58708 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), append_58703, *[result_mod_58706], **kwargs_58707)
        
        
        # Call to append(...): (line 325)
        # Processing the call arguments (line 325)
        str_58711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 20), 'str', '#endif')
        # Processing the call keyword arguments (line 325)
        kwargs_58712 = {}
        # Getting the type of 'body' (line 325)
        body_58709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 325)
        append_58710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), body_58709, 'append')
        # Calling append(args, kwargs) (line 325)
        append_call_result_58713 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), append_58710, *[str_58711], **kwargs_58712)
        
        
        # Call to append(...): (line 326)
        # Processing the call arguments (line 326)
        str_58716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 20), 'str', 'int main (void) {')
        # Processing the call keyword arguments (line 326)
        kwargs_58717 = {}
        # Getting the type of 'body' (line 326)
        body_58714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 326)
        append_58715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), body_58714, 'append')
        # Calling append(args, kwargs) (line 326)
        append_call_result_58718 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), append_58715, *[str_58716], **kwargs_58717)
        
        
        # Getting the type of 'call' (line 327)
        call_58719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'call')
        # Testing the type of an if condition (line 327)
        if_condition_58720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), call_58719)
        # Assigning a type to the variable 'if_condition_58720' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_58720', if_condition_58720)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 328)
        # Getting the type of 'call_args' (line 328)
        call_args_58721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'call_args')
        # Getting the type of 'None' (line 328)
        None_58722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'None')
        
        (may_be_58723, more_types_in_union_58724) = may_be_none(call_args_58721, None_58722)

        if may_be_58723:

            if more_types_in_union_58724:
                # Runtime conditional SSA (line 328)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 329):
            
            # Assigning a Str to a Name (line 329):
            str_58725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 28), 'str', '')
            # Assigning a type to the variable 'call_args' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'call_args', str_58725)

            if more_types_in_union_58724:
                # SSA join for if statement (line 328)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 330)
        # Processing the call arguments (line 330)
        str_58728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'str', '  %s(%s);')
        
        # Obtaining an instance of the builtin type 'tuple' (line 330)
        tuple_58729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 330)
        # Adding element type (line 330)
        # Getting the type of 'func' (line 330)
        func_58730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 39), 'func', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 39), tuple_58729, func_58730)
        # Adding element type (line 330)
        # Getting the type of 'call_args' (line 330)
        call_args_58731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 45), 'call_args', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 39), tuple_58729, call_args_58731)
        
        # Applying the binary operator '%' (line 330)
        result_mod_58732 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 24), '%', str_58728, tuple_58729)
        
        # Processing the call keyword arguments (line 330)
        kwargs_58733 = {}
        # Getting the type of 'body' (line 330)
        body_58726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'body', False)
        # Obtaining the member 'append' of a type (line 330)
        append_58727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), body_58726, 'append')
        # Calling append(args, kwargs) (line 330)
        append_call_result_58734 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), append_58727, *[result_mod_58732], **kwargs_58733)
        
        # SSA branch for the else part of an if statement (line 327)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 332)
        # Processing the call arguments (line 332)
        str_58737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 24), 'str', '  %s;')
        # Getting the type of 'func' (line 332)
        func_58738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'func', False)
        # Applying the binary operator '%' (line 332)
        result_mod_58739 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 24), '%', str_58737, func_58738)
        
        # Processing the call keyword arguments (line 332)
        kwargs_58740 = {}
        # Getting the type of 'body' (line 332)
        body_58735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'body', False)
        # Obtaining the member 'append' of a type (line 332)
        append_58736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), body_58735, 'append')
        # Calling append(args, kwargs) (line 332)
        append_call_result_58741 = invoke(stypy.reporting.localization.Localization(__file__, 332, 12), append_58736, *[result_mod_58739], **kwargs_58740)
        
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 333)
        # Processing the call arguments (line 333)
        str_58744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 20), 'str', '  return 0;')
        # Processing the call keyword arguments (line 333)
        kwargs_58745 = {}
        # Getting the type of 'body' (line 333)
        body_58742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 333)
        append_58743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), body_58742, 'append')
        # Calling append(args, kwargs) (line 333)
        append_call_result_58746 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), append_58743, *[str_58744], **kwargs_58745)
        
        
        # Call to append(...): (line 334)
        # Processing the call arguments (line 334)
        str_58749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 20), 'str', '}')
        # Processing the call keyword arguments (line 334)
        kwargs_58750 = {}
        # Getting the type of 'body' (line 334)
        body_58747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 334)
        append_58748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), body_58747, 'append')
        # Calling append(args, kwargs) (line 334)
        append_call_result_58751 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), append_58748, *[str_58749], **kwargs_58750)
        
        
        # Assigning a BinOp to a Name (line 335):
        
        # Assigning a BinOp to a Name (line 335):
        
        # Call to join(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'body' (line 335)
        body_58754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'body', False)
        # Processing the call keyword arguments (line 335)
        kwargs_58755 = {}
        str_58752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 335)
        join_58753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 15), str_58752, 'join')
        # Calling join(args, kwargs) (line 335)
        join_call_result_58756 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), join_58753, *[body_58754], **kwargs_58755)
        
        str_58757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 33), 'str', '\n')
        # Applying the binary operator '+' (line 335)
        result_add_58758 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 15), '+', join_call_result_58756, str_58757)
        
        # Assigning a type to the variable 'body' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'body', result_add_58758)
        
        # Call to try_link(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'body' (line 337)
        body_58761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'body', False)
        # Getting the type of 'headers' (line 337)
        headers_58762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 35), 'headers', False)
        # Getting the type of 'include_dirs' (line 337)
        include_dirs_58763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 44), 'include_dirs', False)
        # Getting the type of 'libraries' (line 338)
        libraries_58764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'libraries', False)
        # Getting the type of 'library_dirs' (line 338)
        library_dirs_58765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 40), 'library_dirs', False)
        # Processing the call keyword arguments (line 337)
        kwargs_58766 = {}
        # Getting the type of 'self' (line 337)
        self_58759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'self', False)
        # Obtaining the member 'try_link' of a type (line 337)
        try_link_58760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), self_58759, 'try_link')
        # Calling try_link(args, kwargs) (line 337)
        try_link_call_result_58767 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), try_link_58760, *[body_58761, headers_58762, include_dirs_58763, libraries_58764, library_dirs_58765], **kwargs_58766)
        
        # Assigning a type to the variable 'stypy_return_type' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', try_link_call_result_58767)
        
        # ################# End of 'check_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_func' in the type store
        # Getting the type of 'stypy_return_type' (line 306)
        stypy_return_type_58768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_func'
        return stypy_return_type_58768


    @norecursion
    def check_funcs_once(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 341)
        None_58769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 27), 'None')
        # Getting the type of 'None' (line 341)
        None_58770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 46), 'None')
        # Getting the type of 'None' (line 342)
        None_58771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'None')
        # Getting the type of 'None' (line 342)
        None_58772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 48), 'None')
        # Getting the type of 'False' (line 343)
        False_58773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'False')
        # Getting the type of 'False' (line 343)
        False_58774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 36), 'False')
        # Getting the type of 'None' (line 343)
        None_58775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 53), 'None')
        defaults = [None_58769, None_58770, None_58771, None_58772, False_58773, False_58774, None_58775]
        # Create a new context for function 'check_funcs_once'
        module_type_store = module_type_store.open_function_context('check_funcs_once', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_funcs_once.__dict__.__setitem__('stypy_localization', localization)
        config.check_funcs_once.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_funcs_once.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_funcs_once.__dict__.__setitem__('stypy_function_name', 'config.check_funcs_once')
        config.check_funcs_once.__dict__.__setitem__('stypy_param_names_list', ['funcs', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call', 'call_args'])
        config.check_funcs_once.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_funcs_once.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_funcs_once.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_funcs_once.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_funcs_once.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_funcs_once.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_funcs_once', ['funcs', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call', 'call_args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_funcs_once', localization, ['funcs', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call', 'call_args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_funcs_once(...)' code ##################

        str_58776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, (-1)), 'str', 'Check a list of functions at once.\n\n        This is useful to speed up things, since all the functions in the funcs\n        list will be put in one compilation unit.\n\n        Arguments\n        ---------\n        funcs : seq\n            list of functions to test\n        include_dirs : seq\n            list of header paths\n        libraries : seq\n            list of libraries to link the code snippet to\n        libraru_dirs : seq\n            list of library paths\n        decl : dict\n            for every (key, value), the declaration in the value will be\n            used for function in key. If a function is not in the\n            dictionay, no declaration will be used.\n        call : dict\n            for every item (f, value), if the value is True, a call will be\n            done to the function f.\n        ')
        
        # Call to _check_compiler(...): (line 367)
        # Processing the call keyword arguments (line 367)
        kwargs_58779 = {}
        # Getting the type of 'self' (line 367)
        self_58777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 367)
        _check_compiler_58778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 8), self_58777, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 367)
        _check_compiler_call_result_58780 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), _check_compiler_58778, *[], **kwargs_58779)
        
        
        # Assigning a List to a Name (line 368):
        
        # Assigning a List to a Name (line 368):
        
        # Obtaining an instance of the builtin type 'list' (line 368)
        list_58781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 368)
        
        # Assigning a type to the variable 'body' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'body', list_58781)
        
        # Getting the type of 'decl' (line 369)
        decl_58782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'decl')
        # Testing the type of an if condition (line 369)
        if_condition_58783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 8), decl_58782)
        # Assigning a type to the variable 'if_condition_58783' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'if_condition_58783', if_condition_58783)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 370)
        # Processing the call keyword arguments (line 370)
        kwargs_58786 = {}
        # Getting the type of 'decl' (line 370)
        decl_58784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'decl', False)
        # Obtaining the member 'items' of a type (line 370)
        items_58785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), decl_58784, 'items')
        # Calling items(args, kwargs) (line 370)
        items_call_result_58787 = invoke(stypy.reporting.localization.Localization(__file__, 370, 24), items_58785, *[], **kwargs_58786)
        
        # Testing the type of a for loop iterable (line 370)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 370, 12), items_call_result_58787)
        # Getting the type of the for loop variable (line 370)
        for_loop_var_58788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 370, 12), items_call_result_58787)
        # Assigning a type to the variable 'f' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 12), for_loop_var_58788))
        # Assigning a type to the variable 'v' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 12), for_loop_var_58788))
        # SSA begins for a for statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'v' (line 371)
        v_58789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'v')
        # Testing the type of an if condition (line 371)
        if_condition_58790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 16), v_58789)
        # Assigning a type to the variable 'if_condition_58790' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'if_condition_58790', if_condition_58790)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 372)
        # Processing the call arguments (line 372)
        str_58793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 32), 'str', 'int %s (void);')
        # Getting the type of 'f' (line 372)
        f_58794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), 'f', False)
        # Applying the binary operator '%' (line 372)
        result_mod_58795 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 32), '%', str_58793, f_58794)
        
        # Processing the call keyword arguments (line 372)
        kwargs_58796 = {}
        # Getting the type of 'body' (line 372)
        body_58791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'body', False)
        # Obtaining the member 'append' of a type (line 372)
        append_58792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), body_58791, 'append')
        # Calling append(args, kwargs) (line 372)
        append_call_result_58797 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), append_58792, *[result_mod_58795], **kwargs_58796)
        
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 375)
        # Processing the call arguments (line 375)
        str_58800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 20), 'str', '#ifdef _MSC_VER')
        # Processing the call keyword arguments (line 375)
        kwargs_58801 = {}
        # Getting the type of 'body' (line 375)
        body_58798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 375)
        append_58799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), body_58798, 'append')
        # Calling append(args, kwargs) (line 375)
        append_call_result_58802 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), append_58799, *[str_58800], **kwargs_58801)
        
        
        # Getting the type of 'funcs' (line 376)
        funcs_58803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 20), 'funcs')
        # Testing the type of a for loop iterable (line 376)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 376, 8), funcs_58803)
        # Getting the type of the for loop variable (line 376)
        for_loop_var_58804 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 376, 8), funcs_58803)
        # Assigning a type to the variable 'func' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'func', for_loop_var_58804)
        # SSA begins for a for statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 377)
        # Processing the call arguments (line 377)
        str_58807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 24), 'str', '#pragma function(%s)')
        # Getting the type of 'func' (line 377)
        func_58808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 49), 'func', False)
        # Applying the binary operator '%' (line 377)
        result_mod_58809 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 24), '%', str_58807, func_58808)
        
        # Processing the call keyword arguments (line 377)
        kwargs_58810 = {}
        # Getting the type of 'body' (line 377)
        body_58805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'body', False)
        # Obtaining the member 'append' of a type (line 377)
        append_58806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 12), body_58805, 'append')
        # Calling append(args, kwargs) (line 377)
        append_call_result_58811 = invoke(stypy.reporting.localization.Localization(__file__, 377, 12), append_58806, *[result_mod_58809], **kwargs_58810)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 378)
        # Processing the call arguments (line 378)
        str_58814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 20), 'str', '#endif')
        # Processing the call keyword arguments (line 378)
        kwargs_58815 = {}
        # Getting the type of 'body' (line 378)
        body_58812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 378)
        append_58813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), body_58812, 'append')
        # Calling append(args, kwargs) (line 378)
        append_call_result_58816 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), append_58813, *[str_58814], **kwargs_58815)
        
        
        # Call to append(...): (line 380)
        # Processing the call arguments (line 380)
        str_58819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 20), 'str', 'int main (void) {')
        # Processing the call keyword arguments (line 380)
        kwargs_58820 = {}
        # Getting the type of 'body' (line 380)
        body_58817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 380)
        append_58818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), body_58817, 'append')
        # Calling append(args, kwargs) (line 380)
        append_call_result_58821 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), append_58818, *[str_58819], **kwargs_58820)
        
        
        # Getting the type of 'call' (line 381)
        call_58822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'call')
        # Testing the type of an if condition (line 381)
        if_condition_58823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), call_58822)
        # Assigning a type to the variable 'if_condition_58823' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'if_condition_58823', if_condition_58823)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'funcs' (line 382)
        funcs_58824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 21), 'funcs')
        # Testing the type of a for loop iterable (line 382)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 382, 12), funcs_58824)
        # Getting the type of the for loop variable (line 382)
        for_loop_var_58825 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 382, 12), funcs_58824)
        # Assigning a type to the variable 'f' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'f', for_loop_var_58825)
        # SSA begins for a for statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'f' (line 383)
        f_58826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'f')
        # Getting the type of 'call' (line 383)
        call_58827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'call')
        # Applying the binary operator 'in' (line 383)
        result_contains_58828 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 19), 'in', f_58826, call_58827)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'f' (line 383)
        f_58829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 38), 'f')
        # Getting the type of 'call' (line 383)
        call_58830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'call')
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___58831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 33), call_58830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 383)
        subscript_call_result_58832 = invoke(stypy.reporting.localization.Localization(__file__, 383, 33), getitem___58831, f_58829)
        
        # Applying the binary operator 'and' (line 383)
        result_and_keyword_58833 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 19), 'and', result_contains_58828, subscript_call_result_58832)
        
        # Testing the type of an if condition (line 383)
        if_condition_58834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 16), result_and_keyword_58833)
        # Assigning a type to the variable 'if_condition_58834' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'if_condition_58834', if_condition_58834)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'call_args' (line 384)
        call_args_58835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'call_args')
        
        # Getting the type of 'f' (line 384)
        f_58836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 42), 'f')
        # Getting the type of 'call_args' (line 384)
        call_args_58837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 47), 'call_args')
        # Applying the binary operator 'in' (line 384)
        result_contains_58838 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 42), 'in', f_58836, call_args_58837)
        
        # Applying the binary operator 'and' (line 384)
        result_and_keyword_58839 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 28), 'and', call_args_58835, result_contains_58838)
        
        # Obtaining the type of the subscript
        # Getting the type of 'f' (line 384)
        f_58840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 71), 'f')
        # Getting the type of 'call_args' (line 384)
        call_args_58841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 61), 'call_args')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___58842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 61), call_args_58841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_58843 = invoke(stypy.reporting.localization.Localization(__file__, 384, 61), getitem___58842, f_58840)
        
        # Applying the binary operator 'and' (line 384)
        result_and_keyword_58844 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 28), 'and', result_and_keyword_58839, subscript_call_result_58843)
        
        # Applying the 'not' unary operator (line 384)
        result_not__58845 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 23), 'not', result_and_keyword_58844)
        
        # Testing the type of an if condition (line 384)
        if_condition_58846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 20), result_not__58845)
        # Assigning a type to the variable 'if_condition_58846' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'if_condition_58846', if_condition_58846)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 385):
        
        # Assigning a Str to a Name (line 385):
        str_58847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 31), 'str', '')
        # Assigning a type to the variable 'args' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 24), 'args', str_58847)
        # SSA branch for the else part of an if statement (line 384)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 387):
        
        # Assigning a Subscript to a Name (line 387):
        
        # Obtaining the type of the subscript
        # Getting the type of 'f' (line 387)
        f_58848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 41), 'f')
        # Getting the type of 'call_args' (line 387)
        call_args_58849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 31), 'call_args')
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___58850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 31), call_args_58849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_58851 = invoke(stypy.reporting.localization.Localization(__file__, 387, 31), getitem___58850, f_58848)
        
        # Assigning a type to the variable 'args' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'args', subscript_call_result_58851)
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 388)
        # Processing the call arguments (line 388)
        str_58854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 32), 'str', '  %s(%s);')
        
        # Obtaining an instance of the builtin type 'tuple' (line 388)
        tuple_58855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 388)
        # Adding element type (line 388)
        # Getting the type of 'f' (line 388)
        f_58856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 47), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 47), tuple_58855, f_58856)
        # Adding element type (line 388)
        # Getting the type of 'args' (line 388)
        args_58857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 50), 'args', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 47), tuple_58855, args_58857)
        
        # Applying the binary operator '%' (line 388)
        result_mod_58858 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 32), '%', str_58854, tuple_58855)
        
        # Processing the call keyword arguments (line 388)
        kwargs_58859 = {}
        # Getting the type of 'body' (line 388)
        body_58852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'body', False)
        # Obtaining the member 'append' of a type (line 388)
        append_58853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), body_58852, 'append')
        # Calling append(args, kwargs) (line 388)
        append_call_result_58860 = invoke(stypy.reporting.localization.Localization(__file__, 388, 20), append_58853, *[result_mod_58858], **kwargs_58859)
        
        # SSA branch for the else part of an if statement (line 383)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 390)
        # Processing the call arguments (line 390)
        str_58863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 32), 'str', '  %s;')
        # Getting the type of 'f' (line 390)
        f_58864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'f', False)
        # Applying the binary operator '%' (line 390)
        result_mod_58865 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 32), '%', str_58863, f_58864)
        
        # Processing the call keyword arguments (line 390)
        kwargs_58866 = {}
        # Getting the type of 'body' (line 390)
        body_58861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'body', False)
        # Obtaining the member 'append' of a type (line 390)
        append_58862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 20), body_58861, 'append')
        # Calling append(args, kwargs) (line 390)
        append_call_result_58867 = invoke(stypy.reporting.localization.Localization(__file__, 390, 20), append_58862, *[result_mod_58865], **kwargs_58866)
        
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 381)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'funcs' (line 392)
        funcs_58868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'funcs')
        # Testing the type of a for loop iterable (line 392)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 392, 12), funcs_58868)
        # Getting the type of the for loop variable (line 392)
        for_loop_var_58869 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 392, 12), funcs_58868)
        # Assigning a type to the variable 'f' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'f', for_loop_var_58869)
        # SSA begins for a for statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 393)
        # Processing the call arguments (line 393)
        str_58872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 28), 'str', '  %s;')
        # Getting the type of 'f' (line 393)
        f_58873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 38), 'f', False)
        # Applying the binary operator '%' (line 393)
        result_mod_58874 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 28), '%', str_58872, f_58873)
        
        # Processing the call keyword arguments (line 393)
        kwargs_58875 = {}
        # Getting the type of 'body' (line 393)
        body_58870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'body', False)
        # Obtaining the member 'append' of a type (line 393)
        append_58871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), body_58870, 'append')
        # Calling append(args, kwargs) (line 393)
        append_call_result_58876 = invoke(stypy.reporting.localization.Localization(__file__, 393, 16), append_58871, *[result_mod_58874], **kwargs_58875)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 394)
        # Processing the call arguments (line 394)
        str_58879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 20), 'str', '  return 0;')
        # Processing the call keyword arguments (line 394)
        kwargs_58880 = {}
        # Getting the type of 'body' (line 394)
        body_58877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 394)
        append_58878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), body_58877, 'append')
        # Calling append(args, kwargs) (line 394)
        append_call_result_58881 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), append_58878, *[str_58879], **kwargs_58880)
        
        
        # Call to append(...): (line 395)
        # Processing the call arguments (line 395)
        str_58884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 20), 'str', '}')
        # Processing the call keyword arguments (line 395)
        kwargs_58885 = {}
        # Getting the type of 'body' (line 395)
        body_58882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 395)
        append_58883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), body_58882, 'append')
        # Calling append(args, kwargs) (line 395)
        append_call_result_58886 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), append_58883, *[str_58884], **kwargs_58885)
        
        
        # Assigning a BinOp to a Name (line 396):
        
        # Assigning a BinOp to a Name (line 396):
        
        # Call to join(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'body' (line 396)
        body_58889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'body', False)
        # Processing the call keyword arguments (line 396)
        kwargs_58890 = {}
        str_58887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 396)
        join_58888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 15), str_58887, 'join')
        # Calling join(args, kwargs) (line 396)
        join_call_result_58891 = invoke(stypy.reporting.localization.Localization(__file__, 396, 15), join_58888, *[body_58889], **kwargs_58890)
        
        str_58892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 33), 'str', '\n')
        # Applying the binary operator '+' (line 396)
        result_add_58893 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 15), '+', join_call_result_58891, str_58892)
        
        # Assigning a type to the variable 'body' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'body', result_add_58893)
        
        # Call to try_link(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'body' (line 398)
        body_58896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'body', False)
        # Getting the type of 'headers' (line 398)
        headers_58897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 35), 'headers', False)
        # Getting the type of 'include_dirs' (line 398)
        include_dirs_58898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 44), 'include_dirs', False)
        # Getting the type of 'libraries' (line 399)
        libraries_58899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 29), 'libraries', False)
        # Getting the type of 'library_dirs' (line 399)
        library_dirs_58900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 40), 'library_dirs', False)
        # Processing the call keyword arguments (line 398)
        kwargs_58901 = {}
        # Getting the type of 'self' (line 398)
        self_58894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'self', False)
        # Obtaining the member 'try_link' of a type (line 398)
        try_link_58895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 15), self_58894, 'try_link')
        # Calling try_link(args, kwargs) (line 398)
        try_link_call_result_58902 = invoke(stypy.reporting.localization.Localization(__file__, 398, 15), try_link_58895, *[body_58896, headers_58897, include_dirs_58898, libraries_58899, library_dirs_58900], **kwargs_58901)
        
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stypy_return_type', try_link_call_result_58902)
        
        # ################# End of 'check_funcs_once(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_funcs_once' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_58903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58903)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_funcs_once'
        return stypy_return_type_58903


    @norecursion
    def check_inline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_inline'
        module_type_store = module_type_store.open_function_context('check_inline', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_inline.__dict__.__setitem__('stypy_localization', localization)
        config.check_inline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_inline.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_inline.__dict__.__setitem__('stypy_function_name', 'config.check_inline')
        config.check_inline.__dict__.__setitem__('stypy_param_names_list', [])
        config.check_inline.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_inline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_inline.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_inline.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_inline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_inline.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_inline', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_inline', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_inline(...)' code ##################

        str_58904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, (-1)), 'str', 'Return the inline keyword recognized by the compiler, empty string\n        otherwise.')
        
        # Call to check_inline(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'self' (line 404)
        self_58906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 28), 'self', False)
        # Processing the call keyword arguments (line 404)
        kwargs_58907 = {}
        # Getting the type of 'check_inline' (line 404)
        check_inline_58905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'check_inline', False)
        # Calling check_inline(args, kwargs) (line 404)
        check_inline_call_result_58908 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), check_inline_58905, *[self_58906], **kwargs_58907)
        
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', check_inline_call_result_58908)
        
        # ################# End of 'check_inline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_inline' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_58909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_inline'
        return stypy_return_type_58909


    @norecursion
    def check_restrict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_restrict'
        module_type_store = module_type_store.open_function_context('check_restrict', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_restrict.__dict__.__setitem__('stypy_localization', localization)
        config.check_restrict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_restrict.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_restrict.__dict__.__setitem__('stypy_function_name', 'config.check_restrict')
        config.check_restrict.__dict__.__setitem__('stypy_param_names_list', [])
        config.check_restrict.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_restrict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_restrict.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_restrict.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_restrict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_restrict.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_restrict', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_restrict', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_restrict(...)' code ##################

        str_58910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, (-1)), 'str', 'Return the restrict keyword recognized by the compiler, empty string\n        otherwise.')
        
        # Call to check_restrict(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'self' (line 409)
        self_58912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 30), 'self', False)
        # Processing the call keyword arguments (line 409)
        kwargs_58913 = {}
        # Getting the type of 'check_restrict' (line 409)
        check_restrict_58911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 15), 'check_restrict', False)
        # Calling check_restrict(args, kwargs) (line 409)
        check_restrict_call_result_58914 = invoke(stypy.reporting.localization.Localization(__file__, 409, 15), check_restrict_58911, *[self_58912], **kwargs_58913)
        
        # Assigning a type to the variable 'stypy_return_type' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'stypy_return_type', check_restrict_call_result_58914)
        
        # ################# End of 'check_restrict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_restrict' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_58915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58915)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_restrict'
        return stypy_return_type_58915


    @norecursion
    def check_compiler_gcc4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_compiler_gcc4'
        module_type_store = module_type_store.open_function_context('check_compiler_gcc4', 411, 4, False)
        # Assigning a type to the variable 'self' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_localization', localization)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_function_name', 'config.check_compiler_gcc4')
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_param_names_list', [])
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_compiler_gcc4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_compiler_gcc4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_compiler_gcc4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_compiler_gcc4(...)' code ##################

        str_58916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 8), 'str', 'Return True if the C compiler is gcc >= 4.')
        
        # Call to check_compiler_gcc4(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'self' (line 413)
        self_58918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 35), 'self', False)
        # Processing the call keyword arguments (line 413)
        kwargs_58919 = {}
        # Getting the type of 'check_compiler_gcc4' (line 413)
        check_compiler_gcc4_58917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'check_compiler_gcc4', False)
        # Calling check_compiler_gcc4(args, kwargs) (line 413)
        check_compiler_gcc4_call_result_58920 = invoke(stypy.reporting.localization.Localization(__file__, 413, 15), check_compiler_gcc4_58917, *[self_58918], **kwargs_58919)
        
        # Assigning a type to the variable 'stypy_return_type' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'stypy_return_type', check_compiler_gcc4_call_result_58920)
        
        # ################# End of 'check_compiler_gcc4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_compiler_gcc4' in the type store
        # Getting the type of 'stypy_return_type' (line 411)
        stypy_return_type_58921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58921)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_compiler_gcc4'
        return stypy_return_type_58921


    @norecursion
    def check_gcc_function_attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_gcc_function_attribute'
        module_type_store = module_type_store.open_function_context('check_gcc_function_attribute', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_localization', localization)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_function_name', 'config.check_gcc_function_attribute')
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_param_names_list', ['attribute', 'name'])
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_gcc_function_attribute.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_gcc_function_attribute', ['attribute', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_gcc_function_attribute', localization, ['attribute', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_gcc_function_attribute(...)' code ##################

        
        # Call to check_gcc_function_attribute(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'self' (line 416)
        self_58923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 44), 'self', False)
        # Getting the type of 'attribute' (line 416)
        attribute_58924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 50), 'attribute', False)
        # Getting the type of 'name' (line 416)
        name_58925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 61), 'name', False)
        # Processing the call keyword arguments (line 416)
        kwargs_58926 = {}
        # Getting the type of 'check_gcc_function_attribute' (line 416)
        check_gcc_function_attribute_58922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'check_gcc_function_attribute', False)
        # Calling check_gcc_function_attribute(args, kwargs) (line 416)
        check_gcc_function_attribute_call_result_58927 = invoke(stypy.reporting.localization.Localization(__file__, 416, 15), check_gcc_function_attribute_58922, *[self_58923, attribute_58924, name_58925], **kwargs_58926)
        
        # Assigning a type to the variable 'stypy_return_type' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'stypy_return_type', check_gcc_function_attribute_call_result_58927)
        
        # ################# End of 'check_gcc_function_attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_gcc_function_attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_58928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_gcc_function_attribute'
        return stypy_return_type_58928


    @norecursion
    def check_gcc_variable_attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_gcc_variable_attribute'
        module_type_store = module_type_store.open_function_context('check_gcc_variable_attribute', 418, 4, False)
        # Assigning a type to the variable 'self' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_localization', localization)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_function_name', 'config.check_gcc_variable_attribute')
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_param_names_list', ['attribute'])
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_gcc_variable_attribute.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_gcc_variable_attribute', ['attribute'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_gcc_variable_attribute', localization, ['attribute'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_gcc_variable_attribute(...)' code ##################

        
        # Call to check_gcc_variable_attribute(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'self' (line 419)
        self_58930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 44), 'self', False)
        # Getting the type of 'attribute' (line 419)
        attribute_58931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 50), 'attribute', False)
        # Processing the call keyword arguments (line 419)
        kwargs_58932 = {}
        # Getting the type of 'check_gcc_variable_attribute' (line 419)
        check_gcc_variable_attribute_58929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 15), 'check_gcc_variable_attribute', False)
        # Calling check_gcc_variable_attribute(args, kwargs) (line 419)
        check_gcc_variable_attribute_call_result_58933 = invoke(stypy.reporting.localization.Localization(__file__, 419, 15), check_gcc_variable_attribute_58929, *[self_58930, attribute_58931], **kwargs_58932)
        
        # Assigning a type to the variable 'stypy_return_type' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'stypy_return_type', check_gcc_variable_attribute_call_result_58933)
        
        # ################# End of 'check_gcc_variable_attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_gcc_variable_attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 418)
        stypy_return_type_58934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_gcc_variable_attribute'
        return stypy_return_type_58934


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'config' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'config', config)
# Declaration of the 'GrabStdout' class

class GrabStdout(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 424, 4, False)
        # Assigning a type to the variable 'self' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GrabStdout.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 425):
        
        # Assigning a Attribute to a Attribute (line 425):
        # Getting the type of 'sys' (line 425)
        sys_58935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'sys')
        # Obtaining the member 'stdout' of a type (line 425)
        stdout_58936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 26), sys_58935, 'stdout')
        # Getting the type of 'self' (line 425)
        self_58937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'self')
        # Setting the type of the member 'sys_stdout' of a type (line 425)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), self_58937, 'sys_stdout', stdout_58936)
        
        # Assigning a Str to a Attribute (line 426):
        
        # Assigning a Str to a Attribute (line 426):
        str_58938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 20), 'str', '')
        # Getting the type of 'self' (line 426)
        self_58939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self')
        # Setting the type of the member 'data' of a type (line 426)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_58939, 'data', str_58938)
        
        # Assigning a Name to a Attribute (line 427):
        
        # Assigning a Name to a Attribute (line 427):
        # Getting the type of 'self' (line 427)
        self_58940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'self')
        # Getting the type of 'sys' (line 427)
        sys_58941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'sys')
        # Setting the type of the member 'stdout' of a type (line 427)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), sys_58941, 'stdout', self_58940)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 429, 4, False)
        # Assigning a type to the variable 'self' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GrabStdout.write.__dict__.__setitem__('stypy_localization', localization)
        GrabStdout.write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GrabStdout.write.__dict__.__setitem__('stypy_type_store', module_type_store)
        GrabStdout.write.__dict__.__setitem__('stypy_function_name', 'GrabStdout.write')
        GrabStdout.write.__dict__.__setitem__('stypy_param_names_list', ['data'])
        GrabStdout.write.__dict__.__setitem__('stypy_varargs_param_name', None)
        GrabStdout.write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GrabStdout.write.__dict__.__setitem__('stypy_call_defaults', defaults)
        GrabStdout.write.__dict__.__setitem__('stypy_call_varargs', varargs)
        GrabStdout.write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GrabStdout.write.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GrabStdout.write', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        
        # Call to write(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'data' (line 430)
        data_58945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'data', False)
        # Processing the call keyword arguments (line 430)
        kwargs_58946 = {}
        # Getting the type of 'self' (line 430)
        self_58942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self', False)
        # Obtaining the member 'sys_stdout' of a type (line 430)
        sys_stdout_58943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_58942, 'sys_stdout')
        # Obtaining the member 'write' of a type (line 430)
        write_58944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), sys_stdout_58943, 'write')
        # Calling write(args, kwargs) (line 430)
        write_call_result_58947 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), write_58944, *[data_58945], **kwargs_58946)
        
        
        # Getting the type of 'self' (line 431)
        self_58948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self')
        # Obtaining the member 'data' of a type (line 431)
        data_58949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_58948, 'data')
        # Getting the type of 'data' (line 431)
        data_58950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 21), 'data')
        # Applying the binary operator '+=' (line 431)
        result_iadd_58951 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 8), '+=', data_58949, data_58950)
        # Getting the type of 'self' (line 431)
        self_58952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self')
        # Setting the type of the member 'data' of a type (line 431)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_58952, 'data', result_iadd_58951)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 429)
        stypy_return_type_58953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58953)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_58953


    @norecursion
    def flush(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flush'
        module_type_store = module_type_store.open_function_context('flush', 433, 4, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GrabStdout.flush.__dict__.__setitem__('stypy_localization', localization)
        GrabStdout.flush.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GrabStdout.flush.__dict__.__setitem__('stypy_type_store', module_type_store)
        GrabStdout.flush.__dict__.__setitem__('stypy_function_name', 'GrabStdout.flush')
        GrabStdout.flush.__dict__.__setitem__('stypy_param_names_list', [])
        GrabStdout.flush.__dict__.__setitem__('stypy_varargs_param_name', None)
        GrabStdout.flush.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GrabStdout.flush.__dict__.__setitem__('stypy_call_defaults', defaults)
        GrabStdout.flush.__dict__.__setitem__('stypy_call_varargs', varargs)
        GrabStdout.flush.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GrabStdout.flush.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GrabStdout.flush', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flush', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flush(...)' code ##################

        
        # Call to flush(...): (line 434)
        # Processing the call keyword arguments (line 434)
        kwargs_58957 = {}
        # Getting the type of 'self' (line 434)
        self_58954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'self', False)
        # Obtaining the member 'sys_stdout' of a type (line 434)
        sys_stdout_58955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), self_58954, 'sys_stdout')
        # Obtaining the member 'flush' of a type (line 434)
        flush_58956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), sys_stdout_58955, 'flush')
        # Calling flush(args, kwargs) (line 434)
        flush_call_result_58958 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), flush_58956, *[], **kwargs_58957)
        
        
        # ################# End of 'flush(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flush' in the type store
        # Getting the type of 'stypy_return_type' (line 433)
        stypy_return_type_58959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flush'
        return stypy_return_type_58959


    @norecursion
    def restore(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'restore'
        module_type_store = module_type_store.open_function_context('restore', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GrabStdout.restore.__dict__.__setitem__('stypy_localization', localization)
        GrabStdout.restore.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GrabStdout.restore.__dict__.__setitem__('stypy_type_store', module_type_store)
        GrabStdout.restore.__dict__.__setitem__('stypy_function_name', 'GrabStdout.restore')
        GrabStdout.restore.__dict__.__setitem__('stypy_param_names_list', [])
        GrabStdout.restore.__dict__.__setitem__('stypy_varargs_param_name', None)
        GrabStdout.restore.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GrabStdout.restore.__dict__.__setitem__('stypy_call_defaults', defaults)
        GrabStdout.restore.__dict__.__setitem__('stypy_call_varargs', varargs)
        GrabStdout.restore.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GrabStdout.restore.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GrabStdout.restore', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'restore', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'restore(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 437):
        
        # Assigning a Attribute to a Attribute (line 437):
        # Getting the type of 'self' (line 437)
        self_58960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 21), 'self')
        # Obtaining the member 'sys_stdout' of a type (line 437)
        sys_stdout_58961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 21), self_58960, 'sys_stdout')
        # Getting the type of 'sys' (line 437)
        sys_58962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'sys')
        # Setting the type of the member 'stdout' of a type (line 437)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), sys_58962, 'stdout', sys_stdout_58961)
        
        # ################# End of 'restore(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'restore' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_58963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'restore'
        return stypy_return_type_58963


# Assigning a type to the variable 'GrabStdout' (line 422)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 0), 'GrabStdout', GrabStdout)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

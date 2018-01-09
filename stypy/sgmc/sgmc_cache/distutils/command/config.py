
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.config
2: 
3: Implements the Distutils 'config' command, a (mostly) empty command class
4: that exists mainly to be sub-classed by specific module distributions and
5: applications.  The idea is that while every "config" command is different,
6: at least they're all named the same, and users always see "config" in the
7: list of standard commands.  Also, this is a good place to put common
8: configure-like tasks: "try to compile this C code", or "figure out where
9: this header file lives".
10: '''
11: 
12: __revision__ = "$Id$"
13: 
14: import os
15: import re
16: 
17: from distutils.core import Command
18: from distutils.errors import DistutilsExecError
19: from distutils.sysconfig import customize_compiler
20: from distutils import log
21: 
22: LANG_EXT = {'c': '.c', 'c++': '.cxx'}
23: 
24: class config(Command):
25: 
26:     description = "prepare to build"
27: 
28:     user_options = [
29:         ('compiler=', None,
30:          "specify the compiler type"),
31:         ('cc=', None,
32:          "specify the compiler executable"),
33:         ('include-dirs=', 'I',
34:          "list of directories to search for header files"),
35:         ('define=', 'D',
36:          "C preprocessor macros to define"),
37:         ('undef=', 'U',
38:          "C preprocessor macros to undefine"),
39:         ('libraries=', 'l',
40:          "external C libraries to link with"),
41:         ('library-dirs=', 'L',
42:          "directories to search for external C libraries"),
43: 
44:         ('noisy', None,
45:          "show every action (compile, link, run, ...) taken"),
46:         ('dump-source', None,
47:          "dump generated source files before attempting to compile them"),
48:         ]
49: 
50: 
51:     # The three standard command methods: since the "config" command
52:     # does nothing by default, these are empty.
53: 
54:     def initialize_options(self):
55:         self.compiler = None
56:         self.cc = None
57:         self.include_dirs = None
58:         self.libraries = None
59:         self.library_dirs = None
60: 
61:         # maximal output for now
62:         self.noisy = 1
63:         self.dump_source = 1
64: 
65:         # list of temporary files generated along-the-way that we have
66:         # to clean at some point
67:         self.temp_files = []
68: 
69:     def finalize_options(self):
70:         if self.include_dirs is None:
71:             self.include_dirs = self.distribution.include_dirs or []
72:         elif isinstance(self.include_dirs, str):
73:             self.include_dirs = self.include_dirs.split(os.pathsep)
74: 
75:         if self.libraries is None:
76:             self.libraries = []
77:         elif isinstance(self.libraries, str):
78:             self.libraries = [self.libraries]
79: 
80:         if self.library_dirs is None:
81:             self.library_dirs = []
82:         elif isinstance(self.library_dirs, str):
83:             self.library_dirs = self.library_dirs.split(os.pathsep)
84: 
85:     def run(self):
86:         pass
87: 
88: 
89:     # Utility methods for actual "config" commands.  The interfaces are
90:     # loosely based on Autoconf macros of similar names.  Sub-classes
91:     # may use these freely.
92: 
93:     def _check_compiler(self):
94:         '''Check that 'self.compiler' really is a CCompiler object;
95:         if not, make it one.
96:         '''
97:         # We do this late, and only on-demand, because this is an expensive
98:         # import.
99:         from distutils.ccompiler import CCompiler, new_compiler
100:         if not isinstance(self.compiler, CCompiler):
101:             self.compiler = new_compiler(compiler=self.compiler,
102:                                          dry_run=self.dry_run, force=1)
103:             customize_compiler(self.compiler)
104:             if self.include_dirs:
105:                 self.compiler.set_include_dirs(self.include_dirs)
106:             if self.libraries:
107:                 self.compiler.set_libraries(self.libraries)
108:             if self.library_dirs:
109:                 self.compiler.set_library_dirs(self.library_dirs)
110: 
111: 
112:     def _gen_temp_sourcefile(self, body, headers, lang):
113:         filename = "_configtest" + LANG_EXT[lang]
114:         file = open(filename, "w")
115:         if headers:
116:             for header in headers:
117:                 file.write("#include <%s>\n" % header)
118:             file.write("\n")
119:         file.write(body)
120:         if body[-1] != "\n":
121:             file.write("\n")
122:         file.close()
123:         return filename
124: 
125:     def _preprocess(self, body, headers, include_dirs, lang):
126:         src = self._gen_temp_sourcefile(body, headers, lang)
127:         out = "_configtest.i"
128:         self.temp_files.extend([src, out])
129:         self.compiler.preprocess(src, out, include_dirs=include_dirs)
130:         return (src, out)
131: 
132:     def _compile(self, body, headers, include_dirs, lang):
133:         src = self._gen_temp_sourcefile(body, headers, lang)
134:         if self.dump_source:
135:             dump_file(src, "compiling '%s':" % src)
136:         (obj,) = self.compiler.object_filenames([src])
137:         self.temp_files.extend([src, obj])
138:         self.compiler.compile([src], include_dirs=include_dirs)
139:         return (src, obj)
140: 
141:     def _link(self, body, headers, include_dirs, libraries, library_dirs,
142:               lang):
143:         (src, obj) = self._compile(body, headers, include_dirs, lang)
144:         prog = os.path.splitext(os.path.basename(src))[0]
145:         self.compiler.link_executable([obj], prog,
146:                                       libraries=libraries,
147:                                       library_dirs=library_dirs,
148:                                       target_lang=lang)
149: 
150:         if self.compiler.exe_extension is not None:
151:             prog = prog + self.compiler.exe_extension
152:         self.temp_files.append(prog)
153: 
154:         return (src, obj, prog)
155: 
156:     def _clean(self, *filenames):
157:         if not filenames:
158:             filenames = self.temp_files
159:             self.temp_files = []
160:         log.info("removing: %s", ' '.join(filenames))
161:         for filename in filenames:
162:             try:
163:                 os.remove(filename)
164:             except OSError:
165:                 pass
166: 
167: 
168:     # XXX these ignore the dry-run flag: what to do, what to do? even if
169:     # you want a dry-run build, you still need some sort of configuration
170:     # info.  My inclination is to make it up to the real config command to
171:     # consult 'dry_run', and assume a default (minimal) configuration if
172:     # true.  The problem with trying to do it here is that you'd have to
173:     # return either true or false from all the 'try' methods, neither of
174:     # which is correct.
175: 
176:     # XXX need access to the header search path and maybe default macros.
177: 
178:     def try_cpp(self, body=None, headers=None, include_dirs=None, lang="c"):
179:         '''Construct a source file from 'body' (a string containing lines
180:         of C/C++ code) and 'headers' (a list of header files to include)
181:         and run it through the preprocessor.  Return true if the
182:         preprocessor succeeded, false if there were any errors.
183:         ('body' probably isn't of much use, but what the heck.)
184:         '''
185:         from distutils.ccompiler import CompileError
186:         self._check_compiler()
187:         ok = 1
188:         try:
189:             self._preprocess(body, headers, include_dirs, lang)
190:         except CompileError:
191:             ok = 0
192: 
193:         self._clean()
194:         return ok
195: 
196:     def search_cpp(self, pattern, body=None, headers=None, include_dirs=None,
197:                    lang="c"):
198:         '''Construct a source file (just like 'try_cpp()'), run it through
199:         the preprocessor, and return true if any line of the output matches
200:         'pattern'.  'pattern' should either be a compiled regex object or a
201:         string containing a regex.  If both 'body' and 'headers' are None,
202:         preprocesses an empty file -- which can be useful to determine the
203:         symbols the preprocessor and compiler set by default.
204:         '''
205:         self._check_compiler()
206:         src, out = self._preprocess(body, headers, include_dirs, lang)
207: 
208:         if isinstance(pattern, str):
209:             pattern = re.compile(pattern)
210: 
211:         file = open(out)
212:         match = 0
213:         while 1:
214:             line = file.readline()
215:             if line == '':
216:                 break
217:             if pattern.search(line):
218:                 match = 1
219:                 break
220: 
221:         file.close()
222:         self._clean()
223:         return match
224: 
225:     def try_compile(self, body, headers=None, include_dirs=None, lang="c"):
226:         '''Try to compile a source file built from 'body' and 'headers'.
227:         Return true on success, false otherwise.
228:         '''
229:         from distutils.ccompiler import CompileError
230:         self._check_compiler()
231:         try:
232:             self._compile(body, headers, include_dirs, lang)
233:             ok = 1
234:         except CompileError:
235:             ok = 0
236: 
237:         log.info(ok and "success!" or "failure.")
238:         self._clean()
239:         return ok
240: 
241:     def try_link(self, body, headers=None, include_dirs=None, libraries=None,
242:                  library_dirs=None, lang="c"):
243:         '''Try to compile and link a source file, built from 'body' and
244:         'headers', to executable form.  Return true on success, false
245:         otherwise.
246:         '''
247:         from distutils.ccompiler import CompileError, LinkError
248:         self._check_compiler()
249:         try:
250:             self._link(body, headers, include_dirs,
251:                        libraries, library_dirs, lang)
252:             ok = 1
253:         except (CompileError, LinkError):
254:             ok = 0
255: 
256:         log.info(ok and "success!" or "failure.")
257:         self._clean()
258:         return ok
259: 
260:     def try_run(self, body, headers=None, include_dirs=None, libraries=None,
261:                 library_dirs=None, lang="c"):
262:         '''Try to compile, link to an executable, and run a program
263:         built from 'body' and 'headers'.  Return true on success, false
264:         otherwise.
265:         '''
266:         from distutils.ccompiler import CompileError, LinkError
267:         self._check_compiler()
268:         try:
269:             src, obj, exe = self._link(body, headers, include_dirs,
270:                                        libraries, library_dirs, lang)
271:             self.spawn([exe])
272:             ok = 1
273:         except (CompileError, LinkError, DistutilsExecError):
274:             ok = 0
275: 
276:         log.info(ok and "success!" or "failure.")
277:         self._clean()
278:         return ok
279: 
280: 
281:     # -- High-level methods --------------------------------------------
282:     # (these are the ones that are actually likely to be useful
283:     # when implementing a real-world config command!)
284: 
285:     def check_func(self, func, headers=None, include_dirs=None,
286:                    libraries=None, library_dirs=None, decl=0, call=0):
287: 
288:         '''Determine if function 'func' is available by constructing a
289:         source file that refers to 'func', and compiles and links it.
290:         If everything succeeds, returns true; otherwise returns false.
291: 
292:         The constructed source file starts out by including the header
293:         files listed in 'headers'.  If 'decl' is true, it then declares
294:         'func' (as "int func()"); you probably shouldn't supply 'headers'
295:         and set 'decl' true in the same call, or you might get errors about
296:         a conflicting declarations for 'func'.  Finally, the constructed
297:         'main()' function either references 'func' or (if 'call' is true)
298:         calls it.  'libraries' and 'library_dirs' are used when
299:         linking.
300:         '''
301: 
302:         self._check_compiler()
303:         body = []
304:         if decl:
305:             body.append("int %s ();" % func)
306:         body.append("int main () {")
307:         if call:
308:             body.append("  %s();" % func)
309:         else:
310:             body.append("  %s;" % func)
311:         body.append("}")
312:         body = "\n".join(body) + "\n"
313: 
314:         return self.try_link(body, headers, include_dirs,
315:                              libraries, library_dirs)
316: 
317:     # check_func ()
318: 
319:     def check_lib(self, library, library_dirs=None, headers=None,
320:                   include_dirs=None, other_libraries=[]):
321:         '''Determine if 'library' is available to be linked against,
322:         without actually checking that any particular symbols are provided
323:         by it.  'headers' will be used in constructing the source file to
324:         be compiled, but the only effect of this is to check if all the
325:         header files listed are available.  Any libraries listed in
326:         'other_libraries' will be included in the link, in case 'library'
327:         has symbols that depend on other libraries.
328:         '''
329:         self._check_compiler()
330:         return self.try_link("int main (void) { }",
331:                              headers, include_dirs,
332:                              [library]+other_libraries, library_dirs)
333: 
334:     def check_header(self, header, include_dirs=None, library_dirs=None,
335:                      lang="c"):
336:         '''Determine if the system header file named by 'header_file'
337:         exists and can be found by the preprocessor; return true if so,
338:         false otherwise.
339:         '''
340:         return self.try_cpp(body="/* No body */", headers=[header],
341:                             include_dirs=include_dirs)
342: 
343: 
344: def dump_file(filename, head=None):
345:     '''Dumps a file content into log.info.
346: 
347:     If head is not None, will be dumped before the file content.
348:     '''
349:     if head is None:
350:         log.info('%s' % filename)
351:     else:
352:         log.info(head)
353:     file = open(filename)
354:     try:
355:         log.info(file.read())
356:     finally:
357:         file.close()
358: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_21453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', 'distutils.command.config\n\nImplements the Distutils \'config\' command, a (mostly) empty command class\nthat exists mainly to be sub-classed by specific module distributions and\napplications.  The idea is that while every "config" command is different,\nat least they\'re all named the same, and users always see "config" in the\nlist of standard commands.  Also, this is a good place to put common\nconfigure-like tasks: "try to compile this C code", or "figure out where\nthis header file lives".\n')

# Assigning a Str to a Name (line 12):

# Assigning a Str to a Name (line 12):
str_21454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__revision__', str_21454)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import os' statement (line 14)
import os

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import re' statement (line 15)
import re

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.core import Command' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_21455 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.core')

if (type(import_21455) is not StypyTypeError):

    if (import_21455 != 'pyd_module'):
        __import__(import_21455)
        sys_modules_21456 = sys.modules[import_21455]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.core', sys_modules_21456.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_21456, sys_modules_21456.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.core', import_21455)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.errors import DistutilsExecError' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_21457 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors')

if (type(import_21457) is not StypyTypeError):

    if (import_21457 != 'pyd_module'):
        __import__(import_21457)
        sys_modules_21458 = sys.modules[import_21457]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', sys_modules_21458.module_type_store, module_type_store, ['DistutilsExecError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_21458, sys_modules_21458.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError'], [DistutilsExecError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', import_21457)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.sysconfig import customize_compiler' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_21459 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.sysconfig')

if (type(import_21459) is not StypyTypeError):

    if (import_21459 != 'pyd_module'):
        __import__(import_21459)
        sys_modules_21460 = sys.modules[import_21459]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.sysconfig', sys_modules_21460.module_type_store, module_type_store, ['customize_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_21460, sys_modules_21460.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler'], [customize_compiler])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.sysconfig', import_21459)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils import log' statement (line 20)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Dict to a Name (line 22):

# Assigning a Dict to a Name (line 22):

# Obtaining an instance of the builtin type 'dict' (line 22)
dict_21461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 22)
# Adding element type (key, value) (line 22)
str_21462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 12), 'str', 'c')
str_21463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'str', '.c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), dict_21461, (str_21462, str_21463))
# Adding element type (key, value) (line 22)
str_21464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', 'c++')
str_21465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'str', '.cxx')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), dict_21461, (str_21464, str_21465))

# Assigning a type to the variable 'LANG_EXT' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'LANG_EXT', dict_21461)
# Declaration of the 'config' class
# Getting the type of 'Command' (line 24)
Command_21466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'Command')

class config(Command_21466, ):
    
    # Assigning a Str to a Name (line 26):
    
    # Assigning a List to a Name (line 28):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
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

        
        # Assigning a Name to a Attribute (line 55):
        
        # Assigning a Name to a Attribute (line 55):
        # Getting the type of 'None' (line 55)
        None_21467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'None')
        # Getting the type of 'self' (line 55)
        self_21468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_21468, 'compiler', None_21467)
        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'None' (line 56)
        None_21469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'None')
        # Getting the type of 'self' (line 56)
        self_21470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'cc' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_21470, 'cc', None_21469)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'None' (line 57)
        None_21471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'None')
        # Getting the type of 'self' (line 57)
        self_21472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_21472, 'include_dirs', None_21471)
        
        # Assigning a Name to a Attribute (line 58):
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'None' (line 58)
        None_21473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'None')
        # Getting the type of 'self' (line 58)
        self_21474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_21474, 'libraries', None_21473)
        
        # Assigning a Name to a Attribute (line 59):
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'None' (line 59)
        None_21475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'None')
        # Getting the type of 'self' (line 59)
        self_21476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'library_dirs' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_21476, 'library_dirs', None_21475)
        
        # Assigning a Num to a Attribute (line 62):
        
        # Assigning a Num to a Attribute (line 62):
        int_21477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'int')
        # Getting the type of 'self' (line 62)
        self_21478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'noisy' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_21478, 'noisy', int_21477)
        
        # Assigning a Num to a Attribute (line 63):
        
        # Assigning a Num to a Attribute (line 63):
        int_21479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
        # Getting the type of 'self' (line 63)
        self_21480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'dump_source' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_21480, 'dump_source', int_21479)
        
        # Assigning a List to a Attribute (line 67):
        
        # Assigning a List to a Attribute (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_21481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Getting the type of 'self' (line 67)
        self_21482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'temp_files' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_21482, 'temp_files', list_21481)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_21483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_21483


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        config.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.finalize_options.__dict__.__setitem__('stypy_function_name', 'config.finalize_options')
        config.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        config.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 70)
        # Getting the type of 'self' (line 70)
        self_21484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'self')
        # Obtaining the member 'include_dirs' of a type (line 70)
        include_dirs_21485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 11), self_21484, 'include_dirs')
        # Getting the type of 'None' (line 70)
        None_21486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'None')
        
        (may_be_21487, more_types_in_union_21488) = may_be_none(include_dirs_21485, None_21486)

        if may_be_21487:

            if more_types_in_union_21488:
                # Runtime conditional SSA (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BoolOp to a Attribute (line 71):
            
            # Assigning a BoolOp to a Attribute (line 71):
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 71)
            self_21489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'self')
            # Obtaining the member 'distribution' of a type (line 71)
            distribution_21490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), self_21489, 'distribution')
            # Obtaining the member 'include_dirs' of a type (line 71)
            include_dirs_21491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), distribution_21490, 'include_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 71)
            list_21492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 66), 'list')
            # Adding type elements to the builtin type 'list' instance (line 71)
            
            # Applying the binary operator 'or' (line 71)
            result_or_keyword_21493 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 32), 'or', include_dirs_21491, list_21492)
            
            # Getting the type of 'self' (line 71)
            self_21494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'self')
            # Setting the type of the member 'include_dirs' of a type (line 71)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), self_21494, 'include_dirs', result_or_keyword_21493)

            if more_types_in_union_21488:
                # Runtime conditional SSA for else branch (line 70)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_21487) or more_types_in_union_21488):
            
            # Type idiom detected: calculating its left and rigth part (line 72)
            # Getting the type of 'str' (line 72)
            str_21495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'str')
            # Getting the type of 'self' (line 72)
            self_21496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'self')
            # Obtaining the member 'include_dirs' of a type (line 72)
            include_dirs_21497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), self_21496, 'include_dirs')
            
            (may_be_21498, more_types_in_union_21499) = may_be_subtype(str_21495, include_dirs_21497)

            if may_be_21498:

                if more_types_in_union_21499:
                    # Runtime conditional SSA (line 72)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'self' (line 72)
                self_21500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                # Obtaining the member 'include_dirs' of a type (line 72)
                include_dirs_21501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_21500, 'include_dirs')
                # Setting the type of the member 'include_dirs' of a type (line 72)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_21500, 'include_dirs', remove_not_subtype_from_union(include_dirs_21497, str))
                
                # Assigning a Call to a Attribute (line 73):
                
                # Assigning a Call to a Attribute (line 73):
                
                # Call to split(...): (line 73)
                # Processing the call arguments (line 73)
                # Getting the type of 'os' (line 73)
                os_21505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 56), 'os', False)
                # Obtaining the member 'pathsep' of a type (line 73)
                pathsep_21506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 56), os_21505, 'pathsep')
                # Processing the call keyword arguments (line 73)
                kwargs_21507 = {}
                # Getting the type of 'self' (line 73)
                self_21502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'self', False)
                # Obtaining the member 'include_dirs' of a type (line 73)
                include_dirs_21503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 32), self_21502, 'include_dirs')
                # Obtaining the member 'split' of a type (line 73)
                split_21504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 32), include_dirs_21503, 'split')
                # Calling split(args, kwargs) (line 73)
                split_call_result_21508 = invoke(stypy.reporting.localization.Localization(__file__, 73, 32), split_21504, *[pathsep_21506], **kwargs_21507)
                
                # Getting the type of 'self' (line 73)
                self_21509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'self')
                # Setting the type of the member 'include_dirs' of a type (line 73)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), self_21509, 'include_dirs', split_call_result_21508)

                if more_types_in_union_21499:
                    # SSA join for if statement (line 72)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_21487 and more_types_in_union_21488):
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 75)
        # Getting the type of 'self' (line 75)
        self_21510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'self')
        # Obtaining the member 'libraries' of a type (line 75)
        libraries_21511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), self_21510, 'libraries')
        # Getting the type of 'None' (line 75)
        None_21512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'None')
        
        (may_be_21513, more_types_in_union_21514) = may_be_none(libraries_21511, None_21512)

        if may_be_21513:

            if more_types_in_union_21514:
                # Runtime conditional SSA (line 75)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 76):
            
            # Assigning a List to a Attribute (line 76):
            
            # Obtaining an instance of the builtin type 'list' (line 76)
            list_21515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 76)
            
            # Getting the type of 'self' (line 76)
            self_21516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self')
            # Setting the type of the member 'libraries' of a type (line 76)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_21516, 'libraries', list_21515)

            if more_types_in_union_21514:
                # Runtime conditional SSA for else branch (line 75)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_21513) or more_types_in_union_21514):
            
            # Type idiom detected: calculating its left and rigth part (line 77)
            # Getting the type of 'str' (line 77)
            str_21517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 40), 'str')
            # Getting the type of 'self' (line 77)
            self_21518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'self')
            # Obtaining the member 'libraries' of a type (line 77)
            libraries_21519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 24), self_21518, 'libraries')
            
            (may_be_21520, more_types_in_union_21521) = may_be_subtype(str_21517, libraries_21519)

            if may_be_21520:

                if more_types_in_union_21521:
                    # Runtime conditional SSA (line 77)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'self' (line 77)
                self_21522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'self')
                # Obtaining the member 'libraries' of a type (line 77)
                libraries_21523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), self_21522, 'libraries')
                # Setting the type of the member 'libraries' of a type (line 77)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), self_21522, 'libraries', remove_not_subtype_from_union(libraries_21519, str))
                
                # Assigning a List to a Attribute (line 78):
                
                # Assigning a List to a Attribute (line 78):
                
                # Obtaining an instance of the builtin type 'list' (line 78)
                list_21524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'list')
                # Adding type elements to the builtin type 'list' instance (line 78)
                # Adding element type (line 78)
                # Getting the type of 'self' (line 78)
                self_21525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'self')
                # Obtaining the member 'libraries' of a type (line 78)
                libraries_21526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), self_21525, 'libraries')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 29), list_21524, libraries_21526)
                
                # Getting the type of 'self' (line 78)
                self_21527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self')
                # Setting the type of the member 'libraries' of a type (line 78)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), self_21527, 'libraries', list_21524)

                if more_types_in_union_21521:
                    # SSA join for if statement (line 77)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_21513 and more_types_in_union_21514):
                # SSA join for if statement (line 75)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 80)
        # Getting the type of 'self' (line 80)
        self_21528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'library_dirs' of a type (line 80)
        library_dirs_21529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_21528, 'library_dirs')
        # Getting the type of 'None' (line 80)
        None_21530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'None')
        
        (may_be_21531, more_types_in_union_21532) = may_be_none(library_dirs_21529, None_21530)

        if may_be_21531:

            if more_types_in_union_21532:
                # Runtime conditional SSA (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 81):
            
            # Assigning a List to a Attribute (line 81):
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_21533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 32), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            
            # Getting the type of 'self' (line 81)
            self_21534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'self')
            # Setting the type of the member 'library_dirs' of a type (line 81)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), self_21534, 'library_dirs', list_21533)

            if more_types_in_union_21532:
                # Runtime conditional SSA for else branch (line 80)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_21531) or more_types_in_union_21532):
            
            # Type idiom detected: calculating its left and rigth part (line 82)
            # Getting the type of 'str' (line 82)
            str_21535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'str')
            # Getting the type of 'self' (line 82)
            self_21536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'self')
            # Obtaining the member 'library_dirs' of a type (line 82)
            library_dirs_21537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 24), self_21536, 'library_dirs')
            
            (may_be_21538, more_types_in_union_21539) = may_be_subtype(str_21535, library_dirs_21537)

            if may_be_21538:

                if more_types_in_union_21539:
                    # Runtime conditional SSA (line 82)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'self' (line 82)
                self_21540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'self')
                # Obtaining the member 'library_dirs' of a type (line 82)
                library_dirs_21541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), self_21540, 'library_dirs')
                # Setting the type of the member 'library_dirs' of a type (line 82)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), self_21540, 'library_dirs', remove_not_subtype_from_union(library_dirs_21537, str))
                
                # Assigning a Call to a Attribute (line 83):
                
                # Assigning a Call to a Attribute (line 83):
                
                # Call to split(...): (line 83)
                # Processing the call arguments (line 83)
                # Getting the type of 'os' (line 83)
                os_21545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 56), 'os', False)
                # Obtaining the member 'pathsep' of a type (line 83)
                pathsep_21546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 56), os_21545, 'pathsep')
                # Processing the call keyword arguments (line 83)
                kwargs_21547 = {}
                # Getting the type of 'self' (line 83)
                self_21542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'self', False)
                # Obtaining the member 'library_dirs' of a type (line 83)
                library_dirs_21543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 32), self_21542, 'library_dirs')
                # Obtaining the member 'split' of a type (line 83)
                split_21544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 32), library_dirs_21543, 'split')
                # Calling split(args, kwargs) (line 83)
                split_call_result_21548 = invoke(stypy.reporting.localization.Localization(__file__, 83, 32), split_21544, *[pathsep_21546], **kwargs_21547)
                
                # Getting the type of 'self' (line 83)
                self_21549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self')
                # Setting the type of the member 'library_dirs' of a type (line 83)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_21549, 'library_dirs', split_call_result_21548)

                if more_types_in_union_21539:
                    # SSA join for if statement (line 82)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_21531 and more_types_in_union_21532):
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_21550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_21550


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.run.__dict__.__setitem__('stypy_localization', localization)
        config.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.run.__dict__.__setitem__('stypy_function_name', 'config.run')
        config.run.__dict__.__setitem__('stypy_param_names_list', [])
        config.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.run', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_21551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_21551


    @norecursion
    def _check_compiler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_compiler'
        module_type_store = module_type_store.open_function_context('_check_compiler', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
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

        str_21552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'str', "Check that 'self.compiler' really is a CCompiler object;\n        if not, make it one.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 99, 8))
        
        # 'from distutils.ccompiler import CCompiler, new_compiler' statement (line 99)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_21553 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 99, 8), 'distutils.ccompiler')

        if (type(import_21553) is not StypyTypeError):

            if (import_21553 != 'pyd_module'):
                __import__(import_21553)
                sys_modules_21554 = sys.modules[import_21553]
                import_from_module(stypy.reporting.localization.Localization(__file__, 99, 8), 'distutils.ccompiler', sys_modules_21554.module_type_store, module_type_store, ['CCompiler', 'new_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 99, 8), __file__, sys_modules_21554, sys_modules_21554.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import CCompiler, new_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 99, 8), 'distutils.ccompiler', None, module_type_store, ['CCompiler', 'new_compiler'], [CCompiler, new_compiler])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'distutils.ccompiler', import_21553)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        
        
        # Call to isinstance(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'self' (line 100)
        self_21556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'self', False)
        # Obtaining the member 'compiler' of a type (line 100)
        compiler_21557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 26), self_21556, 'compiler')
        # Getting the type of 'CCompiler' (line 100)
        CCompiler_21558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 'CCompiler', False)
        # Processing the call keyword arguments (line 100)
        kwargs_21559 = {}
        # Getting the type of 'isinstance' (line 100)
        isinstance_21555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 100)
        isinstance_call_result_21560 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), isinstance_21555, *[compiler_21557, CCompiler_21558], **kwargs_21559)
        
        # Applying the 'not' unary operator (line 100)
        result_not__21561 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), 'not', isinstance_call_result_21560)
        
        # Testing the type of an if condition (line 100)
        if_condition_21562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_not__21561)
        # Assigning a type to the variable 'if_condition_21562' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_21562', if_condition_21562)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 101):
        
        # Assigning a Call to a Attribute (line 101):
        
        # Call to new_compiler(...): (line 101)
        # Processing the call keyword arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_21564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'self', False)
        # Obtaining the member 'compiler' of a type (line 101)
        compiler_21565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 50), self_21564, 'compiler')
        keyword_21566 = compiler_21565
        # Getting the type of 'self' (line 102)
        self_21567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 102)
        dry_run_21568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 49), self_21567, 'dry_run')
        keyword_21569 = dry_run_21568
        int_21570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 69), 'int')
        keyword_21571 = int_21570
        kwargs_21572 = {'force': keyword_21571, 'dry_run': keyword_21569, 'compiler': keyword_21566}
        # Getting the type of 'new_compiler' (line 101)
        new_compiler_21563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 101)
        new_compiler_call_result_21573 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), new_compiler_21563, *[], **kwargs_21572)
        
        # Getting the type of 'self' (line 101)
        self_21574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self')
        # Setting the type of the member 'compiler' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_21574, 'compiler', new_compiler_call_result_21573)
        
        # Call to customize_compiler(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_21576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'self', False)
        # Obtaining the member 'compiler' of a type (line 103)
        compiler_21577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 31), self_21576, 'compiler')
        # Processing the call keyword arguments (line 103)
        kwargs_21578 = {}
        # Getting the type of 'customize_compiler' (line 103)
        customize_compiler_21575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 103)
        customize_compiler_call_result_21579 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), customize_compiler_21575, *[compiler_21577], **kwargs_21578)
        
        
        # Getting the type of 'self' (line 104)
        self_21580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'self')
        # Obtaining the member 'include_dirs' of a type (line 104)
        include_dirs_21581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), self_21580, 'include_dirs')
        # Testing the type of an if condition (line 104)
        if_condition_21582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 12), include_dirs_21581)
        # Assigning a type to the variable 'if_condition_21582' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'if_condition_21582', if_condition_21582)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_include_dirs(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_21586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 47), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 105)
        include_dirs_21587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 47), self_21586, 'include_dirs')
        # Processing the call keyword arguments (line 105)
        kwargs_21588 = {}
        # Getting the type of 'self' (line 105)
        self_21583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 105)
        compiler_21584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), self_21583, 'compiler')
        # Obtaining the member 'set_include_dirs' of a type (line 105)
        set_include_dirs_21585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), compiler_21584, 'set_include_dirs')
        # Calling set_include_dirs(args, kwargs) (line 105)
        set_include_dirs_call_result_21589 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), set_include_dirs_21585, *[include_dirs_21587], **kwargs_21588)
        
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 106)
        self_21590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'self')
        # Obtaining the member 'libraries' of a type (line 106)
        libraries_21591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 15), self_21590, 'libraries')
        # Testing the type of an if condition (line 106)
        if_condition_21592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 12), libraries_21591)
        # Assigning a type to the variable 'if_condition_21592' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'if_condition_21592', if_condition_21592)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_libraries(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_21596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 'self', False)
        # Obtaining the member 'libraries' of a type (line 107)
        libraries_21597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 44), self_21596, 'libraries')
        # Processing the call keyword arguments (line 107)
        kwargs_21598 = {}
        # Getting the type of 'self' (line 107)
        self_21593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 107)
        compiler_21594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), self_21593, 'compiler')
        # Obtaining the member 'set_libraries' of a type (line 107)
        set_libraries_21595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), compiler_21594, 'set_libraries')
        # Calling set_libraries(args, kwargs) (line 107)
        set_libraries_call_result_21599 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), set_libraries_21595, *[libraries_21597], **kwargs_21598)
        
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 108)
        self_21600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self')
        # Obtaining the member 'library_dirs' of a type (line 108)
        library_dirs_21601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_21600, 'library_dirs')
        # Testing the type of an if condition (line 108)
        if_condition_21602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), library_dirs_21601)
        # Assigning a type to the variable 'if_condition_21602' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_21602', if_condition_21602)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_library_dirs(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_21606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 47), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 109)
        library_dirs_21607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 47), self_21606, 'library_dirs')
        # Processing the call keyword arguments (line 109)
        kwargs_21608 = {}
        # Getting the type of 'self' (line 109)
        self_21603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 109)
        compiler_21604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_21603, 'compiler')
        # Obtaining the member 'set_library_dirs' of a type (line 109)
        set_library_dirs_21605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), compiler_21604, 'set_library_dirs')
        # Calling set_library_dirs(args, kwargs) (line 109)
        set_library_dirs_call_result_21609 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), set_library_dirs_21605, *[library_dirs_21607], **kwargs_21608)
        
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_compiler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_compiler' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_21610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_compiler'
        return stypy_return_type_21610


    @norecursion
    def _gen_temp_sourcefile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_gen_temp_sourcefile'
        module_type_store = module_type_store.open_function_context('_gen_temp_sourcefile', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_localization', localization)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_function_name', 'config._gen_temp_sourcefile')
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'lang'])
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_varargs_param_name', None)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._gen_temp_sourcefile.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._gen_temp_sourcefile', ['body', 'headers', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_gen_temp_sourcefile', localization, ['body', 'headers', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_gen_temp_sourcefile(...)' code ##################

        
        # Assigning a BinOp to a Name (line 113):
        
        # Assigning a BinOp to a Name (line 113):
        str_21611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'str', '_configtest')
        
        # Obtaining the type of the subscript
        # Getting the type of 'lang' (line 113)
        lang_21612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'lang')
        # Getting the type of 'LANG_EXT' (line 113)
        LANG_EXT_21613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'LANG_EXT')
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___21614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 35), LANG_EXT_21613, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_21615 = invoke(stypy.reporting.localization.Localization(__file__, 113, 35), getitem___21614, lang_21612)
        
        # Applying the binary operator '+' (line 113)
        result_add_21616 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 19), '+', str_21611, subscript_call_result_21615)
        
        # Assigning a type to the variable 'filename' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'filename', result_add_21616)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to open(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'filename' (line 114)
        filename_21618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'filename', False)
        str_21619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'str', 'w')
        # Processing the call keyword arguments (line 114)
        kwargs_21620 = {}
        # Getting the type of 'open' (line 114)
        open_21617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'open', False)
        # Calling open(args, kwargs) (line 114)
        open_call_result_21621 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), open_21617, *[filename_21618, str_21619], **kwargs_21620)
        
        # Assigning a type to the variable 'file' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'file', open_call_result_21621)
        
        # Getting the type of 'headers' (line 115)
        headers_21622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'headers')
        # Testing the type of an if condition (line 115)
        if_condition_21623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), headers_21622)
        # Assigning a type to the variable 'if_condition_21623' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_21623', if_condition_21623)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'headers' (line 116)
        headers_21624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'headers')
        # Testing the type of a for loop iterable (line 116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 12), headers_21624)
        # Getting the type of the for loop variable (line 116)
        for_loop_var_21625 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 12), headers_21624)
        # Assigning a type to the variable 'header' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'header', for_loop_var_21625)
        # SSA begins for a for statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 117)
        # Processing the call arguments (line 117)
        str_21628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'str', '#include <%s>\n')
        # Getting the type of 'header' (line 117)
        header_21629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 47), 'header', False)
        # Applying the binary operator '%' (line 117)
        result_mod_21630 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 27), '%', str_21628, header_21629)
        
        # Processing the call keyword arguments (line 117)
        kwargs_21631 = {}
        # Getting the type of 'file' (line 117)
        file_21626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'file', False)
        # Obtaining the member 'write' of a type (line 117)
        write_21627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), file_21626, 'write')
        # Calling write(args, kwargs) (line 117)
        write_call_result_21632 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), write_21627, *[result_mod_21630], **kwargs_21631)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 118)
        # Processing the call arguments (line 118)
        str_21635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'str', '\n')
        # Processing the call keyword arguments (line 118)
        kwargs_21636 = {}
        # Getting the type of 'file' (line 118)
        file_21633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'file', False)
        # Obtaining the member 'write' of a type (line 118)
        write_21634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), file_21633, 'write')
        # Calling write(args, kwargs) (line 118)
        write_call_result_21637 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), write_21634, *[str_21635], **kwargs_21636)
        
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'body' (line 119)
        body_21640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'body', False)
        # Processing the call keyword arguments (line 119)
        kwargs_21641 = {}
        # Getting the type of 'file' (line 119)
        file_21638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'file', False)
        # Obtaining the member 'write' of a type (line 119)
        write_21639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), file_21638, 'write')
        # Calling write(args, kwargs) (line 119)
        write_call_result_21642 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), write_21639, *[body_21640], **kwargs_21641)
        
        
        
        
        # Obtaining the type of the subscript
        int_21643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'int')
        # Getting the type of 'body' (line 120)
        body_21644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'body')
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___21645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), body_21644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_21646 = invoke(stypy.reporting.localization.Localization(__file__, 120, 11), getitem___21645, int_21643)
        
        str_21647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'str', '\n')
        # Applying the binary operator '!=' (line 120)
        result_ne_21648 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '!=', subscript_call_result_21646, str_21647)
        
        # Testing the type of an if condition (line 120)
        if_condition_21649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_ne_21648)
        # Assigning a type to the variable 'if_condition_21649' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_21649', if_condition_21649)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 121)
        # Processing the call arguments (line 121)
        str_21652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'str', '\n')
        # Processing the call keyword arguments (line 121)
        kwargs_21653 = {}
        # Getting the type of 'file' (line 121)
        file_21650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'file', False)
        # Obtaining the member 'write' of a type (line 121)
        write_21651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), file_21650, 'write')
        # Calling write(args, kwargs) (line 121)
        write_call_result_21654 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), write_21651, *[str_21652], **kwargs_21653)
        
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_21657 = {}
        # Getting the type of 'file' (line 122)
        file_21655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'file', False)
        # Obtaining the member 'close' of a type (line 122)
        close_21656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), file_21655, 'close')
        # Calling close(args, kwargs) (line 122)
        close_call_result_21658 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), close_21656, *[], **kwargs_21657)
        
        # Getting the type of 'filename' (line 123)
        filename_21659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'filename')
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', filename_21659)
        
        # ################# End of '_gen_temp_sourcefile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_gen_temp_sourcefile' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_21660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_gen_temp_sourcefile'
        return stypy_return_type_21660


    @norecursion
    def _preprocess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_preprocess'
        module_type_store = module_type_store.open_function_context('_preprocess', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._preprocess.__dict__.__setitem__('stypy_localization', localization)
        config._preprocess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._preprocess.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._preprocess.__dict__.__setitem__('stypy_function_name', 'config._preprocess')
        config._preprocess.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'lang'])
        config._preprocess.__dict__.__setitem__('stypy_varargs_param_name', None)
        config._preprocess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._preprocess.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._preprocess.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._preprocess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._preprocess.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._preprocess', ['body', 'headers', 'include_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_preprocess', localization, ['body', 'headers', 'include_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_preprocess(...)' code ##################

        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to _gen_temp_sourcefile(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'body' (line 126)
        body_21663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 40), 'body', False)
        # Getting the type of 'headers' (line 126)
        headers_21664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 46), 'headers', False)
        # Getting the type of 'lang' (line 126)
        lang_21665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 55), 'lang', False)
        # Processing the call keyword arguments (line 126)
        kwargs_21666 = {}
        # Getting the type of 'self' (line 126)
        self_21661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'self', False)
        # Obtaining the member '_gen_temp_sourcefile' of a type (line 126)
        _gen_temp_sourcefile_21662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 14), self_21661, '_gen_temp_sourcefile')
        # Calling _gen_temp_sourcefile(args, kwargs) (line 126)
        _gen_temp_sourcefile_call_result_21667 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), _gen_temp_sourcefile_21662, *[body_21663, headers_21664, lang_21665], **kwargs_21666)
        
        # Assigning a type to the variable 'src' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'src', _gen_temp_sourcefile_call_result_21667)
        
        # Assigning a Str to a Name (line 127):
        
        # Assigning a Str to a Name (line 127):
        str_21668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 14), 'str', '_configtest.i')
        # Assigning a type to the variable 'out' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'out', str_21668)
        
        # Call to extend(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_21672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'src' (line 128)
        src_21673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 31), list_21672, src_21673)
        # Adding element type (line 128)
        # Getting the type of 'out' (line 128)
        out_21674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'out', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 31), list_21672, out_21674)
        
        # Processing the call keyword arguments (line 128)
        kwargs_21675 = {}
        # Getting the type of 'self' (line 128)
        self_21669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self', False)
        # Obtaining the member 'temp_files' of a type (line 128)
        temp_files_21670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_21669, 'temp_files')
        # Obtaining the member 'extend' of a type (line 128)
        extend_21671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), temp_files_21670, 'extend')
        # Calling extend(args, kwargs) (line 128)
        extend_call_result_21676 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), extend_21671, *[list_21672], **kwargs_21675)
        
        
        # Call to preprocess(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'src' (line 129)
        src_21680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'src', False)
        # Getting the type of 'out' (line 129)
        out_21681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'out', False)
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'include_dirs' (line 129)
        include_dirs_21682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'include_dirs', False)
        keyword_21683 = include_dirs_21682
        kwargs_21684 = {'include_dirs': keyword_21683}
        # Getting the type of 'self' (line 129)
        self_21677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 129)
        compiler_21678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_21677, 'compiler')
        # Obtaining the member 'preprocess' of a type (line 129)
        preprocess_21679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), compiler_21678, 'preprocess')
        # Calling preprocess(args, kwargs) (line 129)
        preprocess_call_result_21685 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), preprocess_21679, *[src_21680, out_21681], **kwargs_21684)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 130)
        tuple_21686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 130)
        # Adding element type (line 130)
        # Getting the type of 'src' (line 130)
        src_21687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_21686, src_21687)
        # Adding element type (line 130)
        # Getting the type of 'out' (line 130)
        out_21688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'out')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_21686, out_21688)
        
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', tuple_21686)
        
        # ################# End of '_preprocess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_preprocess' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_21689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21689)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_preprocess'
        return stypy_return_type_21689


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to _gen_temp_sourcefile(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'body' (line 133)
        body_21692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'body', False)
        # Getting the type of 'headers' (line 133)
        headers_21693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 46), 'headers', False)
        # Getting the type of 'lang' (line 133)
        lang_21694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 55), 'lang', False)
        # Processing the call keyword arguments (line 133)
        kwargs_21695 = {}
        # Getting the type of 'self' (line 133)
        self_21690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'self', False)
        # Obtaining the member '_gen_temp_sourcefile' of a type (line 133)
        _gen_temp_sourcefile_21691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 14), self_21690, '_gen_temp_sourcefile')
        # Calling _gen_temp_sourcefile(args, kwargs) (line 133)
        _gen_temp_sourcefile_call_result_21696 = invoke(stypy.reporting.localization.Localization(__file__, 133, 14), _gen_temp_sourcefile_21691, *[body_21692, headers_21693, lang_21694], **kwargs_21695)
        
        # Assigning a type to the variable 'src' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'src', _gen_temp_sourcefile_call_result_21696)
        
        # Getting the type of 'self' (line 134)
        self_21697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'self')
        # Obtaining the member 'dump_source' of a type (line 134)
        dump_source_21698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), self_21697, 'dump_source')
        # Testing the type of an if condition (line 134)
        if_condition_21699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), dump_source_21698)
        # Assigning a type to the variable 'if_condition_21699' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_21699', if_condition_21699)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dump_file(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'src' (line 135)
        src_21701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'src', False)
        str_21702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'str', "compiling '%s':")
        # Getting the type of 'src' (line 135)
        src_21703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 47), 'src', False)
        # Applying the binary operator '%' (line 135)
        result_mod_21704 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 27), '%', str_21702, src_21703)
        
        # Processing the call keyword arguments (line 135)
        kwargs_21705 = {}
        # Getting the type of 'dump_file' (line 135)
        dump_file_21700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'dump_file', False)
        # Calling dump_file(args, kwargs) (line 135)
        dump_file_call_result_21706 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), dump_file_21700, *[src_21701, result_mod_21704], **kwargs_21705)
        
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 136):
        
        # Assigning a Subscript to a Name (line 136):
        
        # Obtaining the type of the subscript
        int_21707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
        
        # Call to object_filenames(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_21711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        # Getting the type of 'src' (line 136)
        src_21712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), list_21711, src_21712)
        
        # Processing the call keyword arguments (line 136)
        kwargs_21713 = {}
        # Getting the type of 'self' (line 136)
        self_21708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'self', False)
        # Obtaining the member 'compiler' of a type (line 136)
        compiler_21709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 17), self_21708, 'compiler')
        # Obtaining the member 'object_filenames' of a type (line 136)
        object_filenames_21710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 17), compiler_21709, 'object_filenames')
        # Calling object_filenames(args, kwargs) (line 136)
        object_filenames_call_result_21714 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), object_filenames_21710, *[list_21711], **kwargs_21713)
        
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___21715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), object_filenames_call_result_21714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_21716 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), getitem___21715, int_21707)
        
        # Assigning a type to the variable 'tuple_var_assignment_21445' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_21445', subscript_call_result_21716)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_var_assignment_21445' (line 136)
        tuple_var_assignment_21445_21717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_21445')
        # Assigning a type to the variable 'obj' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 9), 'obj', tuple_var_assignment_21445_21717)
        
        # Call to extend(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_21721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'src' (line 137)
        src_21722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 31), list_21721, src_21722)
        # Adding element type (line 137)
        # Getting the type of 'obj' (line 137)
        obj_21723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 31), list_21721, obj_21723)
        
        # Processing the call keyword arguments (line 137)
        kwargs_21724 = {}
        # Getting the type of 'self' (line 137)
        self_21718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self', False)
        # Obtaining the member 'temp_files' of a type (line 137)
        temp_files_21719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_21718, 'temp_files')
        # Obtaining the member 'extend' of a type (line 137)
        extend_21720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), temp_files_21719, 'extend')
        # Calling extend(args, kwargs) (line 137)
        extend_call_result_21725 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), extend_21720, *[list_21721], **kwargs_21724)
        
        
        # Call to compile(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_21729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'src' (line 138)
        src_21730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 30), list_21729, src_21730)
        
        # Processing the call keyword arguments (line 138)
        # Getting the type of 'include_dirs' (line 138)
        include_dirs_21731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 50), 'include_dirs', False)
        keyword_21732 = include_dirs_21731
        kwargs_21733 = {'include_dirs': keyword_21732}
        # Getting the type of 'self' (line 138)
        self_21726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 138)
        compiler_21727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_21726, 'compiler')
        # Obtaining the member 'compile' of a type (line 138)
        compile_21728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), compiler_21727, 'compile')
        # Calling compile(args, kwargs) (line 138)
        compile_call_result_21734 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), compile_21728, *[list_21729], **kwargs_21733)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 139)
        tuple_21735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 139)
        # Adding element type (line 139)
        # Getting the type of 'src' (line 139)
        src_21736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), tuple_21735, src_21736)
        # Adding element type (line 139)
        # Getting the type of 'obj' (line 139)
        obj_21737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'obj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), tuple_21735, obj_21737)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', tuple_21735)
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_21738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_21738


    @norecursion
    def _link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_link'
        module_type_store = module_type_store.open_function_context('_link', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Tuple (line 143):
        
        # Assigning a Subscript to a Name (line 143):
        
        # Obtaining the type of the subscript
        int_21739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'int')
        
        # Call to _compile(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'body' (line 143)
        body_21742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'body', False)
        # Getting the type of 'headers' (line 143)
        headers_21743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'headers', False)
        # Getting the type of 'include_dirs' (line 143)
        include_dirs_21744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 50), 'include_dirs', False)
        # Getting the type of 'lang' (line 143)
        lang_21745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 64), 'lang', False)
        # Processing the call keyword arguments (line 143)
        kwargs_21746 = {}
        # Getting the type of 'self' (line 143)
        self_21740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'self', False)
        # Obtaining the member '_compile' of a type (line 143)
        _compile_21741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 21), self_21740, '_compile')
        # Calling _compile(args, kwargs) (line 143)
        _compile_call_result_21747 = invoke(stypy.reporting.localization.Localization(__file__, 143, 21), _compile_21741, *[body_21742, headers_21743, include_dirs_21744, lang_21745], **kwargs_21746)
        
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___21748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), _compile_call_result_21747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_21749 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), getitem___21748, int_21739)
        
        # Assigning a type to the variable 'tuple_var_assignment_21446' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'tuple_var_assignment_21446', subscript_call_result_21749)
        
        # Assigning a Subscript to a Name (line 143):
        
        # Obtaining the type of the subscript
        int_21750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'int')
        
        # Call to _compile(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'body' (line 143)
        body_21753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'body', False)
        # Getting the type of 'headers' (line 143)
        headers_21754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'headers', False)
        # Getting the type of 'include_dirs' (line 143)
        include_dirs_21755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 50), 'include_dirs', False)
        # Getting the type of 'lang' (line 143)
        lang_21756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 64), 'lang', False)
        # Processing the call keyword arguments (line 143)
        kwargs_21757 = {}
        # Getting the type of 'self' (line 143)
        self_21751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'self', False)
        # Obtaining the member '_compile' of a type (line 143)
        _compile_21752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 21), self_21751, '_compile')
        # Calling _compile(args, kwargs) (line 143)
        _compile_call_result_21758 = invoke(stypy.reporting.localization.Localization(__file__, 143, 21), _compile_21752, *[body_21753, headers_21754, include_dirs_21755, lang_21756], **kwargs_21757)
        
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___21759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), _compile_call_result_21758, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_21760 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), getitem___21759, int_21750)
        
        # Assigning a type to the variable 'tuple_var_assignment_21447' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'tuple_var_assignment_21447', subscript_call_result_21760)
        
        # Assigning a Name to a Name (line 143):
        # Getting the type of 'tuple_var_assignment_21446' (line 143)
        tuple_var_assignment_21446_21761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'tuple_var_assignment_21446')
        # Assigning a type to the variable 'src' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'src', tuple_var_assignment_21446_21761)
        
        # Assigning a Name to a Name (line 143):
        # Getting the type of 'tuple_var_assignment_21447' (line 143)
        tuple_var_assignment_21447_21762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'tuple_var_assignment_21447')
        # Assigning a type to the variable 'obj' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'obj', tuple_var_assignment_21447_21762)
        
        # Assigning a Subscript to a Name (line 144):
        
        # Assigning a Subscript to a Name (line 144):
        
        # Obtaining the type of the subscript
        int_21763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 55), 'int')
        
        # Call to splitext(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to basename(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'src' (line 144)
        src_21770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 49), 'src', False)
        # Processing the call keyword arguments (line 144)
        kwargs_21771 = {}
        # Getting the type of 'os' (line 144)
        os_21767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'os', False)
        # Obtaining the member 'path' of a type (line 144)
        path_21768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 32), os_21767, 'path')
        # Obtaining the member 'basename' of a type (line 144)
        basename_21769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 32), path_21768, 'basename')
        # Calling basename(args, kwargs) (line 144)
        basename_call_result_21772 = invoke(stypy.reporting.localization.Localization(__file__, 144, 32), basename_21769, *[src_21770], **kwargs_21771)
        
        # Processing the call keyword arguments (line 144)
        kwargs_21773 = {}
        # Getting the type of 'os' (line 144)
        os_21764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 144)
        path_21765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), os_21764, 'path')
        # Obtaining the member 'splitext' of a type (line 144)
        splitext_21766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), path_21765, 'splitext')
        # Calling splitext(args, kwargs) (line 144)
        splitext_call_result_21774 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), splitext_21766, *[basename_call_result_21772], **kwargs_21773)
        
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___21775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), splitext_call_result_21774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_21776 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), getitem___21775, int_21763)
        
        # Assigning a type to the variable 'prog' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'prog', subscript_call_result_21776)
        
        # Call to link_executable(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_21780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        # Getting the type of 'obj' (line 145)
        obj_21781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 38), list_21780, obj_21781)
        
        # Getting the type of 'prog' (line 145)
        prog_21782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 45), 'prog', False)
        # Processing the call keyword arguments (line 145)
        # Getting the type of 'libraries' (line 146)
        libraries_21783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 48), 'libraries', False)
        keyword_21784 = libraries_21783
        # Getting the type of 'library_dirs' (line 147)
        library_dirs_21785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'library_dirs', False)
        keyword_21786 = library_dirs_21785
        # Getting the type of 'lang' (line 148)
        lang_21787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 50), 'lang', False)
        keyword_21788 = lang_21787
        kwargs_21789 = {'libraries': keyword_21784, 'target_lang': keyword_21788, 'library_dirs': keyword_21786}
        # Getting the type of 'self' (line 145)
        self_21777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 145)
        compiler_21778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_21777, 'compiler')
        # Obtaining the member 'link_executable' of a type (line 145)
        link_executable_21779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), compiler_21778, 'link_executable')
        # Calling link_executable(args, kwargs) (line 145)
        link_executable_call_result_21790 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), link_executable_21779, *[list_21780, prog_21782], **kwargs_21789)
        
        
        
        # Getting the type of 'self' (line 150)
        self_21791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'self')
        # Obtaining the member 'compiler' of a type (line 150)
        compiler_21792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), self_21791, 'compiler')
        # Obtaining the member 'exe_extension' of a type (line 150)
        exe_extension_21793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), compiler_21792, 'exe_extension')
        # Getting the type of 'None' (line 150)
        None_21794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 46), 'None')
        # Applying the binary operator 'isnot' (line 150)
        result_is_not_21795 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), 'isnot', exe_extension_21793, None_21794)
        
        # Testing the type of an if condition (line 150)
        if_condition_21796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_is_not_21795)
        # Assigning a type to the variable 'if_condition_21796' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_21796', if_condition_21796)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 151):
        
        # Assigning a BinOp to a Name (line 151):
        # Getting the type of 'prog' (line 151)
        prog_21797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'prog')
        # Getting the type of 'self' (line 151)
        self_21798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'self')
        # Obtaining the member 'compiler' of a type (line 151)
        compiler_21799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 26), self_21798, 'compiler')
        # Obtaining the member 'exe_extension' of a type (line 151)
        exe_extension_21800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 26), compiler_21799, 'exe_extension')
        # Applying the binary operator '+' (line 151)
        result_add_21801 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 19), '+', prog_21797, exe_extension_21800)
        
        # Assigning a type to the variable 'prog' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'prog', result_add_21801)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'prog' (line 152)
        prog_21805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'prog', False)
        # Processing the call keyword arguments (line 152)
        kwargs_21806 = {}
        # Getting the type of 'self' (line 152)
        self_21802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self', False)
        # Obtaining the member 'temp_files' of a type (line 152)
        temp_files_21803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_21802, 'temp_files')
        # Obtaining the member 'append' of a type (line 152)
        append_21804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), temp_files_21803, 'append')
        # Calling append(args, kwargs) (line 152)
        append_call_result_21807 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), append_21804, *[prog_21805], **kwargs_21806)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_21808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        # Getting the type of 'src' (line 154)
        src_21809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), tuple_21808, src_21809)
        # Adding element type (line 154)
        # Getting the type of 'obj' (line 154)
        obj_21810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'obj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), tuple_21808, obj_21810)
        # Adding element type (line 154)
        # Getting the type of 'prog' (line 154)
        prog_21811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'prog')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), tuple_21808, prog_21811)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', tuple_21808)
        
        # ################# End of '_link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_link' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_21812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_link'
        return stypy_return_type_21812


    @norecursion
    def _clean(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_clean'
        module_type_store = module_type_store.open_function_context('_clean', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config._clean.__dict__.__setitem__('stypy_localization', localization)
        config._clean.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config._clean.__dict__.__setitem__('stypy_type_store', module_type_store)
        config._clean.__dict__.__setitem__('stypy_function_name', 'config._clean')
        config._clean.__dict__.__setitem__('stypy_param_names_list', [])
        config._clean.__dict__.__setitem__('stypy_varargs_param_name', 'filenames')
        config._clean.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config._clean.__dict__.__setitem__('stypy_call_defaults', defaults)
        config._clean.__dict__.__setitem__('stypy_call_varargs', varargs)
        config._clean.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config._clean.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config._clean', [], 'filenames', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_clean', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_clean(...)' code ##################

        
        
        # Getting the type of 'filenames' (line 157)
        filenames_21813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'filenames')
        # Applying the 'not' unary operator (line 157)
        result_not__21814 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 11), 'not', filenames_21813)
        
        # Testing the type of an if condition (line 157)
        if_condition_21815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), result_not__21814)
        # Assigning a type to the variable 'if_condition_21815' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_21815', if_condition_21815)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 158):
        
        # Assigning a Attribute to a Name (line 158):
        # Getting the type of 'self' (line 158)
        self_21816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'self')
        # Obtaining the member 'temp_files' of a type (line 158)
        temp_files_21817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 24), self_21816, 'temp_files')
        # Assigning a type to the variable 'filenames' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'filenames', temp_files_21817)
        
        # Assigning a List to a Attribute (line 159):
        
        # Assigning a List to a Attribute (line 159):
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_21818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        
        # Getting the type of 'self' (line 159)
        self_21819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self')
        # Setting the type of the member 'temp_files' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_21819, 'temp_files', list_21818)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 160)
        # Processing the call arguments (line 160)
        str_21822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 17), 'str', 'removing: %s')
        
        # Call to join(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'filenames' (line 160)
        filenames_21825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'filenames', False)
        # Processing the call keyword arguments (line 160)
        kwargs_21826 = {}
        str_21823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'str', ' ')
        # Obtaining the member 'join' of a type (line 160)
        join_21824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 33), str_21823, 'join')
        # Calling join(args, kwargs) (line 160)
        join_call_result_21827 = invoke(stypy.reporting.localization.Localization(__file__, 160, 33), join_21824, *[filenames_21825], **kwargs_21826)
        
        # Processing the call keyword arguments (line 160)
        kwargs_21828 = {}
        # Getting the type of 'log' (line 160)
        log_21820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 160)
        info_21821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), log_21820, 'info')
        # Calling info(args, kwargs) (line 160)
        info_call_result_21829 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), info_21821, *[str_21822, join_call_result_21827], **kwargs_21828)
        
        
        # Getting the type of 'filenames' (line 161)
        filenames_21830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'filenames')
        # Testing the type of a for loop iterable (line 161)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 8), filenames_21830)
        # Getting the type of the for loop variable (line 161)
        for_loop_var_21831 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 8), filenames_21830)
        # Assigning a type to the variable 'filename' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'filename', for_loop_var_21831)
        # SSA begins for a for statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to remove(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'filename' (line 163)
        filename_21834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'filename', False)
        # Processing the call keyword arguments (line 163)
        kwargs_21835 = {}
        # Getting the type of 'os' (line 163)
        os_21832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'os', False)
        # Obtaining the member 'remove' of a type (line 163)
        remove_21833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), os_21832, 'remove')
        # Calling remove(args, kwargs) (line 163)
        remove_call_result_21836 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), remove_21833, *[filename_21834], **kwargs_21835)
        
        # SSA branch for the except part of a try statement (line 162)
        # SSA branch for the except 'OSError' branch of a try statement (line 162)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_clean(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_clean' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_21837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_clean'
        return stypy_return_type_21837


    @norecursion
    def try_cpp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 178)
        None_21838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'None')
        # Getting the type of 'None' (line 178)
        None_21839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 41), 'None')
        # Getting the type of 'None' (line 178)
        None_21840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 60), 'None')
        str_21841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 71), 'str', 'c')
        defaults = [None_21838, None_21839, None_21840, str_21841]
        # Create a new context for function 'try_cpp'
        module_type_store = module_type_store.open_function_context('try_cpp', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.try_cpp.__dict__.__setitem__('stypy_localization', localization)
        config.try_cpp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.try_cpp.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.try_cpp.__dict__.__setitem__('stypy_function_name', 'config.try_cpp')
        config.try_cpp.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'lang'])
        config.try_cpp.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.try_cpp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.try_cpp.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.try_cpp.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.try_cpp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.try_cpp.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.try_cpp', ['body', 'headers', 'include_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'try_cpp', localization, ['body', 'headers', 'include_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'try_cpp(...)' code ##################

        str_21842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', "Construct a source file from 'body' (a string containing lines\n        of C/C++ code) and 'headers' (a list of header files to include)\n        and run it through the preprocessor.  Return true if the\n        preprocessor succeeded, false if there were any errors.\n        ('body' probably isn't of much use, but what the heck.)\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 185, 8))
        
        # 'from distutils.ccompiler import CompileError' statement (line 185)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_21843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 185, 8), 'distutils.ccompiler')

        if (type(import_21843) is not StypyTypeError):

            if (import_21843 != 'pyd_module'):
                __import__(import_21843)
                sys_modules_21844 = sys.modules[import_21843]
                import_from_module(stypy.reporting.localization.Localization(__file__, 185, 8), 'distutils.ccompiler', sys_modules_21844.module_type_store, module_type_store, ['CompileError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 185, 8), __file__, sys_modules_21844, sys_modules_21844.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import CompileError

                import_from_module(stypy.reporting.localization.Localization(__file__, 185, 8), 'distutils.ccompiler', None, module_type_store, ['CompileError'], [CompileError])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'distutils.ccompiler', import_21843)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Call to _check_compiler(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_21847 = {}
        # Getting the type of 'self' (line 186)
        self_21845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 186)
        _check_compiler_21846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_21845, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 186)
        _check_compiler_call_result_21848 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), _check_compiler_21846, *[], **kwargs_21847)
        
        
        # Assigning a Num to a Name (line 187):
        
        # Assigning a Num to a Name (line 187):
        int_21849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 13), 'int')
        # Assigning a type to the variable 'ok' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'ok', int_21849)
        
        
        # SSA begins for try-except statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _preprocess(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'body' (line 189)
        body_21852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'body', False)
        # Getting the type of 'headers' (line 189)
        headers_21853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'headers', False)
        # Getting the type of 'include_dirs' (line 189)
        include_dirs_21854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'include_dirs', False)
        # Getting the type of 'lang' (line 189)
        lang_21855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 58), 'lang', False)
        # Processing the call keyword arguments (line 189)
        kwargs_21856 = {}
        # Getting the type of 'self' (line 189)
        self_21850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self', False)
        # Obtaining the member '_preprocess' of a type (line 189)
        _preprocess_21851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_21850, '_preprocess')
        # Calling _preprocess(args, kwargs) (line 189)
        _preprocess_call_result_21857 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), _preprocess_21851, *[body_21852, headers_21853, include_dirs_21854, lang_21855], **kwargs_21856)
        
        # SSA branch for the except part of a try statement (line 188)
        # SSA branch for the except 'CompileError' branch of a try statement (line 188)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 191):
        
        # Assigning a Num to a Name (line 191):
        int_21858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'int')
        # Assigning a type to the variable 'ok' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'ok', int_21858)
        # SSA join for try-except statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _clean(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_21861 = {}
        # Getting the type of 'self' (line 193)
        self_21859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self', False)
        # Obtaining the member '_clean' of a type (line 193)
        _clean_21860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_21859, '_clean')
        # Calling _clean(args, kwargs) (line 193)
        _clean_call_result_21862 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), _clean_21860, *[], **kwargs_21861)
        
        # Getting the type of 'ok' (line 194)
        ok_21863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'ok')
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', ok_21863)
        
        # ################# End of 'try_cpp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'try_cpp' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_21864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'try_cpp'
        return stypy_return_type_21864


    @norecursion
    def search_cpp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 196)
        None_21865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'None')
        # Getting the type of 'None' (line 196)
        None_21866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 53), 'None')
        # Getting the type of 'None' (line 196)
        None_21867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 72), 'None')
        str_21868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 24), 'str', 'c')
        defaults = [None_21865, None_21866, None_21867, str_21868]
        # Create a new context for function 'search_cpp'
        module_type_store = module_type_store.open_function_context('search_cpp', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.search_cpp.__dict__.__setitem__('stypy_localization', localization)
        config.search_cpp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.search_cpp.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.search_cpp.__dict__.__setitem__('stypy_function_name', 'config.search_cpp')
        config.search_cpp.__dict__.__setitem__('stypy_param_names_list', ['pattern', 'body', 'headers', 'include_dirs', 'lang'])
        config.search_cpp.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.search_cpp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.search_cpp.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.search_cpp.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.search_cpp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.search_cpp.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.search_cpp', ['pattern', 'body', 'headers', 'include_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'search_cpp', localization, ['pattern', 'body', 'headers', 'include_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'search_cpp(...)' code ##################

        str_21869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', "Construct a source file (just like 'try_cpp()'), run it through\n        the preprocessor, and return true if any line of the output matches\n        'pattern'.  'pattern' should either be a compiled regex object or a\n        string containing a regex.  If both 'body' and 'headers' are None,\n        preprocesses an empty file -- which can be useful to determine the\n        symbols the preprocessor and compiler set by default.\n        ")
        
        # Call to _check_compiler(...): (line 205)
        # Processing the call keyword arguments (line 205)
        kwargs_21872 = {}
        # Getting the type of 'self' (line 205)
        self_21870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 205)
        _check_compiler_21871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_21870, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 205)
        _check_compiler_call_result_21873 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), _check_compiler_21871, *[], **kwargs_21872)
        
        
        # Assigning a Call to a Tuple (line 206):
        
        # Assigning a Subscript to a Name (line 206):
        
        # Obtaining the type of the subscript
        int_21874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
        
        # Call to _preprocess(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'body' (line 206)
        body_21877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'body', False)
        # Getting the type of 'headers' (line 206)
        headers_21878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 42), 'headers', False)
        # Getting the type of 'include_dirs' (line 206)
        include_dirs_21879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 51), 'include_dirs', False)
        # Getting the type of 'lang' (line 206)
        lang_21880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 65), 'lang', False)
        # Processing the call keyword arguments (line 206)
        kwargs_21881 = {}
        # Getting the type of 'self' (line 206)
        self_21875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'self', False)
        # Obtaining the member '_preprocess' of a type (line 206)
        _preprocess_21876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 19), self_21875, '_preprocess')
        # Calling _preprocess(args, kwargs) (line 206)
        _preprocess_call_result_21882 = invoke(stypy.reporting.localization.Localization(__file__, 206, 19), _preprocess_21876, *[body_21877, headers_21878, include_dirs_21879, lang_21880], **kwargs_21881)
        
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___21883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), _preprocess_call_result_21882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_21884 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___21883, int_21874)
        
        # Assigning a type to the variable 'tuple_var_assignment_21448' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_21448', subscript_call_result_21884)
        
        # Assigning a Subscript to a Name (line 206):
        
        # Obtaining the type of the subscript
        int_21885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
        
        # Call to _preprocess(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'body' (line 206)
        body_21888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'body', False)
        # Getting the type of 'headers' (line 206)
        headers_21889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 42), 'headers', False)
        # Getting the type of 'include_dirs' (line 206)
        include_dirs_21890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 51), 'include_dirs', False)
        # Getting the type of 'lang' (line 206)
        lang_21891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 65), 'lang', False)
        # Processing the call keyword arguments (line 206)
        kwargs_21892 = {}
        # Getting the type of 'self' (line 206)
        self_21886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'self', False)
        # Obtaining the member '_preprocess' of a type (line 206)
        _preprocess_21887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 19), self_21886, '_preprocess')
        # Calling _preprocess(args, kwargs) (line 206)
        _preprocess_call_result_21893 = invoke(stypy.reporting.localization.Localization(__file__, 206, 19), _preprocess_21887, *[body_21888, headers_21889, include_dirs_21890, lang_21891], **kwargs_21892)
        
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___21894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), _preprocess_call_result_21893, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_21895 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___21894, int_21885)
        
        # Assigning a type to the variable 'tuple_var_assignment_21449' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_21449', subscript_call_result_21895)
        
        # Assigning a Name to a Name (line 206):
        # Getting the type of 'tuple_var_assignment_21448' (line 206)
        tuple_var_assignment_21448_21896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_21448')
        # Assigning a type to the variable 'src' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'src', tuple_var_assignment_21448_21896)
        
        # Assigning a Name to a Name (line 206):
        # Getting the type of 'tuple_var_assignment_21449' (line 206)
        tuple_var_assignment_21449_21897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_21449')
        # Assigning a type to the variable 'out' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'out', tuple_var_assignment_21449_21897)
        
        # Type idiom detected: calculating its left and rigth part (line 208)
        # Getting the type of 'str' (line 208)
        str_21898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 31), 'str')
        # Getting the type of 'pattern' (line 208)
        pattern_21899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'pattern')
        
        (may_be_21900, more_types_in_union_21901) = may_be_subtype(str_21898, pattern_21899)

        if may_be_21900:

            if more_types_in_union_21901:
                # Runtime conditional SSA (line 208)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'pattern' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'pattern', remove_not_subtype_from_union(pattern_21899, str))
            
            # Assigning a Call to a Name (line 209):
            
            # Assigning a Call to a Name (line 209):
            
            # Call to compile(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'pattern' (line 209)
            pattern_21904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'pattern', False)
            # Processing the call keyword arguments (line 209)
            kwargs_21905 = {}
            # Getting the type of 're' (line 209)
            re_21902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 're', False)
            # Obtaining the member 'compile' of a type (line 209)
            compile_21903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 22), re_21902, 'compile')
            # Calling compile(args, kwargs) (line 209)
            compile_call_result_21906 = invoke(stypy.reporting.localization.Localization(__file__, 209, 22), compile_21903, *[pattern_21904], **kwargs_21905)
            
            # Assigning a type to the variable 'pattern' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'pattern', compile_call_result_21906)

            if more_types_in_union_21901:
                # SSA join for if statement (line 208)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to open(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'out' (line 211)
        out_21908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'out', False)
        # Processing the call keyword arguments (line 211)
        kwargs_21909 = {}
        # Getting the type of 'open' (line 211)
        open_21907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'open', False)
        # Calling open(args, kwargs) (line 211)
        open_call_result_21910 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), open_21907, *[out_21908], **kwargs_21909)
        
        # Assigning a type to the variable 'file' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'file', open_call_result_21910)
        
        # Assigning a Num to a Name (line 212):
        
        # Assigning a Num to a Name (line 212):
        int_21911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 16), 'int')
        # Assigning a type to the variable 'match' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'match', int_21911)
        
        int_21912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 14), 'int')
        # Testing the type of an if condition (line 213)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), int_21912)
        # SSA begins for while statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to readline(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_21915 = {}
        # Getting the type of 'file' (line 214)
        file_21913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'file', False)
        # Obtaining the member 'readline' of a type (line 214)
        readline_21914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), file_21913, 'readline')
        # Calling readline(args, kwargs) (line 214)
        readline_call_result_21916 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), readline_21914, *[], **kwargs_21915)
        
        # Assigning a type to the variable 'line' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'line', readline_call_result_21916)
        
        
        # Getting the type of 'line' (line 215)
        line_21917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'line')
        str_21918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'str', '')
        # Applying the binary operator '==' (line 215)
        result_eq_21919 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 15), '==', line_21917, str_21918)
        
        # Testing the type of an if condition (line 215)
        if_condition_21920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 12), result_eq_21919)
        # Assigning a type to the variable 'if_condition_21920' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'if_condition_21920', if_condition_21920)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to search(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'line' (line 217)
        line_21923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 30), 'line', False)
        # Processing the call keyword arguments (line 217)
        kwargs_21924 = {}
        # Getting the type of 'pattern' (line 217)
        pattern_21921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'pattern', False)
        # Obtaining the member 'search' of a type (line 217)
        search_21922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), pattern_21921, 'search')
        # Calling search(args, kwargs) (line 217)
        search_call_result_21925 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), search_21922, *[line_21923], **kwargs_21924)
        
        # Testing the type of an if condition (line 217)
        if_condition_21926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), search_call_result_21925)
        # Assigning a type to the variable 'if_condition_21926' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_21926', if_condition_21926)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 218):
        
        # Assigning a Num to a Name (line 218):
        int_21927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 24), 'int')
        # Assigning a type to the variable 'match' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'match', int_21927)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_21930 = {}
        # Getting the type of 'file' (line 221)
        file_21928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'file', False)
        # Obtaining the member 'close' of a type (line 221)
        close_21929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), file_21928, 'close')
        # Calling close(args, kwargs) (line 221)
        close_call_result_21931 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), close_21929, *[], **kwargs_21930)
        
        
        # Call to _clean(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_21934 = {}
        # Getting the type of 'self' (line 222)
        self_21932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member '_clean' of a type (line 222)
        _clean_21933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_21932, '_clean')
        # Calling _clean(args, kwargs) (line 222)
        _clean_call_result_21935 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), _clean_21933, *[], **kwargs_21934)
        
        # Getting the type of 'match' (line 223)
        match_21936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'match')
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', match_21936)
        
        # ################# End of 'search_cpp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'search_cpp' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_21937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21937)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'search_cpp'
        return stypy_return_type_21937


    @norecursion
    def try_compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 225)
        None_21938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 40), 'None')
        # Getting the type of 'None' (line 225)
        None_21939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 59), 'None')
        str_21940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 70), 'str', 'c')
        defaults = [None_21938, None_21939, str_21940]
        # Create a new context for function 'try_compile'
        module_type_store = module_type_store.open_function_context('try_compile', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.try_compile.__dict__.__setitem__('stypy_localization', localization)
        config.try_compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.try_compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.try_compile.__dict__.__setitem__('stypy_function_name', 'config.try_compile')
        config.try_compile.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'lang'])
        config.try_compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.try_compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.try_compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.try_compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.try_compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.try_compile.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.try_compile', ['body', 'headers', 'include_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'try_compile', localization, ['body', 'headers', 'include_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'try_compile(...)' code ##################

        str_21941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'str', "Try to compile a source file built from 'body' and 'headers'.\n        Return true on success, false otherwise.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 229, 8))
        
        # 'from distutils.ccompiler import CompileError' statement (line 229)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_21942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 229, 8), 'distutils.ccompiler')

        if (type(import_21942) is not StypyTypeError):

            if (import_21942 != 'pyd_module'):
                __import__(import_21942)
                sys_modules_21943 = sys.modules[import_21942]
                import_from_module(stypy.reporting.localization.Localization(__file__, 229, 8), 'distutils.ccompiler', sys_modules_21943.module_type_store, module_type_store, ['CompileError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 229, 8), __file__, sys_modules_21943, sys_modules_21943.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import CompileError

                import_from_module(stypy.reporting.localization.Localization(__file__, 229, 8), 'distutils.ccompiler', None, module_type_store, ['CompileError'], [CompileError])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'distutils.ccompiler', import_21942)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Call to _check_compiler(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_21946 = {}
        # Getting the type of 'self' (line 230)
        self_21944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 230)
        _check_compiler_21945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_21944, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 230)
        _check_compiler_call_result_21947 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), _check_compiler_21945, *[], **kwargs_21946)
        
        
        
        # SSA begins for try-except statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _compile(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'body' (line 232)
        body_21950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'body', False)
        # Getting the type of 'headers' (line 232)
        headers_21951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'headers', False)
        # Getting the type of 'include_dirs' (line 232)
        include_dirs_21952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 41), 'include_dirs', False)
        # Getting the type of 'lang' (line 232)
        lang_21953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 55), 'lang', False)
        # Processing the call keyword arguments (line 232)
        kwargs_21954 = {}
        # Getting the type of 'self' (line 232)
        self_21948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', False)
        # Obtaining the member '_compile' of a type (line 232)
        _compile_21949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_21948, '_compile')
        # Calling _compile(args, kwargs) (line 232)
        _compile_call_result_21955 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), _compile_21949, *[body_21950, headers_21951, include_dirs_21952, lang_21953], **kwargs_21954)
        
        
        # Assigning a Num to a Name (line 233):
        
        # Assigning a Num to a Name (line 233):
        int_21956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 17), 'int')
        # Assigning a type to the variable 'ok' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'ok', int_21956)
        # SSA branch for the except part of a try statement (line 231)
        # SSA branch for the except 'CompileError' branch of a try statement (line 231)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 235):
        
        # Assigning a Num to a Name (line 235):
        int_21957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 17), 'int')
        # Assigning a type to the variable 'ok' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'ok', int_21957)
        # SSA join for try-except statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'ok' (line 237)
        ok_21960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'ok', False)
        str_21961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 24), 'str', 'success!')
        # Applying the binary operator 'and' (line 237)
        result_and_keyword_21962 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 17), 'and', ok_21960, str_21961)
        
        str_21963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 38), 'str', 'failure.')
        # Applying the binary operator 'or' (line 237)
        result_or_keyword_21964 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 17), 'or', result_and_keyword_21962, str_21963)
        
        # Processing the call keyword arguments (line 237)
        kwargs_21965 = {}
        # Getting the type of 'log' (line 237)
        log_21958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 237)
        info_21959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), log_21958, 'info')
        # Calling info(args, kwargs) (line 237)
        info_call_result_21966 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), info_21959, *[result_or_keyword_21964], **kwargs_21965)
        
        
        # Call to _clean(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_21969 = {}
        # Getting the type of 'self' (line 238)
        self_21967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member '_clean' of a type (line 238)
        _clean_21968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_21967, '_clean')
        # Calling _clean(args, kwargs) (line 238)
        _clean_call_result_21970 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), _clean_21968, *[], **kwargs_21969)
        
        # Getting the type of 'ok' (line 239)
        ok_21971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'ok')
        # Assigning a type to the variable 'stypy_return_type' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'stypy_return_type', ok_21971)
        
        # ################# End of 'try_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'try_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_21972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21972)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'try_compile'
        return stypy_return_type_21972


    @norecursion
    def try_link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 241)
        None_21973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 37), 'None')
        # Getting the type of 'None' (line 241)
        None_21974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 56), 'None')
        # Getting the type of 'None' (line 241)
        None_21975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 72), 'None')
        # Getting the type of 'None' (line 242)
        None_21976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'None')
        str_21977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 41), 'str', 'c')
        defaults = [None_21973, None_21974, None_21975, None_21976, str_21977]
        # Create a new context for function 'try_link'
        module_type_store = module_type_store.open_function_context('try_link', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.try_link.__dict__.__setitem__('stypy_localization', localization)
        config.try_link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.try_link.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.try_link.__dict__.__setitem__('stypy_function_name', 'config.try_link')
        config.try_link.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'])
        config.try_link.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.try_link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.try_link.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.try_link.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.try_link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.try_link.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.try_link', ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'try_link', localization, ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'try_link(...)' code ##################

        str_21978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, (-1)), 'str', "Try to compile and link a source file, built from 'body' and\n        'headers', to executable form.  Return true on success, false\n        otherwise.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 247, 8))
        
        # 'from distutils.ccompiler import CompileError, LinkError' statement (line 247)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_21979 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 247, 8), 'distutils.ccompiler')

        if (type(import_21979) is not StypyTypeError):

            if (import_21979 != 'pyd_module'):
                __import__(import_21979)
                sys_modules_21980 = sys.modules[import_21979]
                import_from_module(stypy.reporting.localization.Localization(__file__, 247, 8), 'distutils.ccompiler', sys_modules_21980.module_type_store, module_type_store, ['CompileError', 'LinkError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 247, 8), __file__, sys_modules_21980, sys_modules_21980.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import CompileError, LinkError

                import_from_module(stypy.reporting.localization.Localization(__file__, 247, 8), 'distutils.ccompiler', None, module_type_store, ['CompileError', 'LinkError'], [CompileError, LinkError])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'distutils.ccompiler', import_21979)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Call to _check_compiler(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_21983 = {}
        # Getting the type of 'self' (line 248)
        self_21981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 248)
        _check_compiler_21982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_21981, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 248)
        _check_compiler_call_result_21984 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), _check_compiler_21982, *[], **kwargs_21983)
        
        
        
        # SSA begins for try-except statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _link(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'body' (line 250)
        body_21987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 'body', False)
        # Getting the type of 'headers' (line 250)
        headers_21988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'headers', False)
        # Getting the type of 'include_dirs' (line 250)
        include_dirs_21989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 38), 'include_dirs', False)
        # Getting the type of 'libraries' (line 251)
        libraries_21990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'libraries', False)
        # Getting the type of 'library_dirs' (line 251)
        library_dirs_21991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'library_dirs', False)
        # Getting the type of 'lang' (line 251)
        lang_21992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 48), 'lang', False)
        # Processing the call keyword arguments (line 250)
        kwargs_21993 = {}
        # Getting the type of 'self' (line 250)
        self_21985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'self', False)
        # Obtaining the member '_link' of a type (line 250)
        _link_21986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), self_21985, '_link')
        # Calling _link(args, kwargs) (line 250)
        _link_call_result_21994 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), _link_21986, *[body_21987, headers_21988, include_dirs_21989, libraries_21990, library_dirs_21991, lang_21992], **kwargs_21993)
        
        
        # Assigning a Num to a Name (line 252):
        
        # Assigning a Num to a Name (line 252):
        int_21995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 17), 'int')
        # Assigning a type to the variable 'ok' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'ok', int_21995)
        # SSA branch for the except part of a try statement (line 249)
        # SSA branch for the except 'Tuple' branch of a try statement (line 249)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 254):
        
        # Assigning a Num to a Name (line 254):
        int_21996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'int')
        # Assigning a type to the variable 'ok' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'ok', int_21996)
        # SSA join for try-except statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'ok' (line 256)
        ok_21999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'ok', False)
        str_22000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'str', 'success!')
        # Applying the binary operator 'and' (line 256)
        result_and_keyword_22001 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 17), 'and', ok_21999, str_22000)
        
        str_22002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 38), 'str', 'failure.')
        # Applying the binary operator 'or' (line 256)
        result_or_keyword_22003 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 17), 'or', result_and_keyword_22001, str_22002)
        
        # Processing the call keyword arguments (line 256)
        kwargs_22004 = {}
        # Getting the type of 'log' (line 256)
        log_21997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 256)
        info_21998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), log_21997, 'info')
        # Calling info(args, kwargs) (line 256)
        info_call_result_22005 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), info_21998, *[result_or_keyword_22003], **kwargs_22004)
        
        
        # Call to _clean(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_22008 = {}
        # Getting the type of 'self' (line 257)
        self_22006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member '_clean' of a type (line 257)
        _clean_22007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_22006, '_clean')
        # Calling _clean(args, kwargs) (line 257)
        _clean_call_result_22009 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), _clean_22007, *[], **kwargs_22008)
        
        # Getting the type of 'ok' (line 258)
        ok_22010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'ok')
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', ok_22010)
        
        # ################# End of 'try_link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'try_link' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_22011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22011)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'try_link'
        return stypy_return_type_22011


    @norecursion
    def try_run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 260)
        None_22012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 36), 'None')
        # Getting the type of 'None' (line 260)
        None_22013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 55), 'None')
        # Getting the type of 'None' (line 260)
        None_22014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 71), 'None')
        # Getting the type of 'None' (line 261)
        None_22015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 29), 'None')
        str_22016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 40), 'str', 'c')
        defaults = [None_22012, None_22013, None_22014, None_22015, str_22016]
        # Create a new context for function 'try_run'
        module_type_store = module_type_store.open_function_context('try_run', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.try_run.__dict__.__setitem__('stypy_localization', localization)
        config.try_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.try_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.try_run.__dict__.__setitem__('stypy_function_name', 'config.try_run')
        config.try_run.__dict__.__setitem__('stypy_param_names_list', ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'])
        config.try_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.try_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.try_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.try_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.try_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.try_run.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.try_run', ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'try_run', localization, ['body', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'try_run(...)' code ##################

        str_22017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, (-1)), 'str', "Try to compile, link to an executable, and run a program\n        built from 'body' and 'headers'.  Return true on success, false\n        otherwise.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 266, 8))
        
        # 'from distutils.ccompiler import CompileError, LinkError' statement (line 266)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_22018 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 266, 8), 'distutils.ccompiler')

        if (type(import_22018) is not StypyTypeError):

            if (import_22018 != 'pyd_module'):
                __import__(import_22018)
                sys_modules_22019 = sys.modules[import_22018]
                import_from_module(stypy.reporting.localization.Localization(__file__, 266, 8), 'distutils.ccompiler', sys_modules_22019.module_type_store, module_type_store, ['CompileError', 'LinkError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 266, 8), __file__, sys_modules_22019, sys_modules_22019.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import CompileError, LinkError

                import_from_module(stypy.reporting.localization.Localization(__file__, 266, 8), 'distutils.ccompiler', None, module_type_store, ['CompileError', 'LinkError'], [CompileError, LinkError])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'distutils.ccompiler', import_22018)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Call to _check_compiler(...): (line 267)
        # Processing the call keyword arguments (line 267)
        kwargs_22022 = {}
        # Getting the type of 'self' (line 267)
        self_22020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 267)
        _check_compiler_22021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_22020, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 267)
        _check_compiler_call_result_22023 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), _check_compiler_22021, *[], **kwargs_22022)
        
        
        
        # SSA begins for try-except statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 269):
        
        # Assigning a Subscript to a Name (line 269):
        
        # Obtaining the type of the subscript
        int_22024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 12), 'int')
        
        # Call to _link(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'body' (line 269)
        body_22027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'body', False)
        # Getting the type of 'headers' (line 269)
        headers_22028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'headers', False)
        # Getting the type of 'include_dirs' (line 269)
        include_dirs_22029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 54), 'include_dirs', False)
        # Getting the type of 'libraries' (line 270)
        libraries_22030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 39), 'libraries', False)
        # Getting the type of 'library_dirs' (line 270)
        library_dirs_22031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'library_dirs', False)
        # Getting the type of 'lang' (line 270)
        lang_22032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 64), 'lang', False)
        # Processing the call keyword arguments (line 269)
        kwargs_22033 = {}
        # Getting the type of 'self' (line 269)
        self_22025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'self', False)
        # Obtaining the member '_link' of a type (line 269)
        _link_22026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 28), self_22025, '_link')
        # Calling _link(args, kwargs) (line 269)
        _link_call_result_22034 = invoke(stypy.reporting.localization.Localization(__file__, 269, 28), _link_22026, *[body_22027, headers_22028, include_dirs_22029, libraries_22030, library_dirs_22031, lang_22032], **kwargs_22033)
        
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___22035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), _link_call_result_22034, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_22036 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), getitem___22035, int_22024)
        
        # Assigning a type to the variable 'tuple_var_assignment_21450' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_21450', subscript_call_result_22036)
        
        # Assigning a Subscript to a Name (line 269):
        
        # Obtaining the type of the subscript
        int_22037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 12), 'int')
        
        # Call to _link(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'body' (line 269)
        body_22040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'body', False)
        # Getting the type of 'headers' (line 269)
        headers_22041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'headers', False)
        # Getting the type of 'include_dirs' (line 269)
        include_dirs_22042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 54), 'include_dirs', False)
        # Getting the type of 'libraries' (line 270)
        libraries_22043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 39), 'libraries', False)
        # Getting the type of 'library_dirs' (line 270)
        library_dirs_22044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'library_dirs', False)
        # Getting the type of 'lang' (line 270)
        lang_22045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 64), 'lang', False)
        # Processing the call keyword arguments (line 269)
        kwargs_22046 = {}
        # Getting the type of 'self' (line 269)
        self_22038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'self', False)
        # Obtaining the member '_link' of a type (line 269)
        _link_22039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 28), self_22038, '_link')
        # Calling _link(args, kwargs) (line 269)
        _link_call_result_22047 = invoke(stypy.reporting.localization.Localization(__file__, 269, 28), _link_22039, *[body_22040, headers_22041, include_dirs_22042, libraries_22043, library_dirs_22044, lang_22045], **kwargs_22046)
        
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___22048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), _link_call_result_22047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_22049 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), getitem___22048, int_22037)
        
        # Assigning a type to the variable 'tuple_var_assignment_21451' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_21451', subscript_call_result_22049)
        
        # Assigning a Subscript to a Name (line 269):
        
        # Obtaining the type of the subscript
        int_22050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 12), 'int')
        
        # Call to _link(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'body' (line 269)
        body_22053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'body', False)
        # Getting the type of 'headers' (line 269)
        headers_22054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'headers', False)
        # Getting the type of 'include_dirs' (line 269)
        include_dirs_22055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 54), 'include_dirs', False)
        # Getting the type of 'libraries' (line 270)
        libraries_22056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 39), 'libraries', False)
        # Getting the type of 'library_dirs' (line 270)
        library_dirs_22057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'library_dirs', False)
        # Getting the type of 'lang' (line 270)
        lang_22058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 64), 'lang', False)
        # Processing the call keyword arguments (line 269)
        kwargs_22059 = {}
        # Getting the type of 'self' (line 269)
        self_22051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'self', False)
        # Obtaining the member '_link' of a type (line 269)
        _link_22052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 28), self_22051, '_link')
        # Calling _link(args, kwargs) (line 269)
        _link_call_result_22060 = invoke(stypy.reporting.localization.Localization(__file__, 269, 28), _link_22052, *[body_22053, headers_22054, include_dirs_22055, libraries_22056, library_dirs_22057, lang_22058], **kwargs_22059)
        
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___22061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), _link_call_result_22060, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_22062 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), getitem___22061, int_22050)
        
        # Assigning a type to the variable 'tuple_var_assignment_21452' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_21452', subscript_call_result_22062)
        
        # Assigning a Name to a Name (line 269):
        # Getting the type of 'tuple_var_assignment_21450' (line 269)
        tuple_var_assignment_21450_22063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_21450')
        # Assigning a type to the variable 'src' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'src', tuple_var_assignment_21450_22063)
        
        # Assigning a Name to a Name (line 269):
        # Getting the type of 'tuple_var_assignment_21451' (line 269)
        tuple_var_assignment_21451_22064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_21451')
        # Assigning a type to the variable 'obj' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'obj', tuple_var_assignment_21451_22064)
        
        # Assigning a Name to a Name (line 269):
        # Getting the type of 'tuple_var_assignment_21452' (line 269)
        tuple_var_assignment_21452_22065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'tuple_var_assignment_21452')
        # Assigning a type to the variable 'exe' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'exe', tuple_var_assignment_21452_22065)
        
        # Call to spawn(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Obtaining an instance of the builtin type 'list' (line 271)
        list_22068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 271)
        # Adding element type (line 271)
        # Getting the type of 'exe' (line 271)
        exe_22069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'exe', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 23), list_22068, exe_22069)
        
        # Processing the call keyword arguments (line 271)
        kwargs_22070 = {}
        # Getting the type of 'self' (line 271)
        self_22066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'self', False)
        # Obtaining the member 'spawn' of a type (line 271)
        spawn_22067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), self_22066, 'spawn')
        # Calling spawn(args, kwargs) (line 271)
        spawn_call_result_22071 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), spawn_22067, *[list_22068], **kwargs_22070)
        
        
        # Assigning a Num to a Name (line 272):
        
        # Assigning a Num to a Name (line 272):
        int_22072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 17), 'int')
        # Assigning a type to the variable 'ok' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'ok', int_22072)
        # SSA branch for the except part of a try statement (line 268)
        # SSA branch for the except 'Tuple' branch of a try statement (line 268)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 274):
        
        # Assigning a Num to a Name (line 274):
        int_22073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 17), 'int')
        # Assigning a type to the variable 'ok' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'ok', int_22073)
        # SSA join for try-except statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'ok' (line 276)
        ok_22076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 17), 'ok', False)
        str_22077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 24), 'str', 'success!')
        # Applying the binary operator 'and' (line 276)
        result_and_keyword_22078 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 17), 'and', ok_22076, str_22077)
        
        str_22079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 38), 'str', 'failure.')
        # Applying the binary operator 'or' (line 276)
        result_or_keyword_22080 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 17), 'or', result_and_keyword_22078, str_22079)
        
        # Processing the call keyword arguments (line 276)
        kwargs_22081 = {}
        # Getting the type of 'log' (line 276)
        log_22074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 276)
        info_22075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), log_22074, 'info')
        # Calling info(args, kwargs) (line 276)
        info_call_result_22082 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), info_22075, *[result_or_keyword_22080], **kwargs_22081)
        
        
        # Call to _clean(...): (line 277)
        # Processing the call keyword arguments (line 277)
        kwargs_22085 = {}
        # Getting the type of 'self' (line 277)
        self_22083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self', False)
        # Obtaining the member '_clean' of a type (line 277)
        _clean_22084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_22083, '_clean')
        # Calling _clean(args, kwargs) (line 277)
        _clean_call_result_22086 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), _clean_22084, *[], **kwargs_22085)
        
        # Getting the type of 'ok' (line 278)
        ok_22087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'ok')
        # Assigning a type to the variable 'stypy_return_type' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type', ok_22087)
        
        # ################# End of 'try_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'try_run' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_22088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'try_run'
        return stypy_return_type_22088


    @norecursion
    def check_func(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 285)
        None_22089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 39), 'None')
        # Getting the type of 'None' (line 285)
        None_22090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 58), 'None')
        # Getting the type of 'None' (line 286)
        None_22091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 'None')
        # Getting the type of 'None' (line 286)
        None_22092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 48), 'None')
        int_22093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 59), 'int')
        int_22094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 67), 'int')
        defaults = [None_22089, None_22090, None_22091, None_22092, int_22093, int_22094]
        # Create a new context for function 'check_func'
        module_type_store = module_type_store.open_function_context('check_func', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_func.__dict__.__setitem__('stypy_localization', localization)
        config.check_func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_func.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_func.__dict__.__setitem__('stypy_function_name', 'config.check_func')
        config.check_func.__dict__.__setitem__('stypy_param_names_list', ['func', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call'])
        config.check_func.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_func.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_func.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_func.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_func.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_func', ['func', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_func', localization, ['func', 'headers', 'include_dirs', 'libraries', 'library_dirs', 'decl', 'call'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_func(...)' code ##################

        str_22095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, (-1)), 'str', 'Determine if function \'func\' is available by constructing a\n        source file that refers to \'func\', and compiles and links it.\n        If everything succeeds, returns true; otherwise returns false.\n\n        The constructed source file starts out by including the header\n        files listed in \'headers\'.  If \'decl\' is true, it then declares\n        \'func\' (as "int func()"); you probably shouldn\'t supply \'headers\'\n        and set \'decl\' true in the same call, or you might get errors about\n        a conflicting declarations for \'func\'.  Finally, the constructed\n        \'main()\' function either references \'func\' or (if \'call\' is true)\n        calls it.  \'libraries\' and \'library_dirs\' are used when\n        linking.\n        ')
        
        # Call to _check_compiler(...): (line 302)
        # Processing the call keyword arguments (line 302)
        kwargs_22098 = {}
        # Getting the type of 'self' (line 302)
        self_22096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 302)
        _check_compiler_22097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), self_22096, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 302)
        _check_compiler_call_result_22099 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), _check_compiler_22097, *[], **kwargs_22098)
        
        
        # Assigning a List to a Name (line 303):
        
        # Assigning a List to a Name (line 303):
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_22100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        
        # Assigning a type to the variable 'body' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'body', list_22100)
        
        # Getting the type of 'decl' (line 304)
        decl_22101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'decl')
        # Testing the type of an if condition (line 304)
        if_condition_22102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 8), decl_22101)
        # Assigning a type to the variable 'if_condition_22102' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'if_condition_22102', if_condition_22102)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 305)
        # Processing the call arguments (line 305)
        str_22105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 24), 'str', 'int %s ();')
        # Getting the type of 'func' (line 305)
        func_22106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 39), 'func', False)
        # Applying the binary operator '%' (line 305)
        result_mod_22107 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 24), '%', str_22105, func_22106)
        
        # Processing the call keyword arguments (line 305)
        kwargs_22108 = {}
        # Getting the type of 'body' (line 305)
        body_22103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'body', False)
        # Obtaining the member 'append' of a type (line 305)
        append_22104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), body_22103, 'append')
        # Calling append(args, kwargs) (line 305)
        append_call_result_22109 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), append_22104, *[result_mod_22107], **kwargs_22108)
        
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 306)
        # Processing the call arguments (line 306)
        str_22112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 20), 'str', 'int main () {')
        # Processing the call keyword arguments (line 306)
        kwargs_22113 = {}
        # Getting the type of 'body' (line 306)
        body_22110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 306)
        append_22111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), body_22110, 'append')
        # Calling append(args, kwargs) (line 306)
        append_call_result_22114 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), append_22111, *[str_22112], **kwargs_22113)
        
        
        # Getting the type of 'call' (line 307)
        call_22115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'call')
        # Testing the type of an if condition (line 307)
        if_condition_22116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 8), call_22115)
        # Assigning a type to the variable 'if_condition_22116' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'if_condition_22116', if_condition_22116)
        # SSA begins for if statement (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 308)
        # Processing the call arguments (line 308)
        str_22119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'str', '  %s();')
        # Getting the type of 'func' (line 308)
        func_22120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 36), 'func', False)
        # Applying the binary operator '%' (line 308)
        result_mod_22121 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 24), '%', str_22119, func_22120)
        
        # Processing the call keyword arguments (line 308)
        kwargs_22122 = {}
        # Getting the type of 'body' (line 308)
        body_22117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'body', False)
        # Obtaining the member 'append' of a type (line 308)
        append_22118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), body_22117, 'append')
        # Calling append(args, kwargs) (line 308)
        append_call_result_22123 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), append_22118, *[result_mod_22121], **kwargs_22122)
        
        # SSA branch for the else part of an if statement (line 307)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 310)
        # Processing the call arguments (line 310)
        str_22126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 24), 'str', '  %s;')
        # Getting the type of 'func' (line 310)
        func_22127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 34), 'func', False)
        # Applying the binary operator '%' (line 310)
        result_mod_22128 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 24), '%', str_22126, func_22127)
        
        # Processing the call keyword arguments (line 310)
        kwargs_22129 = {}
        # Getting the type of 'body' (line 310)
        body_22124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'body', False)
        # Obtaining the member 'append' of a type (line 310)
        append_22125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), body_22124, 'append')
        # Calling append(args, kwargs) (line 310)
        append_call_result_22130 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), append_22125, *[result_mod_22128], **kwargs_22129)
        
        # SSA join for if statement (line 307)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 311)
        # Processing the call arguments (line 311)
        str_22133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'str', '}')
        # Processing the call keyword arguments (line 311)
        kwargs_22134 = {}
        # Getting the type of 'body' (line 311)
        body_22131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'body', False)
        # Obtaining the member 'append' of a type (line 311)
        append_22132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), body_22131, 'append')
        # Calling append(args, kwargs) (line 311)
        append_call_result_22135 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), append_22132, *[str_22133], **kwargs_22134)
        
        
        # Assigning a BinOp to a Name (line 312):
        
        # Assigning a BinOp to a Name (line 312):
        
        # Call to join(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'body' (line 312)
        body_22138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'body', False)
        # Processing the call keyword arguments (line 312)
        kwargs_22139 = {}
        str_22136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 312)
        join_22137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 15), str_22136, 'join')
        # Calling join(args, kwargs) (line 312)
        join_call_result_22140 = invoke(stypy.reporting.localization.Localization(__file__, 312, 15), join_22137, *[body_22138], **kwargs_22139)
        
        str_22141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 33), 'str', '\n')
        # Applying the binary operator '+' (line 312)
        result_add_22142 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), '+', join_call_result_22140, str_22141)
        
        # Assigning a type to the variable 'body' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'body', result_add_22142)
        
        # Call to try_link(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'body' (line 314)
        body_22145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 29), 'body', False)
        # Getting the type of 'headers' (line 314)
        headers_22146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 35), 'headers', False)
        # Getting the type of 'include_dirs' (line 314)
        include_dirs_22147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 44), 'include_dirs', False)
        # Getting the type of 'libraries' (line 315)
        libraries_22148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 29), 'libraries', False)
        # Getting the type of 'library_dirs' (line 315)
        library_dirs_22149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 40), 'library_dirs', False)
        # Processing the call keyword arguments (line 314)
        kwargs_22150 = {}
        # Getting the type of 'self' (line 314)
        self_22143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'self', False)
        # Obtaining the member 'try_link' of a type (line 314)
        try_link_22144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 15), self_22143, 'try_link')
        # Calling try_link(args, kwargs) (line 314)
        try_link_call_result_22151 = invoke(stypy.reporting.localization.Localization(__file__, 314, 15), try_link_22144, *[body_22145, headers_22146, include_dirs_22147, libraries_22148, library_dirs_22149], **kwargs_22150)
        
        # Assigning a type to the variable 'stypy_return_type' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'stypy_return_type', try_link_call_result_22151)
        
        # ################# End of 'check_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_func' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_22152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_func'
        return stypy_return_type_22152


    @norecursion
    def check_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 319)
        None_22153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 46), 'None')
        # Getting the type of 'None' (line 319)
        None_22154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 60), 'None')
        # Getting the type of 'None' (line 320)
        None_22155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'None')
        
        # Obtaining an instance of the builtin type 'list' (line 320)
        list_22156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 320)
        
        defaults = [None_22153, None_22154, None_22155, list_22156]
        # Create a new context for function 'check_lib'
        module_type_store = module_type_store.open_function_context('check_lib', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config.check_lib.__dict__.__setitem__('stypy_localization', localization)
        config.check_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config.check_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        config.check_lib.__dict__.__setitem__('stypy_function_name', 'config.check_lib')
        config.check_lib.__dict__.__setitem__('stypy_param_names_list', ['library', 'library_dirs', 'headers', 'include_dirs', 'other_libraries'])
        config.check_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        config.check_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config.check_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        config.check_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        config.check_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config.check_lib.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config.check_lib', ['library', 'library_dirs', 'headers', 'include_dirs', 'other_libraries'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_lib', localization, ['library', 'library_dirs', 'headers', 'include_dirs', 'other_libraries'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_lib(...)' code ##################

        str_22157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, (-1)), 'str', "Determine if 'library' is available to be linked against,\n        without actually checking that any particular symbols are provided\n        by it.  'headers' will be used in constructing the source file to\n        be compiled, but the only effect of this is to check if all the\n        header files listed are available.  Any libraries listed in\n        'other_libraries' will be included in the link, in case 'library'\n        has symbols that depend on other libraries.\n        ")
        
        # Call to _check_compiler(...): (line 329)
        # Processing the call keyword arguments (line 329)
        kwargs_22160 = {}
        # Getting the type of 'self' (line 329)
        self_22158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member '_check_compiler' of a type (line 329)
        _check_compiler_22159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_22158, '_check_compiler')
        # Calling _check_compiler(args, kwargs) (line 329)
        _check_compiler_call_result_22161 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), _check_compiler_22159, *[], **kwargs_22160)
        
        
        # Call to try_link(...): (line 330)
        # Processing the call arguments (line 330)
        str_22164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 29), 'str', 'int main (void) { }')
        # Getting the type of 'headers' (line 331)
        headers_22165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'headers', False)
        # Getting the type of 'include_dirs' (line 331)
        include_dirs_22166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 38), 'include_dirs', False)
        
        # Obtaining an instance of the builtin type 'list' (line 332)
        list_22167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'library' (line 332)
        library_22168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'library', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 29), list_22167, library_22168)
        
        # Getting the type of 'other_libraries' (line 332)
        other_libraries_22169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 39), 'other_libraries', False)
        # Applying the binary operator '+' (line 332)
        result_add_22170 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 29), '+', list_22167, other_libraries_22169)
        
        # Getting the type of 'library_dirs' (line 332)
        library_dirs_22171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 56), 'library_dirs', False)
        # Processing the call keyword arguments (line 330)
        kwargs_22172 = {}
        # Getting the type of 'self' (line 330)
        self_22162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'self', False)
        # Obtaining the member 'try_link' of a type (line 330)
        try_link_22163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 15), self_22162, 'try_link')
        # Calling try_link(args, kwargs) (line 330)
        try_link_call_result_22173 = invoke(stypy.reporting.localization.Localization(__file__, 330, 15), try_link_22163, *[str_22164, headers_22165, include_dirs_22166, result_add_22170, library_dirs_22171], **kwargs_22172)
        
        # Assigning a type to the variable 'stypy_return_type' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'stypy_return_type', try_link_call_result_22173)
        
        # ################# End of 'check_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_22174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22174)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_lib'
        return stypy_return_type_22174


    @norecursion
    def check_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 334)
        None_22175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 48), 'None')
        # Getting the type of 'None' (line 334)
        None_22176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 67), 'None')
        str_22177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 26), 'str', 'c')
        defaults = [None_22175, None_22176, str_22177]
        # Create a new context for function 'check_header'
        module_type_store = module_type_store.open_function_context('check_header', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
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

        str_22178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, (-1)), 'str', "Determine if the system header file named by 'header_file'\n        exists and can be found by the preprocessor; return true if so,\n        false otherwise.\n        ")
        
        # Call to try_cpp(...): (line 340)
        # Processing the call keyword arguments (line 340)
        str_22181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 33), 'str', '/* No body */')
        keyword_22182 = str_22181
        
        # Obtaining an instance of the builtin type 'list' (line 340)
        list_22183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 340)
        # Adding element type (line 340)
        # Getting the type of 'header' (line 340)
        header_22184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 59), 'header', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 58), list_22183, header_22184)
        
        keyword_22185 = list_22183
        # Getting the type of 'include_dirs' (line 341)
        include_dirs_22186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 41), 'include_dirs', False)
        keyword_22187 = include_dirs_22186
        kwargs_22188 = {'body': keyword_22182, 'headers': keyword_22185, 'include_dirs': keyword_22187}
        # Getting the type of 'self' (line 340)
        self_22179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'self', False)
        # Obtaining the member 'try_cpp' of a type (line 340)
        try_cpp_22180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 15), self_22179, 'try_cpp')
        # Calling try_cpp(args, kwargs) (line 340)
        try_cpp_call_result_22189 = invoke(stypy.reporting.localization.Localization(__file__, 340, 15), try_cpp_22180, *[], **kwargs_22188)
        
        # Assigning a type to the variable 'stypy_return_type' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'stypy_return_type', try_cpp_call_result_22189)
        
        # ################# End of 'check_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_header' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_22190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_header'
        return stypy_return_type_22190


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 0, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'config' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'config', config)

# Assigning a Str to a Name (line 26):
str_22191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'str', 'prepare to build')
# Getting the type of 'config'
config_22192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_22192, 'description', str_22191)

# Assigning a List to a Name (line 28):

# Obtaining an instance of the builtin type 'list' (line 28)
list_22193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_22194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
str_22195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'str', 'compiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_22194, str_22195)
# Adding element type (line 29)
# Getting the type of 'None' (line 29)
None_22196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_22194, None_22196)
# Adding element type (line 29)
str_22197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'str', 'specify the compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_22194, str_22197)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22194)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_22198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
str_22199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'cc=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_22198, str_22199)
# Adding element type (line 31)
# Getting the type of 'None' (line 31)
None_22200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_22198, None_22200)
# Adding element type (line 31)
str_22201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'str', 'specify the compiler executable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_22198, str_22201)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22198)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_22202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_22203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'str', 'include-dirs=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_22202, str_22203)
# Adding element type (line 33)
str_22204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'str', 'I')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_22202, str_22204)
# Adding element type (line 33)
str_22205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'list of directories to search for header files')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_22202, str_22205)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22202)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_22206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_22207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'define=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_22206, str_22207)
# Adding element type (line 35)
str_22208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_22206, str_22208)
# Adding element type (line 35)
str_22209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'C preprocessor macros to define')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_22206, str_22209)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22206)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_22210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
str_22211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'str', 'undef=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_22210, str_22211)
# Adding element type (line 37)
str_22212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'str', 'U')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_22210, str_22212)
# Adding element type (line 37)
str_22213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', 'C preprocessor macros to undefine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_22210, str_22213)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22210)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_22214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
str_22215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'str', 'libraries=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_22214, str_22215)
# Adding element type (line 39)
str_22216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'str', 'l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_22214, str_22216)
# Adding element type (line 39)
str_22217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'external C libraries to link with')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_22214, str_22217)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22214)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_22218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
str_22219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'library-dirs=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_22218, str_22219)
# Adding element type (line 41)
str_22220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'str', 'L')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_22218, str_22220)
# Adding element type (line 41)
str_22221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'str', 'directories to search for external C libraries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_22218, str_22221)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22218)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_22222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
str_22223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'str', 'noisy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_22222, str_22223)
# Adding element type (line 44)
# Getting the type of 'None' (line 44)
None_22224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_22222, None_22224)
# Adding element type (line 44)
str_22225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'str', 'show every action (compile, link, run, ...) taken')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_22222, str_22225)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22222)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_22226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)
# Adding element type (line 46)
str_22227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'str', 'dump-source')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_22226, str_22227)
# Adding element type (line 46)
# Getting the type of 'None' (line 46)
None_22228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_22226, None_22228)
# Adding element type (line 46)
str_22229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 9), 'str', 'dump generated source files before attempting to compile them')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_22226, str_22229)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_22193, tuple_22226)

# Getting the type of 'config'
config_22230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_22230, 'user_options', list_22193)

@norecursion
def dump_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 344)
    None_22231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 29), 'None')
    defaults = [None_22231]
    # Create a new context for function 'dump_file'
    module_type_store = module_type_store.open_function_context('dump_file', 344, 0, False)
    
    # Passed parameters checking function
    dump_file.stypy_localization = localization
    dump_file.stypy_type_of_self = None
    dump_file.stypy_type_store = module_type_store
    dump_file.stypy_function_name = 'dump_file'
    dump_file.stypy_param_names_list = ['filename', 'head']
    dump_file.stypy_varargs_param_name = None
    dump_file.stypy_kwargs_param_name = None
    dump_file.stypy_call_defaults = defaults
    dump_file.stypy_call_varargs = varargs
    dump_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dump_file', ['filename', 'head'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dump_file', localization, ['filename', 'head'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dump_file(...)' code ##################

    str_22232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, (-1)), 'str', 'Dumps a file content into log.info.\n\n    If head is not None, will be dumped before the file content.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 349)
    # Getting the type of 'head' (line 349)
    head_22233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 7), 'head')
    # Getting the type of 'None' (line 349)
    None_22234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'None')
    
    (may_be_22235, more_types_in_union_22236) = may_be_none(head_22233, None_22234)

    if may_be_22235:

        if more_types_in_union_22236:
            # Runtime conditional SSA (line 349)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to info(...): (line 350)
        # Processing the call arguments (line 350)
        str_22239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 17), 'str', '%s')
        # Getting the type of 'filename' (line 350)
        filename_22240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'filename', False)
        # Applying the binary operator '%' (line 350)
        result_mod_22241 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 17), '%', str_22239, filename_22240)
        
        # Processing the call keyword arguments (line 350)
        kwargs_22242 = {}
        # Getting the type of 'log' (line 350)
        log_22237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 350)
        info_22238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), log_22237, 'info')
        # Calling info(args, kwargs) (line 350)
        info_call_result_22243 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), info_22238, *[result_mod_22241], **kwargs_22242)
        

        if more_types_in_union_22236:
            # Runtime conditional SSA for else branch (line 349)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_22235) or more_types_in_union_22236):
        
        # Call to info(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'head' (line 352)
        head_22246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 17), 'head', False)
        # Processing the call keyword arguments (line 352)
        kwargs_22247 = {}
        # Getting the type of 'log' (line 352)
        log_22244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 352)
        info_22245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), log_22244, 'info')
        # Calling info(args, kwargs) (line 352)
        info_call_result_22248 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), info_22245, *[head_22246], **kwargs_22247)
        

        if (may_be_22235 and more_types_in_union_22236):
            # SSA join for if statement (line 349)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 353):
    
    # Assigning a Call to a Name (line 353):
    
    # Call to open(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'filename' (line 353)
    filename_22250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'filename', False)
    # Processing the call keyword arguments (line 353)
    kwargs_22251 = {}
    # Getting the type of 'open' (line 353)
    open_22249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'open', False)
    # Calling open(args, kwargs) (line 353)
    open_call_result_22252 = invoke(stypy.reporting.localization.Localization(__file__, 353, 11), open_22249, *[filename_22250], **kwargs_22251)
    
    # Assigning a type to the variable 'file' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'file', open_call_result_22252)
    
    # Try-finally block (line 354)
    
    # Call to info(...): (line 355)
    # Processing the call arguments (line 355)
    
    # Call to read(...): (line 355)
    # Processing the call keyword arguments (line 355)
    kwargs_22257 = {}
    # Getting the type of 'file' (line 355)
    file_22255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 17), 'file', False)
    # Obtaining the member 'read' of a type (line 355)
    read_22256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 17), file_22255, 'read')
    # Calling read(args, kwargs) (line 355)
    read_call_result_22258 = invoke(stypy.reporting.localization.Localization(__file__, 355, 17), read_22256, *[], **kwargs_22257)
    
    # Processing the call keyword arguments (line 355)
    kwargs_22259 = {}
    # Getting the type of 'log' (line 355)
    log_22253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'log', False)
    # Obtaining the member 'info' of a type (line 355)
    info_22254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), log_22253, 'info')
    # Calling info(args, kwargs) (line 355)
    info_call_result_22260 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), info_22254, *[read_call_result_22258], **kwargs_22259)
    
    
    # finally branch of the try-finally block (line 354)
    
    # Call to close(...): (line 357)
    # Processing the call keyword arguments (line 357)
    kwargs_22263 = {}
    # Getting the type of 'file' (line 357)
    file_22261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'file', False)
    # Obtaining the member 'close' of a type (line 357)
    close_22262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), file_22261, 'close')
    # Calling close(args, kwargs) (line 357)
    close_call_result_22264 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), close_22262, *[], **kwargs_22263)
    
    
    
    # ################# End of 'dump_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dump_file' in the type store
    # Getting the type of 'stypy_return_type' (line 344)
    stypy_return_type_22265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22265)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dump_file'
    return stypy_return_type_22265

# Assigning a type to the variable 'dump_file' (line 344)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 0), 'dump_file', dump_file)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

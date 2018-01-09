
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.emxccompiler
2: 
3: Provides the EMXCCompiler class, a subclass of UnixCCompiler that
4: handles the EMX port of the GNU C compiler to OS/2.
5: '''
6: 
7: # issues:
8: #
9: # * OS/2 insists that DLLs can have names no longer than 8 characters
10: #   We put export_symbols in a def-file, as though the DLL can have
11: #   an arbitrary length name, but truncate the output filename.
12: #
13: # * only use OMF objects and use LINK386 as the linker (-Zomf)
14: #
15: # * always build for multithreading (-Zmt) as the accompanying OS/2 port
16: #   of Python is only distributed with threads enabled.
17: #
18: # tested configurations:
19: #
20: # * EMX gcc 2.81/EMX 0.9d fix03
21: 
22: __revision__ = "$Id$"
23: 
24: import os,sys,copy
25: from distutils.ccompiler import gen_preprocess_options, gen_lib_options
26: from distutils.unixccompiler import UnixCCompiler
27: from distutils.file_util import write_file
28: from distutils.errors import DistutilsExecError, CompileError, UnknownFileError
29: from distutils import log
30: 
31: class EMXCCompiler (UnixCCompiler):
32: 
33:     compiler_type = 'emx'
34:     obj_extension = ".obj"
35:     static_lib_extension = ".lib"
36:     shared_lib_extension = ".dll"
37:     static_lib_format = "%s%s"
38:     shared_lib_format = "%s%s"
39:     res_extension = ".res"      # compiled resource file
40:     exe_extension = ".exe"
41: 
42:     def __init__ (self,
43:                   verbose=0,
44:                   dry_run=0,
45:                   force=0):
46: 
47:         UnixCCompiler.__init__ (self, verbose, dry_run, force)
48: 
49:         (status, details) = check_config_h()
50:         self.debug_print("Python's GCC status: %s (details: %s)" %
51:                          (status, details))
52:         if status is not CONFIG_H_OK:
53:             self.warn(
54:                 "Python's pyconfig.h doesn't seem to support your compiler.  " +
55:                 ("Reason: %s." % details) +
56:                 "Compiling may fail because of undefined preprocessor macros.")
57: 
58:         (self.gcc_version, self.ld_version) = \
59:             get_versions()
60:         self.debug_print(self.compiler_type + ": gcc %s, ld %s\n" %
61:                          (self.gcc_version,
62:                           self.ld_version) )
63: 
64:         # Hard-code GCC because that's what this is all about.
65:         # XXX optimization, warnings etc. should be customizable.
66:         self.set_executables(compiler='gcc -Zomf -Zmt -O3 -fomit-frame-pointer -mprobe -Wall',
67:                              compiler_so='gcc -Zomf -Zmt -O3 -fomit-frame-pointer -mprobe -Wall',
68:                              linker_exe='gcc -Zomf -Zmt -Zcrtdll',
69:                              linker_so='gcc -Zomf -Zmt -Zcrtdll -Zdll')
70: 
71:         # want the gcc library statically linked (so that we don't have
72:         # to distribute a version dependent on the compiler we have)
73:         self.dll_libraries=["gcc"]
74: 
75:     # __init__ ()
76: 
77:     def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
78:         if ext == '.rc':
79:             # gcc requires '.rc' compiled to binary ('.res') files !!!
80:             try:
81:                 self.spawn(["rc", "-r", src])
82:             except DistutilsExecError, msg:
83:                 raise CompileError, msg
84:         else: # for other files use the C-compiler
85:             try:
86:                 self.spawn(self.compiler_so + cc_args + [src, '-o', obj] +
87:                            extra_postargs)
88:             except DistutilsExecError, msg:
89:                 raise CompileError, msg
90: 
91:     def link (self,
92:               target_desc,
93:               objects,
94:               output_filename,
95:               output_dir=None,
96:               libraries=None,
97:               library_dirs=None,
98:               runtime_library_dirs=None,
99:               export_symbols=None,
100:               debug=0,
101:               extra_preargs=None,
102:               extra_postargs=None,
103:               build_temp=None,
104:               target_lang=None):
105: 
106:         # use separate copies, so we can modify the lists
107:         extra_preargs = copy.copy(extra_preargs or [])
108:         libraries = copy.copy(libraries or [])
109:         objects = copy.copy(objects or [])
110: 
111:         # Additional libraries
112:         libraries.extend(self.dll_libraries)
113: 
114:         # handle export symbols by creating a def-file
115:         # with executables this only works with gcc/ld as linker
116:         if ((export_symbols is not None) and
117:             (target_desc != self.EXECUTABLE)):
118:             # (The linker doesn't do anything if output is up-to-date.
119:             # So it would probably better to check if we really need this,
120:             # but for this we had to insert some unchanged parts of
121:             # UnixCCompiler, and this is not what we want.)
122: 
123:             # we want to put some files in the same directory as the
124:             # object files are, build_temp doesn't help much
125:             # where are the object files
126:             temp_dir = os.path.dirname(objects[0])
127:             # name of dll to give the helper files the same base name
128:             (dll_name, dll_extension) = os.path.splitext(
129:                 os.path.basename(output_filename))
130: 
131:             # generate the filenames for these files
132:             def_file = os.path.join(temp_dir, dll_name + ".def")
133: 
134:             # Generate .def file
135:             contents = [
136:                 "LIBRARY %s INITINSTANCE TERMINSTANCE" % \
137:                 os.path.splitext(os.path.basename(output_filename))[0],
138:                 "DATA MULTIPLE NONSHARED",
139:                 "EXPORTS"]
140:             for sym in export_symbols:
141:                 contents.append('  "%s"' % sym)
142:             self.execute(write_file, (def_file, contents),
143:                          "writing %s" % def_file)
144: 
145:             # next add options for def-file and to creating import libraries
146:             # for gcc/ld the def-file is specified as any other object files
147:             objects.append(def_file)
148: 
149:         #end: if ((export_symbols is not None) and
150:         #        (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):
151: 
152:         # who wants symbols and a many times larger output file
153:         # should explicitly switch the debug mode on
154:         # otherwise we let dllwrap/ld strip the output file
155:         # (On my machine: 10KB < stripped_file < ??100KB
156:         #   unstripped_file = stripped_file + XXX KB
157:         #  ( XXX=254 for a typical python extension))
158:         if not debug:
159:             extra_preargs.append("-s")
160: 
161:         UnixCCompiler.link(self,
162:                            target_desc,
163:                            objects,
164:                            output_filename,
165:                            output_dir,
166:                            libraries,
167:                            library_dirs,
168:                            runtime_library_dirs,
169:                            None, # export_symbols, we do this in our def-file
170:                            debug,
171:                            extra_preargs,
172:                            extra_postargs,
173:                            build_temp,
174:                            target_lang)
175: 
176:     # link ()
177: 
178:     # -- Miscellaneous methods -----------------------------------------
179: 
180:     # override the object_filenames method from CCompiler to
181:     # support rc and res-files
182:     def object_filenames (self,
183:                           source_filenames,
184:                           strip_dir=0,
185:                           output_dir=''):
186:         if output_dir is None: output_dir = ''
187:         obj_names = []
188:         for src_name in source_filenames:
189:             # use normcase to make sure '.rc' is really '.rc' and not '.RC'
190:             (base, ext) = os.path.splitext (os.path.normcase(src_name))
191:             if ext not in (self.src_extensions + ['.rc']):
192:                 raise UnknownFileError, \
193:                       "unknown file type '%s' (from '%s')" % \
194:                       (ext, src_name)
195:             if strip_dir:
196:                 base = os.path.basename (base)
197:             if ext == '.rc':
198:                 # these need to be compiled to object files
199:                 obj_names.append (os.path.join (output_dir,
200:                                             base + self.res_extension))
201:             else:
202:                 obj_names.append (os.path.join (output_dir,
203:                                             base + self.obj_extension))
204:         return obj_names
205: 
206:     # object_filenames ()
207: 
208:     # override the find_library_file method from UnixCCompiler
209:     # to deal with file naming/searching differences
210:     def find_library_file(self, dirs, lib, debug=0):
211:         shortlib = '%s.lib' % lib
212:         longlib = 'lib%s.lib' % lib    # this form very rare
213: 
214:         # get EMX's default library directory search path
215:         try:
216:             emx_dirs = os.environ['LIBRARY_PATH'].split(';')
217:         except KeyError:
218:             emx_dirs = []
219: 
220:         for dir in dirs + emx_dirs:
221:             shortlibp = os.path.join(dir, shortlib)
222:             longlibp = os.path.join(dir, longlib)
223:             if os.path.exists(shortlibp):
224:                 return shortlibp
225:             elif os.path.exists(longlibp):
226:                 return longlibp
227: 
228:         # Oops, didn't find it in *any* of 'dirs'
229:         return None
230: 
231: # class EMXCCompiler
232: 
233: 
234: # Because these compilers aren't configured in Python's pyconfig.h file by
235: # default, we should at least warn the user if he is using a unmodified
236: # version.
237: 
238: CONFIG_H_OK = "ok"
239: CONFIG_H_NOTOK = "not ok"
240: CONFIG_H_UNCERTAIN = "uncertain"
241: 
242: def check_config_h():
243: 
244:     '''Check if the current Python installation (specifically, pyconfig.h)
245:     appears amenable to building extensions with GCC.  Returns a tuple
246:     (status, details), where 'status' is one of the following constants:
247:       CONFIG_H_OK
248:         all is well, go ahead and compile
249:       CONFIG_H_NOTOK
250:         doesn't look good
251:       CONFIG_H_UNCERTAIN
252:         not sure -- unable to read pyconfig.h
253:     'details' is a human-readable string explaining the situation.
254: 
255:     Note there are two ways to conclude "OK": either 'sys.version' contains
256:     the string "GCC" (implying that this Python was built with GCC), or the
257:     installed "pyconfig.h" contains the string "__GNUC__".
258:     '''
259: 
260:     # XXX since this function also checks sys.version, it's not strictly a
261:     # "pyconfig.h" check -- should probably be renamed...
262: 
263:     from distutils import sysconfig
264:     import string
265:     # if sys.version contains GCC then python was compiled with
266:     # GCC, and the pyconfig.h file should be OK
267:     if string.find(sys.version,"GCC") >= 0:
268:         return (CONFIG_H_OK, "sys.version mentions 'GCC'")
269: 
270:     fn = sysconfig.get_config_h_filename()
271:     try:
272:         # It would probably better to read single lines to search.
273:         # But we do this only once, and it is fast enough
274:         f = open(fn)
275:         try:
276:             s = f.read()
277:         finally:
278:             f.close()
279: 
280:     except IOError, exc:
281:         # if we can't read this file, we cannot say it is wrong
282:         # the compiler will complain later about this file as missing
283:         return (CONFIG_H_UNCERTAIN,
284:                 "couldn't read '%s': %s" % (fn, exc.strerror))
285: 
286:     else:
287:         # "pyconfig.h" contains an "#ifdef __GNUC__" or something similar
288:         if string.find(s,"__GNUC__") >= 0:
289:             return (CONFIG_H_OK, "'%s' mentions '__GNUC__'" % fn)
290:         else:
291:             return (CONFIG_H_NOTOK, "'%s' does not mention '__GNUC__'" % fn)
292: 
293: 
294: def get_versions():
295:     ''' Try to find out the versions of gcc and ld.
296:         If not possible it returns None for it.
297:     '''
298:     from distutils.version import StrictVersion
299:     from distutils.spawn import find_executable
300:     import re
301: 
302:     gcc_exe = find_executable('gcc')
303:     if gcc_exe:
304:         out = os.popen(gcc_exe + ' -dumpversion','r')
305:         try:
306:             out_string = out.read()
307:         finally:
308:             out.close()
309:         result = re.search('(\d+\.\d+\.\d+)',out_string)
310:         if result:
311:             gcc_version = StrictVersion(result.group(1))
312:         else:
313:             gcc_version = None
314:     else:
315:         gcc_version = None
316:     # EMX ld has no way of reporting version number, and we use GCC
317:     # anyway - so we can link OMF DLLs
318:     ld_version = None
319:     return (gcc_version, ld_version)
320: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_3103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.emxccompiler\n\nProvides the EMXCCompiler class, a subclass of UnixCCompiler that\nhandles the EMX port of the GNU C compiler to OS/2.\n')

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_3104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__revision__', str_3104)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# Multiple import statement. import os (1/3) (line 24)
import os

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'os', os, module_type_store)
# Multiple import statement. import sys (2/3) (line 24)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'sys', sys, module_type_store)
# Multiple import statement. import copy (3/3) (line 24)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from distutils.ccompiler import gen_preprocess_options, gen_lib_options' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_3105 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.ccompiler')

if (type(import_3105) is not StypyTypeError):

    if (import_3105 != 'pyd_module'):
        __import__(import_3105)
        sys_modules_3106 = sys.modules[import_3105]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.ccompiler', sys_modules_3106.module_type_store, module_type_store, ['gen_preprocess_options', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_3106, sys_modules_3106.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import gen_preprocess_options, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.ccompiler', None, module_type_store, ['gen_preprocess_options', 'gen_lib_options'], [gen_preprocess_options, gen_lib_options])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.ccompiler', import_3105)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from distutils.unixccompiler import UnixCCompiler' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_3107 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.unixccompiler')

if (type(import_3107) is not StypyTypeError):

    if (import_3107 != 'pyd_module'):
        __import__(import_3107)
        sys_modules_3108 = sys.modules[import_3107]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.unixccompiler', sys_modules_3108.module_type_store, module_type_store, ['UnixCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_3108, sys_modules_3108.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import UnixCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.unixccompiler', None, module_type_store, ['UnixCCompiler'], [UnixCCompiler])

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.unixccompiler', import_3107)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from distutils.file_util import write_file' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_3109 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'distutils.file_util')

if (type(import_3109) is not StypyTypeError):

    if (import_3109 != 'pyd_module'):
        __import__(import_3109)
        sys_modules_3110 = sys.modules[import_3109]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'distutils.file_util', sys_modules_3110.module_type_store, module_type_store, ['write_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_3110, sys_modules_3110.module_type_store, module_type_store)
    else:
        from distutils.file_util import write_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'distutils.file_util', None, module_type_store, ['write_file'], [write_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'distutils.file_util', import_3109)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from distutils.errors import DistutilsExecError, CompileError, UnknownFileError' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_3111 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'distutils.errors')

if (type(import_3111) is not StypyTypeError):

    if (import_3111 != 'pyd_module'):
        __import__(import_3111)
        sys_modules_3112 = sys.modules[import_3111]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'distutils.errors', sys_modules_3112.module_type_store, module_type_store, ['DistutilsExecError', 'CompileError', 'UnknownFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_3112, sys_modules_3112.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, CompileError, UnknownFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'CompileError', 'UnknownFileError'], [DistutilsExecError, CompileError, UnknownFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'distutils.errors', import_3111)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from distutils import log' statement (line 29)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'EMXCCompiler' class
# Getting the type of 'UnixCCompiler' (line 31)
UnixCCompiler_3113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'UnixCCompiler')

class EMXCCompiler(UnixCCompiler_3113, ):
    
    # Assigning a Str to a Name (line 33):
    
    # Assigning a Str to a Name (line 34):
    
    # Assigning a Str to a Name (line 35):
    
    # Assigning a Str to a Name (line 36):
    
    # Assigning a Str to a Name (line 37):
    
    # Assigning a Str to a Name (line 38):
    
    # Assigning a Str to a Name (line 39):
    
    # Assigning a Str to a Name (line 40):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'int')
        int_3115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'int')
        int_3116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'int')
        defaults = [int_3114, int_3115, int_3116]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EMXCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_3119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 32), 'self', False)
        # Getting the type of 'verbose' (line 47)
        verbose_3120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'verbose', False)
        # Getting the type of 'dry_run' (line 47)
        dry_run_3121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 47), 'dry_run', False)
        # Getting the type of 'force' (line 47)
        force_3122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 56), 'force', False)
        # Processing the call keyword arguments (line 47)
        kwargs_3123 = {}
        # Getting the type of 'UnixCCompiler' (line 47)
        UnixCCompiler_3117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'UnixCCompiler', False)
        # Obtaining the member '__init__' of a type (line 47)
        init___3118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), UnixCCompiler_3117, '__init__')
        # Calling __init__(args, kwargs) (line 47)
        init___call_result_3124 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), init___3118, *[self_3119, verbose_3120, dry_run_3121, force_3122], **kwargs_3123)
        
        
        # Assigning a Call to a Tuple (line 49):
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_3125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to check_config_h(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_3127 = {}
        # Getting the type of 'check_config_h' (line 49)
        check_config_h_3126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'check_config_h', False)
        # Calling check_config_h(args, kwargs) (line 49)
        check_config_h_call_result_3128 = invoke(stypy.reporting.localization.Localization(__file__, 49, 28), check_config_h_3126, *[], **kwargs_3127)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___3129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), check_config_h_call_result_3128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_3130 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___3129, int_3125)
        
        # Assigning a type to the variable 'tuple_var_assignment_3095' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_3095', subscript_call_result_3130)
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_3131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to check_config_h(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_3133 = {}
        # Getting the type of 'check_config_h' (line 49)
        check_config_h_3132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'check_config_h', False)
        # Calling check_config_h(args, kwargs) (line 49)
        check_config_h_call_result_3134 = invoke(stypy.reporting.localization.Localization(__file__, 49, 28), check_config_h_3132, *[], **kwargs_3133)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___3135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), check_config_h_call_result_3134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_3136 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___3135, int_3131)
        
        # Assigning a type to the variable 'tuple_var_assignment_3096' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_3096', subscript_call_result_3136)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_3095' (line 49)
        tuple_var_assignment_3095_3137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_3095')
        # Assigning a type to the variable 'status' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'status', tuple_var_assignment_3095_3137)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_3096' (line 49)
        tuple_var_assignment_3096_3138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_3096')
        # Assigning a type to the variable 'details' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'details', tuple_var_assignment_3096_3138)
        
        # Call to debug_print(...): (line 50)
        # Processing the call arguments (line 50)
        str_3141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'str', "Python's GCC status: %s (details: %s)")
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_3142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        # Getting the type of 'status' (line 51)
        status_3143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'status', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 26), tuple_3142, status_3143)
        # Adding element type (line 51)
        # Getting the type of 'details' (line 51)
        details_3144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 34), 'details', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 26), tuple_3142, details_3144)
        
        # Applying the binary operator '%' (line 50)
        result_mod_3145 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 25), '%', str_3141, tuple_3142)
        
        # Processing the call keyword arguments (line 50)
        kwargs_3146 = {}
        # Getting the type of 'self' (line 50)
        self_3139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 50)
        debug_print_3140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_3139, 'debug_print')
        # Calling debug_print(args, kwargs) (line 50)
        debug_print_call_result_3147 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), debug_print_3140, *[result_mod_3145], **kwargs_3146)
        
        
        
        # Getting the type of 'status' (line 52)
        status_3148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'status')
        # Getting the type of 'CONFIG_H_OK' (line 52)
        CONFIG_H_OK_3149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'CONFIG_H_OK')
        # Applying the binary operator 'isnot' (line 52)
        result_is_not_3150 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 11), 'isnot', status_3148, CONFIG_H_OK_3149)
        
        # Testing the type of an if condition (line 52)
        if_condition_3151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), result_is_not_3150)
        # Assigning a type to the variable 'if_condition_3151' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_3151', if_condition_3151)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 53)
        # Processing the call arguments (line 53)
        str_3154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'str', "Python's pyconfig.h doesn't seem to support your compiler.  ")
        str_3155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 17), 'str', 'Reason: %s.')
        # Getting the type of 'details' (line 55)
        details_3156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'details', False)
        # Applying the binary operator '%' (line 55)
        result_mod_3157 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 17), '%', str_3155, details_3156)
        
        # Applying the binary operator '+' (line 54)
        result_add_3158 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 16), '+', str_3154, result_mod_3157)
        
        str_3159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'str', 'Compiling may fail because of undefined preprocessor macros.')
        # Applying the binary operator '+' (line 55)
        result_add_3160 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 42), '+', result_add_3158, str_3159)
        
        # Processing the call keyword arguments (line 53)
        kwargs_3161 = {}
        # Getting the type of 'self' (line 53)
        self_3152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 53)
        warn_3153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), self_3152, 'warn')
        # Calling warn(args, kwargs) (line 53)
        warn_call_result_3162 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), warn_3153, *[result_add_3160], **kwargs_3161)
        
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 58):
        
        # Assigning a Subscript to a Name (line 58):
        
        # Obtaining the type of the subscript
        int_3163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'int')
        
        # Call to get_versions(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_3165 = {}
        # Getting the type of 'get_versions' (line 59)
        get_versions_3164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'get_versions', False)
        # Calling get_versions(args, kwargs) (line 59)
        get_versions_call_result_3166 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), get_versions_3164, *[], **kwargs_3165)
        
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___3167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), get_versions_call_result_3166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_3168 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), getitem___3167, int_3163)
        
        # Assigning a type to the variable 'tuple_var_assignment_3097' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'tuple_var_assignment_3097', subscript_call_result_3168)
        
        # Assigning a Subscript to a Name (line 58):
        
        # Obtaining the type of the subscript
        int_3169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'int')
        
        # Call to get_versions(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_3171 = {}
        # Getting the type of 'get_versions' (line 59)
        get_versions_3170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'get_versions', False)
        # Calling get_versions(args, kwargs) (line 59)
        get_versions_call_result_3172 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), get_versions_3170, *[], **kwargs_3171)
        
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___3173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), get_versions_call_result_3172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_3174 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), getitem___3173, int_3169)
        
        # Assigning a type to the variable 'tuple_var_assignment_3098' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'tuple_var_assignment_3098', subscript_call_result_3174)
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'tuple_var_assignment_3097' (line 58)
        tuple_var_assignment_3097_3175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'tuple_var_assignment_3097')
        # Getting the type of 'self' (line 58)
        self_3176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'self')
        # Setting the type of the member 'gcc_version' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 9), self_3176, 'gcc_version', tuple_var_assignment_3097_3175)
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'tuple_var_assignment_3098' (line 58)
        tuple_var_assignment_3098_3177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'tuple_var_assignment_3098')
        # Getting the type of 'self' (line 58)
        self_3178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'self')
        # Setting the type of the member 'ld_version' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 27), self_3178, 'ld_version', tuple_var_assignment_3098_3177)
        
        # Call to debug_print(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_3181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'self', False)
        # Obtaining the member 'compiler_type' of a type (line 60)
        compiler_type_3182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), self_3181, 'compiler_type')
        str_3183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'str', ': gcc %s, ld %s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 61)
        tuple_3184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 61)
        # Adding element type (line 61)
        # Getting the type of 'self' (line 61)
        self_3185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'self', False)
        # Obtaining the member 'gcc_version' of a type (line 61)
        gcc_version_3186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 26), self_3185, 'gcc_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), tuple_3184, gcc_version_3186)
        # Adding element type (line 61)
        # Getting the type of 'self' (line 62)
        self_3187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'self', False)
        # Obtaining the member 'ld_version' of a type (line 62)
        ld_version_3188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 26), self_3187, 'ld_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), tuple_3184, ld_version_3188)
        
        # Applying the binary operator '%' (line 60)
        result_mod_3189 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 46), '%', str_3183, tuple_3184)
        
        # Applying the binary operator '+' (line 60)
        result_add_3190 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 25), '+', compiler_type_3182, result_mod_3189)
        
        # Processing the call keyword arguments (line 60)
        kwargs_3191 = {}
        # Getting the type of 'self' (line 60)
        self_3179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 60)
        debug_print_3180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_3179, 'debug_print')
        # Calling debug_print(args, kwargs) (line 60)
        debug_print_call_result_3192 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), debug_print_3180, *[result_add_3190], **kwargs_3191)
        
        
        # Call to set_executables(...): (line 66)
        # Processing the call keyword arguments (line 66)
        str_3195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'str', 'gcc -Zomf -Zmt -O3 -fomit-frame-pointer -mprobe -Wall')
        keyword_3196 = str_3195
        str_3197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 41), 'str', 'gcc -Zomf -Zmt -O3 -fomit-frame-pointer -mprobe -Wall')
        keyword_3198 = str_3197
        str_3199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 40), 'str', 'gcc -Zomf -Zmt -Zcrtdll')
        keyword_3200 = str_3199
        str_3201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 39), 'str', 'gcc -Zomf -Zmt -Zcrtdll -Zdll')
        keyword_3202 = str_3201
        kwargs_3203 = {'linker_exe': keyword_3200, 'compiler_so': keyword_3198, 'linker_so': keyword_3202, 'compiler': keyword_3196}
        # Getting the type of 'self' (line 66)
        self_3193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 66)
        set_executables_3194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_3193, 'set_executables')
        # Calling set_executables(args, kwargs) (line 66)
        set_executables_call_result_3204 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), set_executables_3194, *[], **kwargs_3203)
        
        
        # Assigning a List to a Attribute (line 73):
        
        # Assigning a List to a Attribute (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_3205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        str_3206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'str', 'gcc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 27), list_3205, str_3206)
        
        # Getting the type of 'self' (line 73)
        self_3207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'dll_libraries' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_3207, 'dll_libraries', list_3205)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EMXCCompiler._compile.__dict__.__setitem__('stypy_localization', localization)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_function_name', 'EMXCCompiler._compile')
        EMXCCompiler._compile.__dict__.__setitem__('stypy_param_names_list', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'])
        EMXCCompiler._compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EMXCCompiler._compile.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EMXCCompiler._compile', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compile', localization, ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compile(...)' code ##################

        
        
        # Getting the type of 'ext' (line 78)
        ext_3208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'ext')
        str_3209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'str', '.rc')
        # Applying the binary operator '==' (line 78)
        result_eq_3210 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '==', ext_3208, str_3209)
        
        # Testing the type of an if condition (line 78)
        if_condition_3211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_eq_3210)
        # Assigning a type to the variable 'if_condition_3211' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_3211', if_condition_3211)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_3214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        str_3215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'str', 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 27), list_3214, str_3215)
        # Adding element type (line 81)
        str_3216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 34), 'str', '-r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 27), list_3214, str_3216)
        # Adding element type (line 81)
        # Getting the type of 'src' (line 81)
        src_3217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 27), list_3214, src_3217)
        
        # Processing the call keyword arguments (line 81)
        kwargs_3218 = {}
        # Getting the type of 'self' (line 81)
        self_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 81)
        spawn_3213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 16), self_3212, 'spawn')
        # Calling spawn(args, kwargs) (line 81)
        spawn_call_result_3219 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), spawn_3213, *[list_3214], **kwargs_3218)
        
        # SSA branch for the except part of a try statement (line 80)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 80)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 82)
        DistutilsExecError_3220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'msg', DistutilsExecError_3220)
        # Getting the type of 'CompileError' (line 83)
        CompileError_3221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 83, 16), CompileError_3221, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 78)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_3224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'self', False)
        # Obtaining the member 'compiler_so' of a type (line 86)
        compiler_so_3225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 27), self_3224, 'compiler_so')
        # Getting the type of 'cc_args' (line 86)
        cc_args_3226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 46), 'cc_args', False)
        # Applying the binary operator '+' (line 86)
        result_add_3227 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 27), '+', compiler_so_3225, cc_args_3226)
        
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_3228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        # Getting the type of 'src' (line 86)
        src_3229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 57), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 56), list_3228, src_3229)
        # Adding element type (line 86)
        str_3230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 62), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 56), list_3228, str_3230)
        # Adding element type (line 86)
        # Getting the type of 'obj' (line 86)
        obj_3231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 68), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 56), list_3228, obj_3231)
        
        # Applying the binary operator '+' (line 86)
        result_add_3232 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 54), '+', result_add_3227, list_3228)
        
        # Getting the type of 'extra_postargs' (line 87)
        extra_postargs_3233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'extra_postargs', False)
        # Applying the binary operator '+' (line 86)
        result_add_3234 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 73), '+', result_add_3232, extra_postargs_3233)
        
        # Processing the call keyword arguments (line 86)
        kwargs_3235 = {}
        # Getting the type of 'self' (line 86)
        self_3222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 86)
        spawn_3223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), self_3222, 'spawn')
        # Calling spawn(args, kwargs) (line 86)
        spawn_call_result_3236 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), spawn_3223, *[result_add_3234], **kwargs_3235)
        
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 85)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 88)
        DistutilsExecError_3237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'msg', DistutilsExecError_3237)
        # Getting the type of 'CompileError' (line 89)
        CompileError_3238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 89, 16), CompileError_3238, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_3239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_3239


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 95)
        None_3240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'None')
        # Getting the type of 'None' (line 96)
        None_3241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'None')
        # Getting the type of 'None' (line 97)
        None_3242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'None')
        # Getting the type of 'None' (line 98)
        None_3243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'None')
        # Getting the type of 'None' (line 99)
        None_3244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'None')
        int_3245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'int')
        # Getting the type of 'None' (line 101)
        None_3246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'None')
        # Getting the type of 'None' (line 102)
        None_3247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'None')
        # Getting the type of 'None' (line 103)
        None_3248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'None')
        # Getting the type of 'None' (line 104)
        None_3249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'None')
        defaults = [None_3240, None_3241, None_3242, None_3243, None_3244, int_3245, None_3246, None_3247, None_3248, None_3249]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EMXCCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        EMXCCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EMXCCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        EMXCCompiler.link.__dict__.__setitem__('stypy_function_name', 'EMXCCompiler.link')
        EMXCCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        EMXCCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        EMXCCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EMXCCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        EMXCCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        EMXCCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EMXCCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EMXCCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'link', localization, ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'link(...)' code ##################

        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to copy(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_preargs' (line 107)
        extra_preargs_3252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'extra_preargs', False)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_3253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        
        # Applying the binary operator 'or' (line 107)
        result_or_keyword_3254 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 34), 'or', extra_preargs_3252, list_3253)
        
        # Processing the call keyword arguments (line 107)
        kwargs_3255 = {}
        # Getting the type of 'copy' (line 107)
        copy_3250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'copy', False)
        # Obtaining the member 'copy' of a type (line 107)
        copy_3251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 24), copy_3250, 'copy')
        # Calling copy(args, kwargs) (line 107)
        copy_call_result_3256 = invoke(stypy.reporting.localization.Localization(__file__, 107, 24), copy_3251, *[result_or_keyword_3254], **kwargs_3255)
        
        # Assigning a type to the variable 'extra_preargs' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'extra_preargs', copy_call_result_3256)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to copy(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Evaluating a boolean operation
        # Getting the type of 'libraries' (line 108)
        libraries_3259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'libraries', False)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_3260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        
        # Applying the binary operator 'or' (line 108)
        result_or_keyword_3261 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 30), 'or', libraries_3259, list_3260)
        
        # Processing the call keyword arguments (line 108)
        kwargs_3262 = {}
        # Getting the type of 'copy' (line 108)
        copy_3257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'copy', False)
        # Obtaining the member 'copy' of a type (line 108)
        copy_3258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), copy_3257, 'copy')
        # Calling copy(args, kwargs) (line 108)
        copy_call_result_3263 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), copy_3258, *[result_or_keyword_3261], **kwargs_3262)
        
        # Assigning a type to the variable 'libraries' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'libraries', copy_call_result_3263)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to copy(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Evaluating a boolean operation
        # Getting the type of 'objects' (line 109)
        objects_3266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'objects', False)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_3267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        
        # Applying the binary operator 'or' (line 109)
        result_or_keyword_3268 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 28), 'or', objects_3266, list_3267)
        
        # Processing the call keyword arguments (line 109)
        kwargs_3269 = {}
        # Getting the type of 'copy' (line 109)
        copy_3264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'copy', False)
        # Obtaining the member 'copy' of a type (line 109)
        copy_3265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 18), copy_3264, 'copy')
        # Calling copy(args, kwargs) (line 109)
        copy_call_result_3270 = invoke(stypy.reporting.localization.Localization(__file__, 109, 18), copy_3265, *[result_or_keyword_3268], **kwargs_3269)
        
        # Assigning a type to the variable 'objects' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'objects', copy_call_result_3270)
        
        # Call to extend(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_3273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'self', False)
        # Obtaining the member 'dll_libraries' of a type (line 112)
        dll_libraries_3274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), self_3273, 'dll_libraries')
        # Processing the call keyword arguments (line 112)
        kwargs_3275 = {}
        # Getting the type of 'libraries' (line 112)
        libraries_3271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'libraries', False)
        # Obtaining the member 'extend' of a type (line 112)
        extend_3272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), libraries_3271, 'extend')
        # Calling extend(args, kwargs) (line 112)
        extend_call_result_3276 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), extend_3272, *[dll_libraries_3274], **kwargs_3275)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'export_symbols' (line 116)
        export_symbols_3277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'export_symbols')
        # Getting the type of 'None' (line 116)
        None_3278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'None')
        # Applying the binary operator 'isnot' (line 116)
        result_is_not_3279 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 13), 'isnot', export_symbols_3277, None_3278)
        
        
        # Getting the type of 'target_desc' (line 117)
        target_desc_3280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'target_desc')
        # Getting the type of 'self' (line 117)
        self_3281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'self')
        # Obtaining the member 'EXECUTABLE' of a type (line 117)
        EXECUTABLE_3282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), self_3281, 'EXECUTABLE')
        # Applying the binary operator '!=' (line 117)
        result_ne_3283 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 13), '!=', target_desc_3280, EXECUTABLE_3282)
        
        # Applying the binary operator 'and' (line 116)
        result_and_keyword_3284 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), 'and', result_is_not_3279, result_ne_3283)
        
        # Testing the type of an if condition (line 116)
        if_condition_3285 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_and_keyword_3284)
        # Assigning a type to the variable 'if_condition_3285' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_3285', if_condition_3285)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to dirname(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining the type of the subscript
        int_3289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 47), 'int')
        # Getting the type of 'objects' (line 126)
        objects_3290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'objects', False)
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___3291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 39), objects_3290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_3292 = invoke(stypy.reporting.localization.Localization(__file__, 126, 39), getitem___3291, int_3289)
        
        # Processing the call keyword arguments (line 126)
        kwargs_3293 = {}
        # Getting the type of 'os' (line 126)
        os_3286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 126)
        path_3287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), os_3286, 'path')
        # Obtaining the member 'dirname' of a type (line 126)
        dirname_3288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), path_3287, 'dirname')
        # Calling dirname(args, kwargs) (line 126)
        dirname_call_result_3294 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), dirname_3288, *[subscript_call_result_3292], **kwargs_3293)
        
        # Assigning a type to the variable 'temp_dir' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'temp_dir', dirname_call_result_3294)
        
        # Assigning a Call to a Tuple (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_3295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 12), 'int')
        
        # Call to splitext(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to basename(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'output_filename' (line 129)
        output_filename_3302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'output_filename', False)
        # Processing the call keyword arguments (line 129)
        kwargs_3303 = {}
        # Getting the type of 'os' (line 129)
        os_3299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 129)
        path_3300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), os_3299, 'path')
        # Obtaining the member 'basename' of a type (line 129)
        basename_3301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), path_3300, 'basename')
        # Calling basename(args, kwargs) (line 129)
        basename_call_result_3304 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), basename_3301, *[output_filename_3302], **kwargs_3303)
        
        # Processing the call keyword arguments (line 128)
        kwargs_3305 = {}
        # Getting the type of 'os' (line 128)
        os_3296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 128)
        path_3297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), os_3296, 'path')
        # Obtaining the member 'splitext' of a type (line 128)
        splitext_3298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), path_3297, 'splitext')
        # Calling splitext(args, kwargs) (line 128)
        splitext_call_result_3306 = invoke(stypy.reporting.localization.Localization(__file__, 128, 40), splitext_3298, *[basename_call_result_3304], **kwargs_3305)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___3307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), splitext_call_result_3306, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_3308 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), getitem___3307, int_3295)
        
        # Assigning a type to the variable 'tuple_var_assignment_3099' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'tuple_var_assignment_3099', subscript_call_result_3308)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_3309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 12), 'int')
        
        # Call to splitext(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to basename(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'output_filename' (line 129)
        output_filename_3316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'output_filename', False)
        # Processing the call keyword arguments (line 129)
        kwargs_3317 = {}
        # Getting the type of 'os' (line 129)
        os_3313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 129)
        path_3314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), os_3313, 'path')
        # Obtaining the member 'basename' of a type (line 129)
        basename_3315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), path_3314, 'basename')
        # Calling basename(args, kwargs) (line 129)
        basename_call_result_3318 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), basename_3315, *[output_filename_3316], **kwargs_3317)
        
        # Processing the call keyword arguments (line 128)
        kwargs_3319 = {}
        # Getting the type of 'os' (line 128)
        os_3310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 128)
        path_3311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), os_3310, 'path')
        # Obtaining the member 'splitext' of a type (line 128)
        splitext_3312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), path_3311, 'splitext')
        # Calling splitext(args, kwargs) (line 128)
        splitext_call_result_3320 = invoke(stypy.reporting.localization.Localization(__file__, 128, 40), splitext_3312, *[basename_call_result_3318], **kwargs_3319)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___3321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), splitext_call_result_3320, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_3322 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), getitem___3321, int_3309)
        
        # Assigning a type to the variable 'tuple_var_assignment_3100' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'tuple_var_assignment_3100', subscript_call_result_3322)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_3099' (line 128)
        tuple_var_assignment_3099_3323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'tuple_var_assignment_3099')
        # Assigning a type to the variable 'dll_name' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'dll_name', tuple_var_assignment_3099_3323)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_3100' (line 128)
        tuple_var_assignment_3100_3324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'tuple_var_assignment_3100')
        # Assigning a type to the variable 'dll_extension' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'dll_extension', tuple_var_assignment_3100_3324)
        
        # Assigning a Call to a Name (line 132):
        
        # Assigning a Call to a Name (line 132):
        
        # Call to join(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'temp_dir' (line 132)
        temp_dir_3328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'temp_dir', False)
        # Getting the type of 'dll_name' (line 132)
        dll_name_3329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 46), 'dll_name', False)
        str_3330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 57), 'str', '.def')
        # Applying the binary operator '+' (line 132)
        result_add_3331 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 46), '+', dll_name_3329, str_3330)
        
        # Processing the call keyword arguments (line 132)
        kwargs_3332 = {}
        # Getting the type of 'os' (line 132)
        os_3325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 132)
        path_3326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), os_3325, 'path')
        # Obtaining the member 'join' of a type (line 132)
        join_3327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), path_3326, 'join')
        # Calling join(args, kwargs) (line 132)
        join_call_result_3333 = invoke(stypy.reporting.localization.Localization(__file__, 132, 23), join_3327, *[temp_dir_3328, result_add_3331], **kwargs_3332)
        
        # Assigning a type to the variable 'def_file' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'def_file', join_call_result_3333)
        
        # Assigning a List to a Name (line 135):
        
        # Assigning a List to a Name (line 135):
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_3334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        str_3335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 16), 'str', 'LIBRARY %s INITINSTANCE TERMINSTANCE')
        
        # Obtaining the type of the subscript
        int_3336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 68), 'int')
        
        # Call to splitext(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to basename(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'output_filename' (line 137)
        output_filename_3343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 137)
        kwargs_3344 = {}
        # Getting the type of 'os' (line 137)
        os_3340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 137)
        path_3341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 33), os_3340, 'path')
        # Obtaining the member 'basename' of a type (line 137)
        basename_3342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 33), path_3341, 'basename')
        # Calling basename(args, kwargs) (line 137)
        basename_call_result_3345 = invoke(stypy.reporting.localization.Localization(__file__, 137, 33), basename_3342, *[output_filename_3343], **kwargs_3344)
        
        # Processing the call keyword arguments (line 137)
        kwargs_3346 = {}
        # Getting the type of 'os' (line 137)
        os_3337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 137)
        path_3338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), os_3337, 'path')
        # Obtaining the member 'splitext' of a type (line 137)
        splitext_3339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), path_3338, 'splitext')
        # Calling splitext(args, kwargs) (line 137)
        splitext_call_result_3347 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), splitext_3339, *[basename_call_result_3345], **kwargs_3346)
        
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___3348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), splitext_call_result_3347, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_3349 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___3348, int_3336)
        
        # Applying the binary operator '%' (line 136)
        result_mod_3350 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 16), '%', str_3335, subscript_call_result_3349)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), list_3334, result_mod_3350)
        # Adding element type (line 135)
        str_3351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'str', 'DATA MULTIPLE NONSHARED')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), list_3334, str_3351)
        # Adding element type (line 135)
        str_3352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 16), 'str', 'EXPORTS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), list_3334, str_3352)
        
        # Assigning a type to the variable 'contents' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'contents', list_3334)
        
        # Getting the type of 'export_symbols' (line 140)
        export_symbols_3353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'export_symbols')
        # Testing the type of a for loop iterable (line 140)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 12), export_symbols_3353)
        # Getting the type of the for loop variable (line 140)
        for_loop_var_3354 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 12), export_symbols_3353)
        # Assigning a type to the variable 'sym' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'sym', for_loop_var_3354)
        # SSA begins for a for statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 141)
        # Processing the call arguments (line 141)
        str_3357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 32), 'str', '  "%s"')
        # Getting the type of 'sym' (line 141)
        sym_3358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'sym', False)
        # Applying the binary operator '%' (line 141)
        result_mod_3359 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 32), '%', str_3357, sym_3358)
        
        # Processing the call keyword arguments (line 141)
        kwargs_3360 = {}
        # Getting the type of 'contents' (line 141)
        contents_3355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'contents', False)
        # Obtaining the member 'append' of a type (line 141)
        append_3356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), contents_3355, 'append')
        # Calling append(args, kwargs) (line 141)
        append_call_result_3361 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), append_3356, *[result_mod_3359], **kwargs_3360)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to execute(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'write_file' (line 142)
        write_file_3364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'write_file', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_3365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'def_file' (line 142)
        def_file_3366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'def_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 38), tuple_3365, def_file_3366)
        # Adding element type (line 142)
        # Getting the type of 'contents' (line 142)
        contents_3367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'contents', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 38), tuple_3365, contents_3367)
        
        str_3368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 25), 'str', 'writing %s')
        # Getting the type of 'def_file' (line 143)
        def_file_3369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'def_file', False)
        # Applying the binary operator '%' (line 143)
        result_mod_3370 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 25), '%', str_3368, def_file_3369)
        
        # Processing the call keyword arguments (line 142)
        kwargs_3371 = {}
        # Getting the type of 'self' (line 142)
        self_3362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 142)
        execute_3363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), self_3362, 'execute')
        # Calling execute(args, kwargs) (line 142)
        execute_call_result_3372 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), execute_3363, *[write_file_3364, tuple_3365, result_mod_3370], **kwargs_3371)
        
        
        # Call to append(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'def_file' (line 147)
        def_file_3375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'def_file', False)
        # Processing the call keyword arguments (line 147)
        kwargs_3376 = {}
        # Getting the type of 'objects' (line 147)
        objects_3373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'objects', False)
        # Obtaining the member 'append' of a type (line 147)
        append_3374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), objects_3373, 'append')
        # Calling append(args, kwargs) (line 147)
        append_call_result_3377 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), append_3374, *[def_file_3375], **kwargs_3376)
        
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'debug' (line 158)
        debug_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'debug')
        # Applying the 'not' unary operator (line 158)
        result_not__3379 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), 'not', debug_3378)
        
        # Testing the type of an if condition (line 158)
        if_condition_3380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_not__3379)
        # Assigning a type to the variable 'if_condition_3380' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_3380', if_condition_3380)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 159)
        # Processing the call arguments (line 159)
        str_3383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 33), 'str', '-s')
        # Processing the call keyword arguments (line 159)
        kwargs_3384 = {}
        # Getting the type of 'extra_preargs' (line 159)
        extra_preargs_3381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'extra_preargs', False)
        # Obtaining the member 'append' of a type (line 159)
        append_3382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), extra_preargs_3381, 'append')
        # Calling append(args, kwargs) (line 159)
        append_call_result_3385 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), append_3382, *[str_3383], **kwargs_3384)
        
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to link(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'self' (line 161)
        self_3388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'self', False)
        # Getting the type of 'target_desc' (line 162)
        target_desc_3389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'target_desc', False)
        # Getting the type of 'objects' (line 163)
        objects_3390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 164)
        output_filename_3391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'output_filename', False)
        # Getting the type of 'output_dir' (line 165)
        output_dir_3392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'output_dir', False)
        # Getting the type of 'libraries' (line 166)
        libraries_3393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'libraries', False)
        # Getting the type of 'library_dirs' (line 167)
        library_dirs_3394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 168)
        runtime_library_dirs_3395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'runtime_library_dirs', False)
        # Getting the type of 'None' (line 169)
        None_3396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 27), 'None', False)
        # Getting the type of 'debug' (line 170)
        debug_3397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'debug', False)
        # Getting the type of 'extra_preargs' (line 171)
        extra_preargs_3398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'extra_preargs', False)
        # Getting the type of 'extra_postargs' (line 172)
        extra_postargs_3399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'extra_postargs', False)
        # Getting the type of 'build_temp' (line 173)
        build_temp_3400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'build_temp', False)
        # Getting the type of 'target_lang' (line 174)
        target_lang_3401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'target_lang', False)
        # Processing the call keyword arguments (line 161)
        kwargs_3402 = {}
        # Getting the type of 'UnixCCompiler' (line 161)
        UnixCCompiler_3386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'UnixCCompiler', False)
        # Obtaining the member 'link' of a type (line 161)
        link_3387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), UnixCCompiler_3386, 'link')
        # Calling link(args, kwargs) (line 161)
        link_call_result_3403 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), link_3387, *[self_3388, target_desc_3389, objects_3390, output_filename_3391, output_dir_3392, libraries_3393, library_dirs_3394, runtime_library_dirs_3395, None_3396, debug_3397, extra_preargs_3398, extra_postargs_3399, build_temp_3400, target_lang_3401], **kwargs_3402)
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_3404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_3404


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 36), 'int')
        str_3406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 37), 'str', '')
        defaults = [int_3405, str_3406]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_localization', localization)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_function_name', 'EMXCCompiler.object_filenames')
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_param_names_list', ['source_filenames', 'strip_dir', 'output_dir'])
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EMXCCompiler.object_filenames.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EMXCCompiler.object_filenames', ['source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'object_filenames', localization, ['source_filenames', 'strip_dir', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'object_filenames(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 186)
        # Getting the type of 'output_dir' (line 186)
        output_dir_3407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'output_dir')
        # Getting the type of 'None' (line 186)
        None_3408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'None')
        
        (may_be_3409, more_types_in_union_3410) = may_be_none(output_dir_3407, None_3408)

        if may_be_3409:

            if more_types_in_union_3410:
                # Runtime conditional SSA (line 186)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 186):
            
            # Assigning a Str to a Name (line 186):
            str_3411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 44), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 31), 'output_dir', str_3411)

            if more_types_in_union_3410:
                # SSA join for if statement (line 186)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 187):
        
        # Assigning a List to a Name (line 187):
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_3412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        
        # Assigning a type to the variable 'obj_names' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'obj_names', list_3412)
        
        # Getting the type of 'source_filenames' (line 188)
        source_filenames_3413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 188)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 8), source_filenames_3413)
        # Getting the type of the for loop variable (line 188)
        for_loop_var_3414 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 8), source_filenames_3413)
        # Assigning a type to the variable 'src_name' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'src_name', for_loop_var_3414)
        # SSA begins for a for statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 190):
        
        # Assigning a Subscript to a Name (line 190):
        
        # Obtaining the type of the subscript
        int_3415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 12), 'int')
        
        # Call to splitext(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to normcase(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'src_name' (line 190)
        src_name_3422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 61), 'src_name', False)
        # Processing the call keyword arguments (line 190)
        kwargs_3423 = {}
        # Getting the type of 'os' (line 190)
        os_3419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 190)
        path_3420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 44), os_3419, 'path')
        # Obtaining the member 'normcase' of a type (line 190)
        normcase_3421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 44), path_3420, 'normcase')
        # Calling normcase(args, kwargs) (line 190)
        normcase_call_result_3424 = invoke(stypy.reporting.localization.Localization(__file__, 190, 44), normcase_3421, *[src_name_3422], **kwargs_3423)
        
        # Processing the call keyword arguments (line 190)
        kwargs_3425 = {}
        # Getting the type of 'os' (line 190)
        os_3416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 190)
        path_3417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 26), os_3416, 'path')
        # Obtaining the member 'splitext' of a type (line 190)
        splitext_3418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 26), path_3417, 'splitext')
        # Calling splitext(args, kwargs) (line 190)
        splitext_call_result_3426 = invoke(stypy.reporting.localization.Localization(__file__, 190, 26), splitext_3418, *[normcase_call_result_3424], **kwargs_3425)
        
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___3427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), splitext_call_result_3426, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_3428 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), getitem___3427, int_3415)
        
        # Assigning a type to the variable 'tuple_var_assignment_3101' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'tuple_var_assignment_3101', subscript_call_result_3428)
        
        # Assigning a Subscript to a Name (line 190):
        
        # Obtaining the type of the subscript
        int_3429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 12), 'int')
        
        # Call to splitext(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to normcase(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'src_name' (line 190)
        src_name_3436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 61), 'src_name', False)
        # Processing the call keyword arguments (line 190)
        kwargs_3437 = {}
        # Getting the type of 'os' (line 190)
        os_3433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 190)
        path_3434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 44), os_3433, 'path')
        # Obtaining the member 'normcase' of a type (line 190)
        normcase_3435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 44), path_3434, 'normcase')
        # Calling normcase(args, kwargs) (line 190)
        normcase_call_result_3438 = invoke(stypy.reporting.localization.Localization(__file__, 190, 44), normcase_3435, *[src_name_3436], **kwargs_3437)
        
        # Processing the call keyword arguments (line 190)
        kwargs_3439 = {}
        # Getting the type of 'os' (line 190)
        os_3430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 190)
        path_3431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 26), os_3430, 'path')
        # Obtaining the member 'splitext' of a type (line 190)
        splitext_3432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 26), path_3431, 'splitext')
        # Calling splitext(args, kwargs) (line 190)
        splitext_call_result_3440 = invoke(stypy.reporting.localization.Localization(__file__, 190, 26), splitext_3432, *[normcase_call_result_3438], **kwargs_3439)
        
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___3441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), splitext_call_result_3440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_3442 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), getitem___3441, int_3429)
        
        # Assigning a type to the variable 'tuple_var_assignment_3102' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'tuple_var_assignment_3102', subscript_call_result_3442)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'tuple_var_assignment_3101' (line 190)
        tuple_var_assignment_3101_3443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'tuple_var_assignment_3101')
        # Assigning a type to the variable 'base' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'base', tuple_var_assignment_3101_3443)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'tuple_var_assignment_3102' (line 190)
        tuple_var_assignment_3102_3444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'tuple_var_assignment_3102')
        # Assigning a type to the variable 'ext' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'ext', tuple_var_assignment_3102_3444)
        
        
        # Getting the type of 'ext' (line 191)
        ext_3445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'ext')
        # Getting the type of 'self' (line 191)
        self_3446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'self')
        # Obtaining the member 'src_extensions' of a type (line 191)
        src_extensions_3447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), self_3446, 'src_extensions')
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_3448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        # Adding element type (line 191)
        str_3449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 50), 'str', '.rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 49), list_3448, str_3449)
        
        # Applying the binary operator '+' (line 191)
        result_add_3450 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 27), '+', src_extensions_3447, list_3448)
        
        # Applying the binary operator 'notin' (line 191)
        result_contains_3451 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), 'notin', ext_3445, result_add_3450)
        
        # Testing the type of an if condition (line 191)
        if_condition_3452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 12), result_contains_3451)
        # Assigning a type to the variable 'if_condition_3452' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'if_condition_3452', if_condition_3452)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'UnknownFileError' (line 192)
        UnknownFileError_3453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 22), 'UnknownFileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 192, 16), UnknownFileError_3453, 'raise parameter', BaseException)
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 195)
        strip_dir_3454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'strip_dir')
        # Testing the type of an if condition (line 195)
        if_condition_3455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), strip_dir_3454)
        # Assigning a type to the variable 'if_condition_3455' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_3455', if_condition_3455)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to basename(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'base' (line 196)
        base_3459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 41), 'base', False)
        # Processing the call keyword arguments (line 196)
        kwargs_3460 = {}
        # Getting the type of 'os' (line 196)
        os_3456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 196)
        path_3457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 23), os_3456, 'path')
        # Obtaining the member 'basename' of a type (line 196)
        basename_3458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 23), path_3457, 'basename')
        # Calling basename(args, kwargs) (line 196)
        basename_call_result_3461 = invoke(stypy.reporting.localization.Localization(__file__, 196, 23), basename_3458, *[base_3459], **kwargs_3460)
        
        # Assigning a type to the variable 'base' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'base', basename_call_result_3461)
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 197)
        ext_3462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'ext')
        str_3463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'str', '.rc')
        # Applying the binary operator '==' (line 197)
        result_eq_3464 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '==', ext_3462, str_3463)
        
        # Testing the type of an if condition (line 197)
        if_condition_3465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_eq_3464)
        # Assigning a type to the variable 'if_condition_3465' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'if_condition_3465', if_condition_3465)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to join(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'output_dir' (line 199)
        output_dir_3471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'output_dir', False)
        # Getting the type of 'base' (line 200)
        base_3472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 44), 'base', False)
        # Getting the type of 'self' (line 200)
        self_3473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 51), 'self', False)
        # Obtaining the member 'res_extension' of a type (line 200)
        res_extension_3474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 51), self_3473, 'res_extension')
        # Applying the binary operator '+' (line 200)
        result_add_3475 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 44), '+', base_3472, res_extension_3474)
        
        # Processing the call keyword arguments (line 199)
        kwargs_3476 = {}
        # Getting the type of 'os' (line 199)
        os_3468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 199)
        path_3469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 34), os_3468, 'path')
        # Obtaining the member 'join' of a type (line 199)
        join_3470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 34), path_3469, 'join')
        # Calling join(args, kwargs) (line 199)
        join_call_result_3477 = invoke(stypy.reporting.localization.Localization(__file__, 199, 34), join_3470, *[output_dir_3471, result_add_3475], **kwargs_3476)
        
        # Processing the call keyword arguments (line 199)
        kwargs_3478 = {}
        # Getting the type of 'obj_names' (line 199)
        obj_names_3466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 199)
        append_3467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), obj_names_3466, 'append')
        # Calling append(args, kwargs) (line 199)
        append_call_result_3479 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), append_3467, *[join_call_result_3477], **kwargs_3478)
        
        # SSA branch for the else part of an if statement (line 197)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to join(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'output_dir' (line 202)
        output_dir_3485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 48), 'output_dir', False)
        # Getting the type of 'base' (line 203)
        base_3486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'base', False)
        # Getting the type of 'self' (line 203)
        self_3487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 51), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 203)
        obj_extension_3488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 51), self_3487, 'obj_extension')
        # Applying the binary operator '+' (line 203)
        result_add_3489 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 44), '+', base_3486, obj_extension_3488)
        
        # Processing the call keyword arguments (line 202)
        kwargs_3490 = {}
        # Getting the type of 'os' (line 202)
        os_3482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 202)
        path_3483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 34), os_3482, 'path')
        # Obtaining the member 'join' of a type (line 202)
        join_3484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 34), path_3483, 'join')
        # Calling join(args, kwargs) (line 202)
        join_call_result_3491 = invoke(stypy.reporting.localization.Localization(__file__, 202, 34), join_3484, *[output_dir_3485, result_add_3489], **kwargs_3490)
        
        # Processing the call keyword arguments (line 202)
        kwargs_3492 = {}
        # Getting the type of 'obj_names' (line 202)
        obj_names_3480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 202)
        append_3481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), obj_names_3480, 'append')
        # Calling append(args, kwargs) (line 202)
        append_call_result_3493 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), append_3481, *[join_call_result_3491], **kwargs_3492)
        
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 204)
        obj_names_3494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'stypy_return_type', obj_names_3494)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_3495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3495)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_3495


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 49), 'int')
        defaults = [int_3496]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_localization', localization)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_function_name', 'EMXCCompiler.find_library_file')
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_param_names_list', ['dirs', 'lib', 'debug'])
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EMXCCompiler.find_library_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EMXCCompiler.find_library_file', ['dirs', 'lib', 'debug'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_library_file', localization, ['dirs', 'lib', 'debug'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_library_file(...)' code ##################

        
        # Assigning a BinOp to a Name (line 211):
        
        # Assigning a BinOp to a Name (line 211):
        str_3497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'str', '%s.lib')
        # Getting the type of 'lib' (line 211)
        lib_3498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'lib')
        # Applying the binary operator '%' (line 211)
        result_mod_3499 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 19), '%', str_3497, lib_3498)
        
        # Assigning a type to the variable 'shortlib' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'shortlib', result_mod_3499)
        
        # Assigning a BinOp to a Name (line 212):
        
        # Assigning a BinOp to a Name (line 212):
        str_3500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'str', 'lib%s.lib')
        # Getting the type of 'lib' (line 212)
        lib_3501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'lib')
        # Applying the binary operator '%' (line 212)
        result_mod_3502 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 18), '%', str_3500, lib_3501)
        
        # Assigning a type to the variable 'longlib' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'longlib', result_mod_3502)
        
        
        # SSA begins for try-except statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to split(...): (line 216)
        # Processing the call arguments (line 216)
        str_3509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 56), 'str', ';')
        # Processing the call keyword arguments (line 216)
        kwargs_3510 = {}
        
        # Obtaining the type of the subscript
        str_3503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 34), 'str', 'LIBRARY_PATH')
        # Getting the type of 'os' (line 216)
        os_3504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'os', False)
        # Obtaining the member 'environ' of a type (line 216)
        environ_3505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 23), os_3504, 'environ')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___3506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 23), environ_3505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_3507 = invoke(stypy.reporting.localization.Localization(__file__, 216, 23), getitem___3506, str_3503)
        
        # Obtaining the member 'split' of a type (line 216)
        split_3508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 23), subscript_call_result_3507, 'split')
        # Calling split(args, kwargs) (line 216)
        split_call_result_3511 = invoke(stypy.reporting.localization.Localization(__file__, 216, 23), split_3508, *[str_3509], **kwargs_3510)
        
        # Assigning a type to the variable 'emx_dirs' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'emx_dirs', split_call_result_3511)
        # SSA branch for the except part of a try statement (line 215)
        # SSA branch for the except 'KeyError' branch of a try statement (line 215)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a List to a Name (line 218):
        
        # Assigning a List to a Name (line 218):
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_3512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        
        # Assigning a type to the variable 'emx_dirs' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'emx_dirs', list_3512)
        # SSA join for try-except statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'dirs' (line 220)
        dirs_3513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'dirs')
        # Getting the type of 'emx_dirs' (line 220)
        emx_dirs_3514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'emx_dirs')
        # Applying the binary operator '+' (line 220)
        result_add_3515 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 19), '+', dirs_3513, emx_dirs_3514)
        
        # Testing the type of a for loop iterable (line 220)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 8), result_add_3515)
        # Getting the type of the for loop variable (line 220)
        for_loop_var_3516 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 8), result_add_3515)
        # Assigning a type to the variable 'dir' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'dir', for_loop_var_3516)
        # SSA begins for a for statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to join(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'dir' (line 221)
        dir_3520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 37), 'dir', False)
        # Getting the type of 'shortlib' (line 221)
        shortlib_3521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 42), 'shortlib', False)
        # Processing the call keyword arguments (line 221)
        kwargs_3522 = {}
        # Getting the type of 'os' (line 221)
        os_3517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 221)
        path_3518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 24), os_3517, 'path')
        # Obtaining the member 'join' of a type (line 221)
        join_3519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 24), path_3518, 'join')
        # Calling join(args, kwargs) (line 221)
        join_call_result_3523 = invoke(stypy.reporting.localization.Localization(__file__, 221, 24), join_3519, *[dir_3520, shortlib_3521], **kwargs_3522)
        
        # Assigning a type to the variable 'shortlibp' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'shortlibp', join_call_result_3523)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to join(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'dir' (line 222)
        dir_3527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'dir', False)
        # Getting the type of 'longlib' (line 222)
        longlib_3528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 41), 'longlib', False)
        # Processing the call keyword arguments (line 222)
        kwargs_3529 = {}
        # Getting the type of 'os' (line 222)
        os_3524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 222)
        path_3525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), os_3524, 'path')
        # Obtaining the member 'join' of a type (line 222)
        join_3526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), path_3525, 'join')
        # Calling join(args, kwargs) (line 222)
        join_call_result_3530 = invoke(stypy.reporting.localization.Localization(__file__, 222, 23), join_3526, *[dir_3527, longlib_3528], **kwargs_3529)
        
        # Assigning a type to the variable 'longlibp' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'longlibp', join_call_result_3530)
        
        
        # Call to exists(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'shortlibp' (line 223)
        shortlibp_3534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 30), 'shortlibp', False)
        # Processing the call keyword arguments (line 223)
        kwargs_3535 = {}
        # Getting the type of 'os' (line 223)
        os_3531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 223)
        path_3532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), os_3531, 'path')
        # Obtaining the member 'exists' of a type (line 223)
        exists_3533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), path_3532, 'exists')
        # Calling exists(args, kwargs) (line 223)
        exists_call_result_3536 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), exists_3533, *[shortlibp_3534], **kwargs_3535)
        
        # Testing the type of an if condition (line 223)
        if_condition_3537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 12), exists_call_result_3536)
        # Assigning a type to the variable 'if_condition_3537' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'if_condition_3537', if_condition_3537)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'shortlibp' (line 224)
        shortlibp_3538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'shortlibp')
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', shortlibp_3538)
        # SSA branch for the else part of an if statement (line 223)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to exists(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'longlibp' (line 225)
        longlibp_3542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 32), 'longlibp', False)
        # Processing the call keyword arguments (line 225)
        kwargs_3543 = {}
        # Getting the type of 'os' (line 225)
        os_3539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 225)
        path_3540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 17), os_3539, 'path')
        # Obtaining the member 'exists' of a type (line 225)
        exists_3541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 17), path_3540, 'exists')
        # Calling exists(args, kwargs) (line 225)
        exists_call_result_3544 = invoke(stypy.reporting.localization.Localization(__file__, 225, 17), exists_3541, *[longlibp_3542], **kwargs_3543)
        
        # Testing the type of an if condition (line 225)
        if_condition_3545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 17), exists_call_result_3544)
        # Assigning a type to the variable 'if_condition_3545' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'if_condition_3545', if_condition_3545)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'longlibp' (line 226)
        longlibp_3546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'longlibp')
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'stypy_return_type', longlibp_3546)
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 229)
        None_3547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', None_3547)
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_3548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_3548


# Assigning a type to the variable 'EMXCCompiler' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'EMXCCompiler', EMXCCompiler)

# Assigning a Str to a Name (line 33):
str_3549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'str', 'emx')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3550, 'compiler_type', str_3549)

# Assigning a Str to a Name (line 34):
str_3551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'str', '.obj')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3552, 'obj_extension', str_3551)

# Assigning a Str to a Name (line 35):
str_3553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'str', '.lib')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3554, 'static_lib_extension', str_3553)

# Assigning a Str to a Name (line 36):
str_3555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'str', '.dll')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3556, 'shared_lib_extension', str_3555)

# Assigning a Str to a Name (line 37):
str_3557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'str', '%s%s')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3558, 'static_lib_format', str_3557)

# Assigning a Str to a Name (line 38):
str_3559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'str', '%s%s')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3560, 'shared_lib_format', str_3559)

# Assigning a Str to a Name (line 39):
str_3561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'str', '.res')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'res_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3562, 'res_extension', str_3561)

# Assigning a Str to a Name (line 40):
str_3563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'str', '.exe')
# Getting the type of 'EMXCCompiler'
EMXCCompiler_3564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EMXCCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EMXCCompiler_3564, 'exe_extension', str_3563)

# Assigning a Str to a Name (line 238):

# Assigning a Str to a Name (line 238):
str_3565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 14), 'str', 'ok')
# Assigning a type to the variable 'CONFIG_H_OK' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'CONFIG_H_OK', str_3565)

# Assigning a Str to a Name (line 239):

# Assigning a Str to a Name (line 239):
str_3566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 17), 'str', 'not ok')
# Assigning a type to the variable 'CONFIG_H_NOTOK' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'CONFIG_H_NOTOK', str_3566)

# Assigning a Str to a Name (line 240):

# Assigning a Str to a Name (line 240):
str_3567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 21), 'str', 'uncertain')
# Assigning a type to the variable 'CONFIG_H_UNCERTAIN' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'CONFIG_H_UNCERTAIN', str_3567)

@norecursion
def check_config_h(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_config_h'
    module_type_store = module_type_store.open_function_context('check_config_h', 242, 0, False)
    
    # Passed parameters checking function
    check_config_h.stypy_localization = localization
    check_config_h.stypy_type_of_self = None
    check_config_h.stypy_type_store = module_type_store
    check_config_h.stypy_function_name = 'check_config_h'
    check_config_h.stypy_param_names_list = []
    check_config_h.stypy_varargs_param_name = None
    check_config_h.stypy_kwargs_param_name = None
    check_config_h.stypy_call_defaults = defaults
    check_config_h.stypy_call_varargs = varargs
    check_config_h.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_config_h', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_config_h', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_config_h(...)' code ##################

    str_3568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'str', 'Check if the current Python installation (specifically, pyconfig.h)\n    appears amenable to building extensions with GCC.  Returns a tuple\n    (status, details), where \'status\' is one of the following constants:\n      CONFIG_H_OK\n        all is well, go ahead and compile\n      CONFIG_H_NOTOK\n        doesn\'t look good\n      CONFIG_H_UNCERTAIN\n        not sure -- unable to read pyconfig.h\n    \'details\' is a human-readable string explaining the situation.\n\n    Note there are two ways to conclude "OK": either \'sys.version\' contains\n    the string "GCC" (implying that this Python was built with GCC), or the\n    installed "pyconfig.h" contains the string "__GNUC__".\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 263, 4))
    
    # 'from distutils import sysconfig' statement (line 263)
    try:
        from distutils import sysconfig

    except:
        sysconfig = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 263, 4), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 264, 4))
    
    # 'import string' statement (line 264)
    import string

    import_module(stypy.reporting.localization.Localization(__file__, 264, 4), 'string', string, module_type_store)
    
    
    
    
    # Call to find(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'sys' (line 267)
    sys_3571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'sys', False)
    # Obtaining the member 'version' of a type (line 267)
    version_3572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 19), sys_3571, 'version')
    str_3573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 31), 'str', 'GCC')
    # Processing the call keyword arguments (line 267)
    kwargs_3574 = {}
    # Getting the type of 'string' (line 267)
    string_3569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 7), 'string', False)
    # Obtaining the member 'find' of a type (line 267)
    find_3570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 7), string_3569, 'find')
    # Calling find(args, kwargs) (line 267)
    find_call_result_3575 = invoke(stypy.reporting.localization.Localization(__file__, 267, 7), find_3570, *[version_3572, str_3573], **kwargs_3574)
    
    int_3576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 41), 'int')
    # Applying the binary operator '>=' (line 267)
    result_ge_3577 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 7), '>=', find_call_result_3575, int_3576)
    
    # Testing the type of an if condition (line 267)
    if_condition_3578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), result_ge_3577)
    # Assigning a type to the variable 'if_condition_3578' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'if_condition_3578', if_condition_3578)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 268)
    tuple_3579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 268)
    # Adding element type (line 268)
    # Getting the type of 'CONFIG_H_OK' (line 268)
    CONFIG_H_OK_3580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'CONFIG_H_OK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 16), tuple_3579, CONFIG_H_OK_3580)
    # Adding element type (line 268)
    str_3581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 29), 'str', "sys.version mentions 'GCC'")
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 16), tuple_3579, str_3581)
    
    # Assigning a type to the variable 'stypy_return_type' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', tuple_3579)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to get_config_h_filename(...): (line 270)
    # Processing the call keyword arguments (line 270)
    kwargs_3584 = {}
    # Getting the type of 'sysconfig' (line 270)
    sysconfig_3582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 9), 'sysconfig', False)
    # Obtaining the member 'get_config_h_filename' of a type (line 270)
    get_config_h_filename_3583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 9), sysconfig_3582, 'get_config_h_filename')
    # Calling get_config_h_filename(args, kwargs) (line 270)
    get_config_h_filename_call_result_3585 = invoke(stypy.reporting.localization.Localization(__file__, 270, 9), get_config_h_filename_3583, *[], **kwargs_3584)
    
    # Assigning a type to the variable 'fn' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'fn', get_config_h_filename_call_result_3585)
    
    
    # SSA begins for try-except statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 274):
    
    # Assigning a Call to a Name (line 274):
    
    # Call to open(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'fn' (line 274)
    fn_3587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 17), 'fn', False)
    # Processing the call keyword arguments (line 274)
    kwargs_3588 = {}
    # Getting the type of 'open' (line 274)
    open_3586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'open', False)
    # Calling open(args, kwargs) (line 274)
    open_call_result_3589 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), open_3586, *[fn_3587], **kwargs_3588)
    
    # Assigning a type to the variable 'f' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'f', open_call_result_3589)
    
    # Try-finally block (line 275)
    
    # Assigning a Call to a Name (line 276):
    
    # Assigning a Call to a Name (line 276):
    
    # Call to read(...): (line 276)
    # Processing the call keyword arguments (line 276)
    kwargs_3592 = {}
    # Getting the type of 'f' (line 276)
    f_3590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'f', False)
    # Obtaining the member 'read' of a type (line 276)
    read_3591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), f_3590, 'read')
    # Calling read(args, kwargs) (line 276)
    read_call_result_3593 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), read_3591, *[], **kwargs_3592)
    
    # Assigning a type to the variable 's' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 's', read_call_result_3593)
    
    # finally branch of the try-finally block (line 275)
    
    # Call to close(...): (line 278)
    # Processing the call keyword arguments (line 278)
    kwargs_3596 = {}
    # Getting the type of 'f' (line 278)
    f_3594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'f', False)
    # Obtaining the member 'close' of a type (line 278)
    close_3595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), f_3594, 'close')
    # Calling close(args, kwargs) (line 278)
    close_call_result_3597 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), close_3595, *[], **kwargs_3596)
    
    
    # SSA branch for the except part of a try statement (line 271)
    # SSA branch for the except 'IOError' branch of a try statement (line 271)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'IOError' (line 280)
    IOError_3598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'IOError')
    # Assigning a type to the variable 'exc' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'exc', IOError_3598)
    
    # Obtaining an instance of the builtin type 'tuple' (line 283)
    tuple_3599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 283)
    # Adding element type (line 283)
    # Getting the type of 'CONFIG_H_UNCERTAIN' (line 283)
    CONFIG_H_UNCERTAIN_3600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'CONFIG_H_UNCERTAIN')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 16), tuple_3599, CONFIG_H_UNCERTAIN_3600)
    # Adding element type (line 283)
    str_3601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'str', "couldn't read '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 284)
    tuple_3602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 284)
    # Adding element type (line 284)
    # Getting the type of 'fn' (line 284)
    fn_3603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 44), 'fn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 44), tuple_3602, fn_3603)
    # Adding element type (line 284)
    # Getting the type of 'exc' (line 284)
    exc_3604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 48), 'exc')
    # Obtaining the member 'strerror' of a type (line 284)
    strerror_3605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 48), exc_3604, 'strerror')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 44), tuple_3602, strerror_3605)
    
    # Applying the binary operator '%' (line 284)
    result_mod_3606 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 16), '%', str_3601, tuple_3602)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 16), tuple_3599, result_mod_3606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type', tuple_3599)
    # SSA branch for the else branch of a try statement (line 271)
    module_type_store.open_ssa_branch('except else')
    
    
    
    # Call to find(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 's' (line 288)
    s_3609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 's', False)
    str_3610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 25), 'str', '__GNUC__')
    # Processing the call keyword arguments (line 288)
    kwargs_3611 = {}
    # Getting the type of 'string' (line 288)
    string_3607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'string', False)
    # Obtaining the member 'find' of a type (line 288)
    find_3608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), string_3607, 'find')
    # Calling find(args, kwargs) (line 288)
    find_call_result_3612 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), find_3608, *[s_3609, str_3610], **kwargs_3611)
    
    int_3613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 40), 'int')
    # Applying the binary operator '>=' (line 288)
    result_ge_3614 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 11), '>=', find_call_result_3612, int_3613)
    
    # Testing the type of an if condition (line 288)
    if_condition_3615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), result_ge_3614)
    # Assigning a type to the variable 'if_condition_3615' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'if_condition_3615', if_condition_3615)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 289)
    tuple_3616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 289)
    # Adding element type (line 289)
    # Getting the type of 'CONFIG_H_OK' (line 289)
    CONFIG_H_OK_3617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'CONFIG_H_OK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 20), tuple_3616, CONFIG_H_OK_3617)
    # Adding element type (line 289)
    str_3618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 33), 'str', "'%s' mentions '__GNUC__'")
    # Getting the type of 'fn' (line 289)
    fn_3619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 62), 'fn')
    # Applying the binary operator '%' (line 289)
    result_mod_3620 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 33), '%', str_3618, fn_3619)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 20), tuple_3616, result_mod_3620)
    
    # Assigning a type to the variable 'stypy_return_type' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'stypy_return_type', tuple_3616)
    # SSA branch for the else part of an if statement (line 288)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 291)
    tuple_3621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 291)
    # Adding element type (line 291)
    # Getting the type of 'CONFIG_H_NOTOK' (line 291)
    CONFIG_H_NOTOK_3622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'CONFIG_H_NOTOK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 20), tuple_3621, CONFIG_H_NOTOK_3622)
    # Adding element type (line 291)
    str_3623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 36), 'str', "'%s' does not mention '__GNUC__'")
    # Getting the type of 'fn' (line 291)
    fn_3624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 73), 'fn')
    # Applying the binary operator '%' (line 291)
    result_mod_3625 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 36), '%', str_3623, fn_3624)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 20), tuple_3621, result_mod_3625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'stypy_return_type', tuple_3621)
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_config_h(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_config_h' in the type store
    # Getting the type of 'stypy_return_type' (line 242)
    stypy_return_type_3626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_config_h'
    return stypy_return_type_3626

# Assigning a type to the variable 'check_config_h' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'check_config_h', check_config_h)

@norecursion
def get_versions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_versions'
    module_type_store = module_type_store.open_function_context('get_versions', 294, 0, False)
    
    # Passed parameters checking function
    get_versions.stypy_localization = localization
    get_versions.stypy_type_of_self = None
    get_versions.stypy_type_store = module_type_store
    get_versions.stypy_function_name = 'get_versions'
    get_versions.stypy_param_names_list = []
    get_versions.stypy_varargs_param_name = None
    get_versions.stypy_kwargs_param_name = None
    get_versions.stypy_call_defaults = defaults
    get_versions.stypy_call_varargs = varargs
    get_versions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_versions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_versions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_versions(...)' code ##################

    str_3627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, (-1)), 'str', ' Try to find out the versions of gcc and ld.\n        If not possible it returns None for it.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 298, 4))
    
    # 'from distutils.version import StrictVersion' statement (line 298)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_3628 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 298, 4), 'distutils.version')

    if (type(import_3628) is not StypyTypeError):

        if (import_3628 != 'pyd_module'):
            __import__(import_3628)
            sys_modules_3629 = sys.modules[import_3628]
            import_from_module(stypy.reporting.localization.Localization(__file__, 298, 4), 'distutils.version', sys_modules_3629.module_type_store, module_type_store, ['StrictVersion'])
            nest_module(stypy.reporting.localization.Localization(__file__, 298, 4), __file__, sys_modules_3629, sys_modules_3629.module_type_store, module_type_store)
        else:
            from distutils.version import StrictVersion

            import_from_module(stypy.reporting.localization.Localization(__file__, 298, 4), 'distutils.version', None, module_type_store, ['StrictVersion'], [StrictVersion])

    else:
        # Assigning a type to the variable 'distutils.version' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'distutils.version', import_3628)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 299, 4))
    
    # 'from distutils.spawn import find_executable' statement (line 299)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_3630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 299, 4), 'distutils.spawn')

    if (type(import_3630) is not StypyTypeError):

        if (import_3630 != 'pyd_module'):
            __import__(import_3630)
            sys_modules_3631 = sys.modules[import_3630]
            import_from_module(stypy.reporting.localization.Localization(__file__, 299, 4), 'distutils.spawn', sys_modules_3631.module_type_store, module_type_store, ['find_executable'])
            nest_module(stypy.reporting.localization.Localization(__file__, 299, 4), __file__, sys_modules_3631, sys_modules_3631.module_type_store, module_type_store)
        else:
            from distutils.spawn import find_executable

            import_from_module(stypy.reporting.localization.Localization(__file__, 299, 4), 'distutils.spawn', None, module_type_store, ['find_executable'], [find_executable])

    else:
        # Assigning a type to the variable 'distutils.spawn' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'distutils.spawn', import_3630)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 300, 4))
    
    # 'import re' statement (line 300)
    import re

    import_module(stypy.reporting.localization.Localization(__file__, 300, 4), 're', re, module_type_store)
    
    
    # Assigning a Call to a Name (line 302):
    
    # Assigning a Call to a Name (line 302):
    
    # Call to find_executable(...): (line 302)
    # Processing the call arguments (line 302)
    str_3633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'str', 'gcc')
    # Processing the call keyword arguments (line 302)
    kwargs_3634 = {}
    # Getting the type of 'find_executable' (line 302)
    find_executable_3632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 302)
    find_executable_call_result_3635 = invoke(stypy.reporting.localization.Localization(__file__, 302, 14), find_executable_3632, *[str_3633], **kwargs_3634)
    
    # Assigning a type to the variable 'gcc_exe' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'gcc_exe', find_executable_call_result_3635)
    
    # Getting the type of 'gcc_exe' (line 303)
    gcc_exe_3636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 7), 'gcc_exe')
    # Testing the type of an if condition (line 303)
    if_condition_3637 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 4), gcc_exe_3636)
    # Assigning a type to the variable 'if_condition_3637' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'if_condition_3637', if_condition_3637)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to popen(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'gcc_exe' (line 304)
    gcc_exe_3640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'gcc_exe', False)
    str_3641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 33), 'str', ' -dumpversion')
    # Applying the binary operator '+' (line 304)
    result_add_3642 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 23), '+', gcc_exe_3640, str_3641)
    
    str_3643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 49), 'str', 'r')
    # Processing the call keyword arguments (line 304)
    kwargs_3644 = {}
    # Getting the type of 'os' (line 304)
    os_3638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 14), 'os', False)
    # Obtaining the member 'popen' of a type (line 304)
    popen_3639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 14), os_3638, 'popen')
    # Calling popen(args, kwargs) (line 304)
    popen_call_result_3645 = invoke(stypy.reporting.localization.Localization(__file__, 304, 14), popen_3639, *[result_add_3642, str_3643], **kwargs_3644)
    
    # Assigning a type to the variable 'out' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'out', popen_call_result_3645)
    
    # Try-finally block (line 305)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to read(...): (line 306)
    # Processing the call keyword arguments (line 306)
    kwargs_3648 = {}
    # Getting the type of 'out' (line 306)
    out_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'out', False)
    # Obtaining the member 'read' of a type (line 306)
    read_3647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 25), out_3646, 'read')
    # Calling read(args, kwargs) (line 306)
    read_call_result_3649 = invoke(stypy.reporting.localization.Localization(__file__, 306, 25), read_3647, *[], **kwargs_3648)
    
    # Assigning a type to the variable 'out_string' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'out_string', read_call_result_3649)
    
    # finally branch of the try-finally block (line 305)
    
    # Call to close(...): (line 308)
    # Processing the call keyword arguments (line 308)
    kwargs_3652 = {}
    # Getting the type of 'out' (line 308)
    out_3650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'out', False)
    # Obtaining the member 'close' of a type (line 308)
    close_3651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), out_3650, 'close')
    # Calling close(args, kwargs) (line 308)
    close_call_result_3653 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), close_3651, *[], **kwargs_3652)
    
    
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to search(...): (line 309)
    # Processing the call arguments (line 309)
    str_3656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'str', '(\\d+\\.\\d+\\.\\d+)')
    # Getting the type of 'out_string' (line 309)
    out_string_3657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 45), 'out_string', False)
    # Processing the call keyword arguments (line 309)
    kwargs_3658 = {}
    # Getting the type of 're' (line 309)
    re_3654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 17), 're', False)
    # Obtaining the member 'search' of a type (line 309)
    search_3655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 17), re_3654, 'search')
    # Calling search(args, kwargs) (line 309)
    search_call_result_3659 = invoke(stypy.reporting.localization.Localization(__file__, 309, 17), search_3655, *[str_3656, out_string_3657], **kwargs_3658)
    
    # Assigning a type to the variable 'result' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'result', search_call_result_3659)
    
    # Getting the type of 'result' (line 310)
    result_3660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'result')
    # Testing the type of an if condition (line 310)
    if_condition_3661 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 8), result_3660)
    # Assigning a type to the variable 'if_condition_3661' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'if_condition_3661', if_condition_3661)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 311):
    
    # Assigning a Call to a Name (line 311):
    
    # Call to StrictVersion(...): (line 311)
    # Processing the call arguments (line 311)
    
    # Call to group(...): (line 311)
    # Processing the call arguments (line 311)
    int_3665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 53), 'int')
    # Processing the call keyword arguments (line 311)
    kwargs_3666 = {}
    # Getting the type of 'result' (line 311)
    result_3663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 40), 'result', False)
    # Obtaining the member 'group' of a type (line 311)
    group_3664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 40), result_3663, 'group')
    # Calling group(args, kwargs) (line 311)
    group_call_result_3667 = invoke(stypy.reporting.localization.Localization(__file__, 311, 40), group_3664, *[int_3665], **kwargs_3666)
    
    # Processing the call keyword arguments (line 311)
    kwargs_3668 = {}
    # Getting the type of 'StrictVersion' (line 311)
    StrictVersion_3662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 26), 'StrictVersion', False)
    # Calling StrictVersion(args, kwargs) (line 311)
    StrictVersion_call_result_3669 = invoke(stypy.reporting.localization.Localization(__file__, 311, 26), StrictVersion_3662, *[group_call_result_3667], **kwargs_3668)
    
    # Assigning a type to the variable 'gcc_version' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'gcc_version', StrictVersion_call_result_3669)
    # SSA branch for the else part of an if statement (line 310)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 313):
    
    # Assigning a Name to a Name (line 313):
    # Getting the type of 'None' (line 313)
    None_3670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'None')
    # Assigning a type to the variable 'gcc_version' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'gcc_version', None_3670)
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 303)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 315):
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'None' (line 315)
    None_3671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'None')
    # Assigning a type to the variable 'gcc_version' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'gcc_version', None_3671)
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 318):
    
    # Assigning a Name to a Name (line 318):
    # Getting the type of 'None' (line 318)
    None_3672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 17), 'None')
    # Assigning a type to the variable 'ld_version' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'ld_version', None_3672)
    
    # Obtaining an instance of the builtin type 'tuple' (line 319)
    tuple_3673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 319)
    # Adding element type (line 319)
    # Getting the type of 'gcc_version' (line 319)
    gcc_version_3674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'gcc_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 12), tuple_3673, gcc_version_3674)
    # Adding element type (line 319)
    # Getting the type of 'ld_version' (line 319)
    ld_version_3675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 25), 'ld_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 12), tuple_3673, ld_version_3675)
    
    # Assigning a type to the variable 'stypy_return_type' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type', tuple_3673)
    
    # ################# End of 'get_versions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_versions' in the type store
    # Getting the type of 'stypy_return_type' (line 294)
    stypy_return_type_3676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3676)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_versions'
    return stypy_return_type_3676

# Assigning a type to the variable 'get_versions' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'get_versions', get_versions)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

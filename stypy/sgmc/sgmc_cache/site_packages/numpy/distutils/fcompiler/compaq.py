
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: #http://www.compaq.com/fortran/docs/
3: from __future__ import division, absolute_import, print_function
4: 
5: import os
6: import sys
7: 
8: from numpy.distutils.fcompiler import FCompiler
9: from numpy.distutils.compat import get_exception
10: from distutils.errors import DistutilsPlatformError
11: 
12: compilers = ['CompaqFCompiler']
13: if os.name != 'posix' or sys.platform[:6] == 'cygwin' :
14:     # Otherwise we'd get a false positive on posix systems with
15:     # case-insensitive filesystems (like darwin), because we'll pick
16:     # up /bin/df
17:     compilers.append('CompaqVisualFCompiler')
18: 
19: class CompaqFCompiler(FCompiler):
20: 
21:     compiler_type = 'compaq'
22:     description = 'Compaq Fortran Compiler'
23:     version_pattern = r'Compaq Fortran (?P<version>[^\s]*).*'
24: 
25:     if sys.platform[:5]=='linux':
26:         fc_exe = 'fort'
27:     else:
28:         fc_exe = 'f90'
29: 
30:     executables = {
31:         'version_cmd'  : ['<F90>', "-version"],
32:         'compiler_f77' : [fc_exe, "-f77rtl", "-fixed"],
33:         'compiler_fix' : [fc_exe, "-fixed"],
34:         'compiler_f90' : [fc_exe],
35:         'linker_so'    : ['<F90>'],
36:         'archiver'     : ["ar", "-cr"],
37:         'ranlib'       : ["ranlib"]
38:         }
39: 
40:     module_dir_switch = '-module ' # not tested
41:     module_include_switch = '-I'
42: 
43:     def get_flags(self):
44:         return ['-assume no2underscore', '-nomixed_str_len_arg']
45:     def get_flags_debug(self):
46:         return ['-g', '-check bounds']
47:     def get_flags_opt(self):
48:         return ['-O4', '-align dcommons', '-assume bigarrays',
49:                 '-assume nozsize', '-math_library fast']
50:     def get_flags_arch(self):
51:         return ['-arch host', '-tune host']
52:     def get_flags_linker_so(self):
53:         if sys.platform[:5]=='linux':
54:             return ['-shared']
55:         return ['-shared', '-Wl,-expect_unresolved,*']
56: 
57: class CompaqVisualFCompiler(FCompiler):
58: 
59:     compiler_type = 'compaqv'
60:     description = 'DIGITAL or Compaq Visual Fortran Compiler'
61:     version_pattern = r'(DIGITAL|Compaq) Visual Fortran Optimizing Compiler'\
62:                       ' Version (?P<version>[^\s]*).*'
63: 
64:     compile_switch = '/compile_only'
65:     object_switch = '/object:'
66:     library_switch = '/OUT:'      #No space after /OUT:!
67: 
68:     static_lib_extension = ".lib"
69:     static_lib_format = "%s%s"
70:     module_dir_switch = '/module:'
71:     module_include_switch = '/I'
72: 
73:     ar_exe = 'lib.exe'
74:     fc_exe = 'DF'
75: 
76:     if sys.platform=='win32':
77:         from numpy.distutils.msvccompiler import MSVCCompiler
78: 
79:         try:
80:             m = MSVCCompiler()
81:             m.initialize()
82:             ar_exe = m.lib
83:         except DistutilsPlatformError:
84:             pass
85:         except AttributeError:
86:             msg = get_exception()
87:             if '_MSVCCompiler__root' in str(msg):
88:                 print('Ignoring "%s" (I think it is msvccompiler.py bug)' % (msg))
89:             else:
90:                 raise
91:         except IOError:
92:             e = get_exception()
93:             if not "vcvarsall.bat" in str(e):
94:                 print("Unexpected IOError in", __file__)
95:                 raise e
96:         except ValueError:
97:             e = get_exception()
98:             if not "path']" in str(e):
99:                 print("Unexpected ValueError in", __file__)
100:                 raise e
101: 
102:     executables = {
103:         'version_cmd'  : ['<F90>', "/what"],
104:         'compiler_f77' : [fc_exe, "/f77rtl", "/fixed"],
105:         'compiler_fix' : [fc_exe, "/fixed"],
106:         'compiler_f90' : [fc_exe],
107:         'linker_so'    : ['<F90>'],
108:         'archiver'     : [ar_exe, "/OUT:"],
109:         'ranlib'       : None
110:         }
111: 
112:     def get_flags(self):
113:         return ['/nologo', '/MD', '/WX', '/iface=(cref,nomixed_str_len_arg)',
114:                 '/names:lowercase', '/assume:underscore']
115:     def get_flags_opt(self):
116:         return ['/Ox', '/fast', '/optimize:5', '/unroll:0', '/math_library:fast']
117:     def get_flags_arch(self):
118:         return ['/threads']
119:     def get_flags_debug(self):
120:         return ['/debug']
121: 
122: if __name__ == '__main__':
123:     from distutils import log
124:     log.set_verbosity(2)
125:     from numpy.distutils.fcompiler import new_fcompiler
126:     compiler = new_fcompiler(compiler='compaq')
127:     compiler.customize()
128:     print(compiler.get_version())
129: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60267 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.fcompiler')

if (type(import_60267) is not StypyTypeError):

    if (import_60267 != 'pyd_module'):
        __import__(import_60267)
        sys_modules_60268 = sys.modules[import_60267]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.fcompiler', sys_modules_60268.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_60268, sys_modules_60268.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.fcompiler', import_60267)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60269 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.compat')

if (type(import_60269) is not StypyTypeError):

    if (import_60269 != 'pyd_module'):
        __import__(import_60269)
        sys_modules_60270 = sys.modules[import_60269]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.compat', sys_modules_60270.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_60270, sys_modules_60270.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.compat', import_60269)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.errors import DistutilsPlatformError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60271 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_60271) is not StypyTypeError):

    if (import_60271 != 'pyd_module'):
        __import__(import_60271)
        sys_modules_60272 = sys.modules[import_60271]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_60272.module_type_store, module_type_store, ['DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_60272, sys_modules_60272.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError'], [DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_60271)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_60273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_60274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'str', 'CompaqFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 12), list_60273, str_60274)

# Assigning a type to the variable 'compilers' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'compilers', list_60273)


# Evaluating a boolean operation

# Getting the type of 'os' (line 13)
os_60275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 3), 'os')
# Obtaining the member 'name' of a type (line 13)
name_60276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), os_60275, 'name')
str_60277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'str', 'posix')
# Applying the binary operator '!=' (line 13)
result_ne_60278 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 3), '!=', name_60276, str_60277)



# Obtaining the type of the subscript
int_60279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 39), 'int')
slice_60280 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 25), None, int_60279, None)
# Getting the type of 'sys' (line 13)
sys_60281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'sys')
# Obtaining the member 'platform' of a type (line 13)
platform_60282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 25), sys_60281, 'platform')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___60283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 25), platform_60282, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_60284 = invoke(stypy.reporting.localization.Localization(__file__, 13, 25), getitem___60283, slice_60280)

str_60285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 45), 'str', 'cygwin')
# Applying the binary operator '==' (line 13)
result_eq_60286 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 25), '==', subscript_call_result_60284, str_60285)

# Applying the binary operator 'or' (line 13)
result_or_keyword_60287 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 3), 'or', result_ne_60278, result_eq_60286)

# Testing the type of an if condition (line 13)
if_condition_60288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 0), result_or_keyword_60287)
# Assigning a type to the variable 'if_condition_60288' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'if_condition_60288', if_condition_60288)
# SSA begins for if statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 17)
# Processing the call arguments (line 17)
str_60291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'str', 'CompaqVisualFCompiler')
# Processing the call keyword arguments (line 17)
kwargs_60292 = {}
# Getting the type of 'compilers' (line 17)
compilers_60289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'compilers', False)
# Obtaining the member 'append' of a type (line 17)
append_60290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), compilers_60289, 'append')
# Calling append(args, kwargs) (line 17)
append_call_result_60293 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), append_60290, *[str_60291], **kwargs_60292)

# SSA join for if statement (line 13)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'CompaqFCompiler' class
# Getting the type of 'FCompiler' (line 19)
FCompiler_60294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'FCompiler')

class CompaqFCompiler(FCompiler_60294, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'CompaqFCompiler.get_flags')
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_60295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        str_60296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'str', '-assume no2underscore')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), list_60295, str_60296)
        # Adding element type (line 44)
        str_60297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'str', '-nomixed_str_len_arg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), list_60295, str_60297)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', list_60295)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_60298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60298)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_60298


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'CompaqFCompiler.get_flags_debug')
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_60299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        str_60300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), list_60299, str_60300)
        # Adding element type (line 46)
        str_60301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'str', '-check bounds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), list_60299, str_60301)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', list_60299)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_60302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_60302


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'CompaqFCompiler.get_flags_opt')
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_60303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        str_60304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'str', '-O4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), list_60303, str_60304)
        # Adding element type (line 48)
        str_60305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'str', '-align dcommons')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), list_60303, str_60305)
        # Adding element type (line 48)
        str_60306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 42), 'str', '-assume bigarrays')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), list_60303, str_60306)
        # Adding element type (line 48)
        str_60307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'str', '-assume nozsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), list_60303, str_60307)
        # Adding element type (line 48)
        str_60308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'str', '-math_library fast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), list_60303, str_60308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', list_60303)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_60309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60309)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_60309


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'CompaqFCompiler.get_flags_arch')
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_60310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_60311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'str', '-arch host')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), list_60310, str_60311)
        # Adding element type (line 51)
        str_60312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'str', '-tune host')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), list_60310, str_60312)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', list_60310)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_60313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60313)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_60313


    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'CompaqFCompiler.get_flags_linker_so')
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqFCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Obtaining the type of the subscript
        int_60314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'int')
        slice_60315 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 11), None, int_60314, None)
        # Getting the type of 'sys' (line 53)
        sys_60316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 53)
        platform_60317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), sys_60316, 'platform')
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___60318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), platform_60317, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_60319 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), getitem___60318, slice_60315)
        
        str_60320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', 'linux')
        # Applying the binary operator '==' (line 53)
        result_eq_60321 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 11), '==', subscript_call_result_60319, str_60320)
        
        # Testing the type of an if condition (line 53)
        if_condition_60322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), result_eq_60321)
        # Assigning a type to the variable 'if_condition_60322' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_60322', if_condition_60322)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_60323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        str_60324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'str', '-shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 19), list_60323, str_60324)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'stypy_return_type', list_60323)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_60325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        str_60326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'str', '-shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 15), list_60325, str_60326)
        # Adding element type (line 55)
        str_60327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 27), 'str', '-Wl,-expect_unresolved,*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 15), list_60325, str_60327)
        
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', list_60325)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_60328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_60328


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CompaqFCompiler' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'CompaqFCompiler', CompaqFCompiler)

# Assigning a Str to a Name (line 21):
str_60329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'str', 'compaq')
# Getting the type of 'CompaqFCompiler'
CompaqFCompiler_60330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqFCompiler_60330, 'compiler_type', str_60329)

# Assigning a Str to a Name (line 22):
str_60331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'str', 'Compaq Fortran Compiler')
# Getting the type of 'CompaqFCompiler'
CompaqFCompiler_60332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqFCompiler_60332, 'description', str_60331)

# Assigning a Str to a Name (line 23):
str_60333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'str', 'Compaq Fortran (?P<version>[^\\s]*).*')
# Getting the type of 'CompaqFCompiler'
CompaqFCompiler_60334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqFCompiler_60334, 'version_pattern', str_60333)



# Obtaining the type of the subscript
int_60335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
slice_60336 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 7), None, int_60335, None)
# Getting the type of 'sys' (line 25)
sys_60337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'sys')
# Obtaining the member 'platform' of a type (line 25)
platform_60338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), sys_60337, 'platform')
# Obtaining the member '__getitem__' of a type (line 25)
getitem___60339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 7), platform_60338, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 25)
subscript_call_result_60340 = invoke(stypy.reporting.localization.Localization(__file__, 25, 7), getitem___60339, slice_60336)

str_60341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'str', 'linux')
# Applying the binary operator '==' (line 25)
result_eq_60342 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), '==', subscript_call_result_60340, str_60341)

# Testing the type of an if condition (line 25)
if_condition_60343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_eq_60342)
# Assigning a type to the variable 'if_condition_60343' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_60343', if_condition_60343)
# SSA begins for if statement (line 25)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 26):
str_60344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'str', 'fort')
# Assigning a type to the variable 'fc_exe' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'fc_exe', str_60344)
# SSA branch for the else part of an if statement (line 25)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 28):
str_60345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'str', 'f90')
# Assigning a type to the variable 'fc_exe' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'fc_exe', str_60345)
# SSA join for if statement (line 25)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 30):

# Obtaining an instance of the builtin type 'dict' (line 30)
dict_60346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 30)
# Adding element type (key, value) (line 30)
str_60347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 31)
list_60348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
str_60349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 25), list_60348, str_60349)
# Adding element type (line 31)
str_60350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'str', '-version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 25), list_60348, str_60350)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60347, list_60348))
# Adding element type (key, value) (line 30)
str_60351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 32)
list_60352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
# Getting the type of 'fc_exe' (line 32)
fc_exe_60353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'fc_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_60352, fc_exe_60353)
# Adding element type (line 32)
str_60354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 34), 'str', '-f77rtl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_60352, str_60354)
# Adding element type (line 32)
str_60355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 45), 'str', '-fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_60352, str_60355)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60351, list_60352))
# Adding element type (key, value) (line 30)
str_60356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 33)
list_60357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
# Getting the type of 'fc_exe' (line 33)
fc_exe_60358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'fc_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 25), list_60357, fc_exe_60358)
# Adding element type (line 33)
str_60359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', '-fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 25), list_60357, str_60359)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60356, list_60357))
# Adding element type (key, value) (line 30)
str_60360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 34)
list_60361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
# Getting the type of 'fc_exe' (line 34)
fc_exe_60362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'fc_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 25), list_60361, fc_exe_60362)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60360, list_60361))
# Adding element type (key, value) (line 30)
str_60363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 35)
list_60364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)
# Adding element type (line 35)
str_60365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_60364, str_60365)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60363, list_60364))
# Adding element type (key, value) (line 30)
str_60366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 36)
list_60367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
str_60368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_60367, str_60368)
# Adding element type (line 36)
str_60369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_60367, str_60369)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60366, list_60367))
# Adding element type (key, value) (line 30)
str_60370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 37)
list_60371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
str_60372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 25), list_60371, str_60372)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 18), dict_60346, (str_60370, list_60371))

# Getting the type of 'CompaqFCompiler'
CompaqFCompiler_60373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqFCompiler_60373, 'executables', dict_60346)

# Assigning a Str to a Name (line 40):
str_60374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'str', '-module ')
# Getting the type of 'CompaqFCompiler'
CompaqFCompiler_60375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqFCompiler_60375, 'module_dir_switch', str_60374)

# Assigning a Str to a Name (line 41):
str_60376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'str', '-I')
# Getting the type of 'CompaqFCompiler'
CompaqFCompiler_60377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqFCompiler_60377, 'module_include_switch', str_60376)
# Declaration of the 'CompaqVisualFCompiler' class
# Getting the type of 'FCompiler' (line 57)
FCompiler_60378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'FCompiler')

class CompaqVisualFCompiler(FCompiler_60378, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'CompaqVisualFCompiler.get_flags')
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqVisualFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqVisualFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_60379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_60380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), list_60379, str_60380)
        # Adding element type (line 113)
        str_60381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), list_60379, str_60381)
        # Adding element type (line 113)
        str_60382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 34), 'str', '/WX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), list_60379, str_60382)
        # Adding element type (line 113)
        str_60383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 41), 'str', '/iface=(cref,nomixed_str_len_arg)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), list_60379, str_60383)
        # Adding element type (line 113)
        str_60384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 16), 'str', '/names:lowercase')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), list_60379, str_60384)
        # Adding element type (line 113)
        str_60385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'str', '/assume:underscore')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), list_60379, str_60385)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', list_60379)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_60386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_60386


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'CompaqVisualFCompiler.get_flags_opt')
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqVisualFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqVisualFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_60387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        str_60388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 16), 'str', '/Ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), list_60387, str_60388)
        # Adding element type (line 116)
        str_60389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'str', '/fast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), list_60387, str_60389)
        # Adding element type (line 116)
        str_60390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'str', '/optimize:5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), list_60387, str_60390)
        # Adding element type (line 116)
        str_60391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'str', '/unroll:0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), list_60387, str_60391)
        # Adding element type (line 116)
        str_60392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 60), 'str', '/math_library:fast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), list_60387, str_60392)
        
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', list_60387)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_60393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_60393


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'CompaqVisualFCompiler.get_flags_arch')
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqVisualFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqVisualFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_60394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        str_60395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'str', '/threads')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 15), list_60394, str_60395)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', list_60394)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_60396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60396)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_60396


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'CompaqVisualFCompiler.get_flags_debug')
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CompaqVisualFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqVisualFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_60397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        str_60398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'str', '/debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), list_60397, str_60398)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', list_60397)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_60399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60399)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_60399


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 57, 0, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompaqVisualFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CompaqVisualFCompiler' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'CompaqVisualFCompiler', CompaqVisualFCompiler)

# Assigning a Str to a Name (line 59):
str_60400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'str', 'compaqv')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60401, 'compiler_type', str_60400)

# Assigning a Str to a Name (line 60):
str_60402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 18), 'str', 'DIGITAL or Compaq Visual Fortran Compiler')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60403, 'description', str_60402)

# Assigning a Str to a Name (line 61):
str_60404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'str', '(DIGITAL|Compaq) Visual Fortran Optimizing Compiler Version (?P<version>[^\\s]*).*')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60405, 'version_pattern', str_60404)

# Assigning a Str to a Name (line 64):
str_60406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'str', '/compile_only')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'compile_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60407, 'compile_switch', str_60406)

# Assigning a Str to a Name (line 65):
str_60408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'str', '/object:')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'object_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60409, 'object_switch', str_60408)

# Assigning a Str to a Name (line 66):
str_60410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'str', '/OUT:')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'library_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60411, 'library_switch', str_60410)

# Assigning a Str to a Name (line 68):
str_60412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'str', '.lib')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60413, 'static_lib_extension', str_60412)

# Assigning a Str to a Name (line 69):
str_60414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'str', '%s%s')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60415, 'static_lib_format', str_60414)

# Assigning a Str to a Name (line 70):
str_60416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'str', '/module:')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60417, 'module_dir_switch', str_60416)

# Assigning a Str to a Name (line 71):
str_60418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'str', '/I')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60419, 'module_include_switch', str_60418)

# Assigning a Str to a Name (line 73):
str_60420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'str', 'lib.exe')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'ar_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60421, 'ar_exe', str_60420)

# Assigning a Str to a Name (line 74):
str_60422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'str', 'DF')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'fc_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60423, 'fc_exe', str_60422)


# Getting the type of 'sys' (line 76)
sys_60424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'sys')
# Obtaining the member 'platform' of a type (line 76)
platform_60425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), sys_60424, 'platform')
str_60426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'str', 'win32')
# Applying the binary operator '==' (line 76)
result_eq_60427 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '==', platform_60425, str_60426)

# Testing the type of an if condition (line 76)
if_condition_60428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_eq_60427)
# Assigning a type to the variable 'if_condition_60428' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_60428', if_condition_60428)
# SSA begins for if statement (line 76)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 77, 8))

# 'from numpy.distutils.msvccompiler import MSVCCompiler' statement (line 77)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60429 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 77, 8), 'numpy.distutils.msvccompiler')

if (type(import_60429) is not StypyTypeError):

    if (import_60429 != 'pyd_module'):
        __import__(import_60429)
        sys_modules_60430 = sys.modules[import_60429]
        import_from_module(stypy.reporting.localization.Localization(__file__, 77, 8), 'numpy.distutils.msvccompiler', sys_modules_60430.module_type_store, module_type_store, ['MSVCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 77, 8), __file__, sys_modules_60430, sys_modules_60430.module_type_store, module_type_store)
    else:
        from numpy.distutils.msvccompiler import MSVCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 77, 8), 'numpy.distutils.msvccompiler', None, module_type_store, ['MSVCCompiler'], [MSVCCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.msvccompiler' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'numpy.distutils.msvccompiler', import_60429)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')



# SSA begins for try-except statement (line 79)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Call to a Name (line 80):

# Call to MSVCCompiler(...): (line 80)
# Processing the call keyword arguments (line 80)
kwargs_60432 = {}
# Getting the type of 'MSVCCompiler' (line 80)
MSVCCompiler_60431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'MSVCCompiler', False)
# Calling MSVCCompiler(args, kwargs) (line 80)
MSVCCompiler_call_result_60433 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), MSVCCompiler_60431, *[], **kwargs_60432)

# Assigning a type to the variable 'm' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'm', MSVCCompiler_call_result_60433)

# Call to initialize(...): (line 81)
# Processing the call keyword arguments (line 81)
kwargs_60436 = {}
# Getting the type of 'm' (line 81)
m_60434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'm', False)
# Obtaining the member 'initialize' of a type (line 81)
initialize_60435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), m_60434, 'initialize')
# Calling initialize(args, kwargs) (line 81)
initialize_call_result_60437 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), initialize_60435, *[], **kwargs_60436)


# Assigning a Attribute to a Name (line 82):
# Getting the type of 'm' (line 82)
m_60438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'm')
# Obtaining the member 'lib' of a type (line 82)
lib_60439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), m_60438, 'lib')
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'ar_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60440, 'ar_exe', lib_60439)
# SSA branch for the except part of a try statement (line 79)
# SSA branch for the except 'DistutilsPlatformError' branch of a try statement (line 79)
module_type_store.open_ssa_branch('except')
pass
# SSA branch for the except 'AttributeError' branch of a try statement (line 79)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 86):

# Call to get_exception(...): (line 86)
# Processing the call keyword arguments (line 86)
kwargs_60442 = {}
# Getting the type of 'get_exception' (line 86)
get_exception_60441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'get_exception', False)
# Calling get_exception(args, kwargs) (line 86)
get_exception_call_result_60443 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), get_exception_60441, *[], **kwargs_60442)

# Assigning a type to the variable 'msg' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'msg', get_exception_call_result_60443)


str_60444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'str', '_MSVCCompiler__root')

# Call to str(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of 'msg' (line 87)
msg_60446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 44), 'msg', False)
# Processing the call keyword arguments (line 87)
kwargs_60447 = {}
# Getting the type of 'str' (line 87)
str_60445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 40), 'str', False)
# Calling str(args, kwargs) (line 87)
str_call_result_60448 = invoke(stypy.reporting.localization.Localization(__file__, 87, 40), str_60445, *[msg_60446], **kwargs_60447)

# Applying the binary operator 'in' (line 87)
result_contains_60449 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), 'in', str_60444, str_call_result_60448)

# Testing the type of an if condition (line 87)
if_condition_60450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 12), result_contains_60449)
# Assigning a type to the variable 'if_condition_60450' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'if_condition_60450', if_condition_60450)
# SSA begins for if statement (line 87)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 88)
# Processing the call arguments (line 88)
str_60452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'str', 'Ignoring "%s" (I think it is msvccompiler.py bug)')
# Getting the type of 'msg' (line 88)
msg_60453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 77), 'msg', False)
# Applying the binary operator '%' (line 88)
result_mod_60454 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 22), '%', str_60452, msg_60453)

# Processing the call keyword arguments (line 88)
kwargs_60455 = {}
# Getting the type of 'print' (line 88)
print_60451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'print', False)
# Calling print(args, kwargs) (line 88)
print_call_result_60456 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), print_60451, *[result_mod_60454], **kwargs_60455)

# SSA branch for the else part of an if statement (line 87)
module_type_store.open_ssa_branch('else')
# SSA join for if statement (line 87)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the except 'IOError' branch of a try statement (line 79)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 92):

# Call to get_exception(...): (line 92)
# Processing the call keyword arguments (line 92)
kwargs_60458 = {}
# Getting the type of 'get_exception' (line 92)
get_exception_60457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'get_exception', False)
# Calling get_exception(args, kwargs) (line 92)
get_exception_call_result_60459 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), get_exception_60457, *[], **kwargs_60458)

# Assigning a type to the variable 'e' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'e', get_exception_call_result_60459)



str_60460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'str', 'vcvarsall.bat')

# Call to str(...): (line 93)
# Processing the call arguments (line 93)
# Getting the type of 'e' (line 93)
e_60462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'e', False)
# Processing the call keyword arguments (line 93)
kwargs_60463 = {}
# Getting the type of 'str' (line 93)
str_60461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'str', False)
# Calling str(args, kwargs) (line 93)
str_call_result_60464 = invoke(stypy.reporting.localization.Localization(__file__, 93, 38), str_60461, *[e_60462], **kwargs_60463)

# Applying the binary operator 'in' (line 93)
result_contains_60465 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 19), 'in', str_60460, str_call_result_60464)

# Applying the 'not' unary operator (line 93)
result_not__60466 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 15), 'not', result_contains_60465)

# Testing the type of an if condition (line 93)
if_condition_60467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 12), result_not__60466)
# Assigning a type to the variable 'if_condition_60467' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'if_condition_60467', if_condition_60467)
# SSA begins for if statement (line 93)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 94)
# Processing the call arguments (line 94)
str_60469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'str', 'Unexpected IOError in')
# Getting the type of '__file__' (line 94)
file___60470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 47), '__file__', False)
# Processing the call keyword arguments (line 94)
kwargs_60471 = {}
# Getting the type of 'print' (line 94)
print_60468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'print', False)
# Calling print(args, kwargs) (line 94)
print_call_result_60472 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), print_60468, *[str_60469, file___60470], **kwargs_60471)

# Getting the type of 'e' (line 95)
e_60473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'e')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 95, 16), e_60473, 'raise parameter', BaseException)
# SSA join for if statement (line 93)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the except 'ValueError' branch of a try statement (line 79)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 97):

# Call to get_exception(...): (line 97)
# Processing the call keyword arguments (line 97)
kwargs_60475 = {}
# Getting the type of 'get_exception' (line 97)
get_exception_60474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'get_exception', False)
# Calling get_exception(args, kwargs) (line 97)
get_exception_call_result_60476 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), get_exception_60474, *[], **kwargs_60475)

# Assigning a type to the variable 'e' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'e', get_exception_call_result_60476)



str_60477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'str', "path']")

# Call to str(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'e' (line 98)
e_60479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'e', False)
# Processing the call keyword arguments (line 98)
kwargs_60480 = {}
# Getting the type of 'str' (line 98)
str_60478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'str', False)
# Calling str(args, kwargs) (line 98)
str_call_result_60481 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), str_60478, *[e_60479], **kwargs_60480)

# Applying the binary operator 'in' (line 98)
result_contains_60482 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 19), 'in', str_60477, str_call_result_60481)

# Applying the 'not' unary operator (line 98)
result_not__60483 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 15), 'not', result_contains_60482)

# Testing the type of an if condition (line 98)
if_condition_60484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 12), result_not__60483)
# Assigning a type to the variable 'if_condition_60484' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'if_condition_60484', if_condition_60484)
# SSA begins for if statement (line 98)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 99)
# Processing the call arguments (line 99)
str_60486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 22), 'str', 'Unexpected ValueError in')
# Getting the type of '__file__' (line 99)
file___60487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 50), '__file__', False)
# Processing the call keyword arguments (line 99)
kwargs_60488 = {}
# Getting the type of 'print' (line 99)
print_60485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'print', False)
# Calling print(args, kwargs) (line 99)
print_call_result_60489 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), print_60485, *[str_60486, file___60487], **kwargs_60488)

# Getting the type of 'e' (line 100)
e_60490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'e')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 16), e_60490, 'raise parameter', BaseException)
# SSA join for if statement (line 98)
module_type_store = module_type_store.join_ssa_context()

# SSA join for try-except statement (line 79)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 76)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 102):

# Obtaining an instance of the builtin type 'dict' (line 102)
dict_60491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 102)
# Adding element type (key, value) (line 102)
str_60492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 103)
list_60493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 103)
# Adding element type (line 103)
str_60494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 25), list_60493, str_60494)
# Adding element type (line 103)
str_60495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 35), 'str', '/what')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 25), list_60493, str_60495)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60492, list_60493))
# Adding element type (key, value) (line 102)
str_60496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 104)
list_60497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 104)
# Adding element type (line 104)
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Obtaining the member 'fc_exe' of a type
fc_exe_60499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60498, 'fc_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 25), list_60497, fc_exe_60499)
# Adding element type (line 104)
str_60500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 34), 'str', '/f77rtl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 25), list_60497, str_60500)
# Adding element type (line 104)
str_60501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 45), 'str', '/fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 25), list_60497, str_60501)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60496, list_60497))
# Adding element type (key, value) (line 102)
str_60502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 105)
list_60503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 105)
# Adding element type (line 105)
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Obtaining the member 'fc_exe' of a type
fc_exe_60505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60504, 'fc_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), list_60503, fc_exe_60505)
# Adding element type (line 105)
str_60506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'str', '/fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 25), list_60503, str_60506)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60502, list_60503))
# Adding element type (key, value) (line 102)
str_60507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 106)
list_60508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 106)
# Adding element type (line 106)
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Obtaining the member 'fc_exe' of a type
fc_exe_60510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60509, 'fc_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 25), list_60508, fc_exe_60510)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60507, list_60508))
# Adding element type (key, value) (line 102)
str_60511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 107)
list_60512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 107)
# Adding element type (line 107)
str_60513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_60512, str_60513)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60511, list_60512))
# Adding element type (key, value) (line 102)
str_60514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 108)
list_60515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 108)
# Adding element type (line 108)
# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Obtaining the member 'ar_exe' of a type
ar_exe_60517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60516, 'ar_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 25), list_60515, ar_exe_60517)
# Adding element type (line 108)
str_60518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'str', '/OUT:')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 25), list_60515, str_60518)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60514, list_60515))
# Adding element type (key, value) (line 102)
str_60519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'str', 'ranlib')
# Getting the type of 'None' (line 109)
None_60520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 18), dict_60491, (str_60519, None_60520))

# Getting the type of 'CompaqVisualFCompiler'
CompaqVisualFCompiler_60521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CompaqVisualFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CompaqVisualFCompiler_60521, 'executables', dict_60491)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 123, 4))
    
    # 'from distutils import log' statement (line 123)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 123, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 124)
    # Processing the call arguments (line 124)
    int_60524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'int')
    # Processing the call keyword arguments (line 124)
    kwargs_60525 = {}
    # Getting the type of 'log' (line 124)
    log_60522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 124)
    set_verbosity_60523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), log_60522, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 124)
    set_verbosity_call_result_60526 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), set_verbosity_60523, *[int_60524], **kwargs_60525)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 125, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 125)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_60527 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 125, 4), 'numpy.distutils.fcompiler')

    if (type(import_60527) is not StypyTypeError):

        if (import_60527 != 'pyd_module'):
            __import__(import_60527)
            sys_modules_60528 = sys.modules[import_60527]
            import_from_module(stypy.reporting.localization.Localization(__file__, 125, 4), 'numpy.distutils.fcompiler', sys_modules_60528.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 125, 4), __file__, sys_modules_60528, sys_modules_60528.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 125, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'numpy.distutils.fcompiler', import_60527)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 126):
    
    # Call to new_fcompiler(...): (line 126)
    # Processing the call keyword arguments (line 126)
    str_60530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 38), 'str', 'compaq')
    keyword_60531 = str_60530
    kwargs_60532 = {'compiler': keyword_60531}
    # Getting the type of 'new_fcompiler' (line 126)
    new_fcompiler_60529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 126)
    new_fcompiler_call_result_60533 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), new_fcompiler_60529, *[], **kwargs_60532)
    
    # Assigning a type to the variable 'compiler' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'compiler', new_fcompiler_call_result_60533)
    
    # Call to customize(...): (line 127)
    # Processing the call keyword arguments (line 127)
    kwargs_60536 = {}
    # Getting the type of 'compiler' (line 127)
    compiler_60534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 127)
    customize_60535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), compiler_60534, 'customize')
    # Calling customize(args, kwargs) (line 127)
    customize_call_result_60537 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), customize_60535, *[], **kwargs_60536)
    
    
    # Call to print(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Call to get_version(...): (line 128)
    # Processing the call keyword arguments (line 128)
    kwargs_60541 = {}
    # Getting the type of 'compiler' (line 128)
    compiler_60539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 128)
    get_version_60540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 10), compiler_60539, 'get_version')
    # Calling get_version(args, kwargs) (line 128)
    get_version_call_result_60542 = invoke(stypy.reporting.localization.Localization(__file__, 128, 10), get_version_60540, *[], **kwargs_60541)
    
    # Processing the call keyword arguments (line 128)
    kwargs_60543 = {}
    # Getting the type of 'print' (line 128)
    print_60538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'print', False)
    # Calling print(args, kwargs) (line 128)
    print_call_result_60544 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), print_60538, *[get_version_call_result_60542], **kwargs_60543)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

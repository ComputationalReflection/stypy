
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.pgroup.com
2: from __future__ import division, absolute_import, print_function
3: 
4: from numpy.distutils.fcompiler import FCompiler
5: from sys import platform
6: 
7: compilers = ['PGroupFCompiler']
8: 
9: class PGroupFCompiler(FCompiler):
10: 
11:     compiler_type = 'pg'
12:     description = 'Portland Group Fortran Compiler'
13:     version_pattern =  r'\s*pg(f77|f90|hpf|fortran) (?P<version>[\d.-]+).*'
14: 
15:     if platform == 'darwin':
16:         executables = {
17:         'version_cmd'  : ["<F77>", "-V"],
18:         'compiler_f77' : ["pgfortran", "-dynamiclib"],
19:         'compiler_fix' : ["pgfortran", "-Mfixed", "-dynamiclib"],
20:         'compiler_f90' : ["pgfortran", "-dynamiclib"],
21:         'linker_so'    : ["libtool"],
22:         'archiver'     : ["ar", "-cr"],
23:         'ranlib'       : ["ranlib"]
24:         }
25:         pic_flags = ['']
26:     else:
27:         executables = {
28:         'version_cmd'  : ["<F77>", "-V"],
29:         'compiler_f77' : ["pgfortran"],
30:         'compiler_fix' : ["pgfortran", "-Mfixed"],
31:         'compiler_f90' : ["pgfortran"],
32:         'linker_so'    : ["pgfortran", "-shared", "-fpic"],
33:         'archiver'     : ["ar", "-cr"],
34:         'ranlib'       : ["ranlib"]
35:         }
36:         pic_flags = ['-fpic']
37: 
38: 
39:     module_dir_switch = '-module '
40:     module_include_switch = '-I'
41: 
42:     def get_flags(self):
43:         opt = ['-Minform=inform', '-Mnosecond_underscore']
44:         return self.pic_flags + opt
45:     def get_flags_opt(self):
46:         return ['-fast']
47:     def get_flags_debug(self):
48:         return ['-g']
49: 
50:     if platform == 'darwin':
51:         def get_flags_linker_so(self):
52:             return ["-dynamic", '-undefined', 'dynamic_lookup']
53: 
54:     def runtime_library_dir_option(self, dir):
55:         return '-R"%s"' % dir
56: 
57: if __name__ == '__main__':
58:     from distutils import log
59:     log.set_verbosity(2)
60:     from numpy.distutils.fcompiler import new_fcompiler
61:     compiler = new_fcompiler(compiler='pg')
62:     compiler.customize()
63:     print(compiler.get_version())
64: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler')

if (type(import_63101) is not StypyTypeError):

    if (import_63101 != 'pyd_module'):
        __import__(import_63101)
        sys_modules_63102 = sys.modules[import_63101]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', sys_modules_63102.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_63102, sys_modules_63102.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', import_63101)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from sys import platform' statement (line 5)
from sys import platform

import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', None, module_type_store, ['platform'], [platform])


# Assigning a List to a Name (line 7):

# Obtaining an instance of the builtin type 'list' (line 7)
list_63103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_63104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'str', 'PGroupFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_63103, str_63104)

# Assigning a type to the variable 'compilers' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'compilers', list_63103)
# Declaration of the 'PGroupFCompiler' class
# Getting the type of 'FCompiler' (line 9)
FCompiler_63105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 22), 'FCompiler')

class PGroupFCompiler(FCompiler_63105, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'PGroupFCompiler.get_flags')
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PGroupFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PGroupFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_63106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        str_63107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'str', '-Minform=inform')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), list_63106, str_63107)
        # Adding element type (line 43)
        str_63108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'str', '-Mnosecond_underscore')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), list_63106, str_63108)
        
        # Assigning a type to the variable 'opt' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'opt', list_63106)
        # Getting the type of 'self' (line 44)
        self_63109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'self')
        # Obtaining the member 'pic_flags' of a type (line 44)
        pic_flags_63110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), self_63109, 'pic_flags')
        # Getting the type of 'opt' (line 44)
        opt_63111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 32), 'opt')
        # Applying the binary operator '+' (line 44)
        result_add_63112 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), '+', pic_flags_63110, opt_63111)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', result_add_63112)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_63113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_63113


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'PGroupFCompiler.get_flags_opt')
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PGroupFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PGroupFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_63114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        str_63115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'str', '-fast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), list_63114, str_63115)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', list_63114)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_63116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_63116


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'PGroupFCompiler.get_flags_debug')
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PGroupFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PGroupFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_63117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        str_63118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), list_63117, str_63118)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', list_63117)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_63119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_63119


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'PGroupFCompiler.runtime_library_dir_option')
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PGroupFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PGroupFCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

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

        str_63120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'str', '-R"%s"')
        # Getting the type of 'dir' (line 55)
        dir_63121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'dir')
        # Applying the binary operator '%' (line 55)
        result_mod_63122 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 15), '%', str_63120, dir_63121)
        
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', result_mod_63122)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_63123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_63123


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PGroupFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PGroupFCompiler' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'PGroupFCompiler', PGroupFCompiler)

# Assigning a Str to a Name (line 11):
str_63124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'pg')
# Getting the type of 'PGroupFCompiler'
PGroupFCompiler_63125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PGroupFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PGroupFCompiler_63125, 'compiler_type', str_63124)

# Assigning a Str to a Name (line 12):
str_63126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'str', 'Portland Group Fortran Compiler')
# Getting the type of 'PGroupFCompiler'
PGroupFCompiler_63127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PGroupFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PGroupFCompiler_63127, 'description', str_63126)

# Assigning a Str to a Name (line 13):
str_63128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', '\\s*pg(f77|f90|hpf|fortran) (?P<version>[\\d.-]+).*')
# Getting the type of 'PGroupFCompiler'
PGroupFCompiler_63129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PGroupFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PGroupFCompiler_63129, 'version_pattern', str_63128)


# Getting the type of 'platform' (line 15)
platform_63130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'platform')
str_63131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'str', 'darwin')
# Applying the binary operator '==' (line 15)
result_eq_63132 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 7), '==', platform_63130, str_63131)

# Testing the type of an if condition (line 15)
if_condition_63133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), result_eq_63132)
# Assigning a type to the variable 'if_condition_63133' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_63133', if_condition_63133)
# SSA begins for if statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Name (line 16):

# Obtaining an instance of the builtin type 'dict' (line 16)
dict_63134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 16)
# Adding element type (key, value) (line 16)
str_63135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 17)
list_63136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_63137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', '<F77>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_63136, str_63137)
# Adding element type (line 17)
str_63138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'str', '-V')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_63136, str_63138)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63135, list_63136))
# Adding element type (key, value) (line 16)
str_63139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 18)
list_63140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_63141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_63140, str_63141)
# Adding element type (line 18)
str_63142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 39), 'str', '-dynamiclib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_63140, str_63142)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63139, list_63140))
# Adding element type (key, value) (line 16)
str_63143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 19)
list_63144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_63145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_63144, str_63145)
# Adding element type (line 19)
str_63146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'str', '-Mfixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_63144, str_63146)
# Adding element type (line 19)
str_63147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 50), 'str', '-dynamiclib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_63144, str_63147)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63143, list_63144))
# Adding element type (key, value) (line 16)
str_63148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 20)
list_63149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_63150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_63149, str_63150)
# Adding element type (line 20)
str_63151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 39), 'str', '-dynamiclib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_63149, str_63151)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63148, list_63149))
# Adding element type (key, value) (line 16)
str_63152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 21)
list_63153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_63154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'libtool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_63153, str_63154)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63152, list_63153))
# Adding element type (key, value) (line 16)
str_63155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 22)
list_63156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_63157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_63156, str_63157)
# Adding element type (line 22)
str_63158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_63156, str_63158)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63155, list_63156))
# Adding element type (key, value) (line 16)
str_63159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 23)
list_63160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_63161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_63160, str_63161)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), dict_63134, (str_63159, list_63160))

# Assigning a type to the variable 'executables' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'executables', dict_63134)

# Assigning a List to a Name (line 25):

# Obtaining an instance of the builtin type 'list' (line 25)
list_63162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_63163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'str', '')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), list_63162, str_63163)

# Assigning a type to the variable 'pic_flags' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'pic_flags', list_63162)
# SSA branch for the else part of an if statement (line 15)
module_type_store.open_ssa_branch('else')

# Assigning a Dict to a Name (line 27):

# Obtaining an instance of the builtin type 'dict' (line 27)
dict_63164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 27)
# Adding element type (key, value) (line 27)
str_63165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 28)
list_63166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_63167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'str', '<F77>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 25), list_63166, str_63167)
# Adding element type (line 28)
str_63168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'str', '-V')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 25), list_63166, str_63168)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63165, list_63166))
# Adding element type (key, value) (line 27)
str_63169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 29)
list_63170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_63171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 25), list_63170, str_63171)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63169, list_63170))
# Adding element type (key, value) (line 27)
str_63172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 30)
list_63173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
str_63174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_63173, str_63174)
# Adding element type (line 30)
str_63175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'str', '-Mfixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_63173, str_63175)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63172, list_63173))
# Adding element type (key, value) (line 27)
str_63176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 31)
list_63177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
str_63178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 25), list_63177, str_63178)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63176, list_63177))
# Adding element type (key, value) (line 27)
str_63179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 32)
list_63180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
str_63181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'str', 'pgfortran')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_63180, str_63181)
# Adding element type (line 32)
str_63182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 39), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_63180, str_63182)
# Adding element type (line 32)
str_63183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'str', '-fpic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_63180, str_63183)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63179, list_63180))
# Adding element type (key, value) (line 27)
str_63184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 33)
list_63185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
str_63186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 25), list_63185, str_63186)
# Adding element type (line 33)
str_63187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 25), list_63185, str_63187)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63184, list_63185))
# Adding element type (key, value) (line 27)
str_63188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 34)
list_63189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
str_63190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 25), list_63189, str_63190)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), dict_63164, (str_63188, list_63189))

# Assigning a type to the variable 'executables' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'executables', dict_63164)

# Assigning a List to a Name (line 36):

# Obtaining an instance of the builtin type 'list' (line 36)
list_63191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
str_63192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'str', '-fpic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 20), list_63191, str_63192)

# Assigning a type to the variable 'pic_flags' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'pic_flags', list_63191)
# SSA join for if statement (line 15)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 39):
str_63193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'str', '-module ')
# Getting the type of 'PGroupFCompiler'
PGroupFCompiler_63194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PGroupFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PGroupFCompiler_63194, 'module_dir_switch', str_63193)

# Assigning a Str to a Name (line 40):
str_63195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'str', '-I')
# Getting the type of 'PGroupFCompiler'
PGroupFCompiler_63196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PGroupFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PGroupFCompiler_63196, 'module_include_switch', str_63195)


# Getting the type of 'platform' (line 50)
platform_63197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'platform')
str_63198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'str', 'darwin')
# Applying the binary operator '==' (line 50)
result_eq_63199 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), '==', platform_63197, str_63198)

# Testing the type of an if condition (line 50)
if_condition_63200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), result_eq_63199)
# Assigning a type to the variable 'if_condition_63200' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_63200', if_condition_63200)
# SSA begins for if statement (line 50)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def get_flags_linker_so(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_flags_linker_so'
    module_type_store = module_type_store.open_function_context('get_flags_linker_so', 51, 8, False)
    
    # Passed parameters checking function
    get_flags_linker_so.stypy_localization = localization
    get_flags_linker_so.stypy_type_of_self = None
    get_flags_linker_so.stypy_type_store = module_type_store
    get_flags_linker_so.stypy_function_name = 'get_flags_linker_so'
    get_flags_linker_so.stypy_param_names_list = ['self']
    get_flags_linker_so.stypy_varargs_param_name = None
    get_flags_linker_so.stypy_kwargs_param_name = None
    get_flags_linker_so.stypy_call_defaults = defaults
    get_flags_linker_so.stypy_call_varargs = varargs
    get_flags_linker_so.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_flags_linker_so', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_flags_linker_so', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_flags_linker_so(...)' code ##################

    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_63201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    str_63202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'str', '-dynamic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), list_63201, str_63202)
    # Adding element type (line 52)
    str_63203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'str', '-undefined')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), list_63201, str_63203)
    # Adding element type (line 52)
    str_63204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 46), 'str', 'dynamic_lookup')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), list_63201, str_63204)
    
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'stypy_return_type', list_63201)
    
    # ################# End of 'get_flags_linker_so(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_flags_linker_so' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_63205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_63205)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_flags_linker_so'
    return stypy_return_type_63205

# Assigning a type to the variable 'get_flags_linker_so' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'get_flags_linker_so', get_flags_linker_so)
# SSA join for if statement (line 50)
module_type_store = module_type_store.join_ssa_context()


if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 4))
    
    # 'from distutils import log' statement (line 58)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 58, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 59)
    # Processing the call arguments (line 59)
    int_63208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_63209 = {}
    # Getting the type of 'log' (line 59)
    log_63206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 59)
    set_verbosity_63207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), log_63206, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 59)
    set_verbosity_call_result_63210 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), set_verbosity_63207, *[int_63208], **kwargs_63209)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 60, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 60)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_63211 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.fcompiler')

    if (type(import_63211) is not StypyTypeError):

        if (import_63211 != 'pyd_module'):
            __import__(import_63211)
            sys_modules_63212 = sys.modules[import_63211]
            import_from_module(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.fcompiler', sys_modules_63212.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 60, 4), __file__, sys_modules_63212, sys_modules_63212.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'numpy.distutils.fcompiler', import_63211)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 61):
    
    # Call to new_fcompiler(...): (line 61)
    # Processing the call keyword arguments (line 61)
    str_63214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'str', 'pg')
    keyword_63215 = str_63214
    kwargs_63216 = {'compiler': keyword_63215}
    # Getting the type of 'new_fcompiler' (line 61)
    new_fcompiler_63213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 61)
    new_fcompiler_call_result_63217 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), new_fcompiler_63213, *[], **kwargs_63216)
    
    # Assigning a type to the variable 'compiler' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'compiler', new_fcompiler_call_result_63217)
    
    # Call to customize(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_63220 = {}
    # Getting the type of 'compiler' (line 62)
    compiler_63218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 62)
    customize_63219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), compiler_63218, 'customize')
    # Calling customize(args, kwargs) (line 62)
    customize_call_result_63221 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), customize_63219, *[], **kwargs_63220)
    
    
    # Call to print(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Call to get_version(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_63225 = {}
    # Getting the type of 'compiler' (line 63)
    compiler_63223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 63)
    get_version_63224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 10), compiler_63223, 'get_version')
    # Calling get_version(args, kwargs) (line 63)
    get_version_call_result_63226 = invoke(stypy.reporting.localization.Localization(__file__, 63, 10), get_version_63224, *[], **kwargs_63225)
    
    # Processing the call keyword arguments (line 63)
    kwargs_63227 = {}
    # Getting the type of 'print' (line 63)
    print_63222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'print', False)
    # Calling print(args, kwargs) (line 63)
    print_call_result_63228 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), print_63222, *[get_version_call_result_63226], **kwargs_63227)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

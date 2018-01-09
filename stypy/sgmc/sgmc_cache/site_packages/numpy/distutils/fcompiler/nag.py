
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: from numpy.distutils.fcompiler import FCompiler
5: 
6: compilers = ['NAGFCompiler']
7: 
8: class NAGFCompiler(FCompiler):
9: 
10:     compiler_type = 'nag'
11:     description = 'NAGWare Fortran 95 Compiler'
12:     version_pattern =  r'NAGWare Fortran 95 compiler Release (?P<version>[^\s]*)'
13: 
14:     executables = {
15:         'version_cmd'  : ["<F90>", "-V"],
16:         'compiler_f77' : ["f95", "-fixed"],
17:         'compiler_fix' : ["f95", "-fixed"],
18:         'compiler_f90' : ["f95"],
19:         'linker_so'    : ["<F90>"],
20:         'archiver'     : ["ar", "-cr"],
21:         'ranlib'       : ["ranlib"]
22:         }
23: 
24:     def get_flags_linker_so(self):
25:         if sys.platform=='darwin':
26:             return ['-unsharedf95', '-Wl,-bundle,-flat_namespace,-undefined,suppress']
27:         return ["-Wl,-shared"]
28:     def get_flags_opt(self):
29:         return ['-O4']
30:     def get_flags_arch(self):
31:         version = self.get_version()
32:         if version and version < '5.1':
33:             return ['-target=native']
34:         else:
35:             return ['']
36:     def get_flags_debug(self):
37:         return ['-g', '-gline', '-g90', '-nan', '-C']
38: 
39: if __name__ == '__main__':
40:     from distutils import log
41:     log.set_verbosity(2)
42:     from numpy.distutils.fcompiler import new_fcompiler
43:     compiler = new_fcompiler(compiler='nag')
44:     compiler.customize()
45:     print(compiler.get_version())
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62882 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler')

if (type(import_62882) is not StypyTypeError):

    if (import_62882 != 'pyd_module'):
        __import__(import_62882)
        sys_modules_62883 = sys.modules[import_62882]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', sys_modules_62883.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_62883, sys_modules_62883.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', import_62882)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_62884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_62885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'str', 'NAGFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_62884, str_62885)

# Assigning a type to the variable 'compilers' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'compilers', list_62884)
# Declaration of the 'NAGFCompiler' class
# Getting the type of 'FCompiler' (line 8)
FCompiler_62886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'FCompiler')

class NAGFCompiler(FCompiler_62886, ):

    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'NAGFCompiler.get_flags_linker_so')
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NAGFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NAGFCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'sys' (line 25)
        sys_62887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 25)
        platform_62888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), sys_62887, 'platform')
        str_62889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'str', 'darwin')
        # Applying the binary operator '==' (line 25)
        result_eq_62890 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), '==', platform_62888, str_62889)
        
        # Testing the type of an if condition (line 25)
        if_condition_62891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 8), result_eq_62890)
        # Assigning a type to the variable 'if_condition_62891' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'if_condition_62891', if_condition_62891)
        # SSA begins for if statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_62892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        str_62893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'str', '-unsharedf95')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_62892, str_62893)
        # Adding element type (line 26)
        str_62894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 36), 'str', '-Wl,-bundle,-flat_namespace,-undefined,suppress')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_62892, str_62894)
        
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'stypy_return_type', list_62892)
        # SSA join for if statement (line 25)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_62895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        str_62896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'str', '-Wl,-shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_62895, str_62896)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', list_62895)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_62897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62897)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_62897


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'NAGFCompiler.get_flags_opt')
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NAGFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NAGFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_62898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        str_62899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'str', '-O4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), list_62898, str_62899)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', list_62898)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_62900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62900)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62900


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'NAGFCompiler.get_flags_arch')
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NAGFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NAGFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 31):
        
        # Call to get_version(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_62903 = {}
        # Getting the type of 'self' (line 31)
        self_62901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'self', False)
        # Obtaining the member 'get_version' of a type (line 31)
        get_version_62902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), self_62901, 'get_version')
        # Calling get_version(args, kwargs) (line 31)
        get_version_call_result_62904 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), get_version_62902, *[], **kwargs_62903)
        
        # Assigning a type to the variable 'version' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'version', get_version_call_result_62904)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'version' (line 32)
        version_62905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'version')
        
        # Getting the type of 'version' (line 32)
        version_62906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'version')
        str_62907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 33), 'str', '5.1')
        # Applying the binary operator '<' (line 32)
        result_lt_62908 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), '<', version_62906, str_62907)
        
        # Applying the binary operator 'and' (line 32)
        result_and_keyword_62909 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), 'and', version_62905, result_lt_62908)
        
        # Testing the type of an if condition (line 32)
        if_condition_62910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_and_keyword_62909)
        # Assigning a type to the variable 'if_condition_62910' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_62910', if_condition_62910)
        # SSA begins for if statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_62911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        str_62912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'str', '-target=native')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_62911, str_62912)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', list_62911)
        # SSA branch for the else part of an if statement (line 32)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_62913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        str_62914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 19), list_62913, str_62914)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'stypy_return_type', list_62913)
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_62915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62915)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_62915


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'NAGFCompiler.get_flags_debug')
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NAGFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NAGFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_62916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        str_62917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 15), list_62916, str_62917)
        # Adding element type (line 37)
        str_62918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'str', '-gline')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 15), list_62916, str_62918)
        # Adding element type (line 37)
        str_62919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'str', '-g90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 15), list_62916, str_62919)
        # Adding element type (line 37)
        str_62920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 40), 'str', '-nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 15), list_62916, str_62920)
        # Adding element type (line 37)
        str_62921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 48), 'str', '-C')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 15), list_62916, str_62921)
        
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', list_62916)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_62922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62922)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_62922


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NAGFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NAGFCompiler' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'NAGFCompiler', NAGFCompiler)

# Assigning a Str to a Name (line 10):
str_62923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'str', 'nag')
# Getting the type of 'NAGFCompiler'
NAGFCompiler_62924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NAGFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NAGFCompiler_62924, 'compiler_type', str_62923)

# Assigning a Str to a Name (line 11):
str_62925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', 'NAGWare Fortran 95 Compiler')
# Getting the type of 'NAGFCompiler'
NAGFCompiler_62926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NAGFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NAGFCompiler_62926, 'description', str_62925)

# Assigning a Str to a Name (line 12):
str_62927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'str', 'NAGWare Fortran 95 compiler Release (?P<version>[^\\s]*)')
# Getting the type of 'NAGFCompiler'
NAGFCompiler_62928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NAGFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NAGFCompiler_62928, 'version_pattern', str_62927)

# Assigning a Dict to a Name (line 14):

# Obtaining an instance of the builtin type 'dict' (line 14)
dict_62929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 14)
# Adding element type (key, value) (line 14)
str_62930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 15)
list_62931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_62932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_62931, str_62932)
# Adding element type (line 15)
str_62933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'str', '-V')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_62931, str_62933)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62930, list_62931))
# Adding element type (key, value) (line 14)
str_62934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 16)
list_62935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_62936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'f95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_62935, str_62936)
# Adding element type (line 16)
str_62937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', '-fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_62935, str_62937)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62934, list_62935))
# Adding element type (key, value) (line 14)
str_62938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 17)
list_62939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_62940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'f95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_62939, str_62940)
# Adding element type (line 17)
str_62941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'str', '-fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_62939, str_62941)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62938, list_62939))
# Adding element type (key, value) (line 14)
str_62942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 18)
list_62943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_62944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'f95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_62943, str_62944)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62942, list_62943))
# Adding element type (key, value) (line 14)
str_62945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 19)
list_62946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_62947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_62946, str_62947)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62945, list_62946))
# Adding element type (key, value) (line 14)
str_62948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 20)
list_62949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_62950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_62949, str_62950)
# Adding element type (line 20)
str_62951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_62949, str_62951)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62948, list_62949))
# Adding element type (key, value) (line 14)
str_62952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 21)
list_62953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_62954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_62953, str_62954)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62929, (str_62952, list_62953))

# Getting the type of 'NAGFCompiler'
NAGFCompiler_62955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NAGFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NAGFCompiler_62955, 'executables', dict_62929)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 4))
    
    # 'from distutils import log' statement (line 40)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 40, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 41)
    # Processing the call arguments (line 41)
    int_62958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_62959 = {}
    # Getting the type of 'log' (line 41)
    log_62956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 41)
    set_verbosity_62957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), log_62956, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 41)
    set_verbosity_call_result_62960 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), set_verbosity_62957, *[int_62958], **kwargs_62959)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 42)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_62961 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler')

    if (type(import_62961) is not StypyTypeError):

        if (import_62961 != 'pyd_module'):
            __import__(import_62961)
            sys_modules_62962 = sys.modules[import_62961]
            import_from_module(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler', sys_modules_62962.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 42, 4), __file__, sys_modules_62962, sys_modules_62962.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler', import_62961)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 43):
    
    # Call to new_fcompiler(...): (line 43)
    # Processing the call keyword arguments (line 43)
    str_62964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'str', 'nag')
    keyword_62965 = str_62964
    kwargs_62966 = {'compiler': keyword_62965}
    # Getting the type of 'new_fcompiler' (line 43)
    new_fcompiler_62963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 43)
    new_fcompiler_call_result_62967 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), new_fcompiler_62963, *[], **kwargs_62966)
    
    # Assigning a type to the variable 'compiler' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'compiler', new_fcompiler_call_result_62967)
    
    # Call to customize(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_62970 = {}
    # Getting the type of 'compiler' (line 44)
    compiler_62968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 44)
    customize_62969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), compiler_62968, 'customize')
    # Calling customize(args, kwargs) (line 44)
    customize_call_result_62971 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), customize_62969, *[], **kwargs_62970)
    
    
    # Call to print(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to get_version(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_62975 = {}
    # Getting the type of 'compiler' (line 45)
    compiler_62973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 45)
    get_version_62974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 10), compiler_62973, 'get_version')
    # Calling get_version(args, kwargs) (line 45)
    get_version_call_result_62976 = invoke(stypy.reporting.localization.Localization(__file__, 45, 10), get_version_62974, *[], **kwargs_62975)
    
    # Processing the call keyword arguments (line 45)
    kwargs_62977 = {}
    # Getting the type of 'print' (line 45)
    print_62972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'print', False)
    # Calling print(args, kwargs) (line 45)
    print_call_result_62978 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), print_62972, *[get_version_call_result_62976], **kwargs_62977)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

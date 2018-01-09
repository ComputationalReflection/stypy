
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: 
5: from numpy.distutils.fcompiler import FCompiler
6: 
7: compilers = ['LaheyFCompiler']
8: 
9: class LaheyFCompiler(FCompiler):
10: 
11:     compiler_type = 'lahey'
12:     description = 'Lahey/Fujitsu Fortran 95 Compiler'
13:     version_pattern =  r'Lahey/Fujitsu Fortran 95 Compiler Release (?P<version>[^\s*]*)'
14: 
15:     executables = {
16:         'version_cmd'  : ["<F90>", "--version"],
17:         'compiler_f77' : ["lf95", "--fix"],
18:         'compiler_fix' : ["lf95", "--fix"],
19:         'compiler_f90' : ["lf95"],
20:         'linker_so'    : ["lf95", "-shared"],
21:         'archiver'     : ["ar", "-cr"],
22:         'ranlib'       : ["ranlib"]
23:         }
24: 
25:     module_dir_switch = None  #XXX Fix me
26:     module_include_switch = None #XXX Fix me
27: 
28:     def get_flags_opt(self):
29:         return ['-O']
30:     def get_flags_debug(self):
31:         return ['-g', '--chk', '--chkglobal']
32:     def get_library_dirs(self):
33:         opt = []
34:         d = os.environ.get('LAHEY')
35:         if d:
36:             opt.append(os.path.join(d, 'lib'))
37:         return opt
38:     def get_libraries(self):
39:         opt = []
40:         opt.extend(['fj9f6', 'fj9i6', 'fj9ipp', 'fj9e6'])
41:         return opt
42: 
43: if __name__ == '__main__':
44:     from distutils import log
45:     log.set_verbosity(2)
46:     from numpy.distutils.fcompiler import new_fcompiler
47:     compiler = new_fcompiler(compiler='lahey')
48:     compiler.customize()
49:     print(compiler.get_version())
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62618 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler')

if (type(import_62618) is not StypyTypeError):

    if (import_62618 != 'pyd_module'):
        __import__(import_62618)
        sys_modules_62619 = sys.modules[import_62618]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler', sys_modules_62619.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_62619, sys_modules_62619.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler', import_62618)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 7):

# Obtaining an instance of the builtin type 'list' (line 7)
list_62620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_62621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'str', 'LaheyFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_62620, str_62621)

# Assigning a type to the variable 'compilers' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'compilers', list_62620)
# Declaration of the 'LaheyFCompiler' class
# Getting the type of 'FCompiler' (line 9)
FCompiler_62622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'FCompiler')

class LaheyFCompiler(FCompiler_62622, ):

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
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'LaheyFCompiler.get_flags_opt')
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LaheyFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LaheyFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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
        list_62623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        str_62624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'str', '-O')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), list_62623, str_62624)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', list_62623)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_62625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62625


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'LaheyFCompiler.get_flags_debug')
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LaheyFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LaheyFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_62626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        str_62627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_62626, str_62627)
        # Adding element type (line 31)
        str_62628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 22), 'str', '--chk')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_62626, str_62628)
        # Adding element type (line 31)
        str_62629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', '--chkglobal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_62626, str_62629)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', list_62626)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_62630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_62630


    @norecursion
    def get_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_dirs'
        module_type_store = module_type_store.open_function_context('get_library_dirs', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_function_name', 'LaheyFCompiler.get_library_dirs')
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LaheyFCompiler.get_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LaheyFCompiler.get_library_dirs', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 33):
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_62631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        
        # Assigning a type to the variable 'opt' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'opt', list_62631)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to get(...): (line 34)
        # Processing the call arguments (line 34)
        str_62635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 27), 'str', 'LAHEY')
        # Processing the call keyword arguments (line 34)
        kwargs_62636 = {}
        # Getting the type of 'os' (line 34)
        os_62632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'os', False)
        # Obtaining the member 'environ' of a type (line 34)
        environ_62633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), os_62632, 'environ')
        # Obtaining the member 'get' of a type (line 34)
        get_62634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), environ_62633, 'get')
        # Calling get(args, kwargs) (line 34)
        get_call_result_62637 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), get_62634, *[str_62635], **kwargs_62636)
        
        # Assigning a type to the variable 'd' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'd', get_call_result_62637)
        
        # Getting the type of 'd' (line 35)
        d_62638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'd')
        # Testing the type of an if condition (line 35)
        if_condition_62639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), d_62638)
        # Assigning a type to the variable 'if_condition_62639' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_62639', if_condition_62639)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Call to join(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'd' (line 36)
        d_62645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'd', False)
        str_62646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 39), 'str', 'lib')
        # Processing the call keyword arguments (line 36)
        kwargs_62647 = {}
        # Getting the type of 'os' (line 36)
        os_62642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 36)
        path_62643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), os_62642, 'path')
        # Obtaining the member 'join' of a type (line 36)
        join_62644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), path_62643, 'join')
        # Calling join(args, kwargs) (line 36)
        join_call_result_62648 = invoke(stypy.reporting.localization.Localization(__file__, 36, 23), join_62644, *[d_62645, str_62646], **kwargs_62647)
        
        # Processing the call keyword arguments (line 36)
        kwargs_62649 = {}
        # Getting the type of 'opt' (line 36)
        opt_62640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 36)
        append_62641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), opt_62640, 'append')
        # Calling append(args, kwargs) (line 36)
        append_call_result_62650 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), append_62641, *[join_call_result_62648], **kwargs_62649)
        
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 37)
        opt_62651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', opt_62651)
        
        # ################# End of 'get_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_62652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_dirs'
        return stypy_return_type_62652


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'LaheyFCompiler.get_libraries')
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LaheyFCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LaheyFCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 39):
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_62653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        
        # Assigning a type to the variable 'opt' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'opt', list_62653)
        
        # Call to extend(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_62656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        str_62657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'str', 'fj9f6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_62656, str_62657)
        # Adding element type (line 40)
        str_62658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 29), 'str', 'fj9i6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_62656, str_62658)
        # Adding element type (line 40)
        str_62659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'str', 'fj9ipp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_62656, str_62659)
        # Adding element type (line 40)
        str_62660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 48), 'str', 'fj9e6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_62656, str_62660)
        
        # Processing the call keyword arguments (line 40)
        kwargs_62661 = {}
        # Getting the type of 'opt' (line 40)
        opt_62654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'opt', False)
        # Obtaining the member 'extend' of a type (line 40)
        extend_62655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), opt_62654, 'extend')
        # Calling extend(args, kwargs) (line 40)
        extend_call_result_62662 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), extend_62655, *[list_62656], **kwargs_62661)
        
        # Getting the type of 'opt' (line 41)
        opt_62663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', opt_62663)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_62664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62664)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_62664


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LaheyFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LaheyFCompiler' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'LaheyFCompiler', LaheyFCompiler)

# Assigning a Str to a Name (line 11):
str_62665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'lahey')
# Getting the type of 'LaheyFCompiler'
LaheyFCompiler_62666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LaheyFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LaheyFCompiler_62666, 'compiler_type', str_62665)

# Assigning a Str to a Name (line 12):
str_62667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'str', 'Lahey/Fujitsu Fortran 95 Compiler')
# Getting the type of 'LaheyFCompiler'
LaheyFCompiler_62668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LaheyFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LaheyFCompiler_62668, 'description', str_62667)

# Assigning a Str to a Name (line 13):
str_62669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'Lahey/Fujitsu Fortran 95 Compiler Release (?P<version>[^\\s*]*)')
# Getting the type of 'LaheyFCompiler'
LaheyFCompiler_62670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LaheyFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LaheyFCompiler_62670, 'version_pattern', str_62669)

# Assigning a Dict to a Name (line 15):

# Obtaining an instance of the builtin type 'dict' (line 15)
dict_62671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 15)
# Adding element type (key, value) (line 15)
str_62672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 16)
list_62673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_62674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_62673, str_62674)
# Adding element type (line 16)
str_62675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'str', '--version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_62673, str_62675)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62672, list_62673))
# Adding element type (key, value) (line 15)
str_62676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 17)
list_62677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_62678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'lf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_62677, str_62678)
# Adding element type (line 17)
str_62679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'str', '--fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_62677, str_62679)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62676, list_62677))
# Adding element type (key, value) (line 15)
str_62680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 18)
list_62681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_62682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'lf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_62681, str_62682)
# Adding element type (line 18)
str_62683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'str', '--fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_62681, str_62683)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62680, list_62681))
# Adding element type (key, value) (line 15)
str_62684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 19)
list_62685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_62686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'lf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_62685, str_62686)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62684, list_62685))
# Adding element type (key, value) (line 15)
str_62687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 20)
list_62688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_62689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'lf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_62688, str_62689)
# Adding element type (line 20)
str_62690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_62688, str_62690)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62687, list_62688))
# Adding element type (key, value) (line 15)
str_62691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 21)
list_62692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_62693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_62692, str_62693)
# Adding element type (line 21)
str_62694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_62692, str_62694)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62691, list_62692))
# Adding element type (key, value) (line 15)
str_62695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 22)
list_62696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_62697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_62696, str_62697)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 18), dict_62671, (str_62695, list_62696))

# Getting the type of 'LaheyFCompiler'
LaheyFCompiler_62698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LaheyFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LaheyFCompiler_62698, 'executables', dict_62671)

# Assigning a Name to a Name (line 25):
# Getting the type of 'None' (line 25)
None_62699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'None')
# Getting the type of 'LaheyFCompiler'
LaheyFCompiler_62700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LaheyFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LaheyFCompiler_62700, 'module_dir_switch', None_62699)

# Assigning a Name to a Name (line 26):
# Getting the type of 'None' (line 26)
None_62701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'None')
# Getting the type of 'LaheyFCompiler'
LaheyFCompiler_62702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LaheyFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LaheyFCompiler_62702, 'module_include_switch', None_62701)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 4))
    
    # 'from distutils import log' statement (line 44)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 44, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 45)
    # Processing the call arguments (line 45)
    int_62705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'int')
    # Processing the call keyword arguments (line 45)
    kwargs_62706 = {}
    # Getting the type of 'log' (line 45)
    log_62703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 45)
    set_verbosity_62704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 4), log_62703, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 45)
    set_verbosity_call_result_62707 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), set_verbosity_62704, *[int_62705], **kwargs_62706)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 46)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_62708 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 4), 'numpy.distutils.fcompiler')

    if (type(import_62708) is not StypyTypeError):

        if (import_62708 != 'pyd_module'):
            __import__(import_62708)
            sys_modules_62709 = sys.modules[import_62708]
            import_from_module(stypy.reporting.localization.Localization(__file__, 46, 4), 'numpy.distutils.fcompiler', sys_modules_62709.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 46, 4), __file__, sys_modules_62709, sys_modules_62709.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 46, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'numpy.distutils.fcompiler', import_62708)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 47):
    
    # Call to new_fcompiler(...): (line 47)
    # Processing the call keyword arguments (line 47)
    str_62711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'str', 'lahey')
    keyword_62712 = str_62711
    kwargs_62713 = {'compiler': keyword_62712}
    # Getting the type of 'new_fcompiler' (line 47)
    new_fcompiler_62710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 47)
    new_fcompiler_call_result_62714 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), new_fcompiler_62710, *[], **kwargs_62713)
    
    # Assigning a type to the variable 'compiler' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'compiler', new_fcompiler_call_result_62714)
    
    # Call to customize(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_62717 = {}
    # Getting the type of 'compiler' (line 48)
    compiler_62715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 48)
    customize_62716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), compiler_62715, 'customize')
    # Calling customize(args, kwargs) (line 48)
    customize_call_result_62718 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), customize_62716, *[], **kwargs_62717)
    
    
    # Call to print(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Call to get_version(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_62722 = {}
    # Getting the type of 'compiler' (line 49)
    compiler_62720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 49)
    get_version_62721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 10), compiler_62720, 'get_version')
    # Calling get_version(args, kwargs) (line 49)
    get_version_call_result_62723 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), get_version_62721, *[], **kwargs_62722)
    
    # Processing the call keyword arguments (line 49)
    kwargs_62724 = {}
    # Getting the type of 'print' (line 49)
    print_62719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'print', False)
    # Calling print(args, kwargs) (line 49)
    print_call_result_62725 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), print_62719, *[get_version_call_result_62723], **kwargs_62724)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

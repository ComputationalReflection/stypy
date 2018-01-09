
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: have_setuptools = ('setuptools' in sys.modules)
5: 
6: from distutils.command.install_data import install_data as old_install_data
7: 
8: #data installer with improved intelligence over distutils
9: #data files are copied into the project directory instead
10: #of willy-nilly
11: class install_data (old_install_data):
12: 
13:     def run(self):
14:         old_install_data.run(self)
15: 
16:         if have_setuptools:
17:             # Run install_clib again, since setuptools does not run sub-commands
18:             # of install automatically
19:             self.run_command('install_clib')
20: 
21:     def finalize_options (self):
22:         self.set_undefined_options('install',
23:                                    ('install_lib', 'install_dir'),
24:                                    ('root', 'root'),
25:                                    ('force', 'force'),
26:                                   )
27: 

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


# Assigning a Compare to a Name (line 4):

str_59582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 19), 'str', 'setuptools')
# Getting the type of 'sys' (line 4)
sys_59583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 35), 'sys')
# Obtaining the member 'modules' of a type (line 4)
modules_59584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 35), sys_59583, 'modules')
# Applying the binary operator 'in' (line 4)
result_contains_59585 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 19), 'in', str_59582, modules_59584)

# Assigning a type to the variable 'have_setuptools' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'have_setuptools', result_contains_59585)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.command.install_data import old_install_data' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59586 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_data')

if (type(import_59586) is not StypyTypeError):

    if (import_59586 != 'pyd_module'):
        __import__(import_59586)
        sys_modules_59587 = sys.modules[import_59586]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_data', sys_modules_59587.module_type_store, module_type_store, ['install_data'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_59587, sys_modules_59587.module_type_store, module_type_store)
    else:
        from distutils.command.install_data import install_data as old_install_data

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_data', None, module_type_store, ['install_data'], [old_install_data])

else:
    # Assigning a type to the variable 'distutils.command.install_data' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_data', import_59586)

# Adding an alias
module_type_store.add_alias('old_install_data', 'install_data')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'install_data' class
# Getting the type of 'old_install_data' (line 11)
old_install_data_59588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'old_install_data')

class install_data(old_install_data_59588, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_data.run.__dict__.__setitem__('stypy_localization', localization)
        install_data.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.run.__dict__.__setitem__('stypy_function_name', 'install_data.run')
        install_data.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to run(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'self' (line 14)
        self_59591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'self', False)
        # Processing the call keyword arguments (line 14)
        kwargs_59592 = {}
        # Getting the type of 'old_install_data' (line 14)
        old_install_data_59589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'old_install_data', False)
        # Obtaining the member 'run' of a type (line 14)
        run_59590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), old_install_data_59589, 'run')
        # Calling run(args, kwargs) (line 14)
        run_call_result_59593 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), run_59590, *[self_59591], **kwargs_59592)
        
        
        # Getting the type of 'have_setuptools' (line 16)
        have_setuptools_59594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'have_setuptools')
        # Testing the type of an if condition (line 16)
        if_condition_59595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 8), have_setuptools_59594)
        # Assigning a type to the variable 'if_condition_59595' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'if_condition_59595', if_condition_59595)
        # SSA begins for if statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 19)
        # Processing the call arguments (line 19)
        str_59598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'str', 'install_clib')
        # Processing the call keyword arguments (line 19)
        kwargs_59599 = {}
        # Getting the type of 'self' (line 19)
        self_59596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 19)
        run_command_59597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), self_59596, 'run_command')
        # Calling run_command(args, kwargs) (line 19)
        run_command_call_result_59600 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), run_command_59597, *[str_59598], **kwargs_59599)
        
        # SSA join for if statement (line 16)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_59601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59601


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_data.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_data.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_data.finalize_options')
        install_data.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 22)
        # Processing the call arguments (line 22)
        str_59604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_59605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        str_59606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'str', 'install_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 36), tuple_59605, str_59606)
        # Adding element type (line 23)
        str_59607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 36), tuple_59605, str_59607)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_59608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        str_59609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'str', 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 36), tuple_59608, str_59609)
        # Adding element type (line 24)
        str_59610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 44), 'str', 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 36), tuple_59608, str_59610)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_59611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        str_59612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), tuple_59611, str_59612)
        # Adding element type (line 25)
        str_59613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), tuple_59611, str_59613)
        
        # Processing the call keyword arguments (line 22)
        kwargs_59614 = {}
        # Getting the type of 'self' (line 22)
        self_59602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 22)
        set_undefined_options_59603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_59602, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 22)
        set_undefined_options_call_result_59615 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), set_undefined_options_59603, *[str_59604, tuple_59605, tuple_59608, tuple_59611], **kwargs_59614)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_59616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59616)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_59616


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_data' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'install_data', install_data)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

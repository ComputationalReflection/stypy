
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import sys
5: if 'setuptools' in sys.modules:
6:     from setuptools.command.bdist_rpm import bdist_rpm as old_bdist_rpm
7: else:
8:     from distutils.command.bdist_rpm import bdist_rpm as old_bdist_rpm
9: 
10: class bdist_rpm(old_bdist_rpm):
11: 
12:     def _make_spec_file(self):
13:         spec_file = old_bdist_rpm._make_spec_file(self)
14: 
15:         # Replace hardcoded setup.py script name
16:         # with the real setup script name.
17:         setup_py = os.path.basename(sys.argv[0])
18:         if setup_py == 'setup.py':
19:             return spec_file
20:         new_spec_file = []
21:         for line in spec_file:
22:             line = line.replace('setup.py', setup_py)
23:             new_spec_file.append(line)
24:         return new_spec_file
25: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)



str_52386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 3), 'str', 'setuptools')
# Getting the type of 'sys' (line 5)
sys_52387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 19), 'sys')
# Obtaining the member 'modules' of a type (line 5)
modules_52388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 19), sys_52387, 'modules')
# Applying the binary operator 'in' (line 5)
result_contains_52389 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 3), 'in', str_52386, modules_52388)

# Testing the type of an if condition (line 5)
if_condition_52390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 0), result_contains_52389)
# Assigning a type to the variable 'if_condition_52390' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'if_condition_52390', if_condition_52390)
# SSA begins for if statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'from setuptools.command.bdist_rpm import old_bdist_rpm' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52391 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'setuptools.command.bdist_rpm')

if (type(import_52391) is not StypyTypeError):

    if (import_52391 != 'pyd_module'):
        __import__(import_52391)
        sys_modules_52392 = sys.modules[import_52391]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'setuptools.command.bdist_rpm', sys_modules_52392.module_type_store, module_type_store, ['bdist_rpm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_52392, sys_modules_52392.module_type_store, module_type_store)
    else:
        from setuptools.command.bdist_rpm import bdist_rpm as old_bdist_rpm

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'setuptools.command.bdist_rpm', None, module_type_store, ['bdist_rpm'], [old_bdist_rpm])

else:
    # Assigning a type to the variable 'setuptools.command.bdist_rpm' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'setuptools.command.bdist_rpm', import_52391)

# Adding an alias
module_type_store.add_alias('old_bdist_rpm', 'bdist_rpm')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# SSA branch for the else part of an if statement (line 5)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from distutils.command.bdist_rpm import old_bdist_rpm' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52393 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.command.bdist_rpm')

if (type(import_52393) is not StypyTypeError):

    if (import_52393 != 'pyd_module'):
        __import__(import_52393)
        sys_modules_52394 = sys.modules[import_52393]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.command.bdist_rpm', sys_modules_52394.module_type_store, module_type_store, ['bdist_rpm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_52394, sys_modules_52394.module_type_store, module_type_store)
    else:
        from distutils.command.bdist_rpm import bdist_rpm as old_bdist_rpm

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.command.bdist_rpm', None, module_type_store, ['bdist_rpm'], [old_bdist_rpm])

else:
    # Assigning a type to the variable 'distutils.command.bdist_rpm' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.command.bdist_rpm', import_52393)

# Adding an alias
module_type_store.add_alias('old_bdist_rpm', 'bdist_rpm')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# SSA join for if statement (line 5)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'bdist_rpm' class
# Getting the type of 'old_bdist_rpm' (line 10)
old_bdist_rpm_52395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'old_bdist_rpm')

class bdist_rpm(old_bdist_rpm_52395, ):

    @norecursion
    def _make_spec_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_make_spec_file'
        module_type_store = module_type_store.open_function_context('_make_spec_file', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_function_name', 'bdist_rpm._make_spec_file')
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm._make_spec_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_make_spec_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_make_spec_file(...)' code ##################

        
        # Assigning a Call to a Name (line 13):
        
        # Call to _make_spec_file(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'self' (line 13)
        self_52398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 50), 'self', False)
        # Processing the call keyword arguments (line 13)
        kwargs_52399 = {}
        # Getting the type of 'old_bdist_rpm' (line 13)
        old_bdist_rpm_52396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'old_bdist_rpm', False)
        # Obtaining the member '_make_spec_file' of a type (line 13)
        _make_spec_file_52397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 20), old_bdist_rpm_52396, '_make_spec_file')
        # Calling _make_spec_file(args, kwargs) (line 13)
        _make_spec_file_call_result_52400 = invoke(stypy.reporting.localization.Localization(__file__, 13, 20), _make_spec_file_52397, *[self_52398], **kwargs_52399)
        
        # Assigning a type to the variable 'spec_file' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'spec_file', _make_spec_file_call_result_52400)
        
        # Assigning a Call to a Name (line 17):
        
        # Call to basename(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Obtaining the type of the subscript
        int_52404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 45), 'int')
        # Getting the type of 'sys' (line 17)
        sys_52405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 36), 'sys', False)
        # Obtaining the member 'argv' of a type (line 17)
        argv_52406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 36), sys_52405, 'argv')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___52407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 36), argv_52406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_52408 = invoke(stypy.reporting.localization.Localization(__file__, 17, 36), getitem___52407, int_52404)
        
        # Processing the call keyword arguments (line 17)
        kwargs_52409 = {}
        # Getting the type of 'os' (line 17)
        os_52401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 17)
        path_52402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), os_52401, 'path')
        # Obtaining the member 'basename' of a type (line 17)
        basename_52403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), path_52402, 'basename')
        # Calling basename(args, kwargs) (line 17)
        basename_call_result_52410 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), basename_52403, *[subscript_call_result_52408], **kwargs_52409)
        
        # Assigning a type to the variable 'setup_py' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'setup_py', basename_call_result_52410)
        
        
        # Getting the type of 'setup_py' (line 18)
        setup_py_52411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'setup_py')
        str_52412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'str', 'setup.py')
        # Applying the binary operator '==' (line 18)
        result_eq_52413 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), '==', setup_py_52411, str_52412)
        
        # Testing the type of an if condition (line 18)
        if_condition_52414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_eq_52413)
        # Assigning a type to the variable 'if_condition_52414' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_52414', if_condition_52414)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'spec_file' (line 19)
        spec_file_52415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'spec_file')
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type', spec_file_52415)
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 20):
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_52416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        
        # Assigning a type to the variable 'new_spec_file' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'new_spec_file', list_52416)
        
        # Getting the type of 'spec_file' (line 21)
        spec_file_52417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'spec_file')
        # Testing the type of a for loop iterable (line 21)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 8), spec_file_52417)
        # Getting the type of the for loop variable (line 21)
        for_loop_var_52418 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 8), spec_file_52417)
        # Assigning a type to the variable 'line' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'line', for_loop_var_52418)
        # SSA begins for a for statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 22):
        
        # Call to replace(...): (line 22)
        # Processing the call arguments (line 22)
        str_52421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', 'setup.py')
        # Getting the type of 'setup_py' (line 22)
        setup_py_52422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 44), 'setup_py', False)
        # Processing the call keyword arguments (line 22)
        kwargs_52423 = {}
        # Getting the type of 'line' (line 22)
        line_52419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'line', False)
        # Obtaining the member 'replace' of a type (line 22)
        replace_52420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), line_52419, 'replace')
        # Calling replace(args, kwargs) (line 22)
        replace_call_result_52424 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), replace_52420, *[str_52421, setup_py_52422], **kwargs_52423)
        
        # Assigning a type to the variable 'line' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'line', replace_call_result_52424)
        
        # Call to append(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'line' (line 23)
        line_52427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 33), 'line', False)
        # Processing the call keyword arguments (line 23)
        kwargs_52428 = {}
        # Getting the type of 'new_spec_file' (line 23)
        new_spec_file_52425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'new_spec_file', False)
        # Obtaining the member 'append' of a type (line 23)
        append_52426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), new_spec_file_52425, 'append')
        # Calling append(args, kwargs) (line 23)
        append_call_result_52429 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), append_52426, *[line_52427], **kwargs_52428)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_spec_file' (line 24)
        new_spec_file_52430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'new_spec_file')
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type', new_spec_file_52430)
        
        # ################# End of '_make_spec_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_make_spec_file' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_52431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_make_spec_file'
        return stypy_return_type_52431


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bdist_rpm' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'bdist_rpm', bdist_rpm)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

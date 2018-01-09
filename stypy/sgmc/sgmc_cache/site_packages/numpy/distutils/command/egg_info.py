
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: 
5: from setuptools.command.egg_info import egg_info as _egg_info
6: 
7: class egg_info(_egg_info):
8:     def run(self):
9:         if 'sdist' in sys.argv:
10:             import warnings
11:             warnings.warn("`build_src` is being run, this may lead to missing "
12:                           "files in your sdist!  See numpy issue gh-7127 for "
13:                           "details", UserWarning)
14: 
15:         # We need to ensure that build_src has been executed in order to give
16:         # setuptools' egg_info command real filenames instead of functions which
17:         # generate files.
18:         self.run_command("build_src")
19:         _egg_info.run(self)
20: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from setuptools.command.egg_info import _egg_info' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59315 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'setuptools.command.egg_info')

if (type(import_59315) is not StypyTypeError):

    if (import_59315 != 'pyd_module'):
        __import__(import_59315)
        sys_modules_59316 = sys.modules[import_59315]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'setuptools.command.egg_info', sys_modules_59316.module_type_store, module_type_store, ['egg_info'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_59316, sys_modules_59316.module_type_store, module_type_store)
    else:
        from setuptools.command.egg_info import egg_info as _egg_info

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'setuptools.command.egg_info', None, module_type_store, ['egg_info'], [_egg_info])

else:
    # Assigning a type to the variable 'setuptools.command.egg_info' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'setuptools.command.egg_info', import_59315)

# Adding an alias
module_type_store.add_alias('_egg_info', 'egg_info')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'egg_info' class
# Getting the type of '_egg_info' (line 7)
_egg_info_59317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), '_egg_info')

class egg_info(_egg_info_59317, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        egg_info.run.__dict__.__setitem__('stypy_localization', localization)
        egg_info.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        egg_info.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        egg_info.run.__dict__.__setitem__('stypy_function_name', 'egg_info.run')
        egg_info.run.__dict__.__setitem__('stypy_param_names_list', [])
        egg_info.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        egg_info.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        egg_info.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        egg_info.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        egg_info.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        egg_info.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'egg_info.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        str_59318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'sdist')
        # Getting the type of 'sys' (line 9)
        sys_59319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 22), 'sys')
        # Obtaining the member 'argv' of a type (line 9)
        argv_59320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 22), sys_59319, 'argv')
        # Applying the binary operator 'in' (line 9)
        result_contains_59321 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 11), 'in', str_59318, argv_59320)
        
        # Testing the type of an if condition (line 9)
        if_condition_59322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 8), result_contains_59321)
        # Assigning a type to the variable 'if_condition_59322' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'if_condition_59322', if_condition_59322)
        # SSA begins for if statement (line 9)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 12))
        
        # 'import warnings' statement (line 10)
        import warnings

        import_module(stypy.reporting.localization.Localization(__file__, 10, 12), 'warnings', warnings, module_type_store)
        
        
        # Call to warn(...): (line 11)
        # Processing the call arguments (line 11)
        str_59325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', '`build_src` is being run, this may lead to missing files in your sdist!  See numpy issue gh-7127 for details')
        # Getting the type of 'UserWarning' (line 13)
        UserWarning_59326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 37), 'UserWarning', False)
        # Processing the call keyword arguments (line 11)
        kwargs_59327 = {}
        # Getting the type of 'warnings' (line 11)
        warnings_59323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 11)
        warn_59324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), warnings_59323, 'warn')
        # Calling warn(args, kwargs) (line 11)
        warn_call_result_59328 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), warn_59324, *[str_59325, UserWarning_59326], **kwargs_59327)
        
        # SSA join for if statement (line 9)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to run_command(...): (line 18)
        # Processing the call arguments (line 18)
        str_59331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'build_src')
        # Processing the call keyword arguments (line 18)
        kwargs_59332 = {}
        # Getting the type of 'self' (line 18)
        self_59329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', False)
        # Obtaining the member 'run_command' of a type (line 18)
        run_command_59330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_59329, 'run_command')
        # Calling run_command(args, kwargs) (line 18)
        run_command_call_result_59333 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), run_command_59330, *[str_59331], **kwargs_59332)
        
        
        # Call to run(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'self' (line 19)
        self_59336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'self', False)
        # Processing the call keyword arguments (line 19)
        kwargs_59337 = {}
        # Getting the type of '_egg_info' (line 19)
        _egg_info_59334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), '_egg_info', False)
        # Obtaining the member 'run' of a type (line 19)
        run_59335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), _egg_info_59334, 'run')
        # Calling run(args, kwargs) (line 19)
        run_call_result_59338 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), run_59335, *[self_59336], **kwargs_59337)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_59339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59339)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59339


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 7, 0, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'egg_info.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'egg_info' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'egg_info', egg_info)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import pytest
5: 
6: from scipy.special import _test_round
7: 
8: 
9: @pytest.mark.skipif(not _test_round.have_fenv(), reason="no fenv()")
10: def test_add_round_up():
11:     np.random.seed(1234)
12:     _test_round.test_add_round(10**5, 'up')
13: 
14: 
15: @pytest.mark.skipif(not _test_round.have_fenv(), reason="no fenv()")
16: def test_add_round_down():
17:     np.random.seed(1234)
18:     _test_round.test_add_round(10**5, 'down')
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559794 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_559794) is not StypyTypeError):

    if (import_559794 != 'pyd_module'):
        __import__(import_559794)
        sys_modules_559795 = sys.modules[import_559794]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_559795.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_559794)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import pytest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559796 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_559796) is not StypyTypeError):

    if (import_559796 != 'pyd_module'):
        __import__(import_559796)
        sys_modules_559797 = sys.modules[import_559796]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_559797.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_559796)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special import _test_round' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559798 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special')

if (type(import_559798) is not StypyTypeError):

    if (import_559798 != 'pyd_module'):
        __import__(import_559798)
        sys_modules_559799 = sys.modules[import_559798]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', sys_modules_559799.module_type_store, module_type_store, ['_test_round'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_559799, sys_modules_559799.module_type_store, module_type_store)
    else:
        from scipy.special import _test_round

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', None, module_type_store, ['_test_round'], [_test_round])

else:
    # Assigning a type to the variable 'scipy.special' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', import_559798)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_add_round_up(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_add_round_up'
    module_type_store = module_type_store.open_function_context('test_add_round_up', 9, 0, False)
    
    # Passed parameters checking function
    test_add_round_up.stypy_localization = localization
    test_add_round_up.stypy_type_of_self = None
    test_add_round_up.stypy_type_store = module_type_store
    test_add_round_up.stypy_function_name = 'test_add_round_up'
    test_add_round_up.stypy_param_names_list = []
    test_add_round_up.stypy_varargs_param_name = None
    test_add_round_up.stypy_kwargs_param_name = None
    test_add_round_up.stypy_call_defaults = defaults
    test_add_round_up.stypy_call_varargs = varargs
    test_add_round_up.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_add_round_up', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_add_round_up', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_add_round_up(...)' code ##################

    
    # Call to seed(...): (line 11)
    # Processing the call arguments (line 11)
    int_559803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_559804 = {}
    # Getting the type of 'np' (line 11)
    np_559800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 11)
    random_559801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_559800, 'random')
    # Obtaining the member 'seed' of a type (line 11)
    seed_559802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), random_559801, 'seed')
    # Calling seed(args, kwargs) (line 11)
    seed_call_result_559805 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), seed_559802, *[int_559803], **kwargs_559804)
    
    
    # Call to test_add_round(...): (line 12)
    # Processing the call arguments (line 12)
    int_559808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
    int_559809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'int')
    # Applying the binary operator '**' (line 12)
    result_pow_559810 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 31), '**', int_559808, int_559809)
    
    str_559811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 38), 'str', 'up')
    # Processing the call keyword arguments (line 12)
    kwargs_559812 = {}
    # Getting the type of '_test_round' (line 12)
    _test_round_559806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), '_test_round', False)
    # Obtaining the member 'test_add_round' of a type (line 12)
    test_add_round_559807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), _test_round_559806, 'test_add_round')
    # Calling test_add_round(args, kwargs) (line 12)
    test_add_round_call_result_559813 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), test_add_round_559807, *[result_pow_559810, str_559811], **kwargs_559812)
    
    
    # ################# End of 'test_add_round_up(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_add_round_up' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_559814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559814)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_add_round_up'
    return stypy_return_type_559814

# Assigning a type to the variable 'test_add_round_up' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_add_round_up', test_add_round_up)

@norecursion
def test_add_round_down(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_add_round_down'
    module_type_store = module_type_store.open_function_context('test_add_round_down', 15, 0, False)
    
    # Passed parameters checking function
    test_add_round_down.stypy_localization = localization
    test_add_round_down.stypy_type_of_self = None
    test_add_round_down.stypy_type_store = module_type_store
    test_add_round_down.stypy_function_name = 'test_add_round_down'
    test_add_round_down.stypy_param_names_list = []
    test_add_round_down.stypy_varargs_param_name = None
    test_add_round_down.stypy_kwargs_param_name = None
    test_add_round_down.stypy_call_defaults = defaults
    test_add_round_down.stypy_call_varargs = varargs
    test_add_round_down.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_add_round_down', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_add_round_down', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_add_round_down(...)' code ##################

    
    # Call to seed(...): (line 17)
    # Processing the call arguments (line 17)
    int_559818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_559819 = {}
    # Getting the type of 'np' (line 17)
    np_559815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 17)
    random_559816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), np_559815, 'random')
    # Obtaining the member 'seed' of a type (line 17)
    seed_559817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), random_559816, 'seed')
    # Calling seed(args, kwargs) (line 17)
    seed_call_result_559820 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), seed_559817, *[int_559818], **kwargs_559819)
    
    
    # Call to test_add_round(...): (line 18)
    # Processing the call arguments (line 18)
    int_559823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'int')
    int_559824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
    # Applying the binary operator '**' (line 18)
    result_pow_559825 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 31), '**', int_559823, int_559824)
    
    str_559826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'str', 'down')
    # Processing the call keyword arguments (line 18)
    kwargs_559827 = {}
    # Getting the type of '_test_round' (line 18)
    _test_round_559821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), '_test_round', False)
    # Obtaining the member 'test_add_round' of a type (line 18)
    test_add_round_559822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), _test_round_559821, 'test_add_round')
    # Calling test_add_round(args, kwargs) (line 18)
    test_add_round_call_result_559828 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), test_add_round_559822, *[result_pow_559825, str_559826], **kwargs_559827)
    
    
    # ################# End of 'test_add_round_down(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_add_round_down' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_559829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_add_round_down'
    return stypy_return_type_559829

# Assigning a type to the variable 'test_add_round_down' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_add_round_down', test_add_round_down)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

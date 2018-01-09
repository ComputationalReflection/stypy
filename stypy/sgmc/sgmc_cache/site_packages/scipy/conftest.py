
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Pytest customization
2: from __future__ import division, absolute_import, print_function
3: 
4: import os
5: import pytest
6: import warnings
7: 
8: from scipy._lib._fpumode import get_fpu_mode
9: from scipy._lib._testutils import FPUModeChangeWarning
10: 
11: 
12: def pytest_runtest_setup(item):
13:     mark = item.get_marker("xslow")
14:     if mark is not None:
15:         try:
16:             v = int(os.environ.get('SCIPY_XSLOW', '0'))
17:         except ValueError:
18:             v = False
19:         if not v:
20:             pytest.skip("very slow test; set environment variable SCIPY_XSLOW=1 to run it")
21: 
22: 
23: @pytest.fixture(scope="function", autouse=True)
24: def check_fpu_mode(request):
25:     '''
26:     Check FPU mode was not changed during the test.
27:     '''
28:     old_mode = get_fpu_mode()
29:     yield
30:     new_mode = get_fpu_mode()
31: 
32:     if old_mode != new_mode:
33:         warnings.warn("FPU mode changed from {0:#x} to {1:#x} during "
34:                       "the test".format(old_mode, new_mode),
35:                       category=FPUModeChangeWarning, stacklevel=0)
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import pytest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_2.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_1)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib._fpumode import get_fpu_mode' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_3 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._fpumode')

if (type(import_3) is not StypyTypeError):

    if (import_3 != 'pyd_module'):
        __import__(import_3)
        sys_modules_4 = sys.modules[import_3]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._fpumode', sys_modules_4.module_type_store, module_type_store, ['get_fpu_mode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_4, sys_modules_4.module_type_store, module_type_store)
    else:
        from scipy._lib._fpumode import get_fpu_mode

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._fpumode', None, module_type_store, ['get_fpu_mode'], [get_fpu_mode])

else:
    # Assigning a type to the variable 'scipy._lib._fpumode' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._fpumode', import_3)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy._lib._testutils import FPUModeChangeWarning' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_5 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._testutils')

if (type(import_5) is not StypyTypeError):

    if (import_5 != 'pyd_module'):
        __import__(import_5)
        sys_modules_6 = sys.modules[import_5]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._testutils', sys_modules_6.module_type_store, module_type_store, ['FPUModeChangeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_6, sys_modules_6.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import FPUModeChangeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._testutils', None, module_type_store, ['FPUModeChangeWarning'], [FPUModeChangeWarning])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._testutils', import_5)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')


@norecursion
def pytest_runtest_setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pytest_runtest_setup'
    module_type_store = module_type_store.open_function_context('pytest_runtest_setup', 12, 0, False)
    
    # Passed parameters checking function
    pytest_runtest_setup.stypy_localization = localization
    pytest_runtest_setup.stypy_type_of_self = None
    pytest_runtest_setup.stypy_type_store = module_type_store
    pytest_runtest_setup.stypy_function_name = 'pytest_runtest_setup'
    pytest_runtest_setup.stypy_param_names_list = ['item']
    pytest_runtest_setup.stypy_varargs_param_name = None
    pytest_runtest_setup.stypy_kwargs_param_name = None
    pytest_runtest_setup.stypy_call_defaults = defaults
    pytest_runtest_setup.stypy_call_varargs = varargs
    pytest_runtest_setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pytest_runtest_setup', ['item'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pytest_runtest_setup', localization, ['item'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pytest_runtest_setup(...)' code ##################

    
    # Assigning a Call to a Name (line 13):
    
    # Call to get_marker(...): (line 13)
    # Processing the call arguments (line 13)
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'str', 'xslow')
    # Processing the call keyword arguments (line 13)
    kwargs_10 = {}
    # Getting the type of 'item' (line 13)
    item_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'item', False)
    # Obtaining the member 'get_marker' of a type (line 13)
    get_marker_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), item_7, 'get_marker')
    # Calling get_marker(args, kwargs) (line 13)
    get_marker_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), get_marker_8, *[str_9], **kwargs_10)
    
    # Assigning a type to the variable 'mark' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'mark', get_marker_call_result_11)
    
    # Type idiom detected: calculating its left and rigth part (line 14)
    # Getting the type of 'mark' (line 14)
    mark_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'mark')
    # Getting the type of 'None' (line 14)
    None_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'None')
    
    (may_be_14, more_types_in_union_15) = may_not_be_none(mark_12, None_13)

    if may_be_14:

        if more_types_in_union_15:
            # Runtime conditional SSA (line 14)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # SSA begins for try-except statement (line 15)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 16):
        
        # Call to int(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Call to get(...): (line 16)
        # Processing the call arguments (line 16)
        str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'str', 'SCIPY_XSLOW')
        str_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 50), 'str', '0')
        # Processing the call keyword arguments (line 16)
        kwargs_22 = {}
        # Getting the type of 'os' (line 16)
        os_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'os', False)
        # Obtaining the member 'environ' of a type (line 16)
        environ_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), os_17, 'environ')
        # Obtaining the member 'get' of a type (line 16)
        get_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), environ_18, 'get')
        # Calling get(args, kwargs) (line 16)
        get_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), get_19, *[str_20, str_21], **kwargs_22)
        
        # Processing the call keyword arguments (line 16)
        kwargs_24 = {}
        # Getting the type of 'int' (line 16)
        int_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'int', False)
        # Calling int(args, kwargs) (line 16)
        int_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), int_16, *[get_call_result_23], **kwargs_24)
        
        # Assigning a type to the variable 'v' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'v', int_call_result_25)
        # SSA branch for the except part of a try statement (line 15)
        # SSA branch for the except 'ValueError' branch of a try statement (line 15)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'False' (line 18)
        False_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'False')
        # Assigning a type to the variable 'v' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'v', False_26)
        # SSA join for try-except statement (line 15)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'v' (line 19)
        v_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'v')
        # Applying the 'not' unary operator (line 19)
        result_not__28 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 11), 'not', v_27)
        
        # Testing the type of an if condition (line 19)
        if_condition_29 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), result_not__28)
        # Assigning a type to the variable 'if_condition_29' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_29', if_condition_29)
        # SSA begins for if statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to skip(...): (line 20)
        # Processing the call arguments (line 20)
        str_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'str', 'very slow test; set environment variable SCIPY_XSLOW=1 to run it')
        # Processing the call keyword arguments (line 20)
        kwargs_33 = {}
        # Getting the type of 'pytest' (line 20)
        pytest_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'pytest', False)
        # Obtaining the member 'skip' of a type (line 20)
        skip_31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), pytest_30, 'skip')
        # Calling skip(args, kwargs) (line 20)
        skip_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), skip_31, *[str_32], **kwargs_33)
        
        # SSA join for if statement (line 19)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_15:
            # SSA join for if statement (line 14)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'pytest_runtest_setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pytest_runtest_setup' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pytest_runtest_setup'
    return stypy_return_type_35

# Assigning a type to the variable 'pytest_runtest_setup' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest_runtest_setup', pytest_runtest_setup)

@norecursion
def check_fpu_mode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_fpu_mode'
    module_type_store = module_type_store.open_function_context('check_fpu_mode', 23, 0, False)
    
    # Passed parameters checking function
    check_fpu_mode.stypy_localization = localization
    check_fpu_mode.stypy_type_of_self = None
    check_fpu_mode.stypy_type_store = module_type_store
    check_fpu_mode.stypy_function_name = 'check_fpu_mode'
    check_fpu_mode.stypy_param_names_list = ['request']
    check_fpu_mode.stypy_varargs_param_name = None
    check_fpu_mode.stypy_kwargs_param_name = None
    check_fpu_mode.stypy_call_defaults = defaults
    check_fpu_mode.stypy_call_varargs = varargs
    check_fpu_mode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_fpu_mode', ['request'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_fpu_mode', localization, ['request'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_fpu_mode(...)' code ##################

    str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n    Check FPU mode was not changed during the test.\n    ')
    
    # Assigning a Call to a Name (line 28):
    
    # Call to get_fpu_mode(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_38 = {}
    # Getting the type of 'get_fpu_mode' (line 28)
    get_fpu_mode_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'get_fpu_mode', False)
    # Calling get_fpu_mode(args, kwargs) (line 28)
    get_fpu_mode_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), get_fpu_mode_37, *[], **kwargs_38)
    
    # Assigning a type to the variable 'old_mode' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'old_mode', get_fpu_mode_call_result_39)
    # Creating a generator
    GeneratorType_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), GeneratorType_40, None)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', GeneratorType_40)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to get_fpu_mode(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_42 = {}
    # Getting the type of 'get_fpu_mode' (line 30)
    get_fpu_mode_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'get_fpu_mode', False)
    # Calling get_fpu_mode(args, kwargs) (line 30)
    get_fpu_mode_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), get_fpu_mode_41, *[], **kwargs_42)
    
    # Assigning a type to the variable 'new_mode' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'new_mode', get_fpu_mode_call_result_43)
    
    
    # Getting the type of 'old_mode' (line 32)
    old_mode_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'old_mode')
    # Getting the type of 'new_mode' (line 32)
    new_mode_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'new_mode')
    # Applying the binary operator '!=' (line 32)
    result_ne_46 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), '!=', old_mode_44, new_mode_45)
    
    # Testing the type of an if condition (line 32)
    if_condition_47 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), result_ne_46)
    # Assigning a type to the variable 'if_condition_47' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_47', if_condition_47)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to format(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'old_mode' (line 34)
    old_mode_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'old_mode', False)
    # Getting the type of 'new_mode' (line 34)
    new_mode_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 50), 'new_mode', False)
    # Processing the call keyword arguments (line 33)
    kwargs_54 = {}
    str_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'str', 'FPU mode changed from {0:#x} to {1:#x} during the test')
    # Obtaining the member 'format' of a type (line 33)
    format_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), str_50, 'format')
    # Calling format(args, kwargs) (line 33)
    format_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), format_51, *[old_mode_52, new_mode_53], **kwargs_54)
    
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'FPUModeChangeWarning' (line 35)
    FPUModeChangeWarning_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'FPUModeChangeWarning', False)
    keyword_57 = FPUModeChangeWarning_56
    int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 64), 'int')
    keyword_59 = int_58
    kwargs_60 = {'category': keyword_57, 'stacklevel': keyword_59}
    # Getting the type of 'warnings' (line 33)
    warnings_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 33)
    warn_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), warnings_48, 'warn')
    # Calling warn(args, kwargs) (line 33)
    warn_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), warn_49, *[format_call_result_55], **kwargs_60)
    
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_fpu_mode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_fpu_mode' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_62)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_fpu_mode'
    return stypy_return_type_62

# Assigning a type to the variable 'check_fpu_mode' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'check_fpu_mode', check_fpu_mode)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

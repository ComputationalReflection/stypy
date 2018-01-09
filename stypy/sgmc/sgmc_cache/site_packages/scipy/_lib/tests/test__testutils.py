
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: from scipy._lib._testutils import _parse_size, _get_mem_available
5: import pytest
6: 
7: 
8: def test__parse_size():
9:     expected = {
10:         '12': 12e6,
11:         '12 b': 12,
12:         '12k': 12e3,
13:         '  12  M  ': 12e6,
14:         '  12  G  ': 12e9,
15:         ' 12Tb ': 12e12,
16:         '12  Mib ': 12 * 1024.0**2,
17:         '12Tib': 12 * 1024.0**4,
18:     }
19: 
20:     for inp, outp in sorted(expected.items()):
21:         if outp is None:
22:             with pytest.raises(ValueError):
23:                 _parse_size(inp)
24:         else:
25:             assert _parse_size(inp) == outp
26: 
27: 
28: def test__mem_available():
29:     # May return None on non-Linux platforms
30:     available = _get_mem_available()
31:     if sys.platform.startswith('linux'):
32:         assert available >= 0
33:     else:
34:         assert available is None or available >= 0
35: 

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

# 'from scipy._lib._testutils import _parse_size, _get_mem_available' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712422 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._testutils')

if (type(import_712422) is not StypyTypeError):

    if (import_712422 != 'pyd_module'):
        __import__(import_712422)
        sys_modules_712423 = sys.modules[import_712422]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._testutils', sys_modules_712423.module_type_store, module_type_store, ['_parse_size', '_get_mem_available'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_712423, sys_modules_712423.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import _parse_size, _get_mem_available

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._testutils', None, module_type_store, ['_parse_size', '_get_mem_available'], [_parse_size, _get_mem_available])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._testutils', import_712422)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import pytest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712424 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_712424) is not StypyTypeError):

    if (import_712424 != 'pyd_module'):
        __import__(import_712424)
        sys_modules_712425 = sys.modules[import_712424]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_712425.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_712424)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


@norecursion
def test__parse_size(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test__parse_size'
    module_type_store = module_type_store.open_function_context('test__parse_size', 8, 0, False)
    
    # Passed parameters checking function
    test__parse_size.stypy_localization = localization
    test__parse_size.stypy_type_of_self = None
    test__parse_size.stypy_type_store = module_type_store
    test__parse_size.stypy_function_name = 'test__parse_size'
    test__parse_size.stypy_param_names_list = []
    test__parse_size.stypy_varargs_param_name = None
    test__parse_size.stypy_kwargs_param_name = None
    test__parse_size.stypy_call_defaults = defaults
    test__parse_size.stypy_call_varargs = varargs
    test__parse_size.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test__parse_size', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test__parse_size', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test__parse_size(...)' code ##################

    
    # Assigning a Dict to a Name (line 9):
    
    # Obtaining an instance of the builtin type 'dict' (line 9)
    dict_712426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 9)
    # Adding element type (key, value) (line 9)
    str_712427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', '12')
    float_712428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712427, float_712428))
    # Adding element type (key, value) (line 9)
    str_712429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'str', '12 b')
    int_712430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712429, int_712430))
    # Adding element type (key, value) (line 9)
    str_712431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'str', '12k')
    float_712432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712431, float_712432))
    # Adding element type (key, value) (line 9)
    str_712433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'str', '  12  M  ')
    float_712434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712433, float_712434))
    # Adding element type (key, value) (line 9)
    str_712435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'str', '  12  G  ')
    float_712436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 21), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712435, float_712436))
    # Adding element type (key, value) (line 9)
    str_712437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'str', ' 12Tb ')
    float_712438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712437, float_712438))
    # Adding element type (key, value) (line 9)
    str_712439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', '12  Mib ')
    int_712440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    float_712441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'float')
    int_712442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'int')
    # Applying the binary operator '**' (line 16)
    result_pow_712443 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 25), '**', float_712441, int_712442)
    
    # Applying the binary operator '*' (line 16)
    result_mul_712444 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 20), '*', int_712440, result_pow_712443)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712439, result_mul_712444))
    # Adding element type (key, value) (line 9)
    str_712445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', '12Tib')
    int_712446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
    float_712447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'float')
    int_712448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'int')
    # Applying the binary operator '**' (line 17)
    result_pow_712449 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 22), '**', float_712447, int_712448)
    
    # Applying the binary operator '*' (line 17)
    result_mul_712450 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 17), '*', int_712446, result_pow_712449)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), dict_712426, (str_712445, result_mul_712450))
    
    # Assigning a type to the variable 'expected' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'expected', dict_712426)
    
    
    # Call to sorted(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to items(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_712454 = {}
    # Getting the type of 'expected' (line 20)
    expected_712452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 28), 'expected', False)
    # Obtaining the member 'items' of a type (line 20)
    items_712453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 28), expected_712452, 'items')
    # Calling items(args, kwargs) (line 20)
    items_call_result_712455 = invoke(stypy.reporting.localization.Localization(__file__, 20, 28), items_712453, *[], **kwargs_712454)
    
    # Processing the call keyword arguments (line 20)
    kwargs_712456 = {}
    # Getting the type of 'sorted' (line 20)
    sorted_712451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'sorted', False)
    # Calling sorted(args, kwargs) (line 20)
    sorted_call_result_712457 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), sorted_712451, *[items_call_result_712455], **kwargs_712456)
    
    # Testing the type of a for loop iterable (line 20)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 4), sorted_call_result_712457)
    # Getting the type of the for loop variable (line 20)
    for_loop_var_712458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 4), sorted_call_result_712457)
    # Assigning a type to the variable 'inp' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'inp', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), for_loop_var_712458))
    # Assigning a type to the variable 'outp' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'outp', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), for_loop_var_712458))
    # SSA begins for a for statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    # Getting the type of 'outp' (line 21)
    outp_712459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'outp')
    # Getting the type of 'None' (line 21)
    None_712460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'None')
    
    (may_be_712461, more_types_in_union_712462) = may_be_none(outp_712459, None_712460)

    if may_be_712461:

        if more_types_in_union_712462:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to raises(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'ValueError' (line 22)
        ValueError_712465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'ValueError', False)
        # Processing the call keyword arguments (line 22)
        kwargs_712466 = {}
        # Getting the type of 'pytest' (line 22)
        pytest_712463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'pytest', False)
        # Obtaining the member 'raises' of a type (line 22)
        raises_712464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), pytest_712463, 'raises')
        # Calling raises(args, kwargs) (line 22)
        raises_call_result_712467 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), raises_712464, *[ValueError_712465], **kwargs_712466)
        
        with_712468 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 22, 17), raises_call_result_712467, 'with parameter', '__enter__', '__exit__')

        if with_712468:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 22)
            enter___712469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), raises_call_result_712467, '__enter__')
            with_enter_712470 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), enter___712469)
            
            # Call to _parse_size(...): (line 23)
            # Processing the call arguments (line 23)
            # Getting the type of 'inp' (line 23)
            inp_712472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'inp', False)
            # Processing the call keyword arguments (line 23)
            kwargs_712473 = {}
            # Getting the type of '_parse_size' (line 23)
            _parse_size_712471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), '_parse_size', False)
            # Calling _parse_size(args, kwargs) (line 23)
            _parse_size_call_result_712474 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), _parse_size_712471, *[inp_712472], **kwargs_712473)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 22)
            exit___712475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), raises_call_result_712467, '__exit__')
            with_exit_712476 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), exit___712475, None, None, None)


        if more_types_in_union_712462:
            # Runtime conditional SSA for else branch (line 21)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_712461) or more_types_in_union_712462):
        # Evaluating assert statement condition
        
        
        # Call to _parse_size(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'inp' (line 25)
        inp_712478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'inp', False)
        # Processing the call keyword arguments (line 25)
        kwargs_712479 = {}
        # Getting the type of '_parse_size' (line 25)
        _parse_size_712477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), '_parse_size', False)
        # Calling _parse_size(args, kwargs) (line 25)
        _parse_size_call_result_712480 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), _parse_size_712477, *[inp_712478], **kwargs_712479)
        
        # Getting the type of 'outp' (line 25)
        outp_712481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'outp')
        # Applying the binary operator '==' (line 25)
        result_eq_712482 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 19), '==', _parse_size_call_result_712480, outp_712481)
        

        if (may_be_712461 and more_types_in_union_712462):
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test__parse_size(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test__parse_size' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_712483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712483)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test__parse_size'
    return stypy_return_type_712483

# Assigning a type to the variable 'test__parse_size' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test__parse_size', test__parse_size)

@norecursion
def test__mem_available(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test__mem_available'
    module_type_store = module_type_store.open_function_context('test__mem_available', 28, 0, False)
    
    # Passed parameters checking function
    test__mem_available.stypy_localization = localization
    test__mem_available.stypy_type_of_self = None
    test__mem_available.stypy_type_store = module_type_store
    test__mem_available.stypy_function_name = 'test__mem_available'
    test__mem_available.stypy_param_names_list = []
    test__mem_available.stypy_varargs_param_name = None
    test__mem_available.stypy_kwargs_param_name = None
    test__mem_available.stypy_call_defaults = defaults
    test__mem_available.stypy_call_varargs = varargs
    test__mem_available.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test__mem_available', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test__mem_available', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test__mem_available(...)' code ##################

    
    # Assigning a Call to a Name (line 30):
    
    # Call to _get_mem_available(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_712485 = {}
    # Getting the type of '_get_mem_available' (line 30)
    _get_mem_available_712484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), '_get_mem_available', False)
    # Calling _get_mem_available(args, kwargs) (line 30)
    _get_mem_available_call_result_712486 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), _get_mem_available_712484, *[], **kwargs_712485)
    
    # Assigning a type to the variable 'available' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'available', _get_mem_available_call_result_712486)
    
    
    # Call to startswith(...): (line 31)
    # Processing the call arguments (line 31)
    str_712490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', 'linux')
    # Processing the call keyword arguments (line 31)
    kwargs_712491 = {}
    # Getting the type of 'sys' (line 31)
    sys_712487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'sys', False)
    # Obtaining the member 'platform' of a type (line 31)
    platform_712488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), sys_712487, 'platform')
    # Obtaining the member 'startswith' of a type (line 31)
    startswith_712489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), platform_712488, 'startswith')
    # Calling startswith(args, kwargs) (line 31)
    startswith_call_result_712492 = invoke(stypy.reporting.localization.Localization(__file__, 31, 7), startswith_712489, *[str_712490], **kwargs_712491)
    
    # Testing the type of an if condition (line 31)
    if_condition_712493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), startswith_call_result_712492)
    # Assigning a type to the variable 'if_condition_712493' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_712493', if_condition_712493)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Evaluating assert statement condition
    
    # Getting the type of 'available' (line 32)
    available_712494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'available')
    int_712495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 28), 'int')
    # Applying the binary operator '>=' (line 32)
    result_ge_712496 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 15), '>=', available_712494, int_712495)
    
    # SSA branch for the else part of an if statement (line 31)
    module_type_store.open_ssa_branch('else')
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 'available' (line 34)
    available_712497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'available')
    # Getting the type of 'None' (line 34)
    None_712498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'None')
    # Applying the binary operator 'is' (line 34)
    result_is__712499 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 15), 'is', available_712497, None_712498)
    
    
    # Getting the type of 'available' (line 34)
    available_712500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'available')
    int_712501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 49), 'int')
    # Applying the binary operator '>=' (line 34)
    result_ge_712502 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 36), '>=', available_712500, int_712501)
    
    # Applying the binary operator 'or' (line 34)
    result_or_keyword_712503 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 15), 'or', result_is__712499, result_ge_712502)
    
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test__mem_available(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test__mem_available' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_712504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712504)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test__mem_available'
    return stypy_return_type_712504

# Assigning a type to the variable 'test__mem_available' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'test__mem_available', test__mem_available)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

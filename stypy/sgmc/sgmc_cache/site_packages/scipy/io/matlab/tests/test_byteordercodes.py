
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Tests for byteorder module '''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: import sys
6: 
7: from numpy.testing import assert_
8: from pytest import raises as assert_raises
9: 
10: import scipy.io.matlab.byteordercodes as sibc
11: 
12: 
13: def test_native():
14:     native_is_le = sys.byteorder == 'little'
15:     assert_(sibc.sys_is_le == native_is_le)
16: 
17: 
18: def test_to_numpy():
19:     if sys.byteorder == 'little':
20:         assert_(sibc.to_numpy_code('native') == '<')
21:         assert_(sibc.to_numpy_code('swapped') == '>')
22:     else:
23:         assert_(sibc.to_numpy_code('native') == '>')
24:         assert_(sibc.to_numpy_code('swapped') == '<')
25:     assert_(sibc.to_numpy_code('native') == sibc.to_numpy_code('='))
26:     assert_(sibc.to_numpy_code('big') == '>')
27:     for code in ('little', '<', 'l', 'L', 'le'):
28:         assert_(sibc.to_numpy_code(code) == '<')
29:     for code in ('big', '>', 'b', 'B', 'be'):
30:         assert_(sibc.to_numpy_code(code) == '>')
31:     assert_raises(ValueError, sibc.to_numpy_code, 'silly string')
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_138017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Tests for byteorder module ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_138018 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_138018) is not StypyTypeError):

    if (import_138018 != 'pyd_module'):
        __import__(import_138018)
        sys_modules_138019 = sys.modules[import_138018]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_138019.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_138019, sys_modules_138019.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_138018)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_138020 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_138020) is not StypyTypeError):

    if (import_138020 != 'pyd_module'):
        __import__(import_138020)
        sys_modules_138021 = sys.modules[import_138020]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_138021.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_138021, sys_modules_138021.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_138020)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy.io.matlab.byteordercodes' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_138022 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.io.matlab.byteordercodes')

if (type(import_138022) is not StypyTypeError):

    if (import_138022 != 'pyd_module'):
        __import__(import_138022)
        sys_modules_138023 = sys.modules[import_138022]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sibc', sys_modules_138023.module_type_store, module_type_store)
    else:
        import scipy.io.matlab.byteordercodes as sibc

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sibc', scipy.io.matlab.byteordercodes, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io.matlab.byteordercodes' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.io.matlab.byteordercodes', import_138022)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


@norecursion
def test_native(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_native'
    module_type_store = module_type_store.open_function_context('test_native', 13, 0, False)
    
    # Passed parameters checking function
    test_native.stypy_localization = localization
    test_native.stypy_type_of_self = None
    test_native.stypy_type_store = module_type_store
    test_native.stypy_function_name = 'test_native'
    test_native.stypy_param_names_list = []
    test_native.stypy_varargs_param_name = None
    test_native.stypy_kwargs_param_name = None
    test_native.stypy_call_defaults = defaults
    test_native.stypy_call_varargs = varargs
    test_native.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_native', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_native', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_native(...)' code ##################

    
    # Assigning a Compare to a Name (line 14):
    
    # Getting the type of 'sys' (line 14)
    sys_138024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'sys')
    # Obtaining the member 'byteorder' of a type (line 14)
    byteorder_138025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 19), sys_138024, 'byteorder')
    str_138026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', 'little')
    # Applying the binary operator '==' (line 14)
    result_eq_138027 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 19), '==', byteorder_138025, str_138026)
    
    # Assigning a type to the variable 'native_is_le' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'native_is_le', result_eq_138027)
    
    # Call to assert_(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Getting the type of 'sibc' (line 15)
    sibc_138029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'sibc', False)
    # Obtaining the member 'sys_is_le' of a type (line 15)
    sys_is_le_138030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 12), sibc_138029, 'sys_is_le')
    # Getting the type of 'native_is_le' (line 15)
    native_is_le_138031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'native_is_le', False)
    # Applying the binary operator '==' (line 15)
    result_eq_138032 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 12), '==', sys_is_le_138030, native_is_le_138031)
    
    # Processing the call keyword arguments (line 15)
    kwargs_138033 = {}
    # Getting the type of 'assert_' (line 15)
    assert__138028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 15)
    assert__call_result_138034 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert__138028, *[result_eq_138032], **kwargs_138033)
    
    
    # ################# End of 'test_native(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_native' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_138035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138035)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_native'
    return stypy_return_type_138035

# Assigning a type to the variable 'test_native' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'test_native', test_native)

@norecursion
def test_to_numpy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_to_numpy'
    module_type_store = module_type_store.open_function_context('test_to_numpy', 18, 0, False)
    
    # Passed parameters checking function
    test_to_numpy.stypy_localization = localization
    test_to_numpy.stypy_type_of_self = None
    test_to_numpy.stypy_type_store = module_type_store
    test_to_numpy.stypy_function_name = 'test_to_numpy'
    test_to_numpy.stypy_param_names_list = []
    test_to_numpy.stypy_varargs_param_name = None
    test_to_numpy.stypy_kwargs_param_name = None
    test_to_numpy.stypy_call_defaults = defaults
    test_to_numpy.stypy_call_varargs = varargs
    test_to_numpy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_to_numpy', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_to_numpy', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_to_numpy(...)' code ##################

    
    
    # Getting the type of 'sys' (line 19)
    sys_138036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'sys')
    # Obtaining the member 'byteorder' of a type (line 19)
    byteorder_138037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 7), sys_138036, 'byteorder')
    str_138038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'str', 'little')
    # Applying the binary operator '==' (line 19)
    result_eq_138039 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 7), '==', byteorder_138037, str_138038)
    
    # Testing the type of an if condition (line 19)
    if_condition_138040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 4), result_eq_138039)
    # Assigning a type to the variable 'if_condition_138040' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'if_condition_138040', if_condition_138040)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    
    # Call to to_numpy_code(...): (line 20)
    # Processing the call arguments (line 20)
    str_138044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'str', 'native')
    # Processing the call keyword arguments (line 20)
    kwargs_138045 = {}
    # Getting the type of 'sibc' (line 20)
    sibc_138042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 20)
    to_numpy_code_138043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), sibc_138042, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 20)
    to_numpy_code_call_result_138046 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), to_numpy_code_138043, *[str_138044], **kwargs_138045)
    
    str_138047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 48), 'str', '<')
    # Applying the binary operator '==' (line 20)
    result_eq_138048 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 16), '==', to_numpy_code_call_result_138046, str_138047)
    
    # Processing the call keyword arguments (line 20)
    kwargs_138049 = {}
    # Getting the type of 'assert_' (line 20)
    assert__138041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_138050 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), assert__138041, *[result_eq_138048], **kwargs_138049)
    
    
    # Call to assert_(...): (line 21)
    # Processing the call arguments (line 21)
    
    
    # Call to to_numpy_code(...): (line 21)
    # Processing the call arguments (line 21)
    str_138054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'str', 'swapped')
    # Processing the call keyword arguments (line 21)
    kwargs_138055 = {}
    # Getting the type of 'sibc' (line 21)
    sibc_138052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 21)
    to_numpy_code_138053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), sibc_138052, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 21)
    to_numpy_code_call_result_138056 = invoke(stypy.reporting.localization.Localization(__file__, 21, 16), to_numpy_code_138053, *[str_138054], **kwargs_138055)
    
    str_138057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 49), 'str', '>')
    # Applying the binary operator '==' (line 21)
    result_eq_138058 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 16), '==', to_numpy_code_call_result_138056, str_138057)
    
    # Processing the call keyword arguments (line 21)
    kwargs_138059 = {}
    # Getting the type of 'assert_' (line 21)
    assert__138051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 21)
    assert__call_result_138060 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert__138051, *[result_eq_138058], **kwargs_138059)
    
    # SSA branch for the else part of an if statement (line 19)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_(...): (line 23)
    # Processing the call arguments (line 23)
    
    
    # Call to to_numpy_code(...): (line 23)
    # Processing the call arguments (line 23)
    str_138064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'str', 'native')
    # Processing the call keyword arguments (line 23)
    kwargs_138065 = {}
    # Getting the type of 'sibc' (line 23)
    sibc_138062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 23)
    to_numpy_code_138063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), sibc_138062, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 23)
    to_numpy_code_call_result_138066 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), to_numpy_code_138063, *[str_138064], **kwargs_138065)
    
    str_138067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 48), 'str', '>')
    # Applying the binary operator '==' (line 23)
    result_eq_138068 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 16), '==', to_numpy_code_call_result_138066, str_138067)
    
    # Processing the call keyword arguments (line 23)
    kwargs_138069 = {}
    # Getting the type of 'assert_' (line 23)
    assert__138061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 23)
    assert__call_result_138070 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assert__138061, *[result_eq_138068], **kwargs_138069)
    
    
    # Call to assert_(...): (line 24)
    # Processing the call arguments (line 24)
    
    
    # Call to to_numpy_code(...): (line 24)
    # Processing the call arguments (line 24)
    str_138074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'str', 'swapped')
    # Processing the call keyword arguments (line 24)
    kwargs_138075 = {}
    # Getting the type of 'sibc' (line 24)
    sibc_138072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 24)
    to_numpy_code_138073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), sibc_138072, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 24)
    to_numpy_code_call_result_138076 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), to_numpy_code_138073, *[str_138074], **kwargs_138075)
    
    str_138077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 49), 'str', '<')
    # Applying the binary operator '==' (line 24)
    result_eq_138078 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 16), '==', to_numpy_code_call_result_138076, str_138077)
    
    # Processing the call keyword arguments (line 24)
    kwargs_138079 = {}
    # Getting the type of 'assert_' (line 24)
    assert__138071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 24)
    assert__call_result_138080 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), assert__138071, *[result_eq_138078], **kwargs_138079)
    
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_(...): (line 25)
    # Processing the call arguments (line 25)
    
    
    # Call to to_numpy_code(...): (line 25)
    # Processing the call arguments (line 25)
    str_138084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'str', 'native')
    # Processing the call keyword arguments (line 25)
    kwargs_138085 = {}
    # Getting the type of 'sibc' (line 25)
    sibc_138082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 25)
    to_numpy_code_138083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), sibc_138082, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 25)
    to_numpy_code_call_result_138086 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), to_numpy_code_138083, *[str_138084], **kwargs_138085)
    
    
    # Call to to_numpy_code(...): (line 25)
    # Processing the call arguments (line 25)
    str_138089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'str', '=')
    # Processing the call keyword arguments (line 25)
    kwargs_138090 = {}
    # Getting the type of 'sibc' (line 25)
    sibc_138087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 44), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 25)
    to_numpy_code_138088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 44), sibc_138087, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 25)
    to_numpy_code_call_result_138091 = invoke(stypy.reporting.localization.Localization(__file__, 25, 44), to_numpy_code_138088, *[str_138089], **kwargs_138090)
    
    # Applying the binary operator '==' (line 25)
    result_eq_138092 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 12), '==', to_numpy_code_call_result_138086, to_numpy_code_call_result_138091)
    
    # Processing the call keyword arguments (line 25)
    kwargs_138093 = {}
    # Getting the type of 'assert_' (line 25)
    assert__138081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 25)
    assert__call_result_138094 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), assert__138081, *[result_eq_138092], **kwargs_138093)
    
    
    # Call to assert_(...): (line 26)
    # Processing the call arguments (line 26)
    
    
    # Call to to_numpy_code(...): (line 26)
    # Processing the call arguments (line 26)
    str_138098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'str', 'big')
    # Processing the call keyword arguments (line 26)
    kwargs_138099 = {}
    # Getting the type of 'sibc' (line 26)
    sibc_138096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 26)
    to_numpy_code_138097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), sibc_138096, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 26)
    to_numpy_code_call_result_138100 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), to_numpy_code_138097, *[str_138098], **kwargs_138099)
    
    str_138101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'str', '>')
    # Applying the binary operator '==' (line 26)
    result_eq_138102 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 12), '==', to_numpy_code_call_result_138100, str_138101)
    
    # Processing the call keyword arguments (line 26)
    kwargs_138103 = {}
    # Getting the type of 'assert_' (line 26)
    assert__138095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 26)
    assert__call_result_138104 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert__138095, *[result_eq_138102], **kwargs_138103)
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_138105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    str_138106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'str', 'little')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 17), tuple_138105, str_138106)
    # Adding element type (line 27)
    str_138107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'str', '<')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 17), tuple_138105, str_138107)
    # Adding element type (line 27)
    str_138108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 32), 'str', 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 17), tuple_138105, str_138108)
    # Adding element type (line 27)
    str_138109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'str', 'L')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 17), tuple_138105, str_138109)
    # Adding element type (line 27)
    str_138110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'str', 'le')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 17), tuple_138105, str_138110)
    
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 4), tuple_138105)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_138111 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 4), tuple_138105)
    # Assigning a type to the variable 'code' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'code', for_loop_var_138111)
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 28)
    # Processing the call arguments (line 28)
    
    
    # Call to to_numpy_code(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'code' (line 28)
    code_138115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'code', False)
    # Processing the call keyword arguments (line 28)
    kwargs_138116 = {}
    # Getting the type of 'sibc' (line 28)
    sibc_138113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 28)
    to_numpy_code_138114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), sibc_138113, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 28)
    to_numpy_code_call_result_138117 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), to_numpy_code_138114, *[code_138115], **kwargs_138116)
    
    str_138118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'str', '<')
    # Applying the binary operator '==' (line 28)
    result_eq_138119 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 16), '==', to_numpy_code_call_result_138117, str_138118)
    
    # Processing the call keyword arguments (line 28)
    kwargs_138120 = {}
    # Getting the type of 'assert_' (line 28)
    assert__138112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 28)
    assert__call_result_138121 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert__138112, *[result_eq_138119], **kwargs_138120)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_138122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    str_138123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'str', 'big')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_138122, str_138123)
    # Adding element type (line 29)
    str_138124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'str', '>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_138122, str_138124)
    # Adding element type (line 29)
    str_138125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'str', 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_138122, str_138125)
    # Adding element type (line 29)
    str_138126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_138122, str_138126)
    # Adding element type (line 29)
    str_138127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 39), 'str', 'be')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_138122, str_138127)
    
    # Testing the type of a for loop iterable (line 29)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 4), tuple_138122)
    # Getting the type of the for loop variable (line 29)
    for_loop_var_138128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 4), tuple_138122)
    # Assigning a type to the variable 'code' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'code', for_loop_var_138128)
    # SSA begins for a for statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 30)
    # Processing the call arguments (line 30)
    
    
    # Call to to_numpy_code(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'code' (line 30)
    code_138132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'code', False)
    # Processing the call keyword arguments (line 30)
    kwargs_138133 = {}
    # Getting the type of 'sibc' (line 30)
    sibc_138130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 30)
    to_numpy_code_138131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), sibc_138130, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 30)
    to_numpy_code_call_result_138134 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), to_numpy_code_138131, *[code_138132], **kwargs_138133)
    
    str_138135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 44), 'str', '>')
    # Applying the binary operator '==' (line 30)
    result_eq_138136 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 16), '==', to_numpy_code_call_result_138134, str_138135)
    
    # Processing the call keyword arguments (line 30)
    kwargs_138137 = {}
    # Getting the type of 'assert_' (line 30)
    assert__138129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 30)
    assert__call_result_138138 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert__138129, *[result_eq_138136], **kwargs_138137)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_raises(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'ValueError' (line 31)
    ValueError_138140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'ValueError', False)
    # Getting the type of 'sibc' (line 31)
    sibc_138141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'sibc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 31)
    to_numpy_code_138142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), sibc_138141, 'to_numpy_code')
    str_138143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 50), 'str', 'silly string')
    # Processing the call keyword arguments (line 31)
    kwargs_138144 = {}
    # Getting the type of 'assert_raises' (line 31)
    assert_raises_138139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 31)
    assert_raises_call_result_138145 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_raises_138139, *[ValueError_138140, to_numpy_code_138142, str_138143], **kwargs_138144)
    
    
    # ################# End of 'test_to_numpy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_to_numpy' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_138146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138146)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_to_numpy'
    return stypy_return_type_138146

# Assigning a type to the variable 'test_to_numpy' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_to_numpy', test_to_numpy)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

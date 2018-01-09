
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Testing
2: 
3: '''
4: 
5: from __future__ import division, print_function, absolute_import
6: 
7: import numpy as np
8: 
9: from numpy.testing import assert_array_equal, assert_array_almost_equal, \
10:      assert_
11: 
12: from scipy.io.matlab.mio_utils import squeeze_element, chars_to_strings
13: 
14: 
15: def test_squeeze_element():
16:     a = np.zeros((1,3))
17:     assert_array_equal(np.squeeze(a), squeeze_element(a))
18:     # 0d output from squeeze gives scalar
19:     sq_int = squeeze_element(np.zeros((1,1), dtype=float))
20:     assert_(isinstance(sq_int, float))
21:     # Unless it's a structured array
22:     sq_sa = squeeze_element(np.zeros((1,1),dtype=[('f1', 'f')]))
23:     assert_(isinstance(sq_sa, np.ndarray))
24: 
25: 
26: def test_chars_strings():
27:     # chars as strings
28:     strings = ['learn ', 'python', 'fast  ', 'here  ']
29:     str_arr = np.array(strings, dtype='U6')  # shape (4,)
30:     chars = [list(s) for s in strings]
31:     char_arr = np.array(chars, dtype='U1')  # shape (4,6)
32:     assert_array_equal(chars_to_strings(char_arr), str_arr)
33:     ca2d = char_arr.reshape((2,2,6))
34:     sa2d = str_arr.reshape((2,2))
35:     assert_array_equal(chars_to_strings(ca2d), sa2d)
36:     ca3d = char_arr.reshape((1,2,2,6))
37:     sa3d = str_arr.reshape((1,2,2))
38:     assert_array_equal(chars_to_strings(ca3d), sa3d)
39:     # Fortran ordered arrays
40:     char_arrf = np.array(chars, dtype='U1', order='F')  # shape (4,6)
41:     assert_array_equal(chars_to_strings(char_arrf), str_arr)
42:     # empty array
43:     arr = np.array([['']], dtype='U1')
44:     out_arr = np.array([''], dtype='U1')
45:     assert_array_equal(chars_to_strings(arr), out_arr)
46: 
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_144382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Testing\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144383 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_144383) is not StypyTypeError):

    if (import_144383 != 'pyd_module'):
        __import__(import_144383)
        sys_modules_144384 = sys.modules[import_144383]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_144384.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_144383)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144385 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_144385) is not StypyTypeError):

    if (import_144385 != 'pyd_module'):
        __import__(import_144385)
        sys_modules_144386 = sys.modules[import_144385]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_144386.module_type_store, module_type_store, ['assert_array_equal', 'assert_array_almost_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_144386, sys_modules_144386.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal', 'assert_array_almost_equal', 'assert_'], [assert_array_equal, assert_array_almost_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_144385)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.io.matlab.mio_utils import squeeze_element, chars_to_strings' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144387 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio_utils')

if (type(import_144387) is not StypyTypeError):

    if (import_144387 != 'pyd_module'):
        __import__(import_144387)
        sys_modules_144388 = sys.modules[import_144387]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio_utils', sys_modules_144388.module_type_store, module_type_store, ['squeeze_element', 'chars_to_strings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_144388, sys_modules_144388.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio_utils import squeeze_element, chars_to_strings

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio_utils', None, module_type_store, ['squeeze_element', 'chars_to_strings'], [squeeze_element, chars_to_strings])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio_utils' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio_utils', import_144387)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


@norecursion
def test_squeeze_element(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_squeeze_element'
    module_type_store = module_type_store.open_function_context('test_squeeze_element', 15, 0, False)
    
    # Passed parameters checking function
    test_squeeze_element.stypy_localization = localization
    test_squeeze_element.stypy_type_of_self = None
    test_squeeze_element.stypy_type_store = module_type_store
    test_squeeze_element.stypy_function_name = 'test_squeeze_element'
    test_squeeze_element.stypy_param_names_list = []
    test_squeeze_element.stypy_varargs_param_name = None
    test_squeeze_element.stypy_kwargs_param_name = None
    test_squeeze_element.stypy_call_defaults = defaults
    test_squeeze_element.stypy_call_varargs = varargs
    test_squeeze_element.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_squeeze_element', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_squeeze_element', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_squeeze_element(...)' code ##################

    
    # Assigning a Call to a Name (line 16):
    
    # Call to zeros(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_144391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    int_144392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 18), tuple_144391, int_144392)
    # Adding element type (line 16)
    int_144393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 18), tuple_144391, int_144393)
    
    # Processing the call keyword arguments (line 16)
    kwargs_144394 = {}
    # Getting the type of 'np' (line 16)
    np_144389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 16)
    zeros_144390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), np_144389, 'zeros')
    # Calling zeros(args, kwargs) (line 16)
    zeros_call_result_144395 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), zeros_144390, *[tuple_144391], **kwargs_144394)
    
    # Assigning a type to the variable 'a' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a', zeros_call_result_144395)
    
    # Call to assert_array_equal(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to squeeze(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'a' (line 17)
    a_144399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 34), 'a', False)
    # Processing the call keyword arguments (line 17)
    kwargs_144400 = {}
    # Getting the type of 'np' (line 17)
    np_144397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'np', False)
    # Obtaining the member 'squeeze' of a type (line 17)
    squeeze_144398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 23), np_144397, 'squeeze')
    # Calling squeeze(args, kwargs) (line 17)
    squeeze_call_result_144401 = invoke(stypy.reporting.localization.Localization(__file__, 17, 23), squeeze_144398, *[a_144399], **kwargs_144400)
    
    
    # Call to squeeze_element(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'a' (line 17)
    a_144403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 54), 'a', False)
    # Processing the call keyword arguments (line 17)
    kwargs_144404 = {}
    # Getting the type of 'squeeze_element' (line 17)
    squeeze_element_144402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 38), 'squeeze_element', False)
    # Calling squeeze_element(args, kwargs) (line 17)
    squeeze_element_call_result_144405 = invoke(stypy.reporting.localization.Localization(__file__, 17, 38), squeeze_element_144402, *[a_144403], **kwargs_144404)
    
    # Processing the call keyword arguments (line 17)
    kwargs_144406 = {}
    # Getting the type of 'assert_array_equal' (line 17)
    assert_array_equal_144396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 17)
    assert_array_equal_call_result_144407 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_array_equal_144396, *[squeeze_call_result_144401, squeeze_element_call_result_144405], **kwargs_144406)
    
    
    # Assigning a Call to a Name (line 19):
    
    # Call to squeeze_element(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to zeros(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining an instance of the builtin type 'tuple' (line 19)
    tuple_144411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 19)
    # Adding element type (line 19)
    int_144412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 39), tuple_144411, int_144412)
    # Adding element type (line 19)
    int_144413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 39), tuple_144411, int_144413)
    
    # Processing the call keyword arguments (line 19)
    # Getting the type of 'float' (line 19)
    float_144414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 51), 'float', False)
    keyword_144415 = float_144414
    kwargs_144416 = {'dtype': keyword_144415}
    # Getting the type of 'np' (line 19)
    np_144409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'np', False)
    # Obtaining the member 'zeros' of a type (line 19)
    zeros_144410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 29), np_144409, 'zeros')
    # Calling zeros(args, kwargs) (line 19)
    zeros_call_result_144417 = invoke(stypy.reporting.localization.Localization(__file__, 19, 29), zeros_144410, *[tuple_144411], **kwargs_144416)
    
    # Processing the call keyword arguments (line 19)
    kwargs_144418 = {}
    # Getting the type of 'squeeze_element' (line 19)
    squeeze_element_144408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'squeeze_element', False)
    # Calling squeeze_element(args, kwargs) (line 19)
    squeeze_element_call_result_144419 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), squeeze_element_144408, *[zeros_call_result_144417], **kwargs_144418)
    
    # Assigning a type to the variable 'sq_int' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'sq_int', squeeze_element_call_result_144419)
    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to isinstance(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'sq_int' (line 20)
    sq_int_144422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'sq_int', False)
    # Getting the type of 'float' (line 20)
    float_144423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 31), 'float', False)
    # Processing the call keyword arguments (line 20)
    kwargs_144424 = {}
    # Getting the type of 'isinstance' (line 20)
    isinstance_144421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 20)
    isinstance_call_result_144425 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), isinstance_144421, *[sq_int_144422, float_144423], **kwargs_144424)
    
    # Processing the call keyword arguments (line 20)
    kwargs_144426 = {}
    # Getting the type of 'assert_' (line 20)
    assert__144420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_144427 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert__144420, *[isinstance_call_result_144425], **kwargs_144426)
    
    
    # Assigning a Call to a Name (line 22):
    
    # Call to squeeze_element(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to zeros(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_144431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    int_144432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 38), tuple_144431, int_144432)
    # Adding element type (line 22)
    int_144433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 38), tuple_144431, int_144433)
    
    # Processing the call keyword arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_144434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_144435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    str_144436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 51), 'str', 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 51), tuple_144435, str_144436)
    # Adding element type (line 22)
    str_144437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 57), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 51), tuple_144435, str_144437)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 49), list_144434, tuple_144435)
    
    keyword_144438 = list_144434
    kwargs_144439 = {'dtype': keyword_144438}
    # Getting the type of 'np' (line 22)
    np_144429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'np', False)
    # Obtaining the member 'zeros' of a type (line 22)
    zeros_144430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 28), np_144429, 'zeros')
    # Calling zeros(args, kwargs) (line 22)
    zeros_call_result_144440 = invoke(stypy.reporting.localization.Localization(__file__, 22, 28), zeros_144430, *[tuple_144431], **kwargs_144439)
    
    # Processing the call keyword arguments (line 22)
    kwargs_144441 = {}
    # Getting the type of 'squeeze_element' (line 22)
    squeeze_element_144428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'squeeze_element', False)
    # Calling squeeze_element(args, kwargs) (line 22)
    squeeze_element_call_result_144442 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), squeeze_element_144428, *[zeros_call_result_144440], **kwargs_144441)
    
    # Assigning a type to the variable 'sq_sa' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'sq_sa', squeeze_element_call_result_144442)
    
    # Call to assert_(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to isinstance(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'sq_sa' (line 23)
    sq_sa_144445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'sq_sa', False)
    # Getting the type of 'np' (line 23)
    np_144446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 23)
    ndarray_144447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 30), np_144446, 'ndarray')
    # Processing the call keyword arguments (line 23)
    kwargs_144448 = {}
    # Getting the type of 'isinstance' (line 23)
    isinstance_144444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 23)
    isinstance_call_result_144449 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), isinstance_144444, *[sq_sa_144445, ndarray_144447], **kwargs_144448)
    
    # Processing the call keyword arguments (line 23)
    kwargs_144450 = {}
    # Getting the type of 'assert_' (line 23)
    assert__144443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 23)
    assert__call_result_144451 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert__144443, *[isinstance_call_result_144449], **kwargs_144450)
    
    
    # ################# End of 'test_squeeze_element(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_squeeze_element' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_144452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144452)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_squeeze_element'
    return stypy_return_type_144452

# Assigning a type to the variable 'test_squeeze_element' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_squeeze_element', test_squeeze_element)

@norecursion
def test_chars_strings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_chars_strings'
    module_type_store = module_type_store.open_function_context('test_chars_strings', 26, 0, False)
    
    # Passed parameters checking function
    test_chars_strings.stypy_localization = localization
    test_chars_strings.stypy_type_of_self = None
    test_chars_strings.stypy_type_store = module_type_store
    test_chars_strings.stypy_function_name = 'test_chars_strings'
    test_chars_strings.stypy_param_names_list = []
    test_chars_strings.stypy_varargs_param_name = None
    test_chars_strings.stypy_kwargs_param_name = None
    test_chars_strings.stypy_call_defaults = defaults
    test_chars_strings.stypy_call_varargs = varargs
    test_chars_strings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_chars_strings', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_chars_strings', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_chars_strings(...)' code ##################

    
    # Assigning a List to a Name (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_144453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    str_144454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'str', 'learn ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_144453, str_144454)
    # Adding element type (line 28)
    str_144455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', 'python')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_144453, str_144455)
    # Adding element type (line 28)
    str_144456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'str', 'fast  ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_144453, str_144456)
    # Adding element type (line 28)
    str_144457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'str', 'here  ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_144453, str_144457)
    
    # Assigning a type to the variable 'strings' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'strings', list_144453)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to array(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'strings' (line 29)
    strings_144460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'strings', False)
    # Processing the call keyword arguments (line 29)
    str_144461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'str', 'U6')
    keyword_144462 = str_144461
    kwargs_144463 = {'dtype': keyword_144462}
    # Getting the type of 'np' (line 29)
    np_144458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 29)
    array_144459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 14), np_144458, 'array')
    # Calling array(args, kwargs) (line 29)
    array_call_result_144464 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), array_144459, *[strings_144460], **kwargs_144463)
    
    # Assigning a type to the variable 'str_arr' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'str_arr', array_call_result_144464)
    
    # Assigning a ListComp to a Name (line 30):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'strings' (line 30)
    strings_144469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'strings')
    comprehension_144470 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), strings_144469)
    # Assigning a type to the variable 's' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 's', comprehension_144470)
    
    # Call to list(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 's' (line 30)
    s_144466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 's', False)
    # Processing the call keyword arguments (line 30)
    kwargs_144467 = {}
    # Getting the type of 'list' (line 30)
    list_144465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'list', False)
    # Calling list(args, kwargs) (line 30)
    list_call_result_144468 = invoke(stypy.reporting.localization.Localization(__file__, 30, 13), list_144465, *[s_144466], **kwargs_144467)
    
    list_144471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), list_144471, list_call_result_144468)
    # Assigning a type to the variable 'chars' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'chars', list_144471)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to array(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'chars' (line 31)
    chars_144474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'chars', False)
    # Processing the call keyword arguments (line 31)
    str_144475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'str', 'U1')
    keyword_144476 = str_144475
    kwargs_144477 = {'dtype': keyword_144476}
    # Getting the type of 'np' (line 31)
    np_144472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 31)
    array_144473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), np_144472, 'array')
    # Calling array(args, kwargs) (line 31)
    array_call_result_144478 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), array_144473, *[chars_144474], **kwargs_144477)
    
    # Assigning a type to the variable 'char_arr' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'char_arr', array_call_result_144478)
    
    # Call to assert_array_equal(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to chars_to_strings(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'char_arr' (line 32)
    char_arr_144481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'char_arr', False)
    # Processing the call keyword arguments (line 32)
    kwargs_144482 = {}
    # Getting the type of 'chars_to_strings' (line 32)
    chars_to_strings_144480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'chars_to_strings', False)
    # Calling chars_to_strings(args, kwargs) (line 32)
    chars_to_strings_call_result_144483 = invoke(stypy.reporting.localization.Localization(__file__, 32, 23), chars_to_strings_144480, *[char_arr_144481], **kwargs_144482)
    
    # Getting the type of 'str_arr' (line 32)
    str_arr_144484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 51), 'str_arr', False)
    # Processing the call keyword arguments (line 32)
    kwargs_144485 = {}
    # Getting the type of 'assert_array_equal' (line 32)
    assert_array_equal_144479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 32)
    assert_array_equal_call_result_144486 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_array_equal_144479, *[chars_to_strings_call_result_144483, str_arr_144484], **kwargs_144485)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Call to reshape(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_144489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    int_144490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 29), tuple_144489, int_144490)
    # Adding element type (line 33)
    int_144491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 29), tuple_144489, int_144491)
    # Adding element type (line 33)
    int_144492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 29), tuple_144489, int_144492)
    
    # Processing the call keyword arguments (line 33)
    kwargs_144493 = {}
    # Getting the type of 'char_arr' (line 33)
    char_arr_144487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'char_arr', False)
    # Obtaining the member 'reshape' of a type (line 33)
    reshape_144488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), char_arr_144487, 'reshape')
    # Calling reshape(args, kwargs) (line 33)
    reshape_call_result_144494 = invoke(stypy.reporting.localization.Localization(__file__, 33, 11), reshape_144488, *[tuple_144489], **kwargs_144493)
    
    # Assigning a type to the variable 'ca2d' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'ca2d', reshape_call_result_144494)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to reshape(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_144497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    int_144498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 28), tuple_144497, int_144498)
    # Adding element type (line 34)
    int_144499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 28), tuple_144497, int_144499)
    
    # Processing the call keyword arguments (line 34)
    kwargs_144500 = {}
    # Getting the type of 'str_arr' (line 34)
    str_arr_144495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'str_arr', False)
    # Obtaining the member 'reshape' of a type (line 34)
    reshape_144496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), str_arr_144495, 'reshape')
    # Calling reshape(args, kwargs) (line 34)
    reshape_call_result_144501 = invoke(stypy.reporting.localization.Localization(__file__, 34, 11), reshape_144496, *[tuple_144497], **kwargs_144500)
    
    # Assigning a type to the variable 'sa2d' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'sa2d', reshape_call_result_144501)
    
    # Call to assert_array_equal(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to chars_to_strings(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'ca2d' (line 35)
    ca2d_144504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'ca2d', False)
    # Processing the call keyword arguments (line 35)
    kwargs_144505 = {}
    # Getting the type of 'chars_to_strings' (line 35)
    chars_to_strings_144503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'chars_to_strings', False)
    # Calling chars_to_strings(args, kwargs) (line 35)
    chars_to_strings_call_result_144506 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), chars_to_strings_144503, *[ca2d_144504], **kwargs_144505)
    
    # Getting the type of 'sa2d' (line 35)
    sa2d_144507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 47), 'sa2d', False)
    # Processing the call keyword arguments (line 35)
    kwargs_144508 = {}
    # Getting the type of 'assert_array_equal' (line 35)
    assert_array_equal_144502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 35)
    assert_array_equal_call_result_144509 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert_array_equal_144502, *[chars_to_strings_call_result_144506, sa2d_144507], **kwargs_144508)
    
    
    # Assigning a Call to a Name (line 36):
    
    # Call to reshape(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_144512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    int_144513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_144512, int_144513)
    # Adding element type (line 36)
    int_144514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_144512, int_144514)
    # Adding element type (line 36)
    int_144515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_144512, int_144515)
    # Adding element type (line 36)
    int_144516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_144512, int_144516)
    
    # Processing the call keyword arguments (line 36)
    kwargs_144517 = {}
    # Getting the type of 'char_arr' (line 36)
    char_arr_144510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'char_arr', False)
    # Obtaining the member 'reshape' of a type (line 36)
    reshape_144511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), char_arr_144510, 'reshape')
    # Calling reshape(args, kwargs) (line 36)
    reshape_call_result_144518 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), reshape_144511, *[tuple_144512], **kwargs_144517)
    
    # Assigning a type to the variable 'ca3d' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'ca3d', reshape_call_result_144518)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to reshape(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_144521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    int_144522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 28), tuple_144521, int_144522)
    # Adding element type (line 37)
    int_144523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 28), tuple_144521, int_144523)
    # Adding element type (line 37)
    int_144524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 28), tuple_144521, int_144524)
    
    # Processing the call keyword arguments (line 37)
    kwargs_144525 = {}
    # Getting the type of 'str_arr' (line 37)
    str_arr_144519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'str_arr', False)
    # Obtaining the member 'reshape' of a type (line 37)
    reshape_144520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 11), str_arr_144519, 'reshape')
    # Calling reshape(args, kwargs) (line 37)
    reshape_call_result_144526 = invoke(stypy.reporting.localization.Localization(__file__, 37, 11), reshape_144520, *[tuple_144521], **kwargs_144525)
    
    # Assigning a type to the variable 'sa3d' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'sa3d', reshape_call_result_144526)
    
    # Call to assert_array_equal(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Call to chars_to_strings(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'ca3d' (line 38)
    ca3d_144529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'ca3d', False)
    # Processing the call keyword arguments (line 38)
    kwargs_144530 = {}
    # Getting the type of 'chars_to_strings' (line 38)
    chars_to_strings_144528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'chars_to_strings', False)
    # Calling chars_to_strings(args, kwargs) (line 38)
    chars_to_strings_call_result_144531 = invoke(stypy.reporting.localization.Localization(__file__, 38, 23), chars_to_strings_144528, *[ca3d_144529], **kwargs_144530)
    
    # Getting the type of 'sa3d' (line 38)
    sa3d_144532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 47), 'sa3d', False)
    # Processing the call keyword arguments (line 38)
    kwargs_144533 = {}
    # Getting the type of 'assert_array_equal' (line 38)
    assert_array_equal_144527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 38)
    assert_array_equal_call_result_144534 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), assert_array_equal_144527, *[chars_to_strings_call_result_144531, sa3d_144532], **kwargs_144533)
    
    
    # Assigning a Call to a Name (line 40):
    
    # Call to array(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'chars' (line 40)
    chars_144537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'chars', False)
    # Processing the call keyword arguments (line 40)
    str_144538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'str', 'U1')
    keyword_144539 = str_144538
    str_144540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'str', 'F')
    keyword_144541 = str_144540
    kwargs_144542 = {'dtype': keyword_144539, 'order': keyword_144541}
    # Getting the type of 'np' (line 40)
    np_144535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 40)
    array_144536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), np_144535, 'array')
    # Calling array(args, kwargs) (line 40)
    array_call_result_144543 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), array_144536, *[chars_144537], **kwargs_144542)
    
    # Assigning a type to the variable 'char_arrf' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'char_arrf', array_call_result_144543)
    
    # Call to assert_array_equal(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to chars_to_strings(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'char_arrf' (line 41)
    char_arrf_144546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'char_arrf', False)
    # Processing the call keyword arguments (line 41)
    kwargs_144547 = {}
    # Getting the type of 'chars_to_strings' (line 41)
    chars_to_strings_144545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'chars_to_strings', False)
    # Calling chars_to_strings(args, kwargs) (line 41)
    chars_to_strings_call_result_144548 = invoke(stypy.reporting.localization.Localization(__file__, 41, 23), chars_to_strings_144545, *[char_arrf_144546], **kwargs_144547)
    
    # Getting the type of 'str_arr' (line 41)
    str_arr_144549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 52), 'str_arr', False)
    # Processing the call keyword arguments (line 41)
    kwargs_144550 = {}
    # Getting the type of 'assert_array_equal' (line 41)
    assert_array_equal_144544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 41)
    assert_array_equal_call_result_144551 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert_array_equal_144544, *[chars_to_strings_call_result_144548, str_arr_144549], **kwargs_144550)
    
    
    # Assigning a Call to a Name (line 43):
    
    # Call to array(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_144554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_144555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    str_144556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), list_144555, str_144556)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_144554, list_144555)
    
    # Processing the call keyword arguments (line 43)
    str_144557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'str', 'U1')
    keyword_144558 = str_144557
    kwargs_144559 = {'dtype': keyword_144558}
    # Getting the type of 'np' (line 43)
    np_144552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 43)
    array_144553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), np_144552, 'array')
    # Calling array(args, kwargs) (line 43)
    array_call_result_144560 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), array_144553, *[list_144554], **kwargs_144559)
    
    # Assigning a type to the variable 'arr' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'arr', array_call_result_144560)
    
    # Assigning a Call to a Name (line 44):
    
    # Call to array(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_144563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    str_144564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_144563, str_144564)
    
    # Processing the call keyword arguments (line 44)
    str_144565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'str', 'U1')
    keyword_144566 = str_144565
    kwargs_144567 = {'dtype': keyword_144566}
    # Getting the type of 'np' (line 44)
    np_144561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 44)
    array_144562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 14), np_144561, 'array')
    # Calling array(args, kwargs) (line 44)
    array_call_result_144568 = invoke(stypy.reporting.localization.Localization(__file__, 44, 14), array_144562, *[list_144563], **kwargs_144567)
    
    # Assigning a type to the variable 'out_arr' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'out_arr', array_call_result_144568)
    
    # Call to assert_array_equal(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to chars_to_strings(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'arr' (line 45)
    arr_144571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'arr', False)
    # Processing the call keyword arguments (line 45)
    kwargs_144572 = {}
    # Getting the type of 'chars_to_strings' (line 45)
    chars_to_strings_144570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'chars_to_strings', False)
    # Calling chars_to_strings(args, kwargs) (line 45)
    chars_to_strings_call_result_144573 = invoke(stypy.reporting.localization.Localization(__file__, 45, 23), chars_to_strings_144570, *[arr_144571], **kwargs_144572)
    
    # Getting the type of 'out_arr' (line 45)
    out_arr_144574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 46), 'out_arr', False)
    # Processing the call keyword arguments (line 45)
    kwargs_144575 = {}
    # Getting the type of 'assert_array_equal' (line 45)
    assert_array_equal_144569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 45)
    assert_array_equal_call_result_144576 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), assert_array_equal_144569, *[chars_to_strings_call_result_144573, out_arr_144574], **kwargs_144575)
    
    
    # ################# End of 'test_chars_strings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_chars_strings' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_144577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144577)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_chars_strings'
    return stypy_return_type_144577

# Assigning a type to the variable 'test_chars_strings' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'test_chars_strings', test_chars_strings)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from scipy.constants import constants, codata, find, value
4: from numpy.testing import (assert_equal, assert_,
5:                            assert_almost_equal)
6: 
7: 
8: def test_find():
9:     keys = find('weak mixing', disp=False)
10:     assert_equal(keys, ['weak mixing angle'])
11: 
12:     keys = find('qwertyuiop', disp=False)
13:     assert_equal(keys, [])
14: 
15:     keys = find('natural unit', disp=False)
16:     assert_equal(keys, sorted(['natural unit of velocity',
17:                                 'natural unit of action',
18:                                 'natural unit of action in eV s',
19:                                 'natural unit of mass',
20:                                 'natural unit of energy',
21:                                 'natural unit of energy in MeV',
22:                                 'natural unit of mom.um',
23:                                 'natural unit of mom.um in MeV/c',
24:                                 'natural unit of length',
25:                                 'natural unit of time']))
26: 
27: 
28: def test_basic_table_parse():
29:     c = 'speed of light in vacuum'
30:     assert_equal(codata.value(c), constants.c)
31:     assert_equal(codata.value(c), constants.speed_of_light)
32: 
33: 
34: def test_basic_lookup():
35:     assert_equal('%d %s' % (codata.c, codata.unit('speed of light in vacuum')),
36:                  '299792458 m s^-1')
37: 
38: 
39: def test_find_all():
40:     assert_(len(codata.find(disp=False)) > 300)
41: 
42: 
43: def test_find_single():
44:     assert_equal(codata.find('Wien freq', disp=False)[0],
45:                  'Wien frequency displacement law constant')
46: 
47: 
48: def test_2002_vs_2006():
49:     assert_almost_equal(codata.value('magn. flux quantum'),
50:                         codata.value('mag. flux quantum'))
51: 
52: 
53: def test_exact_values():
54:     # Check that updating stored values with exact ones worked.
55:     for key in codata.exact_values:
56:         assert_((codata.exact_values[key][0] - value(key)) / value(key) == 0)
57: 
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.constants import constants, codata, find, value' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/tests/')
import_14492 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.constants')

if (type(import_14492) is not StypyTypeError):

    if (import_14492 != 'pyd_module'):
        __import__(import_14492)
        sys_modules_14493 = sys.modules[import_14492]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.constants', sys_modules_14493.module_type_store, module_type_store, ['constants', 'codata', 'find', 'value'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_14493, sys_modules_14493.module_type_store, module_type_store)
    else:
        from scipy.constants import constants, codata, find, value

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.constants', None, module_type_store, ['constants', 'codata', 'find', 'value'], [constants, codata, find, value])

else:
    # Assigning a type to the variable 'scipy.constants' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.constants', import_14492)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_, assert_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/tests/')
import_14494 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_14494) is not StypyTypeError):

    if (import_14494 != 'pyd_module'):
        __import__(import_14494)
        sys_modules_14495 = sys.modules[import_14494]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_14495.module_type_store, module_type_store, ['assert_equal', 'assert_', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_14495, sys_modules_14495.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_', 'assert_almost_equal'], [assert_equal, assert_, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_14494)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/tests/')


@norecursion
def test_find(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_find'
    module_type_store = module_type_store.open_function_context('test_find', 8, 0, False)
    
    # Passed parameters checking function
    test_find.stypy_localization = localization
    test_find.stypy_type_of_self = None
    test_find.stypy_type_store = module_type_store
    test_find.stypy_function_name = 'test_find'
    test_find.stypy_param_names_list = []
    test_find.stypy_varargs_param_name = None
    test_find.stypy_kwargs_param_name = None
    test_find.stypy_call_defaults = defaults
    test_find.stypy_call_varargs = varargs
    test_find.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_find', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_find', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_find(...)' code ##################

    
    # Assigning a Call to a Name (line 9):
    
    # Call to find(...): (line 9)
    # Processing the call arguments (line 9)
    str_14497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'str', 'weak mixing')
    # Processing the call keyword arguments (line 9)
    # Getting the type of 'False' (line 9)
    False_14498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 36), 'False', False)
    keyword_14499 = False_14498
    kwargs_14500 = {'disp': keyword_14499}
    # Getting the type of 'find' (line 9)
    find_14496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'find', False)
    # Calling find(args, kwargs) (line 9)
    find_call_result_14501 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), find_14496, *[str_14497], **kwargs_14500)
    
    # Assigning a type to the variable 'keys' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'keys', find_call_result_14501)
    
    # Call to assert_equal(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'keys' (line 10)
    keys_14503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'keys', False)
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_14504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    str_14505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'weak mixing angle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_14504, str_14505)
    
    # Processing the call keyword arguments (line 10)
    kwargs_14506 = {}
    # Getting the type of 'assert_equal' (line 10)
    assert_equal_14502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 10)
    assert_equal_call_result_14507 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), assert_equal_14502, *[keys_14503, list_14504], **kwargs_14506)
    
    
    # Assigning a Call to a Name (line 12):
    
    # Call to find(...): (line 12)
    # Processing the call arguments (line 12)
    str_14509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'str', 'qwertyuiop')
    # Processing the call keyword arguments (line 12)
    # Getting the type of 'False' (line 12)
    False_14510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 35), 'False', False)
    keyword_14511 = False_14510
    kwargs_14512 = {'disp': keyword_14511}
    # Getting the type of 'find' (line 12)
    find_14508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'find', False)
    # Calling find(args, kwargs) (line 12)
    find_call_result_14513 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), find_14508, *[str_14509], **kwargs_14512)
    
    # Assigning a type to the variable 'keys' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'keys', find_call_result_14513)
    
    # Call to assert_equal(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'keys' (line 13)
    keys_14515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'keys', False)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_14516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    
    # Processing the call keyword arguments (line 13)
    kwargs_14517 = {}
    # Getting the type of 'assert_equal' (line 13)
    assert_equal_14514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 13)
    assert_equal_call_result_14518 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), assert_equal_14514, *[keys_14515, list_14516], **kwargs_14517)
    
    
    # Assigning a Call to a Name (line 15):
    
    # Call to find(...): (line 15)
    # Processing the call arguments (line 15)
    str_14520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'str', 'natural unit')
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'False' (line 15)
    False_14521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 37), 'False', False)
    keyword_14522 = False_14521
    kwargs_14523 = {'disp': keyword_14522}
    # Getting the type of 'find' (line 15)
    find_14519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'find', False)
    # Calling find(args, kwargs) (line 15)
    find_call_result_14524 = invoke(stypy.reporting.localization.Localization(__file__, 15, 11), find_14519, *[str_14520], **kwargs_14523)
    
    # Assigning a type to the variable 'keys' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'keys', find_call_result_14524)
    
    # Call to assert_equal(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'keys' (line 16)
    keys_14526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'keys', False)
    
    # Call to sorted(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_14528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    str_14529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'str', 'natural unit of velocity')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14529)
    # Adding element type (line 16)
    str_14530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'str', 'natural unit of action')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14530)
    # Adding element type (line 16)
    str_14531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'str', 'natural unit of action in eV s')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14531)
    # Adding element type (line 16)
    str_14532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'str', 'natural unit of mass')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14532)
    # Adding element type (line 16)
    str_14533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'str', 'natural unit of energy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14533)
    # Adding element type (line 16)
    str_14534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'str', 'natural unit of energy in MeV')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14534)
    # Adding element type (line 16)
    str_14535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', 'natural unit of mom.um')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14535)
    # Adding element type (line 16)
    str_14536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', 'natural unit of mom.um in MeV/c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14536)
    # Adding element type (line 16)
    str_14537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 32), 'str', 'natural unit of length')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14537)
    # Adding element type (line 16)
    str_14538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'str', 'natural unit of time')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_14528, str_14538)
    
    # Processing the call keyword arguments (line 16)
    kwargs_14539 = {}
    # Getting the type of 'sorted' (line 16)
    sorted_14527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'sorted', False)
    # Calling sorted(args, kwargs) (line 16)
    sorted_call_result_14540 = invoke(stypy.reporting.localization.Localization(__file__, 16, 23), sorted_14527, *[list_14528], **kwargs_14539)
    
    # Processing the call keyword arguments (line 16)
    kwargs_14541 = {}
    # Getting the type of 'assert_equal' (line 16)
    assert_equal_14525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 16)
    assert_equal_call_result_14542 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), assert_equal_14525, *[keys_14526, sorted_call_result_14540], **kwargs_14541)
    
    
    # ################# End of 'test_find(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_find' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_14543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14543)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_find'
    return stypy_return_type_14543

# Assigning a type to the variable 'test_find' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test_find', test_find)

@norecursion
def test_basic_table_parse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_basic_table_parse'
    module_type_store = module_type_store.open_function_context('test_basic_table_parse', 28, 0, False)
    
    # Passed parameters checking function
    test_basic_table_parse.stypy_localization = localization
    test_basic_table_parse.stypy_type_of_self = None
    test_basic_table_parse.stypy_type_store = module_type_store
    test_basic_table_parse.stypy_function_name = 'test_basic_table_parse'
    test_basic_table_parse.stypy_param_names_list = []
    test_basic_table_parse.stypy_varargs_param_name = None
    test_basic_table_parse.stypy_kwargs_param_name = None
    test_basic_table_parse.stypy_call_defaults = defaults
    test_basic_table_parse.stypy_call_varargs = varargs
    test_basic_table_parse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_basic_table_parse', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_basic_table_parse', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_basic_table_parse(...)' code ##################

    
    # Assigning a Str to a Name (line 29):
    str_14544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'str', 'speed of light in vacuum')
    # Assigning a type to the variable 'c' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'c', str_14544)
    
    # Call to assert_equal(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to value(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'c' (line 30)
    c_14548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'c', False)
    # Processing the call keyword arguments (line 30)
    kwargs_14549 = {}
    # Getting the type of 'codata' (line 30)
    codata_14546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'codata', False)
    # Obtaining the member 'value' of a type (line 30)
    value_14547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), codata_14546, 'value')
    # Calling value(args, kwargs) (line 30)
    value_call_result_14550 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), value_14547, *[c_14548], **kwargs_14549)
    
    # Getting the type of 'constants' (line 30)
    constants_14551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'constants', False)
    # Obtaining the member 'c' of a type (line 30)
    c_14552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 34), constants_14551, 'c')
    # Processing the call keyword arguments (line 30)
    kwargs_14553 = {}
    # Getting the type of 'assert_equal' (line 30)
    assert_equal_14545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 30)
    assert_equal_call_result_14554 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_equal_14545, *[value_call_result_14550, c_14552], **kwargs_14553)
    
    
    # Call to assert_equal(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to value(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'c' (line 31)
    c_14558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'c', False)
    # Processing the call keyword arguments (line 31)
    kwargs_14559 = {}
    # Getting the type of 'codata' (line 31)
    codata_14556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'codata', False)
    # Obtaining the member 'value' of a type (line 31)
    value_14557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), codata_14556, 'value')
    # Calling value(args, kwargs) (line 31)
    value_call_result_14560 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), value_14557, *[c_14558], **kwargs_14559)
    
    # Getting the type of 'constants' (line 31)
    constants_14561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'constants', False)
    # Obtaining the member 'speed_of_light' of a type (line 31)
    speed_of_light_14562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 34), constants_14561, 'speed_of_light')
    # Processing the call keyword arguments (line 31)
    kwargs_14563 = {}
    # Getting the type of 'assert_equal' (line 31)
    assert_equal_14555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 31)
    assert_equal_call_result_14564 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_equal_14555, *[value_call_result_14560, speed_of_light_14562], **kwargs_14563)
    
    
    # ################# End of 'test_basic_table_parse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_basic_table_parse' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_14565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14565)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_basic_table_parse'
    return stypy_return_type_14565

# Assigning a type to the variable 'test_basic_table_parse' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'test_basic_table_parse', test_basic_table_parse)

@norecursion
def test_basic_lookup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_basic_lookup'
    module_type_store = module_type_store.open_function_context('test_basic_lookup', 34, 0, False)
    
    # Passed parameters checking function
    test_basic_lookup.stypy_localization = localization
    test_basic_lookup.stypy_type_of_self = None
    test_basic_lookup.stypy_type_store = module_type_store
    test_basic_lookup.stypy_function_name = 'test_basic_lookup'
    test_basic_lookup.stypy_param_names_list = []
    test_basic_lookup.stypy_varargs_param_name = None
    test_basic_lookup.stypy_kwargs_param_name = None
    test_basic_lookup.stypy_call_defaults = defaults
    test_basic_lookup.stypy_call_varargs = varargs
    test_basic_lookup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_basic_lookup', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_basic_lookup', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_basic_lookup(...)' code ##################

    
    # Call to assert_equal(...): (line 35)
    # Processing the call arguments (line 35)
    str_14567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'str', '%d %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_14568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'codata' (line 35)
    codata_14569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'codata', False)
    # Obtaining the member 'c' of a type (line 35)
    c_14570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 28), codata_14569, 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 28), tuple_14568, c_14570)
    # Adding element type (line 35)
    
    # Call to unit(...): (line 35)
    # Processing the call arguments (line 35)
    str_14573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 50), 'str', 'speed of light in vacuum')
    # Processing the call keyword arguments (line 35)
    kwargs_14574 = {}
    # Getting the type of 'codata' (line 35)
    codata_14571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 38), 'codata', False)
    # Obtaining the member 'unit' of a type (line 35)
    unit_14572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 38), codata_14571, 'unit')
    # Calling unit(args, kwargs) (line 35)
    unit_call_result_14575 = invoke(stypy.reporting.localization.Localization(__file__, 35, 38), unit_14572, *[str_14573], **kwargs_14574)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 28), tuple_14568, unit_call_result_14575)
    
    # Applying the binary operator '%' (line 35)
    result_mod_14576 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 17), '%', str_14567, tuple_14568)
    
    str_14577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'str', '299792458 m s^-1')
    # Processing the call keyword arguments (line 35)
    kwargs_14578 = {}
    # Getting the type of 'assert_equal' (line 35)
    assert_equal_14566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 35)
    assert_equal_call_result_14579 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert_equal_14566, *[result_mod_14576, str_14577], **kwargs_14578)
    
    
    # ################# End of 'test_basic_lookup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_basic_lookup' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_14580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14580)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_basic_lookup'
    return stypy_return_type_14580

# Assigning a type to the variable 'test_basic_lookup' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test_basic_lookup', test_basic_lookup)

@norecursion
def test_find_all(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_find_all'
    module_type_store = module_type_store.open_function_context('test_find_all', 39, 0, False)
    
    # Passed parameters checking function
    test_find_all.stypy_localization = localization
    test_find_all.stypy_type_of_self = None
    test_find_all.stypy_type_store = module_type_store
    test_find_all.stypy_function_name = 'test_find_all'
    test_find_all.stypy_param_names_list = []
    test_find_all.stypy_varargs_param_name = None
    test_find_all.stypy_kwargs_param_name = None
    test_find_all.stypy_call_defaults = defaults
    test_find_all.stypy_call_varargs = varargs
    test_find_all.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_find_all', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_find_all', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_find_all(...)' code ##################

    
    # Call to assert_(...): (line 40)
    # Processing the call arguments (line 40)
    
    
    # Call to len(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to find(...): (line 40)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'False' (line 40)
    False_14585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'False', False)
    keyword_14586 = False_14585
    kwargs_14587 = {'disp': keyword_14586}
    # Getting the type of 'codata' (line 40)
    codata_14583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'codata', False)
    # Obtaining the member 'find' of a type (line 40)
    find_14584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), codata_14583, 'find')
    # Calling find(args, kwargs) (line 40)
    find_call_result_14588 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), find_14584, *[], **kwargs_14587)
    
    # Processing the call keyword arguments (line 40)
    kwargs_14589 = {}
    # Getting the type of 'len' (line 40)
    len_14582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'len', False)
    # Calling len(args, kwargs) (line 40)
    len_call_result_14590 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), len_14582, *[find_call_result_14588], **kwargs_14589)
    
    int_14591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'int')
    # Applying the binary operator '>' (line 40)
    result_gt_14592 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 12), '>', len_call_result_14590, int_14591)
    
    # Processing the call keyword arguments (line 40)
    kwargs_14593 = {}
    # Getting the type of 'assert_' (line 40)
    assert__14581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 40)
    assert__call_result_14594 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), assert__14581, *[result_gt_14592], **kwargs_14593)
    
    
    # ################# End of 'test_find_all(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_find_all' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_14595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14595)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_find_all'
    return stypy_return_type_14595

# Assigning a type to the variable 'test_find_all' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'test_find_all', test_find_all)

@norecursion
def test_find_single(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_find_single'
    module_type_store = module_type_store.open_function_context('test_find_single', 43, 0, False)
    
    # Passed parameters checking function
    test_find_single.stypy_localization = localization
    test_find_single.stypy_type_of_self = None
    test_find_single.stypy_type_store = module_type_store
    test_find_single.stypy_function_name = 'test_find_single'
    test_find_single.stypy_param_names_list = []
    test_find_single.stypy_varargs_param_name = None
    test_find_single.stypy_kwargs_param_name = None
    test_find_single.stypy_call_defaults = defaults
    test_find_single.stypy_call_varargs = varargs
    test_find_single.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_find_single', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_find_single', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_find_single(...)' code ##################

    
    # Call to assert_equal(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining the type of the subscript
    int_14597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 54), 'int')
    
    # Call to find(...): (line 44)
    # Processing the call arguments (line 44)
    str_14600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'str', 'Wien freq')
    # Processing the call keyword arguments (line 44)
    # Getting the type of 'False' (line 44)
    False_14601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 47), 'False', False)
    keyword_14602 = False_14601
    kwargs_14603 = {'disp': keyword_14602}
    # Getting the type of 'codata' (line 44)
    codata_14598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'codata', False)
    # Obtaining the member 'find' of a type (line 44)
    find_14599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 17), codata_14598, 'find')
    # Calling find(args, kwargs) (line 44)
    find_call_result_14604 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), find_14599, *[str_14600], **kwargs_14603)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___14605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 17), find_call_result_14604, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_14606 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), getitem___14605, int_14597)
    
    str_14607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'str', 'Wien frequency displacement law constant')
    # Processing the call keyword arguments (line 44)
    kwargs_14608 = {}
    # Getting the type of 'assert_equal' (line 44)
    assert_equal_14596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 44)
    assert_equal_call_result_14609 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), assert_equal_14596, *[subscript_call_result_14606, str_14607], **kwargs_14608)
    
    
    # ################# End of 'test_find_single(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_find_single' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_14610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_find_single'
    return stypy_return_type_14610

# Assigning a type to the variable 'test_find_single' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'test_find_single', test_find_single)

@norecursion
def test_2002_vs_2006(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_2002_vs_2006'
    module_type_store = module_type_store.open_function_context('test_2002_vs_2006', 48, 0, False)
    
    # Passed parameters checking function
    test_2002_vs_2006.stypy_localization = localization
    test_2002_vs_2006.stypy_type_of_self = None
    test_2002_vs_2006.stypy_type_store = module_type_store
    test_2002_vs_2006.stypy_function_name = 'test_2002_vs_2006'
    test_2002_vs_2006.stypy_param_names_list = []
    test_2002_vs_2006.stypy_varargs_param_name = None
    test_2002_vs_2006.stypy_kwargs_param_name = None
    test_2002_vs_2006.stypy_call_defaults = defaults
    test_2002_vs_2006.stypy_call_varargs = varargs
    test_2002_vs_2006.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_2002_vs_2006', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_2002_vs_2006', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_2002_vs_2006(...)' code ##################

    
    # Call to assert_almost_equal(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Call to value(...): (line 49)
    # Processing the call arguments (line 49)
    str_14614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'str', 'magn. flux quantum')
    # Processing the call keyword arguments (line 49)
    kwargs_14615 = {}
    # Getting the type of 'codata' (line 49)
    codata_14612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'codata', False)
    # Obtaining the member 'value' of a type (line 49)
    value_14613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), codata_14612, 'value')
    # Calling value(args, kwargs) (line 49)
    value_call_result_14616 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), value_14613, *[str_14614], **kwargs_14615)
    
    
    # Call to value(...): (line 50)
    # Processing the call arguments (line 50)
    str_14619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 37), 'str', 'mag. flux quantum')
    # Processing the call keyword arguments (line 50)
    kwargs_14620 = {}
    # Getting the type of 'codata' (line 50)
    codata_14617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'codata', False)
    # Obtaining the member 'value' of a type (line 50)
    value_14618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), codata_14617, 'value')
    # Calling value(args, kwargs) (line 50)
    value_call_result_14621 = invoke(stypy.reporting.localization.Localization(__file__, 50, 24), value_14618, *[str_14619], **kwargs_14620)
    
    # Processing the call keyword arguments (line 49)
    kwargs_14622 = {}
    # Getting the type of 'assert_almost_equal' (line 49)
    assert_almost_equal_14611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 49)
    assert_almost_equal_call_result_14623 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), assert_almost_equal_14611, *[value_call_result_14616, value_call_result_14621], **kwargs_14622)
    
    
    # ################# End of 'test_2002_vs_2006(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_2002_vs_2006' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_14624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14624)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_2002_vs_2006'
    return stypy_return_type_14624

# Assigning a type to the variable 'test_2002_vs_2006' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'test_2002_vs_2006', test_2002_vs_2006)

@norecursion
def test_exact_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_exact_values'
    module_type_store = module_type_store.open_function_context('test_exact_values', 53, 0, False)
    
    # Passed parameters checking function
    test_exact_values.stypy_localization = localization
    test_exact_values.stypy_type_of_self = None
    test_exact_values.stypy_type_store = module_type_store
    test_exact_values.stypy_function_name = 'test_exact_values'
    test_exact_values.stypy_param_names_list = []
    test_exact_values.stypy_varargs_param_name = None
    test_exact_values.stypy_kwargs_param_name = None
    test_exact_values.stypy_call_defaults = defaults
    test_exact_values.stypy_call_varargs = varargs
    test_exact_values.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_exact_values', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_exact_values', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_exact_values(...)' code ##################

    
    # Getting the type of 'codata' (line 55)
    codata_14625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'codata')
    # Obtaining the member 'exact_values' of a type (line 55)
    exact_values_14626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), codata_14625, 'exact_values')
    # Testing the type of a for loop iterable (line 55)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 4), exact_values_14626)
    # Getting the type of the for loop variable (line 55)
    for_loop_var_14627 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 4), exact_values_14626)
    # Assigning a type to the variable 'key' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'key', for_loop_var_14627)
    # SSA begins for a for statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 56)
    # Processing the call arguments (line 56)
    
    
    # Obtaining the type of the subscript
    int_14629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 56)
    key_14630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 37), 'key', False)
    # Getting the type of 'codata' (line 56)
    codata_14631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'codata', False)
    # Obtaining the member 'exact_values' of a type (line 56)
    exact_values_14632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), codata_14631, 'exact_values')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___14633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), exact_values_14632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_14634 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___14633, key_14630)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___14635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), subscript_call_result_14634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_14636 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___14635, int_14629)
    
    
    # Call to value(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'key' (line 56)
    key_14638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 53), 'key', False)
    # Processing the call keyword arguments (line 56)
    kwargs_14639 = {}
    # Getting the type of 'value' (line 56)
    value_14637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'value', False)
    # Calling value(args, kwargs) (line 56)
    value_call_result_14640 = invoke(stypy.reporting.localization.Localization(__file__, 56, 47), value_14637, *[key_14638], **kwargs_14639)
    
    # Applying the binary operator '-' (line 56)
    result_sub_14641 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 17), '-', subscript_call_result_14636, value_call_result_14640)
    
    
    # Call to value(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'key' (line 56)
    key_14643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 67), 'key', False)
    # Processing the call keyword arguments (line 56)
    kwargs_14644 = {}
    # Getting the type of 'value' (line 56)
    value_14642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 61), 'value', False)
    # Calling value(args, kwargs) (line 56)
    value_call_result_14645 = invoke(stypy.reporting.localization.Localization(__file__, 56, 61), value_14642, *[key_14643], **kwargs_14644)
    
    # Applying the binary operator 'div' (line 56)
    result_div_14646 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), 'div', result_sub_14641, value_call_result_14645)
    
    int_14647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 75), 'int')
    # Applying the binary operator '==' (line 56)
    result_eq_14648 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '==', result_div_14646, int_14647)
    
    # Processing the call keyword arguments (line 56)
    kwargs_14649 = {}
    # Getting the type of 'assert_' (line 56)
    assert__14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 56)
    assert__call_result_14650 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert__14628, *[result_eq_14648], **kwargs_14649)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_exact_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_exact_values' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_14651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14651)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_exact_values'
    return stypy_return_type_14651

# Assigning a type to the variable 'test_exact_values' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'test_exact_values', test_exact_values)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

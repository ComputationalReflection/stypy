
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy.testing import (assert_array_equal, assert_array_almost_equal)
4: from scipy.interpolate import pade
5: 
6: def test_pade_trivial():
7:     nump, denomp = pade([1.0], 0)
8:     assert_array_equal(nump.c, [1.0])
9:     assert_array_equal(denomp.c, [1.0])
10: 
11: 
12: def test_pade_4term_exp():
13:     # First four Taylor coefficients of exp(x).
14:     # Unlike poly1d, the first array element is the zero-order term.
15:     an = [1.0, 1.0, 0.5, 1.0/6]
16: 
17:     nump, denomp = pade(an, 0)
18:     assert_array_almost_equal(nump.c, [1.0/6, 0.5, 1.0, 1.0])
19:     assert_array_almost_equal(denomp.c, [1.0])
20: 
21:     nump, denomp = pade(an, 1)
22:     assert_array_almost_equal(nump.c, [1.0/6, 2.0/3, 1.0])
23:     assert_array_almost_equal(denomp.c, [-1.0/3, 1.0])
24: 
25:     nump, denomp = pade(an, 2)
26:     assert_array_almost_equal(nump.c, [1.0/3, 1.0])
27:     assert_array_almost_equal(denomp.c, [1.0/6, -2.0/3, 1.0])
28: 
29:     nump, denomp = pade(an, 3)
30:     assert_array_almost_equal(nump.c, [1.0])
31:     assert_array_almost_equal(denomp.c, [-1.0/6, 0.5, -1.0, 1.0])
32: 
33: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_array_equal, assert_array_almost_equal' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_115027 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_115027) is not StypyTypeError):

    if (import_115027 != 'pyd_module'):
        __import__(import_115027)
        sys_modules_115028 = sys.modules[import_115027]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_115028.module_type_store, module_type_store, ['assert_array_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_115028, sys_modules_115028.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal', 'assert_array_almost_equal'], [assert_array_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_115027)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.interpolate import pade' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_115029 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.interpolate')

if (type(import_115029) is not StypyTypeError):

    if (import_115029 != 'pyd_module'):
        __import__(import_115029)
        sys_modules_115030 = sys.modules[import_115029]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.interpolate', sys_modules_115030.module_type_store, module_type_store, ['pade'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_115030, sys_modules_115030.module_type_store, module_type_store)
    else:
        from scipy.interpolate import pade

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.interpolate', None, module_type_store, ['pade'], [pade])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.interpolate', import_115029)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')


@norecursion
def test_pade_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_pade_trivial'
    module_type_store = module_type_store.open_function_context('test_pade_trivial', 6, 0, False)
    
    # Passed parameters checking function
    test_pade_trivial.stypy_localization = localization
    test_pade_trivial.stypy_type_of_self = None
    test_pade_trivial.stypy_type_store = module_type_store
    test_pade_trivial.stypy_function_name = 'test_pade_trivial'
    test_pade_trivial.stypy_param_names_list = []
    test_pade_trivial.stypy_varargs_param_name = None
    test_pade_trivial.stypy_kwargs_param_name = None
    test_pade_trivial.stypy_call_defaults = defaults
    test_pade_trivial.stypy_call_varargs = varargs
    test_pade_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_pade_trivial', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_pade_trivial', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_pade_trivial(...)' code ##################

    
    # Assigning a Call to a Tuple (line 7):
    
    # Assigning a Subscript to a Name (line 7):
    
    # Obtaining the type of the subscript
    int_115031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'int')
    
    # Call to pade(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Obtaining an instance of the builtin type 'list' (line 7)
    list_115033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 7)
    # Adding element type (line 7)
    float_115034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 24), list_115033, float_115034)
    
    int_115035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 31), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_115036 = {}
    # Getting the type of 'pade' (line 7)
    pade_115032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 7)
    pade_call_result_115037 = invoke(stypy.reporting.localization.Localization(__file__, 7, 19), pade_115032, *[list_115033, int_115035], **kwargs_115036)
    
    # Obtaining the member '__getitem__' of a type (line 7)
    getitem___115038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), pade_call_result_115037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 7)
    subscript_call_result_115039 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), getitem___115038, int_115031)
    
    # Assigning a type to the variable 'tuple_var_assignment_115017' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'tuple_var_assignment_115017', subscript_call_result_115039)
    
    # Assigning a Subscript to a Name (line 7):
    
    # Obtaining the type of the subscript
    int_115040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'int')
    
    # Call to pade(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Obtaining an instance of the builtin type 'list' (line 7)
    list_115042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 7)
    # Adding element type (line 7)
    float_115043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 24), list_115042, float_115043)
    
    int_115044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 31), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_115045 = {}
    # Getting the type of 'pade' (line 7)
    pade_115041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 7)
    pade_call_result_115046 = invoke(stypy.reporting.localization.Localization(__file__, 7, 19), pade_115041, *[list_115042, int_115044], **kwargs_115045)
    
    # Obtaining the member '__getitem__' of a type (line 7)
    getitem___115047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), pade_call_result_115046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 7)
    subscript_call_result_115048 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), getitem___115047, int_115040)
    
    # Assigning a type to the variable 'tuple_var_assignment_115018' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'tuple_var_assignment_115018', subscript_call_result_115048)
    
    # Assigning a Name to a Name (line 7):
    # Getting the type of 'tuple_var_assignment_115017' (line 7)
    tuple_var_assignment_115017_115049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'tuple_var_assignment_115017')
    # Assigning a type to the variable 'nump' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'nump', tuple_var_assignment_115017_115049)
    
    # Assigning a Name to a Name (line 7):
    # Getting the type of 'tuple_var_assignment_115018' (line 7)
    tuple_var_assignment_115018_115050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'tuple_var_assignment_115018')
    # Assigning a type to the variable 'denomp' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'denomp', tuple_var_assignment_115018_115050)
    
    # Call to assert_array_equal(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'nump' (line 8)
    nump_115052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'nump', False)
    # Obtaining the member 'c' of a type (line 8)
    c_115053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 23), nump_115052, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 8)
    list_115054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 8)
    # Adding element type (line 8)
    float_115055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 31), list_115054, float_115055)
    
    # Processing the call keyword arguments (line 8)
    kwargs_115056 = {}
    # Getting the type of 'assert_array_equal' (line 8)
    assert_array_equal_115051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 8)
    assert_array_equal_call_result_115057 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), assert_array_equal_115051, *[c_115053, list_115054], **kwargs_115056)
    
    
    # Call to assert_array_equal(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'denomp' (line 9)
    denomp_115059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'denomp', False)
    # Obtaining the member 'c' of a type (line 9)
    c_115060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 23), denomp_115059, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_115061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    float_115062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 33), list_115061, float_115062)
    
    # Processing the call keyword arguments (line 9)
    kwargs_115063 = {}
    # Getting the type of 'assert_array_equal' (line 9)
    assert_array_equal_115058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 9)
    assert_array_equal_call_result_115064 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), assert_array_equal_115058, *[c_115060, list_115061], **kwargs_115063)
    
    
    # ################# End of 'test_pade_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_pade_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_115065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_pade_trivial'
    return stypy_return_type_115065

# Assigning a type to the variable 'test_pade_trivial' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'test_pade_trivial', test_pade_trivial)

@norecursion
def test_pade_4term_exp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_pade_4term_exp'
    module_type_store = module_type_store.open_function_context('test_pade_4term_exp', 12, 0, False)
    
    # Passed parameters checking function
    test_pade_4term_exp.stypy_localization = localization
    test_pade_4term_exp.stypy_type_of_self = None
    test_pade_4term_exp.stypy_type_store = module_type_store
    test_pade_4term_exp.stypy_function_name = 'test_pade_4term_exp'
    test_pade_4term_exp.stypy_param_names_list = []
    test_pade_4term_exp.stypy_varargs_param_name = None
    test_pade_4term_exp.stypy_kwargs_param_name = None
    test_pade_4term_exp.stypy_call_defaults = defaults
    test_pade_4term_exp.stypy_call_varargs = varargs
    test_pade_4term_exp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_pade_4term_exp', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_pade_4term_exp', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_pade_4term_exp(...)' code ##################

    
    # Assigning a List to a Name (line 15):
    
    # Assigning a List to a Name (line 15):
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_115066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    float_115067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_115066, float_115067)
    # Adding element type (line 15)
    float_115068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_115066, float_115068)
    # Adding element type (line 15)
    float_115069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_115066, float_115069)
    # Adding element type (line 15)
    float_115070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'float')
    int_115071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_115072 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 25), 'div', float_115070, int_115071)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_115066, result_div_115072)
    
    # Assigning a type to the variable 'an' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'an', list_115066)
    
    # Assigning a Call to a Tuple (line 17):
    
    # Assigning a Subscript to a Name (line 17):
    
    # Obtaining the type of the subscript
    int_115073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'int')
    
    # Call to pade(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'an' (line 17)
    an_115075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'an', False)
    int_115076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_115077 = {}
    # Getting the type of 'pade' (line 17)
    pade_115074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 17)
    pade_call_result_115078 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), pade_115074, *[an_115075, int_115076], **kwargs_115077)
    
    # Obtaining the member '__getitem__' of a type (line 17)
    getitem___115079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), pade_call_result_115078, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 17)
    subscript_call_result_115080 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), getitem___115079, int_115073)
    
    # Assigning a type to the variable 'tuple_var_assignment_115019' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'tuple_var_assignment_115019', subscript_call_result_115080)
    
    # Assigning a Subscript to a Name (line 17):
    
    # Obtaining the type of the subscript
    int_115081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'int')
    
    # Call to pade(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'an' (line 17)
    an_115083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'an', False)
    int_115084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_115085 = {}
    # Getting the type of 'pade' (line 17)
    pade_115082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 17)
    pade_call_result_115086 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), pade_115082, *[an_115083, int_115084], **kwargs_115085)
    
    # Obtaining the member '__getitem__' of a type (line 17)
    getitem___115087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), pade_call_result_115086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 17)
    subscript_call_result_115088 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), getitem___115087, int_115081)
    
    # Assigning a type to the variable 'tuple_var_assignment_115020' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'tuple_var_assignment_115020', subscript_call_result_115088)
    
    # Assigning a Name to a Name (line 17):
    # Getting the type of 'tuple_var_assignment_115019' (line 17)
    tuple_var_assignment_115019_115089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'tuple_var_assignment_115019')
    # Assigning a type to the variable 'nump' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'nump', tuple_var_assignment_115019_115089)
    
    # Assigning a Name to a Name (line 17):
    # Getting the type of 'tuple_var_assignment_115020' (line 17)
    tuple_var_assignment_115020_115090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'tuple_var_assignment_115020')
    # Assigning a type to the variable 'denomp' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'denomp', tuple_var_assignment_115020_115090)
    
    # Call to assert_array_almost_equal(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'nump' (line 18)
    nump_115092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'nump', False)
    # Obtaining the member 'c' of a type (line 18)
    c_115093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 30), nump_115092, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_115094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    float_115095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 39), 'float')
    int_115096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 43), 'int')
    # Applying the binary operator 'div' (line 18)
    result_div_115097 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 39), 'div', float_115095, int_115096)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_115094, result_div_115097)
    # Adding element type (line 18)
    float_115098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_115094, float_115098)
    # Adding element type (line 18)
    float_115099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 51), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_115094, float_115099)
    # Adding element type (line 18)
    float_115100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 56), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_115094, float_115100)
    
    # Processing the call keyword arguments (line 18)
    kwargs_115101 = {}
    # Getting the type of 'assert_array_almost_equal' (line 18)
    assert_array_almost_equal_115091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 18)
    assert_array_almost_equal_call_result_115102 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), assert_array_almost_equal_115091, *[c_115093, list_115094], **kwargs_115101)
    
    
    # Call to assert_array_almost_equal(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'denomp' (line 19)
    denomp_115104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 30), 'denomp', False)
    # Obtaining the member 'c' of a type (line 19)
    c_115105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 30), denomp_115104, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_115106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    float_115107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 40), list_115106, float_115107)
    
    # Processing the call keyword arguments (line 19)
    kwargs_115108 = {}
    # Getting the type of 'assert_array_almost_equal' (line 19)
    assert_array_almost_equal_115103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 19)
    assert_array_almost_equal_call_result_115109 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_array_almost_equal_115103, *[c_115105, list_115106], **kwargs_115108)
    
    
    # Assigning a Call to a Tuple (line 21):
    
    # Assigning a Subscript to a Name (line 21):
    
    # Obtaining the type of the subscript
    int_115110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
    
    # Call to pade(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'an' (line 21)
    an_115112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'an', False)
    int_115113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_115114 = {}
    # Getting the type of 'pade' (line 21)
    pade_115111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 21)
    pade_call_result_115115 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), pade_115111, *[an_115112, int_115113], **kwargs_115114)
    
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___115116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), pade_call_result_115115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_115117 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), getitem___115116, int_115110)
    
    # Assigning a type to the variable 'tuple_var_assignment_115021' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_115021', subscript_call_result_115117)
    
    # Assigning a Subscript to a Name (line 21):
    
    # Obtaining the type of the subscript
    int_115118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
    
    # Call to pade(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'an' (line 21)
    an_115120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'an', False)
    int_115121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_115122 = {}
    # Getting the type of 'pade' (line 21)
    pade_115119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 21)
    pade_call_result_115123 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), pade_115119, *[an_115120, int_115121], **kwargs_115122)
    
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___115124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), pade_call_result_115123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_115125 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), getitem___115124, int_115118)
    
    # Assigning a type to the variable 'tuple_var_assignment_115022' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_115022', subscript_call_result_115125)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_var_assignment_115021' (line 21)
    tuple_var_assignment_115021_115126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_115021')
    # Assigning a type to the variable 'nump' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'nump', tuple_var_assignment_115021_115126)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_var_assignment_115022' (line 21)
    tuple_var_assignment_115022_115127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_115022')
    # Assigning a type to the variable 'denomp' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'denomp', tuple_var_assignment_115022_115127)
    
    # Call to assert_array_almost_equal(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'nump' (line 22)
    nump_115129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'nump', False)
    # Obtaining the member 'c' of a type (line 22)
    c_115130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 30), nump_115129, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_115131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    float_115132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 39), 'float')
    int_115133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'int')
    # Applying the binary operator 'div' (line 22)
    result_div_115134 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 39), 'div', float_115132, int_115133)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 38), list_115131, result_div_115134)
    # Adding element type (line 22)
    float_115135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'float')
    int_115136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 50), 'int')
    # Applying the binary operator 'div' (line 22)
    result_div_115137 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 46), 'div', float_115135, int_115136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 38), list_115131, result_div_115137)
    # Adding element type (line 22)
    float_115138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 38), list_115131, float_115138)
    
    # Processing the call keyword arguments (line 22)
    kwargs_115139 = {}
    # Getting the type of 'assert_array_almost_equal' (line 22)
    assert_array_almost_equal_115128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 22)
    assert_array_almost_equal_call_result_115140 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_array_almost_equal_115128, *[c_115130, list_115131], **kwargs_115139)
    
    
    # Call to assert_array_almost_equal(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'denomp' (line 23)
    denomp_115142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'denomp', False)
    # Obtaining the member 'c' of a type (line 23)
    c_115143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 30), denomp_115142, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_115144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    float_115145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 41), 'float')
    int_115146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 46), 'int')
    # Applying the binary operator 'div' (line 23)
    result_div_115147 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 41), 'div', float_115145, int_115146)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 40), list_115144, result_div_115147)
    # Adding element type (line 23)
    float_115148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 40), list_115144, float_115148)
    
    # Processing the call keyword arguments (line 23)
    kwargs_115149 = {}
    # Getting the type of 'assert_array_almost_equal' (line 23)
    assert_array_almost_equal_115141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 23)
    assert_array_almost_equal_call_result_115150 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_array_almost_equal_115141, *[c_115143, list_115144], **kwargs_115149)
    
    
    # Assigning a Call to a Tuple (line 25):
    
    # Assigning a Subscript to a Name (line 25):
    
    # Obtaining the type of the subscript
    int_115151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'int')
    
    # Call to pade(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'an' (line 25)
    an_115153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'an', False)
    int_115154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_115155 = {}
    # Getting the type of 'pade' (line 25)
    pade_115152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 25)
    pade_call_result_115156 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), pade_115152, *[an_115153, int_115154], **kwargs_115155)
    
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___115157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), pade_call_result_115156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_115158 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), getitem___115157, int_115151)
    
    # Assigning a type to the variable 'tuple_var_assignment_115023' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_115023', subscript_call_result_115158)
    
    # Assigning a Subscript to a Name (line 25):
    
    # Obtaining the type of the subscript
    int_115159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'int')
    
    # Call to pade(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'an' (line 25)
    an_115161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'an', False)
    int_115162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_115163 = {}
    # Getting the type of 'pade' (line 25)
    pade_115160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 25)
    pade_call_result_115164 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), pade_115160, *[an_115161, int_115162], **kwargs_115163)
    
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___115165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), pade_call_result_115164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_115166 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), getitem___115165, int_115159)
    
    # Assigning a type to the variable 'tuple_var_assignment_115024' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_115024', subscript_call_result_115166)
    
    # Assigning a Name to a Name (line 25):
    # Getting the type of 'tuple_var_assignment_115023' (line 25)
    tuple_var_assignment_115023_115167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_115023')
    # Assigning a type to the variable 'nump' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'nump', tuple_var_assignment_115023_115167)
    
    # Assigning a Name to a Name (line 25):
    # Getting the type of 'tuple_var_assignment_115024' (line 25)
    tuple_var_assignment_115024_115168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_115024')
    # Assigning a type to the variable 'denomp' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'denomp', tuple_var_assignment_115024_115168)
    
    # Call to assert_array_almost_equal(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'nump' (line 26)
    nump_115170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'nump', False)
    # Obtaining the member 'c' of a type (line 26)
    c_115171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), nump_115170, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_115172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    float_115173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'float')
    int_115174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_115175 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 39), 'div', float_115173, int_115174)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_115172, result_div_115175)
    # Adding element type (line 26)
    float_115176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_115172, float_115176)
    
    # Processing the call keyword arguments (line 26)
    kwargs_115177 = {}
    # Getting the type of 'assert_array_almost_equal' (line 26)
    assert_array_almost_equal_115169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 26)
    assert_array_almost_equal_call_result_115178 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert_array_almost_equal_115169, *[c_115171, list_115172], **kwargs_115177)
    
    
    # Call to assert_array_almost_equal(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'denomp' (line 27)
    denomp_115180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'denomp', False)
    # Obtaining the member 'c' of a type (line 27)
    c_115181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 30), denomp_115180, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_115182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    float_115183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 41), 'float')
    int_115184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 45), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_115185 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 41), 'div', float_115183, int_115184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 40), list_115182, result_div_115185)
    # Adding element type (line 27)
    float_115186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 48), 'float')
    int_115187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 53), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_115188 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 48), 'div', float_115186, int_115187)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 40), list_115182, result_div_115188)
    # Adding element type (line 27)
    float_115189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 56), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 40), list_115182, float_115189)
    
    # Processing the call keyword arguments (line 27)
    kwargs_115190 = {}
    # Getting the type of 'assert_array_almost_equal' (line 27)
    assert_array_almost_equal_115179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 27)
    assert_array_almost_equal_call_result_115191 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), assert_array_almost_equal_115179, *[c_115181, list_115182], **kwargs_115190)
    
    
    # Assigning a Call to a Tuple (line 29):
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_115192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    
    # Call to pade(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'an' (line 29)
    an_115194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'an', False)
    int_115195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_115196 = {}
    # Getting the type of 'pade' (line 29)
    pade_115193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 29)
    pade_call_result_115197 = invoke(stypy.reporting.localization.Localization(__file__, 29, 19), pade_115193, *[an_115194, int_115195], **kwargs_115196)
    
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___115198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), pade_call_result_115197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_115199 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___115198, int_115192)
    
    # Assigning a type to the variable 'tuple_var_assignment_115025' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_115025', subscript_call_result_115199)
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_115200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    
    # Call to pade(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'an' (line 29)
    an_115202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'an', False)
    int_115203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_115204 = {}
    # Getting the type of 'pade' (line 29)
    pade_115201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'pade', False)
    # Calling pade(args, kwargs) (line 29)
    pade_call_result_115205 = invoke(stypy.reporting.localization.Localization(__file__, 29, 19), pade_115201, *[an_115202, int_115203], **kwargs_115204)
    
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___115206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), pade_call_result_115205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_115207 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___115206, int_115200)
    
    # Assigning a type to the variable 'tuple_var_assignment_115026' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_115026', subscript_call_result_115207)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_115025' (line 29)
    tuple_var_assignment_115025_115208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_115025')
    # Assigning a type to the variable 'nump' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'nump', tuple_var_assignment_115025_115208)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_115026' (line 29)
    tuple_var_assignment_115026_115209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_115026')
    # Assigning a type to the variable 'denomp' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'denomp', tuple_var_assignment_115026_115209)
    
    # Call to assert_array_almost_equal(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'nump' (line 30)
    nump_115211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'nump', False)
    # Obtaining the member 'c' of a type (line 30)
    c_115212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 30), nump_115211, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_115213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    float_115214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 38), list_115213, float_115214)
    
    # Processing the call keyword arguments (line 30)
    kwargs_115215 = {}
    # Getting the type of 'assert_array_almost_equal' (line 30)
    assert_array_almost_equal_115210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 30)
    assert_array_almost_equal_call_result_115216 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_array_almost_equal_115210, *[c_115212, list_115213], **kwargs_115215)
    
    
    # Call to assert_array_almost_equal(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'denomp' (line 31)
    denomp_115218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'denomp', False)
    # Obtaining the member 'c' of a type (line 31)
    c_115219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), denomp_115218, 'c')
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_115220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    float_115221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'float')
    int_115222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_115223 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 41), 'div', float_115221, int_115222)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 40), list_115220, result_div_115223)
    # Adding element type (line 31)
    float_115224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 40), list_115220, float_115224)
    # Adding element type (line 31)
    float_115225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 54), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 40), list_115220, float_115225)
    # Adding element type (line 31)
    float_115226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 60), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 40), list_115220, float_115226)
    
    # Processing the call keyword arguments (line 31)
    kwargs_115227 = {}
    # Getting the type of 'assert_array_almost_equal' (line 31)
    assert_array_almost_equal_115217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 31)
    assert_array_almost_equal_call_result_115228 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_array_almost_equal_115217, *[c_115219, list_115220], **kwargs_115227)
    
    
    # ################# End of 'test_pade_4term_exp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_pade_4term_exp' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_115229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_pade_4term_exp'
    return stypy_return_type_115229

# Assigning a type to the variable 'test_pade_4term_exp' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'test_pade_4term_exp', test_pade_4term_exp)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

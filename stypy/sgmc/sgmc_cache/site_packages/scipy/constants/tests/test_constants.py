
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy.testing import assert_equal, assert_allclose
4: import scipy.constants as sc
5: 
6: 
7: def test_convert_temperature():
8:     assert_equal(sc.convert_temperature(32, 'f', 'Celsius'), 0)
9:     assert_equal(sc.convert_temperature([0, 0], 'celsius', 'Kelvin'),
10:                  [273.15, 273.15])
11:     assert_equal(sc.convert_temperature([0, 0], 'kelvin', 'c'),
12:                  [-273.15, -273.15])
13:     assert_equal(sc.convert_temperature([32, 32], 'f', 'k'), [273.15, 273.15])
14:     assert_equal(sc.convert_temperature([273.15, 273.15], 'kelvin', 'F'),
15:                  [32, 32])
16:     assert_equal(sc.convert_temperature([0, 0], 'C', 'fahrenheit'), [32, 32])
17:     assert_allclose(sc.convert_temperature([0, 0], 'c', 'r'), [491.67, 491.67],
18:                     rtol=0., atol=1e-13)
19:     assert_allclose(sc.convert_temperature([491.67, 491.67], 'Rankine', 'C'),
20:                     [0., 0.], rtol=0., atol=1e-13)
21:     assert_allclose(sc.convert_temperature([491.67, 491.67], 'r', 'F'),
22:                     [32., 32.], rtol=0., atol=1e-13)
23:     assert_allclose(sc.convert_temperature([32, 32], 'fahrenheit', 'R'),
24:                     [491.67, 491.67], rtol=0., atol=1e-13)
25:     assert_allclose(sc.convert_temperature([273.15, 273.15], 'K', 'R'),
26:                     [491.67, 491.67], rtol=0., atol=1e-13)
27:     assert_allclose(sc.convert_temperature([491.67, 0.], 'rankine', 'kelvin'),
28:                     [273.15, 0.], rtol=0., atol=1e-13)
29: 
30: 
31: def test_lambda_to_nu():
32:     assert_equal(sc.lambda2nu(sc.speed_of_light), 1)
33: 
34: 
35: def test_nu_to_lambda():
36:     assert_equal(sc.nu2lambda(1), sc.speed_of_light)
37: 
38: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/tests/')
import_14652 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_14652) is not StypyTypeError):

    if (import_14652 != 'pyd_module'):
        __import__(import_14652)
        sys_modules_14653 = sys.modules[import_14652]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_14653.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_14653, sys_modules_14653.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_14652)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import scipy.constants' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/tests/')
import_14654 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.constants')

if (type(import_14654) is not StypyTypeError):

    if (import_14654 != 'pyd_module'):
        __import__(import_14654)
        sys_modules_14655 = sys.modules[import_14654]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sc', sys_modules_14655.module_type_store, module_type_store)
    else:
        import scipy.constants as sc

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sc', scipy.constants, module_type_store)

else:
    # Assigning a type to the variable 'scipy.constants' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.constants', import_14654)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/tests/')


@norecursion
def test_convert_temperature(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_convert_temperature'
    module_type_store = module_type_store.open_function_context('test_convert_temperature', 7, 0, False)
    
    # Passed parameters checking function
    test_convert_temperature.stypy_localization = localization
    test_convert_temperature.stypy_type_of_self = None
    test_convert_temperature.stypy_type_store = module_type_store
    test_convert_temperature.stypy_function_name = 'test_convert_temperature'
    test_convert_temperature.stypy_param_names_list = []
    test_convert_temperature.stypy_varargs_param_name = None
    test_convert_temperature.stypy_kwargs_param_name = None
    test_convert_temperature.stypy_call_defaults = defaults
    test_convert_temperature.stypy_call_varargs = varargs
    test_convert_temperature.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_convert_temperature', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_convert_temperature', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_convert_temperature(...)' code ##################

    
    # Call to assert_equal(...): (line 8)
    # Processing the call arguments (line 8)
    
    # Call to convert_temperature(...): (line 8)
    # Processing the call arguments (line 8)
    int_14659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 40), 'int')
    str_14660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 44), 'str', 'f')
    str_14661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 49), 'str', 'Celsius')
    # Processing the call keyword arguments (line 8)
    kwargs_14662 = {}
    # Getting the type of 'sc' (line 8)
    sc_14657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 8)
    convert_temperature_14658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 17), sc_14657, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 8)
    convert_temperature_call_result_14663 = invoke(stypy.reporting.localization.Localization(__file__, 8, 17), convert_temperature_14658, *[int_14659, str_14660, str_14661], **kwargs_14662)
    
    int_14664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 61), 'int')
    # Processing the call keyword arguments (line 8)
    kwargs_14665 = {}
    # Getting the type of 'assert_equal' (line 8)
    assert_equal_14656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 8)
    assert_equal_call_result_14666 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), assert_equal_14656, *[convert_temperature_call_result_14663, int_14664], **kwargs_14665)
    
    
    # Call to assert_equal(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Call to convert_temperature(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_14670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    int_14671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 40), list_14670, int_14671)
    # Adding element type (line 9)
    int_14672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 40), list_14670, int_14672)
    
    str_14673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 48), 'str', 'celsius')
    str_14674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 59), 'str', 'Kelvin')
    # Processing the call keyword arguments (line 9)
    kwargs_14675 = {}
    # Getting the type of 'sc' (line 9)
    sc_14668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 9)
    convert_temperature_14669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 17), sc_14668, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 9)
    convert_temperature_call_result_14676 = invoke(stypy.reporting.localization.Localization(__file__, 9, 17), convert_temperature_14669, *[list_14670, str_14673, str_14674], **kwargs_14675)
    
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_14677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    float_14678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_14677, float_14678)
    # Adding element type (line 10)
    float_14679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_14677, float_14679)
    
    # Processing the call keyword arguments (line 9)
    kwargs_14680 = {}
    # Getting the type of 'assert_equal' (line 9)
    assert_equal_14667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 9)
    assert_equal_call_result_14681 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), assert_equal_14667, *[convert_temperature_call_result_14676, list_14677], **kwargs_14680)
    
    
    # Call to assert_equal(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to convert_temperature(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_14685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    int_14686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 40), list_14685, int_14686)
    # Adding element type (line 11)
    int_14687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 40), list_14685, int_14687)
    
    str_14688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 48), 'str', 'kelvin')
    str_14689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 58), 'str', 'c')
    # Processing the call keyword arguments (line 11)
    kwargs_14690 = {}
    # Getting the type of 'sc' (line 11)
    sc_14683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 11)
    convert_temperature_14684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 17), sc_14683, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 11)
    convert_temperature_call_result_14691 = invoke(stypy.reporting.localization.Localization(__file__, 11, 17), convert_temperature_14684, *[list_14685, str_14688, str_14689], **kwargs_14690)
    
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_14692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    float_14693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_14692, float_14693)
    # Adding element type (line 12)
    float_14694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_14692, float_14694)
    
    # Processing the call keyword arguments (line 11)
    kwargs_14695 = {}
    # Getting the type of 'assert_equal' (line 11)
    assert_equal_14682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 11)
    assert_equal_call_result_14696 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), assert_equal_14682, *[convert_temperature_call_result_14691, list_14692], **kwargs_14695)
    
    
    # Call to assert_equal(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to convert_temperature(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_14700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    int_14701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 40), list_14700, int_14701)
    # Adding element type (line 13)
    int_14702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 40), list_14700, int_14702)
    
    str_14703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 50), 'str', 'f')
    str_14704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 55), 'str', 'k')
    # Processing the call keyword arguments (line 13)
    kwargs_14705 = {}
    # Getting the type of 'sc' (line 13)
    sc_14698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 13)
    convert_temperature_14699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 17), sc_14698, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 13)
    convert_temperature_call_result_14706 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), convert_temperature_14699, *[list_14700, str_14703, str_14704], **kwargs_14705)
    
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_14707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    float_14708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 62), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 61), list_14707, float_14708)
    # Adding element type (line 13)
    float_14709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 70), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 61), list_14707, float_14709)
    
    # Processing the call keyword arguments (line 13)
    kwargs_14710 = {}
    # Getting the type of 'assert_equal' (line 13)
    assert_equal_14697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 13)
    assert_equal_call_result_14711 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), assert_equal_14697, *[convert_temperature_call_result_14706, list_14707], **kwargs_14710)
    
    
    # Call to assert_equal(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to convert_temperature(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_14715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    float_14716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 41), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 40), list_14715, float_14716)
    # Adding element type (line 14)
    float_14717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 40), list_14715, float_14717)
    
    str_14718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 58), 'str', 'kelvin')
    str_14719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 68), 'str', 'F')
    # Processing the call keyword arguments (line 14)
    kwargs_14720 = {}
    # Getting the type of 'sc' (line 14)
    sc_14713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 14)
    convert_temperature_14714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 17), sc_14713, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 14)
    convert_temperature_call_result_14721 = invoke(stypy.reporting.localization.Localization(__file__, 14, 17), convert_temperature_14714, *[list_14715, str_14718, str_14719], **kwargs_14720)
    
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_14722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_14723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 17), list_14722, int_14723)
    # Adding element type (line 15)
    int_14724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 17), list_14722, int_14724)
    
    # Processing the call keyword arguments (line 14)
    kwargs_14725 = {}
    # Getting the type of 'assert_equal' (line 14)
    assert_equal_14712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 14)
    assert_equal_call_result_14726 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), assert_equal_14712, *[convert_temperature_call_result_14721, list_14722], **kwargs_14725)
    
    
    # Call to assert_equal(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to convert_temperature(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_14730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_14731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 40), list_14730, int_14731)
    # Adding element type (line 16)
    int_14732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 40), list_14730, int_14732)
    
    str_14733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 48), 'str', 'C')
    str_14734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 53), 'str', 'fahrenheit')
    # Processing the call keyword arguments (line 16)
    kwargs_14735 = {}
    # Getting the type of 'sc' (line 16)
    sc_14728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 16)
    convert_temperature_14729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), sc_14728, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 16)
    convert_temperature_call_result_14736 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), convert_temperature_14729, *[list_14730, str_14733, str_14734], **kwargs_14735)
    
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_14737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 68), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_14738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 69), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 68), list_14737, int_14738)
    # Adding element type (line 16)
    int_14739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 73), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 68), list_14737, int_14739)
    
    # Processing the call keyword arguments (line 16)
    kwargs_14740 = {}
    # Getting the type of 'assert_equal' (line 16)
    assert_equal_14727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 16)
    assert_equal_call_result_14741 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), assert_equal_14727, *[convert_temperature_call_result_14736, list_14737], **kwargs_14740)
    
    
    # Call to assert_allclose(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to convert_temperature(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_14745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_14746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 43), list_14745, int_14746)
    # Adding element type (line 17)
    int_14747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 43), list_14745, int_14747)
    
    str_14748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 51), 'str', 'c')
    str_14749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 56), 'str', 'r')
    # Processing the call keyword arguments (line 17)
    kwargs_14750 = {}
    # Getting the type of 'sc' (line 17)
    sc_14743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 17)
    convert_temperature_14744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), sc_14743, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 17)
    convert_temperature_call_result_14751 = invoke(stypy.reporting.localization.Localization(__file__, 17, 20), convert_temperature_14744, *[list_14745, str_14748, str_14749], **kwargs_14750)
    
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_14752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 62), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    float_14753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 63), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 62), list_14752, float_14753)
    # Adding element type (line 17)
    float_14754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 71), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 62), list_14752, float_14754)
    
    # Processing the call keyword arguments (line 17)
    float_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'float')
    keyword_14756 = float_14755
    float_14757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'float')
    keyword_14758 = float_14757
    kwargs_14759 = {'rtol': keyword_14756, 'atol': keyword_14758}
    # Getting the type of 'assert_allclose' (line 17)
    assert_allclose_14742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 17)
    assert_allclose_call_result_14760 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_allclose_14742, *[convert_temperature_call_result_14751, list_14752], **kwargs_14759)
    
    
    # Call to assert_allclose(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to convert_temperature(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_14764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    float_14765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 43), list_14764, float_14765)
    # Adding element type (line 19)
    float_14766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 43), list_14764, float_14766)
    
    str_14767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 61), 'str', 'Rankine')
    str_14768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 72), 'str', 'C')
    # Processing the call keyword arguments (line 19)
    kwargs_14769 = {}
    # Getting the type of 'sc' (line 19)
    sc_14762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 19)
    convert_temperature_14763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), sc_14762, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 19)
    convert_temperature_call_result_14770 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), convert_temperature_14763, *[list_14764, str_14767, str_14768], **kwargs_14769)
    
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_14771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    float_14772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), list_14771, float_14772)
    # Adding element type (line 20)
    float_14773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), list_14771, float_14773)
    
    # Processing the call keyword arguments (line 19)
    float_14774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'float')
    keyword_14775 = float_14774
    float_14776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 44), 'float')
    keyword_14777 = float_14776
    kwargs_14778 = {'rtol': keyword_14775, 'atol': keyword_14777}
    # Getting the type of 'assert_allclose' (line 19)
    assert_allclose_14761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 19)
    assert_allclose_call_result_14779 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_allclose_14761, *[convert_temperature_call_result_14770, list_14771], **kwargs_14778)
    
    
    # Call to assert_allclose(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to convert_temperature(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_14783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    float_14784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 43), list_14783, float_14784)
    # Adding element type (line 21)
    float_14785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 43), list_14783, float_14785)
    
    str_14786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 61), 'str', 'r')
    str_14787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 66), 'str', 'F')
    # Processing the call keyword arguments (line 21)
    kwargs_14788 = {}
    # Getting the type of 'sc' (line 21)
    sc_14781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 21)
    convert_temperature_14782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), sc_14781, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 21)
    convert_temperature_call_result_14789 = invoke(stypy.reporting.localization.Localization(__file__, 21, 20), convert_temperature_14782, *[list_14783, str_14786, str_14787], **kwargs_14788)
    
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_14790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    float_14791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_14790, float_14791)
    # Adding element type (line 22)
    float_14792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_14790, float_14792)
    
    # Processing the call keyword arguments (line 21)
    float_14793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 37), 'float')
    keyword_14794 = float_14793
    float_14795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'float')
    keyword_14796 = float_14795
    kwargs_14797 = {'rtol': keyword_14794, 'atol': keyword_14796}
    # Getting the type of 'assert_allclose' (line 21)
    assert_allclose_14780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 21)
    assert_allclose_call_result_14798 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert_allclose_14780, *[convert_temperature_call_result_14789, list_14790], **kwargs_14797)
    
    
    # Call to assert_allclose(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to convert_temperature(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_14802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    int_14803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 43), list_14802, int_14803)
    # Adding element type (line 23)
    int_14804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 43), list_14802, int_14804)
    
    str_14805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 53), 'str', 'fahrenheit')
    str_14806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 67), 'str', 'R')
    # Processing the call keyword arguments (line 23)
    kwargs_14807 = {}
    # Getting the type of 'sc' (line 23)
    sc_14800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 23)
    convert_temperature_14801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), sc_14800, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 23)
    convert_temperature_call_result_14808 = invoke(stypy.reporting.localization.Localization(__file__, 23, 20), convert_temperature_14801, *[list_14802, str_14805, str_14806], **kwargs_14807)
    
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_14809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    float_14810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 20), list_14809, float_14810)
    # Adding element type (line 24)
    float_14811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 20), list_14809, float_14811)
    
    # Processing the call keyword arguments (line 23)
    float_14812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'float')
    keyword_14813 = float_14812
    float_14814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 52), 'float')
    keyword_14815 = float_14814
    kwargs_14816 = {'rtol': keyword_14813, 'atol': keyword_14815}
    # Getting the type of 'assert_allclose' (line 23)
    assert_allclose_14799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 23)
    assert_allclose_call_result_14817 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_allclose_14799, *[convert_temperature_call_result_14808, list_14809], **kwargs_14816)
    
    
    # Call to assert_allclose(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to convert_temperature(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_14821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    float_14822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 43), list_14821, float_14822)
    # Adding element type (line 25)
    float_14823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 43), list_14821, float_14823)
    
    str_14824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 61), 'str', 'K')
    str_14825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 66), 'str', 'R')
    # Processing the call keyword arguments (line 25)
    kwargs_14826 = {}
    # Getting the type of 'sc' (line 25)
    sc_14819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 25)
    convert_temperature_14820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), sc_14819, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 25)
    convert_temperature_call_result_14827 = invoke(stypy.reporting.localization.Localization(__file__, 25, 20), convert_temperature_14820, *[list_14821, str_14824, str_14825], **kwargs_14826)
    
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_14828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    float_14829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 20), list_14828, float_14829)
    # Adding element type (line 26)
    float_14830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 20), list_14828, float_14830)
    
    # Processing the call keyword arguments (line 25)
    float_14831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'float')
    keyword_14832 = float_14831
    float_14833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 52), 'float')
    keyword_14834 = float_14833
    kwargs_14835 = {'rtol': keyword_14832, 'atol': keyword_14834}
    # Getting the type of 'assert_allclose' (line 25)
    assert_allclose_14818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 25)
    assert_allclose_call_result_14836 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), assert_allclose_14818, *[convert_temperature_call_result_14827, list_14828], **kwargs_14835)
    
    
    # Call to assert_allclose(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to convert_temperature(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_14840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    float_14841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 43), list_14840, float_14841)
    # Adding element type (line 27)
    float_14842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 43), list_14840, float_14842)
    
    str_14843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'str', 'rankine')
    str_14844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 68), 'str', 'kelvin')
    # Processing the call keyword arguments (line 27)
    kwargs_14845 = {}
    # Getting the type of 'sc' (line 27)
    sc_14838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'sc', False)
    # Obtaining the member 'convert_temperature' of a type (line 27)
    convert_temperature_14839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 20), sc_14838, 'convert_temperature')
    # Calling convert_temperature(args, kwargs) (line 27)
    convert_temperature_call_result_14846 = invoke(stypy.reporting.localization.Localization(__file__, 27, 20), convert_temperature_14839, *[list_14840, str_14843, str_14844], **kwargs_14845)
    
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_14847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    float_14848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), list_14847, float_14848)
    # Adding element type (line 28)
    float_14849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), list_14847, float_14849)
    
    # Processing the call keyword arguments (line 27)
    float_14850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'float')
    keyword_14851 = float_14850
    float_14852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'float')
    keyword_14853 = float_14852
    kwargs_14854 = {'rtol': keyword_14851, 'atol': keyword_14853}
    # Getting the type of 'assert_allclose' (line 27)
    assert_allclose_14837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 27)
    assert_allclose_call_result_14855 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), assert_allclose_14837, *[convert_temperature_call_result_14846, list_14847], **kwargs_14854)
    
    
    # ################# End of 'test_convert_temperature(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_convert_temperature' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_14856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14856)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_convert_temperature'
    return stypy_return_type_14856

# Assigning a type to the variable 'test_convert_temperature' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'test_convert_temperature', test_convert_temperature)

@norecursion
def test_lambda_to_nu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_lambda_to_nu'
    module_type_store = module_type_store.open_function_context('test_lambda_to_nu', 31, 0, False)
    
    # Passed parameters checking function
    test_lambda_to_nu.stypy_localization = localization
    test_lambda_to_nu.stypy_type_of_self = None
    test_lambda_to_nu.stypy_type_store = module_type_store
    test_lambda_to_nu.stypy_function_name = 'test_lambda_to_nu'
    test_lambda_to_nu.stypy_param_names_list = []
    test_lambda_to_nu.stypy_varargs_param_name = None
    test_lambda_to_nu.stypy_kwargs_param_name = None
    test_lambda_to_nu.stypy_call_defaults = defaults
    test_lambda_to_nu.stypy_call_varargs = varargs
    test_lambda_to_nu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_lambda_to_nu', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_lambda_to_nu', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_lambda_to_nu(...)' code ##################

    
    # Call to assert_equal(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to lambda2nu(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'sc' (line 32)
    sc_14860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'sc', False)
    # Obtaining the member 'speed_of_light' of a type (line 32)
    speed_of_light_14861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 30), sc_14860, 'speed_of_light')
    # Processing the call keyword arguments (line 32)
    kwargs_14862 = {}
    # Getting the type of 'sc' (line 32)
    sc_14858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'sc', False)
    # Obtaining the member 'lambda2nu' of a type (line 32)
    lambda2nu_14859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 17), sc_14858, 'lambda2nu')
    # Calling lambda2nu(args, kwargs) (line 32)
    lambda2nu_call_result_14863 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), lambda2nu_14859, *[speed_of_light_14861], **kwargs_14862)
    
    int_14864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'int')
    # Processing the call keyword arguments (line 32)
    kwargs_14865 = {}
    # Getting the type of 'assert_equal' (line 32)
    assert_equal_14857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 32)
    assert_equal_call_result_14866 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_equal_14857, *[lambda2nu_call_result_14863, int_14864], **kwargs_14865)
    
    
    # ################# End of 'test_lambda_to_nu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_lambda_to_nu' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_14867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_lambda_to_nu'
    return stypy_return_type_14867

# Assigning a type to the variable 'test_lambda_to_nu' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'test_lambda_to_nu', test_lambda_to_nu)

@norecursion
def test_nu_to_lambda(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_nu_to_lambda'
    module_type_store = module_type_store.open_function_context('test_nu_to_lambda', 35, 0, False)
    
    # Passed parameters checking function
    test_nu_to_lambda.stypy_localization = localization
    test_nu_to_lambda.stypy_type_of_self = None
    test_nu_to_lambda.stypy_type_store = module_type_store
    test_nu_to_lambda.stypy_function_name = 'test_nu_to_lambda'
    test_nu_to_lambda.stypy_param_names_list = []
    test_nu_to_lambda.stypy_varargs_param_name = None
    test_nu_to_lambda.stypy_kwargs_param_name = None
    test_nu_to_lambda.stypy_call_defaults = defaults
    test_nu_to_lambda.stypy_call_varargs = varargs
    test_nu_to_lambda.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_nu_to_lambda', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_nu_to_lambda', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_nu_to_lambda(...)' code ##################

    
    # Call to assert_equal(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to nu2lambda(...): (line 36)
    # Processing the call arguments (line 36)
    int_14871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_14872 = {}
    # Getting the type of 'sc' (line 36)
    sc_14869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'sc', False)
    # Obtaining the member 'nu2lambda' of a type (line 36)
    nu2lambda_14870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 17), sc_14869, 'nu2lambda')
    # Calling nu2lambda(args, kwargs) (line 36)
    nu2lambda_call_result_14873 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), nu2lambda_14870, *[int_14871], **kwargs_14872)
    
    # Getting the type of 'sc' (line 36)
    sc_14874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'sc', False)
    # Obtaining the member 'speed_of_light' of a type (line 36)
    speed_of_light_14875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 34), sc_14874, 'speed_of_light')
    # Processing the call keyword arguments (line 36)
    kwargs_14876 = {}
    # Getting the type of 'assert_equal' (line 36)
    assert_equal_14868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 36)
    assert_equal_call_result_14877 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert_equal_14868, *[nu2lambda_call_result_14873, speed_of_light_14875], **kwargs_14876)
    
    
    # ################# End of 'test_nu_to_lambda(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_nu_to_lambda' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_14878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_nu_to_lambda'
    return stypy_return_type_14878

# Assigning a type to the variable 'test_nu_to_lambda' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'test_nu_to_lambda', test_nu_to_lambda)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

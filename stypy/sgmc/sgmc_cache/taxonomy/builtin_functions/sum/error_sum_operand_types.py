
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sum builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableDataStructureWithTypedElements(Integer)) -> DynamicType
7:     # (IterableDataStructureWithTypedElements(Integer), Integer) -> DynamicType
8: 
9: 
10:     # Call the builtin with correct parameters
11:     ret = sum([1, 2], 3)
12: 
13:     # Call the builtin with incorrect types of parameters
14:     # Type error
15:     ret = sum([1, 2], [1, 2])
16:     # Type error
17:     ret = sum(3, [1, 2])
18:     # Type error
19:     ret = sum()
20:     # Type error
21:     ret = sum([dict(), dict()])
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sum builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to sum(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 14), list_3, int_4)
    # Adding element type (line 11)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 14), list_3, int_5)
    
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_7 = {}
    # Getting the type of 'sum' (line 11)
    sum_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 11)
    sum_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), sum_2, *[list_3, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', sum_call_result_8)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to sum(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 14), list_10, int_11)
    # Adding element type (line 15)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 14), list_10, int_12)
    
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_13, int_14)
    # Adding element type (line 15)
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_13, int_15)
    
    # Processing the call keyword arguments (line 15)
    kwargs_16 = {}
    # Getting the type of 'sum' (line 15)
    sum_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 15)
    sum_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), sum_9, *[list_10, list_13], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', sum_call_result_17)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to sum(...): (line 17)
    # Processing the call arguments (line 17)
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), list_20, int_21)
    # Adding element type (line 17)
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), list_20, int_22)
    
    # Processing the call keyword arguments (line 17)
    kwargs_23 = {}
    # Getting the type of 'sum' (line 17)
    sum_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 17)
    sum_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), sum_18, *[int_19, list_20], **kwargs_23)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', sum_call_result_24)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to sum(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_26 = {}
    # Getting the type of 'sum' (line 19)
    sum_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 19)
    sum_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), sum_25, *[], **kwargs_26)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', sum_call_result_27)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to sum(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Call to dict(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_31 = {}
    # Getting the type of 'dict' (line 21)
    dict_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'dict', False)
    # Calling dict(args, kwargs) (line 21)
    dict_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), dict_30, *[], **kwargs_31)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_29, dict_call_result_32)
    # Adding element type (line 21)
    
    # Call to dict(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_34 = {}
    # Getting the type of 'dict' (line 21)
    dict_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'dict', False)
    # Calling dict(args, kwargs) (line 21)
    dict_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 21, 23), dict_33, *[], **kwargs_34)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 14), list_29, dict_call_result_35)
    
    # Processing the call keyword arguments (line 21)
    kwargs_36 = {}
    # Getting the type of 'sum' (line 21)
    sum_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 21)
    sum_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), sum_28, *[list_29], **kwargs_36)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', sum_call_result_37)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

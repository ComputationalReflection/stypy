
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of homogeneous dicts created as literals"
4: 
5: if __name__ == '__main__':
6:     d = {
7:         "one": 1,
8:         "two": 2,
9:         "three": 3,
10:     }
11: 
12:     it_keys = d.keys()
13:     it_values = d.values()
14:     it_items = d.items()
15: 
16:     # Type error
17:     print it_keys[0] + 3
18:     # Type error
19:     print it_values[0] + "str"
20:     # Type error
21:     print it_items[0] + 3
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of homogeneous dicts created as literals')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Dict to a Name (line 6):
    
    # Obtaining an instance of the builtin type 'dict' (line 6)
    dict_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 6)
    # Adding element type (key, value) (line 6)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'str', 'one')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_3, int_4))
    # Adding element type (key, value) (line 6)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'str', 'two')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_5, int_6))
    # Adding element type (key, value) (line 6)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'three')
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_7, int_8))
    
    # Assigning a type to the variable 'd' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'd', dict_2)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to keys(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_11 = {}
    # Getting the type of 'd' (line 12)
    d_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'd', False)
    # Obtaining the member 'keys' of a type (line 12)
    keys_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 14), d_9, 'keys')
    # Calling keys(args, kwargs) (line 12)
    keys_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), keys_10, *[], **kwargs_11)
    
    # Assigning a type to the variable 'it_keys' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'it_keys', keys_call_result_12)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to values(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_15 = {}
    # Getting the type of 'd' (line 13)
    d_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'd', False)
    # Obtaining the member 'values' of a type (line 13)
    values_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), d_13, 'values')
    # Calling values(args, kwargs) (line 13)
    values_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), values_14, *[], **kwargs_15)
    
    # Assigning a type to the variable 'it_values' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'it_values', values_call_result_16)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to items(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_19 = {}
    # Getting the type of 'd' (line 14)
    d_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'd', False)
    # Obtaining the member 'items' of a type (line 14)
    items_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 15), d_17, 'items')
    # Calling items(args, kwargs) (line 14)
    items_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), items_18, *[], **kwargs_19)
    
    # Assigning a type to the variable 'it_items' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'it_items', items_call_result_20)
    
    # Obtaining the type of the subscript
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
    # Getting the type of 'it_keys' (line 17)
    it_keys_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'it_keys')
    # Obtaining the member '__getitem__' of a type (line 17)
    getitem___23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 10), it_keys_22, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 17)
    subscript_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), getitem___23, int_21)
    
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
    # Applying the binary operator '+' (line 17)
    result_add_26 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 10), '+', subscript_call_result_24, int_25)
    
    
    # Obtaining the type of the subscript
    int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
    # Getting the type of 'it_values' (line 19)
    it_values_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'it_values')
    # Obtaining the member '__getitem__' of a type (line 19)
    getitem___29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), it_values_28, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 19)
    subscript_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), getitem___29, int_27)
    
    str_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', 'str')
    # Applying the binary operator '+' (line 19)
    result_add_32 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '+', subscript_call_result_30, str_31)
    
    
    # Obtaining the type of the subscript
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'int')
    # Getting the type of 'it_items' (line 21)
    it_items_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'it_items')
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), it_items_34, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), getitem___35, int_33)
    
    int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
    # Applying the binary operator '+' (line 21)
    result_add_38 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), '+', subscript_call_result_36, int_37)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of homogeneous dicts created with list methods"
4: 
5: if __name__ == '__main__':
6:     d = dict.fromkeys([
7:         "one",
8:         "two",
9:         "three"], [
10:         1,
11:         2,
12:         3]
13:     )
14: 
15:     print d
16:     it_keys = d.keys()
17:     it_values = d.values()
18:     it_items = d.items()
19: 
20:     # Type error
21:     print it_keys[0] + 3
22:     # Type error
23:     print it_values[0] + "str"
24:     # Type error
25:     print it_items[0] + 3
26: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of homogeneous dicts created with list methods')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to fromkeys(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Obtaining an instance of the builtin type 'list' (line 6)
    list_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 6)
    # Adding element type (line 6)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'str', 'one')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_4, str_5)
    # Adding element type (line 6)
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'str', 'two')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_4, str_6)
    # Adding element type (line 6)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'three')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_4, str_7)
    
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 18), list_8, int_9)
    # Adding element type (line 9)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 18), list_8, int_10)
    # Adding element type (line 9)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 18), list_8, int_11)
    
    # Processing the call keyword arguments (line 6)
    kwargs_12 = {}
    # Getting the type of 'dict' (line 6)
    dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'dict', False)
    # Obtaining the member 'fromkeys' of a type (line 6)
    fromkeys_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, 'fromkeys')
    # Calling fromkeys(args, kwargs) (line 6)
    fromkeys_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 6, 8), fromkeys_3, *[list_4, list_8], **kwargs_12)
    
    # Assigning a type to the variable 'd' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'd', fromkeys_call_result_13)
    # Getting the type of 'd' (line 15)
    d_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'd')
    
    # Assigning a Call to a Name (line 16):
    
    # Call to keys(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_17 = {}
    # Getting the type of 'd' (line 16)
    d_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'd', False)
    # Obtaining the member 'keys' of a type (line 16)
    keys_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), d_15, 'keys')
    # Calling keys(args, kwargs) (line 16)
    keys_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), keys_16, *[], **kwargs_17)
    
    # Assigning a type to the variable 'it_keys' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'it_keys', keys_call_result_18)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to values(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_21 = {}
    # Getting the type of 'd' (line 17)
    d_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'd', False)
    # Obtaining the member 'values' of a type (line 17)
    values_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), d_19, 'values')
    # Calling values(args, kwargs) (line 17)
    values_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), values_20, *[], **kwargs_21)
    
    # Assigning a type to the variable 'it_values' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'it_values', values_call_result_22)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to items(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_25 = {}
    # Getting the type of 'd' (line 18)
    d_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'd', False)
    # Obtaining the member 'items' of a type (line 18)
    items_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 15), d_23, 'items')
    # Calling items(args, kwargs) (line 18)
    items_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 18, 15), items_24, *[], **kwargs_25)
    
    # Assigning a type to the variable 'it_items' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'it_items', items_call_result_26)
    
    # Obtaining the type of the subscript
    int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    # Getting the type of 'it_keys' (line 21)
    it_keys_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'it_keys')
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), it_keys_28, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), getitem___29, int_27)
    
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'int')
    # Applying the binary operator '+' (line 21)
    result_add_32 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), '+', subscript_call_result_30, int_31)
    
    
    # Obtaining the type of the subscript
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'int')
    # Getting the type of 'it_values' (line 23)
    it_values_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'it_values')
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 10), it_values_34, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), getitem___35, int_33)
    
    str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', 'str')
    # Applying the binary operator '+' (line 23)
    result_add_38 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 10), '+', subscript_call_result_36, str_37)
    
    
    # Obtaining the type of the subscript
    int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
    # Getting the type of 'it_items' (line 25)
    it_items_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'it_items')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), it_items_40, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), getitem___41, int_39)
    
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'int')
    # Applying the binary operator '+' (line 25)
    result_add_44 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 10), '+', subscript_call_result_42, int_43)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of heterogeneous keys and values created with dict methods"
4: 
5: if __name__ == '__main__':
6:     d = dict.fromkeys([
7:         "one",
8:         2,
9:         "three"], "None"
10:     )
11: 
12:     d["two"] = 2
13: 
14:     it_keys = d.keys()
15:     it_values = d.values()
16:     it_items = d.items()
17: 
18:     # Type warning
19:     print it_keys[0] + 3
20:     # Type warning
21:     print it_values[0] + "str"
22:     # Type error
23:     print it_items[0] + 3
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of heterogeneous keys and values created with dict methods')
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
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_4, int_6)
    # Adding element type (line 6)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'three')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_4, str_7)
    
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'str', 'None')
    # Processing the call keyword arguments (line 6)
    kwargs_9 = {}
    # Getting the type of 'dict' (line 6)
    dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'dict', False)
    # Obtaining the member 'fromkeys' of a type (line 6)
    fromkeys_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, 'fromkeys')
    # Calling fromkeys(args, kwargs) (line 6)
    fromkeys_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 6, 8), fromkeys_3, *[list_4, str_8], **kwargs_9)
    
    # Assigning a type to the variable 'd' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'd', fromkeys_call_result_10)
    
    # Assigning a Num to a Subscript (line 12):
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
    # Getting the type of 'd' (line 12)
    d_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'd')
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 6), 'str', 'two')
    # Storing an element on a container (line 12)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), d_12, (str_13, int_11))
    
    # Assigning a Call to a Name (line 14):
    
    # Call to keys(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_16 = {}
    # Getting the type of 'd' (line 14)
    d_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'd', False)
    # Obtaining the member 'keys' of a type (line 14)
    keys_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), d_14, 'keys')
    # Calling keys(args, kwargs) (line 14)
    keys_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), keys_15, *[], **kwargs_16)
    
    # Assigning a type to the variable 'it_keys' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'it_keys', keys_call_result_17)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to values(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_20 = {}
    # Getting the type of 'd' (line 15)
    d_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'd', False)
    # Obtaining the member 'values' of a type (line 15)
    values_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 16), d_18, 'values')
    # Calling values(args, kwargs) (line 15)
    values_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 15, 16), values_19, *[], **kwargs_20)
    
    # Assigning a type to the variable 'it_values' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'it_values', values_call_result_21)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to items(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_24 = {}
    # Getting the type of 'd' (line 16)
    d_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'd', False)
    # Obtaining the member 'items' of a type (line 16)
    items_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 15), d_22, 'items')
    # Calling items(args, kwargs) (line 16)
    items_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), items_23, *[], **kwargs_24)
    
    # Assigning a type to the variable 'it_items' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'it_items', items_call_result_25)
    
    # Obtaining the type of the subscript
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'int')
    # Getting the type of 'it_keys' (line 19)
    it_keys_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'it_keys')
    # Obtaining the member '__getitem__' of a type (line 19)
    getitem___28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), it_keys_27, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 19)
    subscript_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), getitem___28, int_26)
    
    int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'int')
    # Applying the binary operator '+' (line 19)
    result_add_31 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '+', subscript_call_result_29, int_30)
    
    
    # Obtaining the type of the subscript
    int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
    # Getting the type of 'it_values' (line 21)
    it_values_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'it_values')
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), it_values_33, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), getitem___34, int_32)
    
    str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', 'str')
    # Applying the binary operator '+' (line 21)
    result_add_37 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), '+', subscript_call_result_35, str_36)
    
    
    # Obtaining the type of the subscript
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'int')
    # Getting the type of 'it_items' (line 23)
    it_items_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'it_items')
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 10), it_items_39, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), getitem___40, int_38)
    
    int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    # Applying the binary operator '+' (line 23)
    result_add_43 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 10), '+', subscript_call_result_41, int_42)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

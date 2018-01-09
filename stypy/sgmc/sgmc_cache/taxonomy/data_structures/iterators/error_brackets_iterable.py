
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking that the [] operation is applicable to an iterable"
4: 
5: if __name__ == '__main__':
6:     it_list = iter(range(5))
7:     it_tuple = iter(tuple(range(5)))
8:     d = {
9:         "one": 1,
10:         "two": 2,
11:         "three": 3,
12:     }
13: 
14:     it_keys = iter(d.keys())
15:     it_values = iter(d.values())
16:     it_items = iter(d.items())
17: 
18:     # Type error
19:     print it_list[3]
20:     # Type error
21:     print it_tuple[3]
22:     # Type error
23:     print it_keys[3]
24:     # Type error
25:     print it_values[3]
26:     # Type error
27:     print it_items[3]
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Checking that the [] operation is applicable to an iterable')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to iter(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_5 = {}
    # Getting the type of 'range' (line 6)
    range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 19), range_3, *[int_4], **kwargs_5)
    
    # Processing the call keyword arguments (line 6)
    kwargs_7 = {}
    # Getting the type of 'iter' (line 6)
    iter_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'iter', False)
    # Calling iter(args, kwargs) (line 6)
    iter_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), iter_2, *[range_call_result_6], **kwargs_7)
    
    # Assigning a type to the variable 'it_list' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_list', iter_call_result_8)
    
    # Assigning a Call to a Name (line 7):
    
    # Call to iter(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Call to tuple(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Call to range(...): (line 7)
    # Processing the call arguments (line 7)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_13 = {}
    # Getting the type of 'range' (line 7)
    range_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'range', False)
    # Calling range(args, kwargs) (line 7)
    range_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 7, 26), range_11, *[int_12], **kwargs_13)
    
    # Processing the call keyword arguments (line 7)
    kwargs_15 = {}
    # Getting the type of 'tuple' (line 7)
    tuple_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 7)
    tuple_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 7, 20), tuple_10, *[range_call_result_14], **kwargs_15)
    
    # Processing the call keyword arguments (line 7)
    kwargs_17 = {}
    # Getting the type of 'iter' (line 7)
    iter_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 7)
    iter_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 7, 15), iter_9, *[tuple_call_result_16], **kwargs_17)
    
    # Assigning a type to the variable 'it_tuple' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'it_tuple', iter_call_result_18)
    
    # Assigning a Dict to a Name (line 8):
    
    # Obtaining an instance of the builtin type 'dict' (line 8)
    dict_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 8)
    # Adding element type (key, value) (line 8)
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'one')
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), dict_19, (str_20, int_21))
    # Adding element type (key, value) (line 8)
    str_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'two')
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), dict_19, (str_22, int_23))
    # Adding element type (key, value) (line 8)
    str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'str', 'three')
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), dict_19, (str_24, int_25))
    
    # Assigning a type to the variable 'd' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'd', dict_19)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to iter(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to keys(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_29 = {}
    # Getting the type of 'd' (line 14)
    d_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'd', False)
    # Obtaining the member 'keys' of a type (line 14)
    keys_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 19), d_27, 'keys')
    # Calling keys(args, kwargs) (line 14)
    keys_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 14, 19), keys_28, *[], **kwargs_29)
    
    # Processing the call keyword arguments (line 14)
    kwargs_31 = {}
    # Getting the type of 'iter' (line 14)
    iter_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'iter', False)
    # Calling iter(args, kwargs) (line 14)
    iter_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), iter_26, *[keys_call_result_30], **kwargs_31)
    
    # Assigning a type to the variable 'it_keys' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'it_keys', iter_call_result_32)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to iter(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to values(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_36 = {}
    # Getting the type of 'd' (line 15)
    d_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'd', False)
    # Obtaining the member 'values' of a type (line 15)
    values_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), d_34, 'values')
    # Calling values(args, kwargs) (line 15)
    values_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), values_35, *[], **kwargs_36)
    
    # Processing the call keyword arguments (line 15)
    kwargs_38 = {}
    # Getting the type of 'iter' (line 15)
    iter_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'iter', False)
    # Calling iter(args, kwargs) (line 15)
    iter_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 15, 16), iter_33, *[values_call_result_37], **kwargs_38)
    
    # Assigning a type to the variable 'it_values' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'it_values', iter_call_result_39)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to iter(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to items(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_43 = {}
    # Getting the type of 'd' (line 16)
    d_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'd', False)
    # Obtaining the member 'items' of a type (line 16)
    items_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), d_41, 'items')
    # Calling items(args, kwargs) (line 16)
    items_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), items_42, *[], **kwargs_43)
    
    # Processing the call keyword arguments (line 16)
    kwargs_45 = {}
    # Getting the type of 'iter' (line 16)
    iter_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 16)
    iter_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), iter_40, *[items_call_result_44], **kwargs_45)
    
    # Assigning a type to the variable 'it_items' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'it_items', iter_call_result_46)
    
    # Obtaining the type of the subscript
    int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'int')
    # Getting the type of 'it_list' (line 19)
    it_list_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'it_list')
    # Obtaining the member '__getitem__' of a type (line 19)
    getitem___49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), it_list_48, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 19)
    subscript_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), getitem___49, int_47)
    
    
    # Obtaining the type of the subscript
    int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'int')
    # Getting the type of 'it_tuple' (line 21)
    it_tuple_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'it_tuple')
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), it_tuple_52, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), getitem___53, int_51)
    
    
    # Obtaining the type of the subscript
    int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
    # Getting the type of 'it_keys' (line 23)
    it_keys_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'it_keys')
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 10), it_keys_56, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), getitem___57, int_55)
    
    
    # Obtaining the type of the subscript
    int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
    # Getting the type of 'it_values' (line 25)
    it_values_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'it_values')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), it_values_60, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), getitem___61, int_59)
    
    
    # Obtaining the type of the subscript
    int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'int')
    # Getting the type of 'it_items' (line 27)
    it_items_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'it_items')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 10), it_items_64, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), getitem___65, int_63)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

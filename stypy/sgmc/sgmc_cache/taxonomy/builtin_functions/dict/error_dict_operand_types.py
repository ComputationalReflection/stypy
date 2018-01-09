
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "dict builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'dict'>
7:     # (<type dict>) -> <type 'dict'>
8:     # (IterableObject) -> <type 'dict'>
9: 
10: 
11:     # Call the builtin with correct parameters
12:     # No error
13:     ret = dict()
14: 
15:     d = {1: 2,
16:          3: 4}
17: 
18:     # No error
19:     ret = dict(d)
20:     # No error
21:     ret = dict([(1, 2), (3, 4)])
22: 
23:     # Call the builtin with incorrect types of parameters
24:     # Type error
25:     ret = dict([1, 2, 3, 4])
26: 
27:     # Type error
28:     ret = dict(4)
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'dict builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to dict(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_3 = {}
    # Getting the type of 'dict' (line 13)
    dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 13)
    dict_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), dict_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', dict_call_result_4)
    
    # Assigning a Dict to a Name (line 15):
    
    # Obtaining an instance of the builtin type 'dict' (line 15)
    dict_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 15)
    # Adding element type (key, value) (line 15)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'int')
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 8), dict_5, (int_6, int_7))
    # Adding element type (key, value) (line 15)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'int')
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 8), dict_5, (int_8, int_9))
    
    # Assigning a type to the variable 'd' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'd', dict_5)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to dict(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'd' (line 19)
    d_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'd', False)
    # Processing the call keyword arguments (line 19)
    kwargs_12 = {}
    # Getting the type of 'dict' (line 19)
    dict_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 19)
    dict_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), dict_10, *[d_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', dict_call_result_13)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to dict(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 17), tuple_16, int_17)
    # Adding element type (line 21)
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 17), tuple_16, int_18)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_15, tuple_16)
    # Adding element type (line 21)
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), tuple_19, int_20)
    # Adding element type (line 21)
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), tuple_19, int_21)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_15, tuple_19)
    
    # Processing the call keyword arguments (line 21)
    kwargs_22 = {}
    # Getting the type of 'dict' (line 21)
    dict_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 21)
    dict_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), dict_14, *[list_15], **kwargs_22)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', dict_call_result_23)
    
    # Assigning a Call to a Name (line 25):
    
    # Call to dict(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_25, int_26)
    # Adding element type (line 25)
    int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_25, int_27)
    # Adding element type (line 25)
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_25, int_28)
    # Adding element type (line 25)
    int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_25, int_29)
    
    # Processing the call keyword arguments (line 25)
    kwargs_30 = {}
    # Getting the type of 'dict' (line 25)
    dict_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 25)
    dict_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), dict_24, *[list_25], **kwargs_30)
    
    # Assigning a type to the variable 'ret' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'ret', dict_call_result_31)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to dict(...): (line 28)
    # Processing the call arguments (line 28)
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_34 = {}
    # Getting the type of 'dict' (line 28)
    dict_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 28)
    dict_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), dict_32, *[int_33], **kwargs_34)
    
    # Assigning a type to the variable 'ret' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ret', dict_call_result_35)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

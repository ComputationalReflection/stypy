
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "enumerate builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'enumerate'>
7:     # (IterableObject) -> <type 'enumerate'>
8:     # (Has__iter__) -> <type 'enumerate'>
9:     # (IterableObject, Integer) -> <type 'enumerate'>
10:     # (Has__iter__, Integer) -> <type 'enumerate'>
11: 
12: 
13:     # Call the builtin with correct parameters
14:     # No error
15:     ret = enumerate("str")
16:     # No error
17:     ret = enumerate([1, 2, 3])
18:     # No error
19:     ret = enumerate([1, 2, 3], 3)
20: 
21:     # Call the builtin with incorrect types of parameters
22:     # Type error
23:     ret = enumerate(3)
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'enumerate builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to enumerate(...): (line 15)
    # Processing the call arguments (line 15)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'str', 'str')
    # Processing the call keyword arguments (line 15)
    kwargs_4 = {}
    # Getting the type of 'enumerate' (line 15)
    enumerate_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 15)
    enumerate_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), enumerate_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', enumerate_call_result_5)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to enumerate(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), list_7, int_8)
    # Adding element type (line 17)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), list_7, int_9)
    # Adding element type (line 17)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), list_7, int_10)
    
    # Processing the call keyword arguments (line 17)
    kwargs_11 = {}
    # Getting the type of 'enumerate' (line 17)
    enumerate_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 17)
    enumerate_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), enumerate_6, *[list_7], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', enumerate_call_result_12)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to enumerate(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), list_14, int_15)
    # Adding element type (line 19)
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), list_14, int_16)
    # Adding element type (line 19)
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), list_14, int_17)
    
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_19 = {}
    # Getting the type of 'enumerate' (line 19)
    enumerate_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 19)
    enumerate_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), enumerate_13, *[list_14, int_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', enumerate_call_result_20)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to enumerate(...): (line 23)
    # Processing the call arguments (line 23)
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_23 = {}
    # Getting the type of 'enumerate' (line 23)
    enumerate_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 23)
    enumerate_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), enumerate_21, *[int_22], **kwargs_23)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', enumerate_call_result_24)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

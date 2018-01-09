
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "iter builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> DynamicType
7:     # (IterableObject) -> DynamicType
8:     # (IterableObject, AnyType) -> DynamicType
9:     # (Has__call__, AnyType) -> DynamicType
10: 
11: 
12:     # Call the builtin with correct parameters
13:     # No error
14:     ret = iter("str")
15:     # No error
16:     ret = iter([1, 2, 3])
17: 
18:     # Call the builtin with incorrect types of parameters
19:     # Type error
20:     ret = iter(3)
21:     # Type error
22:     ret = iter([1, 2, 3], 3)
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'iter builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to iter(...): (line 14)
    # Processing the call arguments (line 14)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', 'str')
    # Processing the call keyword arguments (line 14)
    kwargs_4 = {}
    # Getting the type of 'iter' (line 14)
    iter_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 14)
    iter_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), iter_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', iter_call_result_5)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to iter(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), list_7, int_8)
    # Adding element type (line 16)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), list_7, int_9)
    # Adding element type (line 16)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), list_7, int_10)
    
    # Processing the call keyword arguments (line 16)
    kwargs_11 = {}
    # Getting the type of 'iter' (line 16)
    iter_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 16)
    iter_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), iter_6, *[list_7], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', iter_call_result_12)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to iter(...): (line 20)
    # Processing the call arguments (line 20)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_15 = {}
    # Getting the type of 'iter' (line 20)
    iter_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 20)
    iter_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), iter_13, *[int_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', iter_call_result_16)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to iter(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_18, int_19)
    # Adding element type (line 22)
    int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_18, int_20)
    # Adding element type (line 22)
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_18, int_21)
    
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_23 = {}
    # Getting the type of 'iter' (line 22)
    iter_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 22)
    iter_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), iter_17, *[list_18, int_22], **kwargs_23)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', iter_call_result_24)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

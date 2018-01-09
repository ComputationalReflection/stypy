
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "max builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> DynamicType
7:     # (Str) -> DynamicType
8:     # (IterableObject, Has__call__) -> DynamicType
9:     # (Str, Has__call__) -> DynamicType
10:     # (AnyType, AnyType) -> DynamicType
11:     # (AnyType, AnyType, Has__call__) -> DynamicType
12:     # (AnyType, AnyType, AnyType) -> DynamicType
13:     # (AnyType, AnyType, AnyType, Has__call__) -> DynamicType
14:     # (AnyType, VarArgs) -> DynamicType
15: 
16:     # Call the builtin with correct parameters
17:     ret = max(2, 3)
18:     ret = max([1, 2, 3])
19:     ret = max("str")
20: 
21:     # Call the builtin with incorrect types of parameters
22:     # Type error
23:     ret = max(3)
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'max builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 17):
    
    # Call to max(...): (line 17)
    # Processing the call arguments (line 17)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_5 = {}
    # Getting the type of 'max' (line 17)
    max_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'max', False)
    # Calling max(args, kwargs) (line 17)
    max_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), max_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', max_call_result_6)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to max(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_8, int_9)
    # Adding element type (line 18)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_8, int_10)
    # Adding element type (line 18)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_8, int_11)
    
    # Processing the call keyword arguments (line 18)
    kwargs_12 = {}
    # Getting the type of 'max' (line 18)
    max_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'max', False)
    # Calling max(args, kwargs) (line 18)
    max_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), max_7, *[list_8], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', max_call_result_13)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to max(...): (line 19)
    # Processing the call arguments (line 19)
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'str', 'str')
    # Processing the call keyword arguments (line 19)
    kwargs_16 = {}
    # Getting the type of 'max' (line 19)
    max_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'max', False)
    # Calling max(args, kwargs) (line 19)
    max_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), max_14, *[str_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', max_call_result_17)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to max(...): (line 23)
    # Processing the call arguments (line 23)
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_20 = {}
    # Getting the type of 'max' (line 23)
    max_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'max', False)
    # Calling max(args, kwargs) (line 23)
    max_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), max_18, *[int_19], **kwargs_20)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', max_call_result_21)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

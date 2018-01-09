
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "type builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> TypeObjectOfParam(1)
7:     # (Str, <type tuple>, <type dict>) -> DynamicType
8: 
9: 
10:     # Call the builtin with correct parameters
11:     ret = type(3)
12:     # Type warning
13:     ret = type("TypeName", (list,), {'a': 0})
14: 
15:     # Call the builtin with incorrect types of parameters
16:     # Type error
17:     ret = type("TypeName2", (list,), ())
18:     # Type error
19:     ret = type("TypeName2", [])
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'type builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to type(...): (line 11)
    # Processing the call arguments (line 11)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_4 = {}
    # Getting the type of 'type' (line 11)
    type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'type', False)
    # Calling type(args, kwargs) (line 11)
    type_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), type_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', type_call_result_5)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to type(...): (line 13)
    # Processing the call arguments (line 13)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'TypeName')
    
    # Obtaining an instance of the builtin type 'tuple' (line 13)
    tuple_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 13)
    # Adding element type (line 13)
    # Getting the type of 'list' (line 13)
    list_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 28), tuple_8, list_9)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 13)
    dict_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 36), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 13)
    # Adding element type (key, value) (line 13)
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 37), 'str', 'a')
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 42), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 36), dict_10, (str_11, int_12))
    
    # Processing the call keyword arguments (line 13)
    kwargs_13 = {}
    # Getting the type of 'type' (line 13)
    type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'type', False)
    # Calling type(args, kwargs) (line 13)
    type_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), type_6, *[str_7, tuple_8, dict_10], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', type_call_result_14)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to type(...): (line 17)
    # Processing the call arguments (line 17)
    str_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'str', 'TypeName2')
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    # Getting the type of 'list' (line 17)
    list_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 29), tuple_17, list_18)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    
    # Processing the call keyword arguments (line 17)
    kwargs_20 = {}
    # Getting the type of 'type' (line 17)
    type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'type', False)
    # Calling type(args, kwargs) (line 17)
    type_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), type_15, *[str_16, tuple_17, tuple_19], **kwargs_20)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', type_call_result_21)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to type(...): (line 19)
    # Processing the call arguments (line 19)
    str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'str', 'TypeName2')
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    
    # Processing the call keyword arguments (line 19)
    kwargs_25 = {}
    # Getting the type of 'type' (line 19)
    type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'type', False)
    # Calling type(args, kwargs) (line 19)
    type_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), type_22, *[str_23, list_24], **kwargs_25)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', type_call_result_26)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

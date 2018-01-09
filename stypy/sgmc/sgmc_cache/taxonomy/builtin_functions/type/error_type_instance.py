
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "type builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> TypeObjectOfParam(1)
7:     # (Str, <type tuple>, <type dict>) -> DynamicType
8: 
9: 
10:     # Type error
11:     ret = type(str, (list,), ())
12:     # Type error
13:     ret = type("TypeName2", tuple)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'type builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to type(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'str' (line 11)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 11)
    tuple_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 11)
    # Adding element type (line 11)
    # Getting the type of 'list' (line 11)
    list_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 21), tuple_4, list_5)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 11)
    tuple_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 11)
    
    # Processing the call keyword arguments (line 11)
    kwargs_7 = {}
    # Getting the type of 'type' (line 11)
    type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'type', False)
    # Calling type(args, kwargs) (line 11)
    type_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), type_2, *[str_3, tuple_4, tuple_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', type_call_result_8)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to type(...): (line 13)
    # Processing the call arguments (line 13)
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'TypeName2')
    # Getting the type of 'tuple' (line 13)
    tuple_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'tuple', False)
    # Processing the call keyword arguments (line 13)
    kwargs_12 = {}
    # Getting the type of 'type' (line 13)
    type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'type', False)
    # Calling type(args, kwargs) (line 13)
    type_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), type_9, *[str_10, tuple_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', type_call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

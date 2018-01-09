
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "tuple builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'tuple'>
7:     # (Str) -> <type 'tuple'>
8:     # (IterableObject) -> <type 'tuple'>
9: 
10: 
11:     # Call the builtin with correct parameters
12:     ret = tuple()
13:     ret = tuple("str")
14:     ret = tuple([1, 2])
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = tuple(3)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'tuple builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to tuple(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_3 = {}
    # Getting the type of 'tuple' (line 12)
    tuple_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 12)
    tuple_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), tuple_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', tuple_call_result_4)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to tuple(...): (line 13)
    # Processing the call arguments (line 13)
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'str', 'str')
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'tuple' (line 13)
    tuple_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 13)
    tuple_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_5, *[str_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', tuple_call_result_8)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to tuple(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_10, int_11)
    # Adding element type (line 14)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_10, int_12)
    
    # Processing the call keyword arguments (line 14)
    kwargs_13 = {}
    # Getting the type of 'tuple' (line 14)
    tuple_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 14)
    tuple_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), tuple_9, *[list_10], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', tuple_call_result_14)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to tuple(...): (line 18)
    # Processing the call arguments (line 18)
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_17 = {}
    # Getting the type of 'tuple' (line 18)
    tuple_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 18)
    tuple_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), tuple_15, *[int_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', tuple_call_result_18)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

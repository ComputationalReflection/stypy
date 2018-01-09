
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "list builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'list'>
7:     # (IterableObject) -> <type 'list'>
8:     # (Str) -> <type 'list'>
9: 
10: 
11:     # Call the builtin with correct parameters
12:     # No error
13:     ret = list()
14:     # No error
15:     ret = list("str")
16:     # No error
17:     ret = list([1, 2])
18: 
19:     # Call the builtin with incorrect types of parameters
20:     # Type error
21:     ret = list(4)
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'list builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to list(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_3 = {}
    # Getting the type of 'list' (line 13)
    list_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'list', False)
    # Calling list(args, kwargs) (line 13)
    list_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), list_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', list_call_result_4)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to list(...): (line 15)
    # Processing the call arguments (line 15)
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'str')
    # Processing the call keyword arguments (line 15)
    kwargs_7 = {}
    # Getting the type of 'list' (line 15)
    list_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'list', False)
    # Calling list(args, kwargs) (line 15)
    list_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), list_5, *[str_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', list_call_result_8)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to list(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 15), list_10, int_11)
    # Adding element type (line 17)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 15), list_10, int_12)
    
    # Processing the call keyword arguments (line 17)
    kwargs_13 = {}
    # Getting the type of 'list' (line 17)
    list_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'list', False)
    # Calling list(args, kwargs) (line 17)
    list_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), list_9, *[list_10], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', list_call_result_14)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to list(...): (line 21)
    # Processing the call arguments (line 21)
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_17 = {}
    # Getting the type of 'list' (line 21)
    list_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'list', False)
    # Calling list(args, kwargs) (line 21)
    list_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), list_15, *[int_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', list_call_result_18)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

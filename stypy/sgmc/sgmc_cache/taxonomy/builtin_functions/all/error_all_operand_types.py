
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "all builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'bool'>
7:     # (Str) -> <type 'bool'>
8: 
9: 
10:     # Call the builtin with correct parameters
11:     # No error
12:     ret = all([1, 2, 3])
13:     # No error
14:     ret = all("str")
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = all(4)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'all builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to all(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_3, int_4)
    # Adding element type (line 12)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_3, int_5)
    # Adding element type (line 12)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_3, int_6)
    
    # Processing the call keyword arguments (line 12)
    kwargs_7 = {}
    # Getting the type of 'all' (line 12)
    all_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'all', False)
    # Calling all(args, kwargs) (line 12)
    all_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), all_2, *[list_3], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', all_call_result_8)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to all(...): (line 14)
    # Processing the call arguments (line 14)
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'str', 'str')
    # Processing the call keyword arguments (line 14)
    kwargs_11 = {}
    # Getting the type of 'all' (line 14)
    all_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'all', False)
    # Calling all(args, kwargs) (line 14)
    all_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), all_9, *[str_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', all_call_result_12)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to all(...): (line 18)
    # Processing the call arguments (line 18)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_15 = {}
    # Getting the type of 'all' (line 18)
    all_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'all', False)
    # Calling all(args, kwargs) (line 18)
    all_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), all_13, *[int_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', all_call_result_16)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

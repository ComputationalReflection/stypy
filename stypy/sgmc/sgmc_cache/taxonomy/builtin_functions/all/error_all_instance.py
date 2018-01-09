
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "all builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'bool'>
7:     # (Str) -> <type 'bool'>
8: 
9: 
10:     # Type error
11:     ret = all(list)
12:     # Type error
13:     ret = all(str)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'all builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to all(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'list' (line 11)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'list', False)
    # Processing the call keyword arguments (line 11)
    kwargs_4 = {}
    # Getting the type of 'all' (line 11)
    all_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'all', False)
    # Calling all(args, kwargs) (line 11)
    all_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), all_2, *[list_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', all_call_result_5)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to all(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'str' (line 13)
    str_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'str', False)
    # Processing the call keyword arguments (line 13)
    kwargs_8 = {}
    # Getting the type of 'all' (line 13)
    all_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'all', False)
    # Calling all(args, kwargs) (line 13)
    all_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), all_6, *[str_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', all_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

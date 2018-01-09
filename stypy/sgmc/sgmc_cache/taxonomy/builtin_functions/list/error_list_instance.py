
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "list builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'list'>
7:     # (IterableObject) -> <type 'list'>
8:     # (Str) -> <type 'list'>
9: 
10: 
11:     # Type error
12:     ret = list(str)
13:     # Type error
14:     ret = list(list)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'list builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to list(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'str' (line 12)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', False)
    # Processing the call keyword arguments (line 12)
    kwargs_4 = {}
    # Getting the type of 'list' (line 12)
    list_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'list', False)
    # Calling list(args, kwargs) (line 12)
    list_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), list_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', list_call_result_5)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to list(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'list' (line 14)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'list', False)
    # Processing the call keyword arguments (line 14)
    kwargs_8 = {}
    # Getting the type of 'list' (line 14)
    list_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'list', False)
    # Calling list(args, kwargs) (line 14)
    list_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), list_6, *[list_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', list_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "tuple builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'tuple'>
7:     # (Str) -> <type 'tuple'>
8:     # (IterableObject) -> <type 'tuple'>
9: 
10:     # Type error
11:     ret = tuple(str)
12:     # Type error
13:     ret = tuple(list)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'tuple builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to tuple(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'str' (line 11)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'str', False)
    # Processing the call keyword arguments (line 11)
    kwargs_4 = {}
    # Getting the type of 'tuple' (line 11)
    tuple_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 11)
    tuple_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), tuple_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', tuple_call_result_5)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to tuple(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'list' (line 13)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'list', False)
    # Processing the call keyword arguments (line 13)
    kwargs_8 = {}
    # Getting the type of 'tuple' (line 13)
    tuple_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 13)
    tuple_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_6, *[list_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', tuple_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

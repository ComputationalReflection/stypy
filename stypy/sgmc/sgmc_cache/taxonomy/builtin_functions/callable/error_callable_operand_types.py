
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "callable builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> <type 'bool'>
7: 
8:     # No error
9:     ret = callable(list)
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'callable builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 9):
    
    # Call to callable(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'list' (line 9)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'list', False)
    # Processing the call keyword arguments (line 9)
    kwargs_4 = {}
    # Getting the type of 'callable' (line 9)
    callable_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'callable', False)
    # Calling callable(args, kwargs) (line 9)
    callable_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 9, 10), callable_2, *[list_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'ret', callable_call_result_5)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "format builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> <type 'str'>
7:     # (AnyType, Str) -> <type 'str'>
8: 
9: 
10:     # Type error
11:     ret = format(int, str)
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'format builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to format(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'int' (line 11)
    int_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'int', False)
    # Getting the type of 'str' (line 11)
    str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', False)
    # Processing the call keyword arguments (line 11)
    kwargs_5 = {}
    # Getting the type of 'format' (line 11)
    format_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'format', False)
    # Calling format(args, kwargs) (line 11)
    format_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), format_2, *[int_3, str_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', format_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

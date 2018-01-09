
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "tuple method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'tuple'>
7:     # (Str) -> <type 'tuple'>
8:     # (IterableObject) -> <type 'tuple'>
9: 
10: 
11:     # Call the builtin with incorrect number of parameters
12:     # Type error
13:     ret = tuple("str", "str")
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'tuple method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to tuple(...): (line 13)
    # Processing the call arguments (line 13)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'str', 'str')
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'str')
    # Processing the call keyword arguments (line 13)
    kwargs_5 = {}
    # Getting the type of 'tuple' (line 13)
    tuple_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 13)
    tuple_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_2, *[str_3, str_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', tuple_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "slice method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> <type 'slice'>
7:     # (AnyType, AnyType) -> <type 'slice'>
8:     # (AnyType, AnyType, AnyType) -> <type 'slice'>
9: 
10: 
11:     # Call the builtin with incorrect number of parameters
12:     # Type error
13:     ret = slice(3, 4, 5, 6)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'slice method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to slice(...): (line 13)
    # Processing the call arguments (line 13)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'slice' (line 13)
    slice_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'slice', False)
    # Calling slice(args, kwargs) (line 13)
    slice_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), slice_2, *[int_3, int_4, int_5, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', slice_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

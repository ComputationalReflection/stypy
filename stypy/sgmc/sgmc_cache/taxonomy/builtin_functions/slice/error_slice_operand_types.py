
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "slice builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> <type 'slice'>
7:     # (AnyType, AnyType) -> <type 'slice'>
8:     # (AnyType, AnyType, AnyType) -> <type 'slice'>
9: 
10: 
11:     # Call the builtin with correct parameters
12:     ret = slice(3, 4, 5)
13: 
14:     # Call the builtin with incorrect types of parameters
15:     # Type error
16:     ret = slice()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'slice builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to slice(...): (line 12)
    # Processing the call arguments (line 12)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_6 = {}
    # Getting the type of 'slice' (line 12)
    slice_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'slice', False)
    # Calling slice(args, kwargs) (line 12)
    slice_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), slice_2, *[int_3, int_4, int_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', slice_call_result_7)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to slice(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_9 = {}
    # Getting the type of 'slice' (line 16)
    slice_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'slice', False)
    # Calling slice(args, kwargs) (line 16)
    slice_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), slice_8, *[], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', slice_call_result_10)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

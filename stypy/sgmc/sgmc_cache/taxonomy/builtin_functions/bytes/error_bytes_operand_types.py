
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "bytes builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'str'>
7:     # (AnyType) -> <type 'str'>
8: 
9: 
10:     # Call the builtin with correct parameters
11:     # No error
12:     ret = bytes()
13:     # No error
14:     ret = bytes(5)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'bytes builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to bytes(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_3 = {}
    # Getting the type of 'bytes' (line 12)
    bytes_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'bytes', False)
    # Calling bytes(args, kwargs) (line 12)
    bytes_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), bytes_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', bytes_call_result_4)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to bytes(...): (line 14)
    # Processing the call arguments (line 14)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_7 = {}
    # Getting the type of 'bytes' (line 14)
    bytes_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'bytes', False)
    # Calling bytes(args, kwargs) (line 14)
    bytes_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), bytes_5, *[int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', bytes_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

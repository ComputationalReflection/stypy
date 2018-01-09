
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "bin builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <type 'str'>
7: 
8: 
9:     # Call the builtin with correct parameters
10:     # No error
11:     ret = bin(4)
12: 
13:     # Call the builtin with incorrect types of parameters
14:     # Type error
15:     ret = bin("str")
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'bin builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to bin(...): (line 11)
    # Processing the call arguments (line 11)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_4 = {}
    # Getting the type of 'bin' (line 11)
    bin_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'bin', False)
    # Calling bin(args, kwargs) (line 11)
    bin_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), bin_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', bin_call_result_5)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to bin(...): (line 15)
    # Processing the call arguments (line 15)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'str', 'str')
    # Processing the call keyword arguments (line 15)
    kwargs_8 = {}
    # Getting the type of 'bin' (line 15)
    bin_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'bin', False)
    # Calling bin(args, kwargs) (line 15)
    bin_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), bin_6, *[str_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', bin_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

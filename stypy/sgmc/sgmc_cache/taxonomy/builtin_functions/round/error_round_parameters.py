
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "round method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (RealNumber) -> <type 'float'>
7:     # (RealNumber, Integer) -> <type 'float'>
8:     # (RealNumber, CastsToIndex) -> <type 'float'>
9:     # (CastsToFloat) -> <type 'float'>
10:     # (CastsToFloat, Integer) -> <type 'float'>
11:     # (CastsToFloat, CastsToIndex) -> <type 'float'>
12: 
13: 
14:     # Call the builtin with incorrect number of parameters
15:     # Type error
16:     ret = round(3.4, 10, 10)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'round method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to round(...): (line 16)
    # Processing the call arguments (line 16)
    float_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'float')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_6 = {}
    # Getting the type of 'round' (line 16)
    round_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'round', False)
    # Calling round(args, kwargs) (line 16)
    round_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), round_2, *[float_3, int_4, int_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', round_call_result_7)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "round builtin is invoked, but incorrect parameter types are passed"
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
14: 
15:     # Call the builtin with correct parameters
16:     ret = round(3.4)
17:     ret = round(3.4, 10)
18: 
19:     # Call the builtin with incorrect types of parameters
20:     # Type error
21:     ret = round("str")
22:     # Type error
23:     ret = round()
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'round builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to round(...): (line 16)
    # Processing the call arguments (line 16)
    float_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'float')
    # Processing the call keyword arguments (line 16)
    kwargs_4 = {}
    # Getting the type of 'round' (line 16)
    round_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'round', False)
    # Calling round(args, kwargs) (line 16)
    round_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), round_2, *[float_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', round_call_result_5)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to round(...): (line 17)
    # Processing the call arguments (line 17)
    float_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'float')
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_9 = {}
    # Getting the type of 'round' (line 17)
    round_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'round', False)
    # Calling round(args, kwargs) (line 17)
    round_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), round_6, *[float_7, int_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', round_call_result_10)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to round(...): (line 21)
    # Processing the call arguments (line 21)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'str', 'str')
    # Processing the call keyword arguments (line 21)
    kwargs_13 = {}
    # Getting the type of 'round' (line 21)
    round_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'round', False)
    # Calling round(args, kwargs) (line 21)
    round_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), round_11, *[str_12], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', round_call_result_14)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to round(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_16 = {}
    # Getting the type of 'round' (line 23)
    round_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'round', False)
    # Calling round(args, kwargs) (line 23)
    round_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), round_15, *[], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', round_call_result_17)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

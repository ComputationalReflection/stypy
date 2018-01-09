
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Operations with None variables and builtin functions"
4: 
5: if __name__ == '__main__':
6:     import math
7: 
8:     none1 = None
9: 
10:     # Type error
11:     a = math.sin(none1)
12: 
13:     # Type error
14:     x = math.pow(none1, 4)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Operations with None variables and builtin functions')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'import math' statement (line 6)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'math', math, module_type_store)
    
    
    # Assigning a Name to a Name (line 8):
    # Getting the type of 'None' (line 8)
    None_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'None')
    # Assigning a type to the variable 'none1' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'none1', None_2)
    
    # Assigning a Call to a Name (line 11):
    
    # Call to sin(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'none1' (line 11)
    none1_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'none1', False)
    # Processing the call keyword arguments (line 11)
    kwargs_6 = {}
    # Getting the type of 'math' (line 11)
    math_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'math', False)
    # Obtaining the member 'sin' of a type (line 11)
    sin_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), math_3, 'sin')
    # Calling sin(args, kwargs) (line 11)
    sin_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), sin_4, *[none1_5], **kwargs_6)
    
    # Assigning a type to the variable 'a' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', sin_call_result_7)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to pow(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'none1' (line 14)
    none1_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'none1', False)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_12 = {}
    # Getting the type of 'math' (line 14)
    math_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'math', False)
    # Obtaining the member 'pow' of a type (line 14)
    pow_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), math_8, 'pow')
    # Calling pow(args, kwargs) (line 14)
    pow_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), pow_9, *[none1_10, int_11], **kwargs_12)
    
    # Assigning a type to the variable 'x' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'x', pow_call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

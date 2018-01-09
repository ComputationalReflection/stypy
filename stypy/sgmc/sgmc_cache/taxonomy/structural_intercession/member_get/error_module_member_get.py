
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Get the type of a member of a module"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7:     r = getattr(math, 'cos')
8:     # Type error
9:     print r + "str"
10: 
11:     r = getattr(math, 'pi')
12:     print r / 2
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Get the type of a member of a module')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to getattr(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'math' (line 7)
    math_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'math', False)
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'str', 'cos')
    # Processing the call keyword arguments (line 7)
    kwargs_5 = {}
    # Getting the type of 'getattr' (line 7)
    getattr_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 7)
    getattr_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), getattr_2, *[math_3, str_4], **kwargs_5)
    
    # Assigning a type to the variable 'r' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'r', getattr_call_result_6)
    # Getting the type of 'r' (line 9)
    r_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'r')
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'str', 'str')
    # Applying the binary operator '+' (line 9)
    result_add_9 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 10), '+', r_7, str_8)
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to getattr(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'math' (line 11)
    math_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'math', False)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', 'pi')
    # Processing the call keyword arguments (line 11)
    kwargs_13 = {}
    # Getting the type of 'getattr' (line 11)
    getattr_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 11)
    getattr_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), getattr_10, *[math_11, str_12], **kwargs_13)
    
    # Assigning a type to the variable 'r' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'r', getattr_call_result_14)
    # Getting the type of 'r' (line 12)
    r_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'r')
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
    # Applying the binary operator 'div' (line 12)
    result_div_17 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), 'div', r_15, int_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

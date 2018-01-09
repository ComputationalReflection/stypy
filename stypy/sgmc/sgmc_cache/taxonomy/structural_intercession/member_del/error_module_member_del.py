
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Del a member of a module"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7:     delattr(math, 'cos')
8:     # Type error
9:     print math.cos(3)
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Del a member of a module')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    
    
    # Call to delattr(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'math' (line 7)
    math_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'math', False)
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'str', 'cos')
    # Processing the call keyword arguments (line 7)
    kwargs_5 = {}
    # Getting the type of 'delattr' (line 7)
    delattr_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'delattr', False)
    # Calling delattr(args, kwargs) (line 7)
    delattr_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), delattr_2, *[math_3, str_4], **kwargs_5)
    
    
    # Call to cos(...): (line 9)
    # Processing the call arguments (line 9)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
    # Processing the call keyword arguments (line 9)
    kwargs_10 = {}
    # Getting the type of 'math' (line 9)
    math_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'math', False)
    # Obtaining the member 'cos' of a type (line 9)
    cos_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 10), math_7, 'cos')
    # Calling cos(args, kwargs) (line 9)
    cos_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 9, 10), cos_8, *[int_9], **kwargs_10)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

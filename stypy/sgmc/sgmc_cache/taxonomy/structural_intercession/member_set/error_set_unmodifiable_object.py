
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set a member on a unmodifiable object"
3: 
4: if __name__ == '__main__':
5:     # Type error
6:     setattr(dict, 'new_attribute', 0)
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set a member on a unmodifiable object')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Call to setattr(...): (line 6)
    # Processing the call arguments (line 6)
    # Getting the type of 'dict' (line 6)
    dict_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'dict', False)
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'str', 'new_attribute')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 35), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_6 = {}
    # Getting the type of 'setattr' (line 6)
    setattr_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 6)
    setattr_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), setattr_2, *[dict_3, str_4, int_5], **kwargs_6)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

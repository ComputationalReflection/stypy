
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "coerce builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Number, Number) -> <type 'tuple'>
7: 
8: 
9:     # Call the builtin with correct parameters
10:     # No error
11:     ret = coerce(3, 4)
12: 
13:     # Call the builtin with incorrect types of parameters
14:     # Type error
15:     ret = coerce("str", "str")
16:     # Type error
17:     ret = coerce("3", "4")
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'coerce builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to coerce(...): (line 11)
    # Processing the call arguments (line 11)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_5 = {}
    # Getting the type of 'coerce' (line 11)
    coerce_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'coerce', False)
    # Calling coerce(args, kwargs) (line 11)
    coerce_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), coerce_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', coerce_call_result_6)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to coerce(...): (line 15)
    # Processing the call arguments (line 15)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'str', 'str')
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'str', 'str')
    # Processing the call keyword arguments (line 15)
    kwargs_10 = {}
    # Getting the type of 'coerce' (line 15)
    coerce_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'coerce', False)
    # Calling coerce(args, kwargs) (line 15)
    coerce_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), coerce_7, *[str_8, str_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', coerce_call_result_11)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to coerce(...): (line 17)
    # Processing the call arguments (line 17)
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'str', '3')
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'str', '4')
    # Processing the call keyword arguments (line 17)
    kwargs_15 = {}
    # Getting the type of 'coerce' (line 17)
    coerce_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'coerce', False)
    # Calling coerce(args, kwargs) (line 17)
    coerce_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), coerce_12, *[str_13, str_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', coerce_call_result_16)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

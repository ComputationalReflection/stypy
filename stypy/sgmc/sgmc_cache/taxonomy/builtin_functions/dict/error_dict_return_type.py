
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "dict builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'dict'>
7:     # (<type dict>) -> <type 'dict'>
8:     # (IterableObject) -> <type 'dict'>
9: 
10: 
11:     # Call the builtin
12:     # No error
13:     ret = dict()
14: 
15:     # Type error
16:     ret.unexisting_method()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'dict builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to dict(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_3 = {}
    # Getting the type of 'dict' (line 13)
    dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 13)
    dict_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), dict_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', dict_call_result_4)
    
    # Call to unexisting_method(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_7 = {}
    # Getting the type of 'ret' (line 16)
    ret_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 16)
    unexisting_method_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), ret_5, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 16)
    unexisting_method_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), unexisting_method_6, *[], **kwargs_7)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

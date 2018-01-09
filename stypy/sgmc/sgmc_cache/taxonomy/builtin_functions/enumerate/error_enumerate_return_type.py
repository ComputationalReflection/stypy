
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "enumerate builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'enumerate'>
7:     # (IterableObject) -> <type 'enumerate'>
8:     # (Has__iter__) -> <type 'enumerate'>
9:     # (IterableObject, Integer) -> <type 'enumerate'>
10:     # (Has__iter__, Integer) -> <type 'enumerate'>
11: 
12: 
13:     # Call the builtin
14:     # No error
15:     ret = enumerate("str")
16: 
17:     # Type error
18:     ret.unexisting_method()
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'enumerate builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to enumerate(...): (line 15)
    # Processing the call arguments (line 15)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'str', 'str')
    # Processing the call keyword arguments (line 15)
    kwargs_4 = {}
    # Getting the type of 'enumerate' (line 15)
    enumerate_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 15)
    enumerate_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), enumerate_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', enumerate_call_result_5)
    
    # Call to unexisting_method(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_8 = {}
    # Getting the type of 'ret' (line 18)
    ret_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 18)
    unexisting_method_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), ret_6, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 18)
    unexisting_method_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), unexisting_method_7, *[], **kwargs_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__import__ builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> types.ModuleType
7:     # (Str, AnyType) -> types.ModuleType
8:     # (Str, AnyType, AnyType) -> types.ModuleType
9:     # (Str, AnyType, AnyType, AnyType) -> types.ModuleType
10:     # (Str, AnyType, AnyType, AnyType, Integer) -> types.ModuleType
11: 
12: 
13:     # Call the builtin
14:     # No error
15:     ret = __import__("math")
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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__import__ builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to __import__(...): (line 15)
    # Processing the call arguments (line 15)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'str', 'math')
    # Processing the call keyword arguments (line 15)
    kwargs_4 = {}
    # Getting the type of '__import__' (line 15)
    import___2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), '__import__', False)
    # Calling __import__(args, kwargs) (line 15)
    import___call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), import___2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', import___call_result_5)
    
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

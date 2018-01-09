
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__import__ builtin is invoked, but incorrect parameter types are passed"
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
13:     # Call the builtin with correct parameters
14:     # No error
15:     ret = __import__("math")
16: 
17:     # Call the builtin with incorrect types of parameters
18:     # Type error
19:     ret = __import__(3)
20: 
21:     # Type error
22:     ret = __import__("invented")
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__import__ builtin is invoked, but incorrect parameter types are passed')
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
    
    # Assigning a Call to a Name (line 19):
    
    # Call to __import__(...): (line 19)
    # Processing the call arguments (line 19)
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_8 = {}
    # Getting the type of '__import__' (line 19)
    import___6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), '__import__', False)
    # Calling __import__(args, kwargs) (line 19)
    import___call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), import___6, *[int_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', import___call_result_9)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to __import__(...): (line 22)
    # Processing the call arguments (line 22)
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'str', 'invented')
    # Processing the call keyword arguments (line 22)
    kwargs_12 = {}
    # Getting the type of '__import__' (line 22)
    import___10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), '__import__', False)
    # Calling __import__(args, kwargs) (line 22)
    import___call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), import___10, *[str_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', import___call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

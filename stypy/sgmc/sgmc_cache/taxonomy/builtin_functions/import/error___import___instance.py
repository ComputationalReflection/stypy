
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__import__ builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> types.ModuleType
7:     # (Str, AnyType) -> types.ModuleType
8:     # (Str, AnyType, AnyType) -> types.ModuleType
9:     # (Str, AnyType, AnyType, AnyType) -> types.ModuleType
10:     # (Str, AnyType, AnyType, AnyType, Integer) -> types.ModuleType
11: 
12:     # Type error
13:     ret = __import__(str)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__import__ builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to __import__(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'str' (line 13)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'str', False)
    # Processing the call keyword arguments (line 13)
    kwargs_4 = {}
    # Getting the type of '__import__' (line 13)
    import___2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), '__import__', False)
    # Calling __import__(args, kwargs) (line 13)
    import___call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), import___2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', import___call_result_5)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

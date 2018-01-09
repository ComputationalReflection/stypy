
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reload builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (types.ModuleType) -> types.ModuleType
7:     import math
8: 
9:     # Call the builtin with correct parameters
10:     ret = reload(math)
11: 
12:     # Call the builtin with incorrect types of parameters
13:     # Type error
14:     ret = reload(list)
15:     # Type error
16:     ret = reload()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reload builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))
    
    # 'import math' statement (line 7)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'math', math, module_type_store)
    
    
    # Assigning a Call to a Name (line 10):
    
    # Call to reload(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'math' (line 10)
    math_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'math', False)
    # Processing the call keyword arguments (line 10)
    kwargs_4 = {}
    # Getting the type of 'reload' (line 10)
    reload_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'reload', False)
    # Calling reload(args, kwargs) (line 10)
    reload_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), reload_2, *[math_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'ret', reload_call_result_5)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to reload(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'list' (line 14)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'list', False)
    # Processing the call keyword arguments (line 14)
    kwargs_8 = {}
    # Getting the type of 'reload' (line 14)
    reload_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'reload', False)
    # Calling reload(args, kwargs) (line 14)
    reload_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), reload_6, *[list_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', reload_call_result_9)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to reload(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_11 = {}
    # Getting the type of 'reload' (line 16)
    reload_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'reload', False)
    # Calling reload(args, kwargs) (line 16)
    reload_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), reload_10, *[], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', reload_call_result_12)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

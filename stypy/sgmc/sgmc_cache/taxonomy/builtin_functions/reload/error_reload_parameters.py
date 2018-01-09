
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reload method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (types.ModuleType) -> types.ModuleType
7:     import math
8: 
9:     # Call the builtin with incorrect number of parameters
10:     # Type error
11:     ret = reload(math, math)
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reload method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))
    
    # 'import math' statement (line 7)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'math', math, module_type_store)
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to reload(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'math' (line 11)
    math_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'math', False)
    # Getting the type of 'math' (line 11)
    math_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 23), 'math', False)
    # Processing the call keyword arguments (line 11)
    kwargs_5 = {}
    # Getting the type of 'reload' (line 11)
    reload_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'reload', False)
    # Calling reload(args, kwargs) (line 11)
    reload_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), reload_2, *[math_3, math_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', reload_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

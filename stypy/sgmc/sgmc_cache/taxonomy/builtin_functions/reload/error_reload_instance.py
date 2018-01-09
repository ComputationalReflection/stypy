
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reload builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (types.ModuleType) -> types.ModuleType
7:     import types
8: 
9:     # Type error
10:     ret = reload(types.ModuleType)
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reload builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))
    
    # 'import types' statement (line 7)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'types', types, module_type_store)
    
    
    # Assigning a Call to a Name (line 10):
    
    # Call to reload(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'types' (line 10)
    types_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'types', False)
    # Obtaining the member 'ModuleType' of a type (line 10)
    ModuleType_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 17), types_3, 'ModuleType')
    # Processing the call keyword arguments (line 10)
    kwargs_5 = {}
    # Getting the type of 'reload' (line 10)
    reload_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'reload', False)
    # Calling reload(args, kwargs) (line 10)
    reload_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), reload_2, *[ModuleType_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'ret', reload_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

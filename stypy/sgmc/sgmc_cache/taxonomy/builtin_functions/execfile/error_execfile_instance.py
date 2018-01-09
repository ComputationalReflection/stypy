
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "execfile builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> DynamicType
7:     # (Str, <type list>) -> DynamicType
8:     # (Str, <type list>, <type dict>) -> DynamicType
9: 
10: 
11:     # Type error
12:     ret = execfile(str, dict, dict)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'execfile builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to execfile(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'str' (line 12)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'str', False)
    # Getting the type of 'dict' (line 12)
    dict_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'dict', False)
    # Getting the type of 'dict' (line 12)
    dict_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 30), 'dict', False)
    # Processing the call keyword arguments (line 12)
    kwargs_6 = {}
    # Getting the type of 'execfile' (line 12)
    execfile_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'execfile', False)
    # Calling execfile(args, kwargs) (line 12)
    execfile_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), execfile_2, *[str_3, dict_4, dict_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', execfile_call_result_7)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

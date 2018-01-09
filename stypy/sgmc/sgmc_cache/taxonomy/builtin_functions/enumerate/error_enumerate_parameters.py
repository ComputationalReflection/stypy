
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "enumerate method is present, but is invoked with a wrong number of parameters"
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
13:     # Call the builtin with incorrect number of parameters
14:     # Type error
15:     ret = enumerate("str", list())
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'enumerate method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to enumerate(...): (line 15)
    # Processing the call arguments (line 15)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'str', 'str')
    
    # Call to list(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_5 = {}
    # Getting the type of 'list' (line 15)
    list_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'list', False)
    # Calling list(args, kwargs) (line 15)
    list_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 15, 27), list_4, *[], **kwargs_5)
    
    # Processing the call keyword arguments (line 15)
    kwargs_7 = {}
    # Getting the type of 'enumerate' (line 15)
    enumerate_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 15)
    enumerate_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), enumerate_2, *[str_3, list_call_result_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', enumerate_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

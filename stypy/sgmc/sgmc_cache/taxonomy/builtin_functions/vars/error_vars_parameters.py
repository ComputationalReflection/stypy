
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "vars method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'dict'>
7:     # (AnyType) -> <type 'dict'>
8: 
9: 
10:     # Call the builtin with incorrect number of parameters
11:     # Type error
12:     ret = vars(list, 5)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'vars method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to vars(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'list' (line 12)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'list', False)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_5 = {}
    # Getting the type of 'vars' (line 12)
    vars_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'vars', False)
    # Calling vars(args, kwargs) (line 12)
    vars_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), vars_2, *[list_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', vars_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

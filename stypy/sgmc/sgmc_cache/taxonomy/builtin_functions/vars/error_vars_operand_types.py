
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "vars builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'dict'>
7:     # (AnyType) -> <type 'dict'>
8: 
9: 
10:     # Call the builtin with correct parameters
11:     ret = vars(list)
12: 
13:     # Call the builtin with incorrect types of parameters
14:     # Type error
15:     ret = vars(list())
16:     # Type error
17:     ret = vars(3)
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'vars builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to vars(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'list' (line 11)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'list', False)
    # Processing the call keyword arguments (line 11)
    kwargs_4 = {}
    # Getting the type of 'vars' (line 11)
    vars_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'vars', False)
    # Calling vars(args, kwargs) (line 11)
    vars_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), vars_2, *[list_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', vars_call_result_5)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to vars(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to list(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_8 = {}
    # Getting the type of 'list' (line 15)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'list', False)
    # Calling list(args, kwargs) (line 15)
    list_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), list_7, *[], **kwargs_8)
    
    # Processing the call keyword arguments (line 15)
    kwargs_10 = {}
    # Getting the type of 'vars' (line 15)
    vars_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'vars', False)
    # Calling vars(args, kwargs) (line 15)
    vars_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), vars_6, *[list_call_result_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', vars_call_result_11)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to vars(...): (line 17)
    # Processing the call arguments (line 17)
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_14 = {}
    # Getting the type of 'vars' (line 17)
    vars_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'vars', False)
    # Calling vars(args, kwargs) (line 17)
    vars_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), vars_12, *[int_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', vars_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

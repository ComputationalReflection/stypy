
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "divmod builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Number, Number) -> <type 'tuple'>
7:     # (Overloads__divmod__, Number) -> <type 'tuple'>
8:     # (Number, Overloads__rdivmod__) -> <type 'tuple'>
9: 
10: 
11: 
12:     # Call the builtin with correct parameters
13:     # No error
14:     ret = divmod(3, 4)
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = divmod(list(), 4)
19:     # Type error
20:     ret = divmod(4, tuple())
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'divmod builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to divmod(...): (line 14)
    # Processing the call arguments (line 14)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_5 = {}
    # Getting the type of 'divmod' (line 14)
    divmod_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 14)
    divmod_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), divmod_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', divmod_call_result_6)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to divmod(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to list(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_9 = {}
    # Getting the type of 'list' (line 18)
    list_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'list', False)
    # Calling list(args, kwargs) (line 18)
    list_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), list_8, *[], **kwargs_9)
    
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_12 = {}
    # Getting the type of 'divmod' (line 18)
    divmod_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 18)
    divmod_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), divmod_7, *[list_call_result_10, int_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', divmod_call_result_13)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to divmod(...): (line 20)
    # Processing the call arguments (line 20)
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'int')
    
    # Call to tuple(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_17 = {}
    # Getting the type of 'tuple' (line 20)
    tuple_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 20)
    tuple_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 20, 20), tuple_16, *[], **kwargs_17)
    
    # Processing the call keyword arguments (line 20)
    kwargs_19 = {}
    # Getting the type of 'divmod' (line 20)
    divmod_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 20)
    divmod_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), divmod_14, *[int_15, tuple_call_result_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', divmod_call_result_20)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

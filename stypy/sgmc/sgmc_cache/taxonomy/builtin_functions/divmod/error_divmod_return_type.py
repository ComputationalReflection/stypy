
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "divmod builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Number, Number) -> <type 'tuple'>
7:     # (Overloads__divmod__, Number) -> <type 'tuple'>
8:     # (Number, Overloads__rdivmod__) -> <type 'tuple'>
9: 
10: 
11:     # Call the builtin
12:     # No error
13:     ret = divmod(3, 4)
14: 
15:     # Type error
16:     ret.unexisting_method()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'divmod builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to divmod(...): (line 13)
    # Processing the call arguments (line 13)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_5 = {}
    # Getting the type of 'divmod' (line 13)
    divmod_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 13)
    divmod_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), divmod_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', divmod_call_result_6)
    
    # Call to unexisting_method(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_9 = {}
    # Getting the type of 'ret' (line 16)
    ret_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 16)
    unexisting_method_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), ret_7, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 16)
    unexisting_method_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), unexisting_method_8, *[], **kwargs_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

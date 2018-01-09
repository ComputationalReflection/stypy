
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "cmp builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, AnyType) -> <type 'int'>
7: 
8: 
9:     # Call the builtin
10:     # No error
11:     ret = cmp(4, 4)
12: 
13:     # Type error
14:     ret.unexisting_method()
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'cmp builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to cmp(...): (line 11)
    # Processing the call arguments (line 11)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_5 = {}
    # Getting the type of 'cmp' (line 11)
    cmp_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'cmp', False)
    # Calling cmp(args, kwargs) (line 11)
    cmp_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), cmp_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', cmp_call_result_6)
    
    # Call to unexisting_method(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_9 = {}
    # Getting the type of 'ret' (line 14)
    ret_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 14)
    unexisting_method_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), ret_7, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 14)
    unexisting_method_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), unexisting_method_8, *[], **kwargs_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

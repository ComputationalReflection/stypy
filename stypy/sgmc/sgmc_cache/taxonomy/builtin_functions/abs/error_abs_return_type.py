
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "abs builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type bool>) -> <type 'int'>
7:     # (<type complex>) -> <type 'float'>
8:     # (Number) -> TypeOfParam(1)
9:     # (Overloads__trunc__) -> <type 'int'>
10: 
11: 
12:     # Call the builtin
13:     # No error
14:     ret = abs(3)
15: 
16:     # Type error
17:     ret.unexisting_method()
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'abs builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to abs(...): (line 14)
    # Processing the call arguments (line 14)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_4 = {}
    # Getting the type of 'abs' (line 14)
    abs_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 14)
    abs_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), abs_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', abs_call_result_5)
    
    # Call to unexisting_method(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_8 = {}
    # Getting the type of 'ret' (line 17)
    ret_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 17)
    unexisting_method_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), ret_6, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 17)
    unexisting_method_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), unexisting_method_7, *[], **kwargs_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

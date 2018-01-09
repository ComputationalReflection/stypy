
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "max builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> DynamicType
7:     # (Str) -> DynamicType
8:     # (IterableObject, Has__call__) -> DynamicType
9:     # (Str, Has__call__) -> DynamicType
10:     # (AnyType, AnyType) -> DynamicType
11:     # (AnyType, AnyType, Has__call__) -> DynamicType
12:     # (AnyType, AnyType, AnyType) -> DynamicType
13:     # (AnyType, AnyType, AnyType, Has__call__) -> DynamicType
14:     # (AnyType, VarArgs) -> DynamicType
15: 
16: 
17:     # Call the builtin
18:     ret = max(2, 3)
19: 
20:     # Type error
21:     ret.unexisting_method()
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'max builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to max(...): (line 18)
    # Processing the call arguments (line 18)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_5 = {}
    # Getting the type of 'max' (line 18)
    max_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'max', False)
    # Calling max(args, kwargs) (line 18)
    max_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), max_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', max_call_result_6)
    
    # Call to unexisting_method(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_9 = {}
    # Getting the type of 'ret' (line 21)
    ret_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 21)
    unexisting_method_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), ret_7, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 21)
    unexisting_method_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), unexisting_method_8, *[], **kwargs_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

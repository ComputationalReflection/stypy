
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "eval builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (types.CodeType) -> DynamicType
7:     # (types.CodeType, types.NoneType) -> DynamicType
8:     # (types.CodeType, <type dict>) -> DynamicType
9:     # (types.CodeType, <type dict>, types.NoneType) -> DynamicType
10:     # (Str) -> DynamicType
11:     # (Str, <type dict>) -> DynamicType
12:     # (Str, <type dict>, types.NoneType) -> DynamicType
13:     # (Str, <type dict>, <type dict>) -> DynamicType
14:     # (Str, types.NoneType, <type dict>) -> DynamicType
15: 
16: 
17:     # Call the builtin with correct parameters
18:     # Type warning
19:     ret = eval("a=5")
20: 
21:     # Call the builtin with incorrect types of parameters
22:     # Type error
23:     ret = eval(3)
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'eval builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to eval(...): (line 19)
    # Processing the call arguments (line 19)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'str', 'a=5')
    # Processing the call keyword arguments (line 19)
    kwargs_4 = {}
    # Getting the type of 'eval' (line 19)
    eval_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'eval', False)
    # Calling eval(args, kwargs) (line 19)
    eval_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), eval_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', eval_call_result_5)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to eval(...): (line 23)
    # Processing the call arguments (line 23)
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_8 = {}
    # Getting the type of 'eval' (line 23)
    eval_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'eval', False)
    # Calling eval(args, kwargs) (line 23)
    eval_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), eval_6, *[int_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', eval_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

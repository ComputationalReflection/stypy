
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "eval builtin is invoked, but a class is used instead of an instance"
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
15:     import types
16: 
17:     # Type error
18:     ret = eval(str)
19: 
20:     # Type error
21:     ret = eval(types.CodeType)
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'eval builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))
    
    # 'import types' statement (line 15)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'types', types, module_type_store)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to eval(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'str' (line 18)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'str', False)
    # Processing the call keyword arguments (line 18)
    kwargs_4 = {}
    # Getting the type of 'eval' (line 18)
    eval_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'eval', False)
    # Calling eval(args, kwargs) (line 18)
    eval_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), eval_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', eval_call_result_5)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to eval(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'types' (line 21)
    types_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'types', False)
    # Obtaining the member 'CodeType' of a type (line 21)
    CodeType_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), types_7, 'CodeType')
    # Processing the call keyword arguments (line 21)
    kwargs_9 = {}
    # Getting the type of 'eval' (line 21)
    eval_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'eval', False)
    # Calling eval(args, kwargs) (line 21)
    eval_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), eval_6, *[CodeType_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', eval_call_result_10)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

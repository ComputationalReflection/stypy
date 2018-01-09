
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sorted builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'list'>
7:     # (IterableObject, Has__call__) -> <type 'list'>
8:     # (IterableObject, Has__call__, Has__call__) -> <type 'list'>
9:     # (IterableObject, Has__call__, Has__call__, <type bool>) -> <type 'list'>
10:     # (Str) -> <type 'list'>
11:     # (Str, Has__call__) -> <type 'list'>
12:     # (Str, Has__call__, Has__call__) -> <type 'list'>
13:     # (Str, Has__call__, Has__call__, <type bool>) -> <type 'list'>
14: 
15: 
16:     # Call the builtin
17:     ret = sorted("str")
18: 
19:     # Type error
20:     ret.unexisting_method()
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sorted builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 17):
    
    # Call to sorted(...): (line 17)
    # Processing the call arguments (line 17)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'str', 'str')
    # Processing the call keyword arguments (line 17)
    kwargs_4 = {}
    # Getting the type of 'sorted' (line 17)
    sorted_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 17)
    sorted_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), sorted_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', sorted_call_result_5)
    
    # Call to unexisting_method(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_8 = {}
    # Getting the type of 'ret' (line 20)
    ret_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 20)
    unexisting_method_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), ret_6, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 20)
    unexisting_method_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), unexisting_method_7, *[], **kwargs_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

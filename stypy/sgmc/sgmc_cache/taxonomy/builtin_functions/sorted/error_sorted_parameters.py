
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sorted method is present, but is invoked with a wrong number of parameters"
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
16:     # Call the builtin with incorrect number of parameters
17:     # Type error
18:     ret = sorted("str", "str", "str", "str", "str")
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sorted method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to sorted(...): (line 18)
    # Processing the call arguments (line 18)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'str', 'str')
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'str')
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'str', 'str')
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'str', 'str')
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 45), 'str', 'str')
    # Processing the call keyword arguments (line 18)
    kwargs_8 = {}
    # Getting the type of 'sorted' (line 18)
    sorted_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 18)
    sorted_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), sorted_2, *[str_3, str_4, str_5, str_6, str_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', sorted_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

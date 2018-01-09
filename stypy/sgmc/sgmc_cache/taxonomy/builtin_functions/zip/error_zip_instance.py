
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "zip builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'list'>
7:     # (Str) -> <type 'list'>
8:     # (IterableObject, IterableObject) -> <type 'list'>
9:     # (Str, IterableObject) -> <type 'list'>
10:     # (IterableObject, Str) -> <type 'list'>
11:     # (Str, Str) -> <type 'list'>
12:     # (IterableObject, IterableObject, IterableObject) -> <type 'list'>
13:     # (Str, IterableObject, IterableObject) -> <type 'list'>
14:     # (IterableObject, Str, IterableObject) -> <type 'list'>
15:     # (IterableObject, IterableObject, Str) -> <type 'list'>
16:     # (Str, Str, IterableObject) -> <type 'list'>
17:     # (IterableObject, Str, Str) -> <type 'list'>
18:     # (Str, IterableObject, Str) -> <type 'list'>
19:     # (Str, Str, Str) -> <type 'list'>
20:     # (IterableObject, VarArgs) -> <type 'list'>
21:     # (Str, VarArgs) -> <type 'list'>
22: 
23: 
24:     # Type error
25:     ret = zip(str)
26:     # Type error
27:     ret = zip(list)
28:     # Type error
29:     ret = zip(list, list)
30:     # Type error
31:     ret = zip(str, list, list)
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'zip builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to zip(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'str' (line 25)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'str', False)
    # Processing the call keyword arguments (line 25)
    kwargs_4 = {}
    # Getting the type of 'zip' (line 25)
    zip_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 25)
    zip_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), zip_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'ret', zip_call_result_5)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to zip(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'list' (line 27)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'list', False)
    # Processing the call keyword arguments (line 27)
    kwargs_8 = {}
    # Getting the type of 'zip' (line 27)
    zip_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 27)
    zip_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), zip_6, *[list_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', zip_call_result_9)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to zip(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'list' (line 29)
    list_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'list', False)
    # Getting the type of 'list' (line 29)
    list_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'list', False)
    # Processing the call keyword arguments (line 29)
    kwargs_13 = {}
    # Getting the type of 'zip' (line 29)
    zip_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 29)
    zip_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), zip_10, *[list_11, list_12], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'ret', zip_call_result_14)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to zip(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'str' (line 31)
    str_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'str', False)
    # Getting the type of 'list' (line 31)
    list_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'list', False)
    # Getting the type of 'list' (line 31)
    list_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'list', False)
    # Processing the call keyword arguments (line 31)
    kwargs_19 = {}
    # Getting the type of 'zip' (line 31)
    zip_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 31)
    zip_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), zip_15, *[str_16, list_17, list_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ret', zip_call_result_20)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

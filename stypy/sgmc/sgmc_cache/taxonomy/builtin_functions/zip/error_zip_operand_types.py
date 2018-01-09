
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "zip builtin is invoked, but incorrect parameter types are passed"
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
24:     # Call the builtin with correct parameters
25:     ret = zip("str")
26:     ret = zip([1, 2])
27:     ret = zip([1, 2], [1, 2])
28:     ret = zip("str", [1, 2], [1, 2])
29: 
30:     # Call the builtin with incorrect types of parameters
31:     # Type error
32:     ret = zip(3)
33: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'zip builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to zip(...): (line 25)
    # Processing the call arguments (line 25)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'str', 'str')
    # Processing the call keyword arguments (line 25)
    kwargs_4 = {}
    # Getting the type of 'zip' (line 25)
    zip_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 25)
    zip_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), zip_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'ret', zip_call_result_5)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to zip(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_7, int_8)
    # Adding element type (line 26)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_7, int_9)
    
    # Processing the call keyword arguments (line 26)
    kwargs_10 = {}
    # Getting the type of 'zip' (line 26)
    zip_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 26)
    zip_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), zip_6, *[list_7], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'ret', zip_call_result_11)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to zip(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_13, int_14)
    # Adding element type (line 27)
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_13, int_15)
    
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_16, int_17)
    # Adding element type (line 27)
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_16, int_18)
    
    # Processing the call keyword arguments (line 27)
    kwargs_19 = {}
    # Getting the type of 'zip' (line 27)
    zip_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 27)
    zip_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), zip_12, *[list_13, list_16], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', zip_call_result_20)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to zip(...): (line 28)
    # Processing the call arguments (line 28)
    str_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'str', 'str')
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), list_23, int_24)
    # Adding element type (line 28)
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), list_23, int_25)
    
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 29), list_26, int_27)
    # Adding element type (line 28)
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 29), list_26, int_28)
    
    # Processing the call keyword arguments (line 28)
    kwargs_29 = {}
    # Getting the type of 'zip' (line 28)
    zip_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 28)
    zip_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), zip_21, *[str_22, list_23, list_26], **kwargs_29)
    
    # Assigning a type to the variable 'ret' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ret', zip_call_result_30)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to zip(...): (line 32)
    # Processing the call arguments (line 32)
    int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'int')
    # Processing the call keyword arguments (line 32)
    kwargs_33 = {}
    # Getting the type of 'zip' (line 32)
    zip_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'zip', False)
    # Calling zip(args, kwargs) (line 32)
    zip_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), zip_31, *[int_32], **kwargs_33)
    
    # Assigning a type to the variable 'ret' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'ret', zip_call_result_34)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

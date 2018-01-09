
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "bytearray builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'bytearray'>
7:     # (IterableDataStructureWithTypedElements(Integer, Overloads__trunc__)) -> <type 'bytearray'>
8:     # (Integer) -> <type 'bytearray'>
9:     # (Str) -> <type 'bytearray'>
10: 
11: 
12:     # Call the builtin with correct parameters
13:     # No error
14:     ret = bytearray()
15:     # No error
16:     ret = bytearray("str")
17:     # No error
18:     ret = bytearray(4)
19:     list_int = [1, 2, 3]
20:     # No error
21:     ret = bytearray(list_int)
22: 
23:     # Call the builtin with incorrect types of parameters
24:     # Type error
25:     ret = bytearray(3.4)
26:     # Type error
27:     ret = bytearray(list())
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'bytearray builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to bytearray(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_3 = {}
    # Getting the type of 'bytearray' (line 14)
    bytearray_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 14)
    bytearray_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), bytearray_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', bytearray_call_result_4)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to bytearray(...): (line 16)
    # Processing the call arguments (line 16)
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'str', 'str')
    # Processing the call keyword arguments (line 16)
    kwargs_7 = {}
    # Getting the type of 'bytearray' (line 16)
    bytearray_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 16)
    bytearray_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), bytearray_5, *[str_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', bytearray_call_result_8)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to bytearray(...): (line 18)
    # Processing the call arguments (line 18)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_11 = {}
    # Getting the type of 'bytearray' (line 18)
    bytearray_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 18)
    bytearray_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), bytearray_9, *[int_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', bytearray_call_result_12)
    
    # Assigning a List to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 15), list_13, int_14)
    # Adding element type (line 19)
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 15), list_13, int_15)
    # Adding element type (line 19)
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 15), list_13, int_16)
    
    # Assigning a type to the variable 'list_int' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'list_int', list_13)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to bytearray(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'list_int' (line 21)
    list_int_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'list_int', False)
    # Processing the call keyword arguments (line 21)
    kwargs_19 = {}
    # Getting the type of 'bytearray' (line 21)
    bytearray_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 21)
    bytearray_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), bytearray_17, *[list_int_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', bytearray_call_result_20)
    
    # Assigning a Call to a Name (line 25):
    
    # Call to bytearray(...): (line 25)
    # Processing the call arguments (line 25)
    float_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'float')
    # Processing the call keyword arguments (line 25)
    kwargs_23 = {}
    # Getting the type of 'bytearray' (line 25)
    bytearray_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 25)
    bytearray_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), bytearray_21, *[float_22], **kwargs_23)
    
    # Assigning a type to the variable 'ret' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'ret', bytearray_call_result_24)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to bytearray(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to list(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_27 = {}
    # Getting the type of 'list' (line 27)
    list_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'list', False)
    # Calling list(args, kwargs) (line 27)
    list_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 27, 20), list_26, *[], **kwargs_27)
    
    # Processing the call keyword arguments (line 27)
    kwargs_29 = {}
    # Getting the type of 'bytearray' (line 27)
    bytearray_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 27)
    bytearray_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), bytearray_25, *[list_call_result_28], **kwargs_29)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', bytearray_call_result_30)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

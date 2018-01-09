
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Inferring the type of the resulting iterator when slicing a homogeneous list"
3: 
4: if __name__ == '__main__':
5:     l = range(5)
6: 
7:     sl = l[0:3]
8: 
9:     r = sl[0]
10: 
11:     # Type error
12:     print r + "str"
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Inferring the type of the resulting iterator when slicing a homogeneous list')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 5):
    
    # Call to range(...): (line 5)
    # Processing the call arguments (line 5)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
    # Processing the call keyword arguments (line 5)
    kwargs_4 = {}
    # Getting the type of 'range' (line 5)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'range', False)
    # Calling range(args, kwargs) (line 5)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 5, 8), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'l' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'l', range_call_result_5)
    
    # Assigning a Subscript to a Name (line 7):
    
    # Obtaining the type of the subscript
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'int')
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
    slice_8 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 9), int_6, int_7, None)
    # Getting the type of 'l' (line 7)
    l_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'l')
    # Obtaining the member '__getitem__' of a type (line 7)
    getitem___10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), l_9, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 7)
    subscript_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), getitem___10, slice_8)
    
    # Assigning a type to the variable 'sl' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'sl', subscript_call_result_11)
    
    # Assigning a Subscript to a Name (line 9):
    
    # Obtaining the type of the subscript
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'int')
    # Getting the type of 'sl' (line 9)
    sl_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'sl')
    # Obtaining the member '__getitem__' of a type (line 9)
    getitem___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), sl_13, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 9)
    subscript_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), getitem___14, int_12)
    
    # Assigning a type to the variable 'r' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'r', subscript_call_result_15)
    # Getting the type of 'r' (line 12)
    r_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'r')
    str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'str', 'str')
    # Applying the binary operator '+' (line 12)
    result_add_18 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), '+', r_16, str_17)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

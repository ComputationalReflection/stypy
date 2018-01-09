
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Inferring the type of the resulting iterator when slicing a homogeneous tuple"
3: 
4: if __name__ == '__main__':
5:     l = (1, 2, 3, 4, 5, 6)
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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Inferring the type of the resulting iterator when slicing a homogeneous tuple')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Tuple to a Name (line 5):
    
    # Obtaining an instance of the builtin type 'tuple' (line 5)
    tuple_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 5)
    # Adding element type (line 5)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 9), tuple_2, int_3)
    # Adding element type (line 5)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 9), tuple_2, int_4)
    # Adding element type (line 5)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 9), tuple_2, int_5)
    # Adding element type (line 5)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 9), tuple_2, int_6)
    # Adding element type (line 5)
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 9), tuple_2, int_7)
    # Adding element type (line 5)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 9), tuple_2, int_8)
    
    # Assigning a type to the variable 'l' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'l', tuple_2)
    
    # Assigning a Subscript to a Name (line 7):
    
    # Obtaining the type of the subscript
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'int')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
    slice_11 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 9), int_9, int_10, None)
    # Getting the type of 'l' (line 7)
    l_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'l')
    # Obtaining the member '__getitem__' of a type (line 7)
    getitem___13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), l_12, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 7)
    subscript_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), getitem___13, slice_11)
    
    # Assigning a type to the variable 'sl' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'sl', subscript_call_result_14)
    
    # Assigning a Subscript to a Name (line 9):
    
    # Obtaining the type of the subscript
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'int')
    # Getting the type of 'sl' (line 9)
    sl_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'sl')
    # Obtaining the member '__getitem__' of a type (line 9)
    getitem___17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), sl_16, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 9)
    subscript_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), getitem___17, int_15)
    
    # Assigning a type to the variable 'r' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'r', subscript_call_result_18)
    # Getting the type of 'r' (line 12)
    r_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'r')
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'str', 'str')
    # Applying the binary operator '+' (line 12)
    result_add_21 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), '+', r_19, str_20)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

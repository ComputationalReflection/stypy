
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of heterogeneous lists created with list methods"
4: 
5: if __name__ == '__main__':
6:     l = (["s", 1, "t", 2] * 3)[3:]
7: 
8:     for elem in l:
9:         # Type warning
10:         print "|" + elem + "|"
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of heterogeneous lists created with list methods')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Subscript to a Name (line 6):
    
    # Obtaining the type of the subscript
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 31), 'int')
    slice_3 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 6, 9), int_2, None, None)
    
    # Obtaining an instance of the builtin type 'list' (line 6)
    list_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 6)
    # Adding element type (line 6)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'str', 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), list_4, str_5)
    # Adding element type (line 6)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), list_4, int_6)
    # Adding element type (line 6)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'str', 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), list_4, str_7)
    # Adding element type (line 6)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), list_4, int_8)
    
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 28), 'int')
    # Applying the binary operator '*' (line 6)
    result_mul_10 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 9), '*', list_4, int_9)
    
    # Obtaining the member '__getitem__' of a type (line 6)
    getitem___11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 9), result_mul_10, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 6)
    subscript_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 6, 9), getitem___11, slice_3)
    
    # Assigning a type to the variable 'l' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'l', subscript_call_result_12)
    
    # Getting the type of 'l' (line 8)
    l_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 16), 'l')
    # Testing the type of a for loop iterable (line 8)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 4), l_13)
    # Getting the type of the for loop variable (line 8)
    for_loop_var_14 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 4), l_13)
    # Assigning a type to the variable 'elem' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'elem', for_loop_var_14)
    # SSA begins for a for statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'str', '|')
    # Getting the type of 'elem' (line 10)
    elem_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'elem')
    # Applying the binary operator '+' (line 10)
    result_add_17 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 14), '+', str_15, elem_16)
    
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'str', '|')
    # Applying the binary operator '+' (line 10)
    result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 25), '+', result_add_17, str_18)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

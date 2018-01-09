
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of heterogeneous tuples created as literals"
4: 
5: if __name__ == '__main__':
6:     l = ("s", 1, "t", 2)
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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of heterogeneous tuples created as literals')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Tuple to a Name (line 6):
    
    # Obtaining an instance of the builtin type 'tuple' (line 6)
    tuple_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 6)
    # Adding element type (line 6)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'str', 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), tuple_2, str_3)
    # Adding element type (line 6)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), tuple_2, int_4)
    # Adding element type (line 6)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 17), 'str', 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), tuple_2, str_5)
    # Adding element type (line 6)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), tuple_2, int_6)
    
    # Assigning a type to the variable 'l' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'l', tuple_2)
    
    # Getting the type of 'l' (line 8)
    l_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 16), 'l')
    # Testing the type of a for loop iterable (line 8)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 4), l_7)
    # Getting the type of the for loop variable (line 8)
    for_loop_var_8 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 4), l_7)
    # Assigning a type to the variable 'elem' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'elem', for_loop_var_8)
    # SSA begins for a for statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'str', '|')
    # Getting the type of 'elem' (line 10)
    elem_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'elem')
    # Applying the binary operator '+' (line 10)
    result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 14), '+', str_9, elem_10)
    
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'str', '|')
    # Applying the binary operator '+' (line 10)
    result_add_13 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 25), '+', result_add_11, str_12)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "A variable can be potentially undefined, but the condition is an error"
3: 
4: if __name__ == '__main__':
5:     a = "str"
6: 
7:     # Type error
8:     for r in a / 3:
9:         resul = 4
10: 
11:     # Type warning
12:     print resul + 2
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'A variable can be potentially undefined, but the condition is an error')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Str to a Name (line 5):
    str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'str', 'str')
    # Assigning a type to the variable 'a' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'a', str_2)
    
    # Getting the type of 'a' (line 8)
    a_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'a')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'int')
    # Applying the binary operator 'div' (line 8)
    result_div_5 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 13), 'div', a_3, int_4)
    
    # Testing the type of a for loop iterable (line 8)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 4), result_div_5)
    # Getting the type of the for loop variable (line 8)
    for_loop_var_6 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 4), result_div_5)
    # Assigning a type to the variable 'r' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'r', for_loop_var_6)
    # SSA begins for a for statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 9):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'int')
    # Assigning a type to the variable 'resul' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'resul', int_7)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'resul' (line 12)
    resul_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'resul')
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    # Applying the binary operator '+' (line 12)
    result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), '+', resul_8, int_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

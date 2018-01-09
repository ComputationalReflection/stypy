
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "A variable can be potentially undefined"
3: 
4: if __name__ == '__main__':
5:     a = "str"
6: 
7:     for x in range(2):
8:         resul = 4
9: 
10:     # Type warning
11:     print resul + 2
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'A variable can be potentially undefined')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Str to a Name (line 5):
    str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'str', 'str')
    # Assigning a type to the variable 'a' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'a', str_2)
    
    
    # Call to range(...): (line 7)
    # Processing the call arguments (line 7)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_5 = {}
    # Getting the type of 'range' (line 7)
    range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'range', False)
    # Calling range(args, kwargs) (line 7)
    range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), range_3, *[int_4], **kwargs_5)
    
    # Testing the type of a for loop iterable (line 7)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 7, 4), range_call_result_6)
    # Getting the type of the for loop variable (line 7)
    for_loop_var_7 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 7, 4), range_call_result_6)
    # Assigning a type to the variable 'x' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'x', for_loop_var_7)
    # SSA begins for a for statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 8):
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 16), 'int')
    # Assigning a type to the variable 'resul' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'resul', int_8)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'resul' (line 11)
    resul_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'resul')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    # Applying the binary operator '+' (line 11)
    result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), '+', resul_9, int_10)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

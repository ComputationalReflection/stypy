
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of the element of the list"
4: 
5: if __name__ == '__main__':
6:     it_list = range(5)
7: 
8:     for i in it_list:
9:         # Type error
10:         print i + "str"
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of the element of the list')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_4 = {}
    # Getting the type of 'range' (line 6)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'it_list' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_list', range_call_result_5)
    
    # Getting the type of 'it_list' (line 8)
    it_list_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'it_list')
    # Testing the type of a for loop iterable (line 8)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 4), it_list_6)
    # Getting the type of the for loop variable (line 8)
    for_loop_var_7 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 4), it_list_6)
    # Assigning a type to the variable 'i' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'i', for_loop_var_7)
    # SSA begins for a for statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 10)
    i_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'i')
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'str', 'str')
    # Applying the binary operator '+' (line 10)
    result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 14), '+', i_8, str_9)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Use an iterator type instead of an iterator instance"
4: 
5: if __name__ == '__main__':
6:     it_list = type(iter(range(5)))
7: 
8:     # Type error
9:     for i in it_list:
10:         print i
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Use an iterator type instead of an iterator instance')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to type(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to iter(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 30), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_6 = {}
    # Getting the type of 'range' (line 6)
    range_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 6, 24), range_4, *[int_5], **kwargs_6)
    
    # Processing the call keyword arguments (line 6)
    kwargs_8 = {}
    # Getting the type of 'iter' (line 6)
    iter_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'iter', False)
    # Calling iter(args, kwargs) (line 6)
    iter_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 6, 19), iter_3, *[range_call_result_7], **kwargs_8)
    
    # Processing the call keyword arguments (line 6)
    kwargs_10 = {}
    # Getting the type of 'type' (line 6)
    type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'type', False)
    # Calling type(args, kwargs) (line 6)
    type_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), type_2, *[iter_call_result_9], **kwargs_10)
    
    # Assigning a type to the variable 'it_list' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_list', type_call_result_11)
    
    # Getting the type of 'it_list' (line 9)
    it_list_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'it_list')
    # Testing the type of a for loop iterable (line 9)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 4), it_list_12)
    # Getting the type of the for loop variable (line 9)
    for_loop_var_13 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 4), it_list_12)
    # Assigning a type to the variable 'i' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'i', for_loop_var_13)
    # SSA begins for a for statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 10)
    i_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'i')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

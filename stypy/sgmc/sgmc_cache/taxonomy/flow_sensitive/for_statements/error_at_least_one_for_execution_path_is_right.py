
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "At least one (but not all) execution paths has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5:     var = 3
6:     for i in range(4):
7:         var = str(var) + str(i)
8: 
9:     # Type warning
10:     print var[0]
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'At least one (but not all) execution paths has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 5):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 10), 'int')
    # Assigning a type to the variable 'var' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'var', int_2)
    
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_5 = {}
    # Getting the type of 'range' (line 6)
    range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), range_3, *[int_4], **kwargs_5)
    
    # Testing the type of a for loop iterable (line 6)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 6, 4), range_call_result_6)
    # Getting the type of the for loop variable (line 6)
    for_loop_var_7 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 6, 4), range_call_result_6)
    # Assigning a type to the variable 'i' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'i', for_loop_var_7)
    # SSA begins for a for statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 7):
    
    # Call to str(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'var' (line 7)
    var_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'var', False)
    # Processing the call keyword arguments (line 7)
    kwargs_10 = {}
    # Getting the type of 'str' (line 7)
    str_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'str', False)
    # Calling str(args, kwargs) (line 7)
    str_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 7, 14), str_8, *[var_9], **kwargs_10)
    
    
    # Call to str(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'i' (line 7)
    i_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 29), 'i', False)
    # Processing the call keyword arguments (line 7)
    kwargs_14 = {}
    # Getting the type of 'str' (line 7)
    str_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', False)
    # Calling str(args, kwargs) (line 7)
    str_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 7, 25), str_12, *[i_13], **kwargs_14)
    
    # Applying the binary operator '+' (line 7)
    result_add_16 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 14), '+', str_call_result_11, str_call_result_15)
    
    # Assigning a type to the variable 'var' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'var', result_add_16)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
    # Getting the type of 'var' (line 10)
    var_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'var')
    # Obtaining the member '__getitem__' of a type (line 10)
    getitem___19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), var_18, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 10)
    subscript_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), getitem___19, int_17)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

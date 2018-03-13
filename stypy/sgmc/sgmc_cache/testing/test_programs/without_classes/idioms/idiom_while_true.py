
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: x = 1
4: y = 1
5: 
6: while True:
7:     z = x + y
8:     s = str(z)
9:     if z > 10:
10:         break
11: 
12: 
13: 
14: 
15: 
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', int_1)

# Assigning a Num to a Name (line 4):
int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'y' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'y', int_2)

# Getting the type of 'True' (line 6)
True_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'True')
# Assigning a type to the variable 'True_3' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'True_3', True_3)
# Testing if the while is going to be iterated (line 6)
# Testing the type of an if condition (line 6)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 0), True_3)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 6, 0), True_3):
    
    # Assigning a BinOp to a Name (line 7):
    # Getting the type of 'x' (line 7)
    x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'x')
    # Getting the type of 'y' (line 7)
    y_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'y')
    # Applying the binary operator '+' (line 7)
    result_add_6 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 8), '+', x_4, y_5)
    
    # Assigning a type to the variable 'z' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'z', result_add_6)
    
    # Assigning a Call to a Name (line 8):
    
    # Call to str(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'z' (line 8)
    z_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'z', False)
    # Processing the call keyword arguments (line 8)
    kwargs_9 = {}
    # Getting the type of 'str' (line 8)
    str_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'str', False)
    # Calling str(args, kwargs) (line 8)
    str_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), str_7, *[z_8], **kwargs_9)
    
    # Assigning a type to the variable 's' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 's', str_call_result_10)
    
    # Getting the type of 'z' (line 9)
    z_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 7), 'z')
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'int')
    # Applying the binary operator '>' (line 9)
    result_gt_13 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 7), '>', z_11, int_12)
    
    # Testing if the type of an if condition is none (line 9)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 9, 4), result_gt_13):
        pass
    else:
        
        # Testing the type of an if condition (line 9)
        if_condition_14 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 4), result_gt_13)
        # Assigning a type to the variable 'if_condition_14' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'if_condition_14', if_condition_14)
        # SSA begins for if statement (line 9)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 9)
        module_type_store = module_type_store.join_ssa_context()
        




# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

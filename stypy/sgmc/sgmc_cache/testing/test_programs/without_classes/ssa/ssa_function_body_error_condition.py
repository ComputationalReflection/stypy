
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def f(param):
2:     if len('*' + param) > 3:
3:         return 3
4:     return None
5: 
6: 
7: def g(param):
8:     accum = 0
9:     for i in range(len(3 + param)):
10:         accum += i
11:     return accum
12: 
13: if True:
14:     f(None)
15: else:
16:     g(None)
17: 
18: if True:
19:     r = f(None)
20: else:
21:     r2 = g(None)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 1, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = ['param']
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', ['param'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, ['param'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    
    
    
    # Call to len(...): (line 2)
    # Processing the call arguments (line 2)
    str_6663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', '*')
    # Getting the type of 'param' (line 2)
    param_6664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 17), 'param', False)
    # Applying the binary operator '+' (line 2)
    result_add_6665 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), '+', str_6663, param_6664)
    
    # Processing the call keyword arguments (line 2)
    kwargs_6666 = {}
    # Getting the type of 'len' (line 2)
    len_6662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 7), 'len', False)
    # Calling len(args, kwargs) (line 2)
    len_call_result_6667 = invoke(stypy.reporting.localization.Localization(__file__, 2, 7), len_6662, *[result_add_6665], **kwargs_6666)
    
    int_6668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 26), 'int')
    # Applying the binary operator '>' (line 2)
    result_gt_6669 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 7), '>', len_call_result_6667, int_6668)
    
    # Testing the type of an if condition (line 2)
    if_condition_6670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2, 4), result_gt_6669)
    # Assigning a type to the variable 'if_condition_6670' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'if_condition_6670', if_condition_6670)
    # SSA begins for if statement (line 2)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_6671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', int_6671)
    # SSA join for if statement (line 2)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 4)
    None_6672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type', None_6672)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_6673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6673)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_6673

# Assigning a type to the variable 'f' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'f', f)

@norecursion
def g(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'g'
    module_type_store = module_type_store.open_function_context('g', 7, 0, False)
    
    # Passed parameters checking function
    g.stypy_localization = localization
    g.stypy_type_of_self = None
    g.stypy_type_store = module_type_store
    g.stypy_function_name = 'g'
    g.stypy_param_names_list = ['param']
    g.stypy_varargs_param_name = None
    g.stypy_kwargs_param_name = None
    g.stypy_call_defaults = defaults
    g.stypy_call_varargs = varargs
    g.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'g', ['param'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'g', localization, ['param'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'g(...)' code ##################

    
    # Assigning a Num to a Name (line 8):
    int_6674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'int')
    # Assigning a type to the variable 'accum' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'accum', int_6674)
    
    
    # Call to range(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Call to len(...): (line 9)
    # Processing the call arguments (line 9)
    int_6677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
    # Getting the type of 'param' (line 9)
    param_6678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 27), 'param', False)
    # Applying the binary operator '+' (line 9)
    result_add_6679 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 23), '+', int_6677, param_6678)
    
    # Processing the call keyword arguments (line 9)
    kwargs_6680 = {}
    # Getting the type of 'len' (line 9)
    len_6676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'len', False)
    # Calling len(args, kwargs) (line 9)
    len_call_result_6681 = invoke(stypy.reporting.localization.Localization(__file__, 9, 19), len_6676, *[result_add_6679], **kwargs_6680)
    
    # Processing the call keyword arguments (line 9)
    kwargs_6682 = {}
    # Getting the type of 'range' (line 9)
    range_6675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'range', False)
    # Calling range(args, kwargs) (line 9)
    range_call_result_6683 = invoke(stypy.reporting.localization.Localization(__file__, 9, 13), range_6675, *[len_call_result_6681], **kwargs_6682)
    
    # Testing the type of a for loop iterable (line 9)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 4), range_call_result_6683)
    # Getting the type of the for loop variable (line 9)
    for_loop_var_6684 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 4), range_call_result_6683)
    # Assigning a type to the variable 'i' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'i', for_loop_var_6684)
    # SSA begins for a for statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'accum' (line 10)
    accum_6685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'accum')
    # Getting the type of 'i' (line 10)
    i_6686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'i')
    # Applying the binary operator '+=' (line 10)
    result_iadd_6687 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 8), '+=', accum_6685, i_6686)
    # Assigning a type to the variable 'accum' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'accum', result_iadd_6687)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'accum' (line 11)
    accum_6688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'accum')
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', accum_6688)
    
    # ################# End of 'g(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'g' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_6689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6689)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'g'
    return stypy_return_type_6689

# Assigning a type to the variable 'g' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'g', g)

# Getting the type of 'True' (line 13)
True_6690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 3), 'True')
# Testing the type of an if condition (line 13)
if_condition_6691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 0), True_6690)
# Assigning a type to the variable 'if_condition_6691' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'if_condition_6691', if_condition_6691)
# SSA begins for if statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to f(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'None' (line 14)
None_6693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'None', False)
# Processing the call keyword arguments (line 14)
kwargs_6694 = {}
# Getting the type of 'f' (line 14)
f_6692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'f', False)
# Calling f(args, kwargs) (line 14)
f_call_result_6695 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), f_6692, *[None_6693], **kwargs_6694)

# SSA branch for the else part of an if statement (line 13)
module_type_store.open_ssa_branch('else')

# Call to g(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'None' (line 16)
None_6697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 6), 'None', False)
# Processing the call keyword arguments (line 16)
kwargs_6698 = {}
# Getting the type of 'g' (line 16)
g_6696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'g', False)
# Calling g(args, kwargs) (line 16)
g_call_result_6699 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), g_6696, *[None_6697], **kwargs_6698)

# SSA join for if statement (line 13)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'True' (line 18)
True_6700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 3), 'True')
# Testing the type of an if condition (line 18)
if_condition_6701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 0), True_6700)
# Assigning a type to the variable 'if_condition_6701' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'if_condition_6701', if_condition_6701)
# SSA begins for if statement (line 18)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 19):

# Call to f(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'None' (line 19)
None_6703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'None', False)
# Processing the call keyword arguments (line 19)
kwargs_6704 = {}
# Getting the type of 'f' (line 19)
f_6702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'f', False)
# Calling f(args, kwargs) (line 19)
f_call_result_6705 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), f_6702, *[None_6703], **kwargs_6704)

# Assigning a type to the variable 'r' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'r', f_call_result_6705)
# SSA branch for the else part of an if statement (line 18)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 21):

# Call to g(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'None' (line 21)
None_6707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'None', False)
# Processing the call keyword arguments (line 21)
kwargs_6708 = {}
# Getting the type of 'g' (line 21)
g_6706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'g', False)
# Calling g(args, kwargs) (line 21)
g_call_result_6709 = invoke(stypy.reporting.localization.Localization(__file__, 21, 9), g_6706, *[None_6707], **kwargs_6708)

# Assigning a type to the variable 'r2' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'r2', g_call_result_6709)
# SSA join for if statement (line 18)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

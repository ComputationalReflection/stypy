
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def f(param):
2:     return '*' + param
3: 
4: 
5: def g(param):
6:     return 3 + param
7: 
8: 
9: if True:
10:     f(None)
11: else:
12:     g(None)
13: 
14: 
15: if True:
16:     r = f(None)
17: else:
18:     r2 = g(None)

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

    str_6634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', '*')
    # Getting the type of 'param' (line 2)
    param_6635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 17), 'param')
    # Applying the binary operator '+' (line 2)
    result_add_6636 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), '+', str_6634, param_6635)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', result_add_6636)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_6637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_6637

# Assigning a type to the variable 'f' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'f', f)

@norecursion
def g(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'g'
    module_type_store = module_type_store.open_function_context('g', 5, 0, False)
    
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

    int_6638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'int')
    # Getting the type of 'param' (line 6)
    param_6639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'param')
    # Applying the binary operator '+' (line 6)
    result_add_6640 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 11), '+', int_6638, param_6639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type', result_add_6640)
    
    # ################# End of 'g(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'g' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_6641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6641)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'g'
    return stypy_return_type_6641

# Assigning a type to the variable 'g' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'g', g)

# Getting the type of 'True' (line 9)
True_6642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 3), 'True')
# Testing the type of an if condition (line 9)
if_condition_6643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 0), True_6642)
# Assigning a type to the variable 'if_condition_6643' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'if_condition_6643', if_condition_6643)
# SSA begins for if statement (line 9)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to f(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'None' (line 10)
None_6645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'None', False)
# Processing the call keyword arguments (line 10)
kwargs_6646 = {}
# Getting the type of 'f' (line 10)
f_6644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'f', False)
# Calling f(args, kwargs) (line 10)
f_call_result_6647 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), f_6644, *[None_6645], **kwargs_6646)

# SSA branch for the else part of an if statement (line 9)
module_type_store.open_ssa_branch('else')

# Call to g(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'None' (line 12)
None_6649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'None', False)
# Processing the call keyword arguments (line 12)
kwargs_6650 = {}
# Getting the type of 'g' (line 12)
g_6648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'g', False)
# Calling g(args, kwargs) (line 12)
g_call_result_6651 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), g_6648, *[None_6649], **kwargs_6650)

# SSA join for if statement (line 9)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'True' (line 15)
True_6652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 3), 'True')
# Testing the type of an if condition (line 15)
if_condition_6653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 0), True_6652)
# Assigning a type to the variable 'if_condition_6653' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'if_condition_6653', if_condition_6653)
# SSA begins for if statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 16):

# Call to f(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'None' (line 16)
None_6655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'None', False)
# Processing the call keyword arguments (line 16)
kwargs_6656 = {}
# Getting the type of 'f' (line 16)
f_6654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'f', False)
# Calling f(args, kwargs) (line 16)
f_call_result_6657 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), f_6654, *[None_6655], **kwargs_6656)

# Assigning a type to the variable 'r' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r', f_call_result_6657)
# SSA branch for the else part of an if statement (line 15)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 18):

# Call to g(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'None' (line 18)
None_6659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'None', False)
# Processing the call keyword arguments (line 18)
kwargs_6660 = {}
# Getting the type of 'g' (line 18)
g_6658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'g', False)
# Calling g(args, kwargs) (line 18)
g_call_result_6661 = invoke(stypy.reporting.localization.Localization(__file__, 18, 9), g_6658, *[None_6659], **kwargs_6660)

# Assigning a type to the variable 'r2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r2', g_call_result_6661)
# SSA join for if statement (line 15)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

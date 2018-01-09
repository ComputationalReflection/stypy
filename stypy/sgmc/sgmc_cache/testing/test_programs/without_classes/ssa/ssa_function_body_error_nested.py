
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def f(param):
2:     return '*' + param
3: 
4: 
5: def g(param):
6:     return f(param)
7: 
8: if True:
9:     g(None)
10: 
11: if True:
12:     r = g(None)
13: 
14: 

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

    str_6710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', '*')
    # Getting the type of 'param' (line 2)
    param_6711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 17), 'param')
    # Applying the binary operator '+' (line 2)
    result_add_6712 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), '+', str_6710, param_6711)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', result_add_6712)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_6713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_6713

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

    
    # Call to f(...): (line 6)
    # Processing the call arguments (line 6)
    # Getting the type of 'param' (line 6)
    param_6715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'param', False)
    # Processing the call keyword arguments (line 6)
    kwargs_6716 = {}
    # Getting the type of 'f' (line 6)
    f_6714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 11), 'f', False)
    # Calling f(args, kwargs) (line 6)
    f_call_result_6717 = invoke(stypy.reporting.localization.Localization(__file__, 6, 11), f_6714, *[param_6715], **kwargs_6716)
    
    # Assigning a type to the variable 'stypy_return_type' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type', f_call_result_6717)
    
    # ################# End of 'g(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'g' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_6718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'g'
    return stypy_return_type_6718

# Assigning a type to the variable 'g' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'g', g)

# Getting the type of 'True' (line 8)
True_6719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 3), 'True')
# Testing the type of an if condition (line 8)
if_condition_6720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 0), True_6719)
# Assigning a type to the variable 'if_condition_6720' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'if_condition_6720', if_condition_6720)
# SSA begins for if statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to g(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'None' (line 9)
None_6722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 6), 'None', False)
# Processing the call keyword arguments (line 9)
kwargs_6723 = {}
# Getting the type of 'g' (line 9)
g_6721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'g', False)
# Calling g(args, kwargs) (line 9)
g_call_result_6724 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), g_6721, *[None_6722], **kwargs_6723)

# SSA join for if statement (line 8)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'True' (line 11)
True_6725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 3), 'True')
# Testing the type of an if condition (line 11)
if_condition_6726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 0), True_6725)
# Assigning a type to the variable 'if_condition_6726' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'if_condition_6726', if_condition_6726)
# SSA begins for if statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 12):

# Call to g(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'None' (line 12)
None_6728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'None', False)
# Processing the call keyword arguments (line 12)
kwargs_6729 = {}
# Getting the type of 'g' (line 12)
g_6727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'g', False)
# Calling g(args, kwargs) (line 12)
g_call_result_6730 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), g_6727, *[None_6728], **kwargs_6729)

# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'r', g_call_result_6730)
# SSA join for if statement (line 11)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

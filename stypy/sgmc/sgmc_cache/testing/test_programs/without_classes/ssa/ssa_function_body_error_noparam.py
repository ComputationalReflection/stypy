
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def f():
2:     arr = [None] * 4
3:     return arr[1]
4: 
5: 
6: def g():
7:     arr = [None] * 4
8:     return arr[1] / 2
9: 
10: 
11: if True:
12:     f()
13: else:
14:     g()
15: 
16: if True:
17:     r = f()
18: else:
19:     r2 = g()
20: 

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
    f.stypy_param_names_list = []
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    
    # Assigning a BinOp to a Name (line 2):
    
    # Obtaining an instance of the builtin type 'list' (line 2)
    list_6731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 2)
    # Adding element type (line 2)
    # Getting the type of 'None' (line 2)
    None_6732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 11), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_6731, None_6732)
    
    int_6733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 19), 'int')
    # Applying the binary operator '*' (line 2)
    result_mul_6734 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 10), '*', list_6731, int_6733)
    
    # Assigning a type to the variable 'arr' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'arr', result_mul_6734)
    
    # Obtaining the type of the subscript
    int_6735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'int')
    # Getting the type of 'arr' (line 3)
    arr_6736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 11), 'arr')
    # Obtaining the member '__getitem__' of a type (line 3)
    getitem___6737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 11), arr_6736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 3)
    subscript_call_result_6738 = invoke(stypy.reporting.localization.Localization(__file__, 3, 11), getitem___6737, int_6735)
    
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'stypy_return_type', subscript_call_result_6738)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_6739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_6739

# Assigning a type to the variable 'f' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'f', f)

@norecursion
def g(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'g'
    module_type_store = module_type_store.open_function_context('g', 6, 0, False)
    
    # Passed parameters checking function
    g.stypy_localization = localization
    g.stypy_type_of_self = None
    g.stypy_type_store = module_type_store
    g.stypy_function_name = 'g'
    g.stypy_param_names_list = []
    g.stypy_varargs_param_name = None
    g.stypy_kwargs_param_name = None
    g.stypy_call_defaults = defaults
    g.stypy_call_varargs = varargs
    g.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'g', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'g', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'g(...)' code ##################

    
    # Assigning a BinOp to a Name (line 7):
    
    # Obtaining an instance of the builtin type 'list' (line 7)
    list_6740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 7)
    # Adding element type (line 7)
    # Getting the type of 'None' (line 7)
    None_6741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_6740, None_6741)
    
    int_6742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
    # Applying the binary operator '*' (line 7)
    result_mul_6743 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 10), '*', list_6740, int_6742)
    
    # Assigning a type to the variable 'arr' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'arr', result_mul_6743)
    
    # Obtaining the type of the subscript
    int_6744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
    # Getting the type of 'arr' (line 8)
    arr_6745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'arr')
    # Obtaining the member '__getitem__' of a type (line 8)
    getitem___6746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 11), arr_6745, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 8)
    subscript_call_result_6747 = invoke(stypy.reporting.localization.Localization(__file__, 8, 11), getitem___6746, int_6744)
    
    int_6748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
    # Applying the binary operator 'div' (line 8)
    result_div_6749 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 11), 'div', subscript_call_result_6747, int_6748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', result_div_6749)
    
    # ################# End of 'g(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'g' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_6750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'g'
    return stypy_return_type_6750

# Assigning a type to the variable 'g' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'g', g)

# Getting the type of 'True' (line 11)
True_6751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 3), 'True')
# Testing the type of an if condition (line 11)
if_condition_6752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 0), True_6751)
# Assigning a type to the variable 'if_condition_6752' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'if_condition_6752', if_condition_6752)
# SSA begins for if statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to f(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_6754 = {}
# Getting the type of 'f' (line 12)
f_6753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'f', False)
# Calling f(args, kwargs) (line 12)
f_call_result_6755 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), f_6753, *[], **kwargs_6754)

# SSA branch for the else part of an if statement (line 11)
module_type_store.open_ssa_branch('else')

# Call to g(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_6757 = {}
# Getting the type of 'g' (line 14)
g_6756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'g', False)
# Calling g(args, kwargs) (line 14)
g_call_result_6758 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), g_6756, *[], **kwargs_6757)

# SSA join for if statement (line 11)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'True' (line 16)
True_6759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 3), 'True')
# Testing the type of an if condition (line 16)
if_condition_6760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 0), True_6759)
# Assigning a type to the variable 'if_condition_6760' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'if_condition_6760', if_condition_6760)
# SSA begins for if statement (line 16)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 17):

# Call to f(...): (line 17)
# Processing the call keyword arguments (line 17)
kwargs_6762 = {}
# Getting the type of 'f' (line 17)
f_6761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'f', False)
# Calling f(args, kwargs) (line 17)
f_call_result_6763 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), f_6761, *[], **kwargs_6762)

# Assigning a type to the variable 'r' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'r', f_call_result_6763)
# SSA branch for the else part of an if statement (line 16)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 19):

# Call to g(...): (line 19)
# Processing the call keyword arguments (line 19)
kwargs_6765 = {}
# Getting the type of 'g' (line 19)
g_6764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'g', False)
# Calling g(args, kwargs) (line 19)
g_call_result_6766 = invoke(stypy.reporting.localization.Localization(__file__, 19, 9), g_6764, *[], **kwargs_6765)

# Assigning a type to the variable 'r2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'r2', g_call_result_6766)
# SSA join for if statement (line 16)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: str_l = ["1", "2", "3", "4"]
2: l = range(5)
3: 
4: other_l = map(lambda x: x, str_l)
5: r1 = other_l[0] + 6  # Unreported
6: 
7: 
8: def f(x):
9:     return str(x)
10: 
11: 
12: other_l2 = map(lambda x: f(x), l)
13: r2 = other_l2[0] + 6  # Reported
14: 
15: 
16: def f2(x):
17:     if True:
18:         return "3"
19:     else:
20:         return 3
21: 
22: 
23: other_l3 = map(lambda x: f2(x), l)
24: x = other_l3[0] + 6  # Not reported
25: 
26: other_l4 = map(lambda x, y: f(x), l)  # Runtime crash
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):

# Obtaining an instance of the builtin type 'list' (line 1)
list_7789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_7790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'str', '1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 8), list_7789, str_7790)
# Adding element type (line 1)
str_7791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 14), 'str', '2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 8), list_7789, str_7791)
# Adding element type (line 1)
str_7792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 19), 'str', '3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 8), list_7789, str_7792)
# Adding element type (line 1)
str_7793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 24), 'str', '4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 8), list_7789, str_7793)

# Assigning a type to the variable 'str_l' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'str_l', list_7789)

# Assigning a Call to a Name (line 2):

# Call to range(...): (line 2)
# Processing the call arguments (line 2)
int_7795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Processing the call keyword arguments (line 2)
kwargs_7796 = {}
# Getting the type of 'range' (line 2)
range_7794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'range', False)
# Calling range(args, kwargs) (line 2)
range_call_result_7797 = invoke(stypy.reporting.localization.Localization(__file__, 2, 4), range_7794, *[int_7795], **kwargs_7796)

# Assigning a type to the variable 'l' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'l', range_call_result_7797)

# Assigning a Call to a Name (line 4):

# Call to map(...): (line 4)
# Processing the call arguments (line 4)

@norecursion
def _stypy_temp_lambda_14(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_14'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_14', 4, 14, True)
    # Passed parameters checking function
    _stypy_temp_lambda_14.stypy_localization = localization
    _stypy_temp_lambda_14.stypy_type_of_self = None
    _stypy_temp_lambda_14.stypy_type_store = module_type_store
    _stypy_temp_lambda_14.stypy_function_name = '_stypy_temp_lambda_14'
    _stypy_temp_lambda_14.stypy_param_names_list = ['x']
    _stypy_temp_lambda_14.stypy_varargs_param_name = None
    _stypy_temp_lambda_14.stypy_kwargs_param_name = None
    _stypy_temp_lambda_14.stypy_call_defaults = defaults
    _stypy_temp_lambda_14.stypy_call_varargs = varargs
    _stypy_temp_lambda_14.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_14', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_14', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 4)
    x_7799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 24), 'x', False)
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), 'stypy_return_type', x_7799)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_14' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_7800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_14'
    return stypy_return_type_7800

# Assigning a type to the variable '_stypy_temp_lambda_14' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), '_stypy_temp_lambda_14', _stypy_temp_lambda_14)
# Getting the type of '_stypy_temp_lambda_14' (line 4)
_stypy_temp_lambda_14_7801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), '_stypy_temp_lambda_14')
# Getting the type of 'str_l' (line 4)
str_l_7802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 27), 'str_l', False)
# Processing the call keyword arguments (line 4)
kwargs_7803 = {}
# Getting the type of 'map' (line 4)
map_7798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 10), 'map', False)
# Calling map(args, kwargs) (line 4)
map_call_result_7804 = invoke(stypy.reporting.localization.Localization(__file__, 4, 10), map_7798, *[_stypy_temp_lambda_14_7801, str_l_7802], **kwargs_7803)

# Assigning a type to the variable 'other_l' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'other_l', map_call_result_7804)

# Assigning a BinOp to a Name (line 5):

# Obtaining the type of the subscript
int_7805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Getting the type of 'other_l' (line 5)
other_l_7806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'other_l')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___7807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), other_l_7806, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_7808 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), getitem___7807, int_7805)

int_7809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
# Applying the binary operator '+' (line 5)
result_add_7810 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 5), '+', subscript_call_result_7808, int_7809)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', result_add_7810)

@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 8, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = ['x']
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    
    # Call to str(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'x' (line 9)
    x_7812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'x', False)
    # Processing the call keyword arguments (line 9)
    kwargs_7813 = {}
    # Getting the type of 'str' (line 9)
    str_7811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', False)
    # Calling str(args, kwargs) (line 9)
    str_call_result_7814 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), str_7811, *[x_7812], **kwargs_7813)
    
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', str_call_result_7814)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_7815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7815)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_7815

# Assigning a type to the variable 'f' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'f', f)

# Assigning a Call to a Name (line 12):

# Call to map(...): (line 12)
# Processing the call arguments (line 12)

@norecursion
def _stypy_temp_lambda_15(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_15'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_15', 12, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_15.stypy_localization = localization
    _stypy_temp_lambda_15.stypy_type_of_self = None
    _stypy_temp_lambda_15.stypy_type_store = module_type_store
    _stypy_temp_lambda_15.stypy_function_name = '_stypy_temp_lambda_15'
    _stypy_temp_lambda_15.stypy_param_names_list = ['x']
    _stypy_temp_lambda_15.stypy_varargs_param_name = None
    _stypy_temp_lambda_15.stypy_kwargs_param_name = None
    _stypy_temp_lambda_15.stypy_call_defaults = defaults
    _stypy_temp_lambda_15.stypy_call_varargs = varargs
    _stypy_temp_lambda_15.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_15', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_15', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to f(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'x' (line 12)
    x_7818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 27), 'x', False)
    # Processing the call keyword arguments (line 12)
    kwargs_7819 = {}
    # Getting the type of 'f' (line 12)
    f_7817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'f', False)
    # Calling f(args, kwargs) (line 12)
    f_call_result_7820 = invoke(stypy.reporting.localization.Localization(__file__, 12, 25), f_7817, *[x_7818], **kwargs_7819)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'stypy_return_type', f_call_result_7820)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_15' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_7821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7821)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_15'
    return stypy_return_type_7821

# Assigning a type to the variable '_stypy_temp_lambda_15' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), '_stypy_temp_lambda_15', _stypy_temp_lambda_15)
# Getting the type of '_stypy_temp_lambda_15' (line 12)
_stypy_temp_lambda_15_7822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), '_stypy_temp_lambda_15')
# Getting the type of 'l' (line 12)
l_7823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 31), 'l', False)
# Processing the call keyword arguments (line 12)
kwargs_7824 = {}
# Getting the type of 'map' (line 12)
map_7816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'map', False)
# Calling map(args, kwargs) (line 12)
map_call_result_7825 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), map_7816, *[_stypy_temp_lambda_15_7822, l_7823], **kwargs_7824)

# Assigning a type to the variable 'other_l2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'other_l2', map_call_result_7825)

# Assigning a BinOp to a Name (line 13):

# Obtaining the type of the subscript
int_7826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'int')
# Getting the type of 'other_l2' (line 13)
other_l2_7827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'other_l2')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___7828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), other_l2_7827, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_7829 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), getitem___7828, int_7826)

int_7830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
# Applying the binary operator '+' (line 13)
result_add_7831 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 5), '+', subscript_call_result_7829, int_7830)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', result_add_7831)

@norecursion
def f2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f2'
    module_type_store = module_type_store.open_function_context('f2', 16, 0, False)
    
    # Passed parameters checking function
    f2.stypy_localization = localization
    f2.stypy_type_of_self = None
    f2.stypy_type_store = module_type_store
    f2.stypy_function_name = 'f2'
    f2.stypy_param_names_list = ['x']
    f2.stypy_varargs_param_name = None
    f2.stypy_kwargs_param_name = None
    f2.stypy_call_defaults = defaults
    f2.stypy_call_varargs = varargs
    f2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f2(...)' code ##################

    
    # Getting the type of 'True' (line 17)
    True_7832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'True')
    # Testing the type of an if condition (line 17)
    if_condition_7833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), True_7832)
    # Assigning a type to the variable 'if_condition_7833' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_7833', if_condition_7833)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_7834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'str', '3')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', str_7834)
    # SSA branch for the else part of an if statement (line 17)
    module_type_store.open_ssa_branch('else')
    int_7835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', int_7835)
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'f2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f2' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_7836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f2'
    return stypy_return_type_7836

# Assigning a type to the variable 'f2' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'f2', f2)

# Assigning a Call to a Name (line 23):

# Call to map(...): (line 23)
# Processing the call arguments (line 23)

@norecursion
def _stypy_temp_lambda_16(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_16'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_16', 23, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_16.stypy_localization = localization
    _stypy_temp_lambda_16.stypy_type_of_self = None
    _stypy_temp_lambda_16.stypy_type_store = module_type_store
    _stypy_temp_lambda_16.stypy_function_name = '_stypy_temp_lambda_16'
    _stypy_temp_lambda_16.stypy_param_names_list = ['x']
    _stypy_temp_lambda_16.stypy_varargs_param_name = None
    _stypy_temp_lambda_16.stypy_kwargs_param_name = None
    _stypy_temp_lambda_16.stypy_call_defaults = defaults
    _stypy_temp_lambda_16.stypy_call_varargs = varargs
    _stypy_temp_lambda_16.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_16', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_16', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to f2(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'x' (line 23)
    x_7839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'x', False)
    # Processing the call keyword arguments (line 23)
    kwargs_7840 = {}
    # Getting the type of 'f2' (line 23)
    f2_7838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'f2', False)
    # Calling f2(args, kwargs) (line 23)
    f2_call_result_7841 = invoke(stypy.reporting.localization.Localization(__file__, 23, 25), f2_7838, *[x_7839], **kwargs_7840)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'stypy_return_type', f2_call_result_7841)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_16' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_7842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_16'
    return stypy_return_type_7842

# Assigning a type to the variable '_stypy_temp_lambda_16' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), '_stypy_temp_lambda_16', _stypy_temp_lambda_16)
# Getting the type of '_stypy_temp_lambda_16' (line 23)
_stypy_temp_lambda_16_7843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), '_stypy_temp_lambda_16')
# Getting the type of 'l' (line 23)
l_7844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'l', False)
# Processing the call keyword arguments (line 23)
kwargs_7845 = {}
# Getting the type of 'map' (line 23)
map_7837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'map', False)
# Calling map(args, kwargs) (line 23)
map_call_result_7846 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), map_7837, *[_stypy_temp_lambda_16_7843, l_7844], **kwargs_7845)

# Assigning a type to the variable 'other_l3' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'other_l3', map_call_result_7846)

# Assigning a BinOp to a Name (line 24):

# Obtaining the type of the subscript
int_7847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'int')
# Getting the type of 'other_l3' (line 24)
other_l3_7848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'other_l3')
# Obtaining the member '__getitem__' of a type (line 24)
getitem___7849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), other_l3_7848, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 24)
subscript_call_result_7850 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), getitem___7849, int_7847)

int_7851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'int')
# Applying the binary operator '+' (line 24)
result_add_7852 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 4), '+', subscript_call_result_7850, int_7851)

# Assigning a type to the variable 'x' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'x', result_add_7852)

# Assigning a Call to a Name (line 26):

# Call to map(...): (line 26)
# Processing the call arguments (line 26)

@norecursion
def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_17'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 26, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_17.stypy_localization = localization
    _stypy_temp_lambda_17.stypy_type_of_self = None
    _stypy_temp_lambda_17.stypy_type_store = module_type_store
    _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
    _stypy_temp_lambda_17.stypy_param_names_list = ['x', 'y']
    _stypy_temp_lambda_17.stypy_varargs_param_name = None
    _stypy_temp_lambda_17.stypy_kwargs_param_name = None
    _stypy_temp_lambda_17.stypy_call_defaults = defaults
    _stypy_temp_lambda_17.stypy_call_varargs = varargs
    _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_17', ['x', 'y'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to f(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'x' (line 26)
    x_7855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'x', False)
    # Processing the call keyword arguments (line 26)
    kwargs_7856 = {}
    # Getting the type of 'f' (line 26)
    f_7854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'f', False)
    # Calling f(args, kwargs) (line 26)
    f_call_result_7857 = invoke(stypy.reporting.localization.Localization(__file__, 26, 28), f_7854, *[x_7855], **kwargs_7856)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'stypy_return_type', f_call_result_7857)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_17' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_7858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7858)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_17'
    return stypy_return_type_7858

# Assigning a type to the variable '_stypy_temp_lambda_17' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
# Getting the type of '_stypy_temp_lambda_17' (line 26)
_stypy_temp_lambda_17_7859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), '_stypy_temp_lambda_17')
# Getting the type of 'l' (line 26)
l_7860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 34), 'l', False)
# Processing the call keyword arguments (line 26)
kwargs_7861 = {}
# Getting the type of 'map' (line 26)
map_7853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'map', False)
# Calling map(args, kwargs) (line 26)
map_call_result_7862 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), map_7853, *[_stypy_temp_lambda_17_7859, l_7860], **kwargs_7861)

# Assigning a type to the variable 'other_l4' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'other_l4', map_call_result_7862)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

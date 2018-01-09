
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: l2 = [False, 1, "string"]
2: 
3: 
4: def f2(x):
5:     return str(x)
6: 
7: 
8: other_l = filter(lambda x: f2(x), l2)
9: 
10: r1 = other_l[2] + 6  # Not reported
11: 
12: l3 = ["False", "1", "string"]
13: other_l2 = filter(lambda x: f2(x), l3)
14: r2 = other_l2[2] + 6  # Reported
15: 
16: other_l3 = filter(lambda x, y: f2(x), l2)  # Not reported
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):

# Obtaining an instance of the builtin type 'list' (line 1)
list_7303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
# Getting the type of 'False' (line 1)
False_7304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 6), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 5), list_7303, False_7304)
# Adding element type (line 1)
int_7305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 5), list_7303, int_7305)
# Adding element type (line 1)
str_7306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 16), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 5), list_7303, str_7306)

# Assigning a type to the variable 'l2' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'l2', list_7303)

@norecursion
def f2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f2'
    module_type_store = module_type_store.open_function_context('f2', 4, 0, False)
    
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

    
    # Call to str(...): (line 5)
    # Processing the call arguments (line 5)
    # Getting the type of 'x' (line 5)
    x_7308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), 'x', False)
    # Processing the call keyword arguments (line 5)
    kwargs_7309 = {}
    # Getting the type of 'str' (line 5)
    str_7307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 11), 'str', False)
    # Calling str(args, kwargs) (line 5)
    str_call_result_7310 = invoke(stypy.reporting.localization.Localization(__file__, 5, 11), str_7307, *[x_7308], **kwargs_7309)
    
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type', str_call_result_7310)
    
    # ################# End of 'f2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f2' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_7311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7311)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f2'
    return stypy_return_type_7311

# Assigning a type to the variable 'f2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'f2', f2)

# Assigning a Call to a Name (line 8):

# Call to filter(...): (line 8)
# Processing the call arguments (line 8)

@norecursion
def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_11'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 8, 17, True)
    # Passed parameters checking function
    _stypy_temp_lambda_11.stypy_localization = localization
    _stypy_temp_lambda_11.stypy_type_of_self = None
    _stypy_temp_lambda_11.stypy_type_store = module_type_store
    _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
    _stypy_temp_lambda_11.stypy_param_names_list = ['x']
    _stypy_temp_lambda_11.stypy_varargs_param_name = None
    _stypy_temp_lambda_11.stypy_kwargs_param_name = None
    _stypy_temp_lambda_11.stypy_call_defaults = defaults
    _stypy_temp_lambda_11.stypy_call_varargs = varargs
    _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_11', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to f2(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'x' (line 8)
    x_7314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 30), 'x', False)
    # Processing the call keyword arguments (line 8)
    kwargs_7315 = {}
    # Getting the type of 'f2' (line 8)
    f2_7313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 27), 'f2', False)
    # Calling f2(args, kwargs) (line 8)
    f2_call_result_7316 = invoke(stypy.reporting.localization.Localization(__file__, 8, 27), f2_7313, *[x_7314], **kwargs_7315)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'stypy_return_type', f2_call_result_7316)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_11' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_7317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7317)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_11'
    return stypy_return_type_7317

# Assigning a type to the variable '_stypy_temp_lambda_11' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
# Getting the type of '_stypy_temp_lambda_11' (line 8)
_stypy_temp_lambda_11_7318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), '_stypy_temp_lambda_11')
# Getting the type of 'l2' (line 8)
l2_7319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 34), 'l2', False)
# Processing the call keyword arguments (line 8)
kwargs_7320 = {}
# Getting the type of 'filter' (line 8)
filter_7312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'filter', False)
# Calling filter(args, kwargs) (line 8)
filter_call_result_7321 = invoke(stypy.reporting.localization.Localization(__file__, 8, 10), filter_7312, *[_stypy_temp_lambda_11_7318, l2_7319], **kwargs_7320)

# Assigning a type to the variable 'other_l' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'other_l', filter_call_result_7321)

# Assigning a BinOp to a Name (line 10):

# Obtaining the type of the subscript
int_7322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'int')
# Getting the type of 'other_l' (line 10)
other_l_7323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'other_l')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___7324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), other_l_7323, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_7325 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), getitem___7324, int_7322)

int_7326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'int')
# Applying the binary operator '+' (line 10)
result_add_7327 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '+', subscript_call_result_7325, int_7326)

# Assigning a type to the variable 'r1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r1', result_add_7327)

# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_7328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_7329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 6), 'str', 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 5), list_7328, str_7329)
# Adding element type (line 12)
str_7330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', '1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 5), list_7328, str_7330)
# Adding element type (line 12)
str_7331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 5), list_7328, str_7331)

# Assigning a type to the variable 'l3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'l3', list_7328)

# Assigning a Call to a Name (line 13):

# Call to filter(...): (line 13)
# Processing the call arguments (line 13)

@norecursion
def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_12'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 13, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_12.stypy_localization = localization
    _stypy_temp_lambda_12.stypy_type_of_self = None
    _stypy_temp_lambda_12.stypy_type_store = module_type_store
    _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
    _stypy_temp_lambda_12.stypy_param_names_list = ['x']
    _stypy_temp_lambda_12.stypy_varargs_param_name = None
    _stypy_temp_lambda_12.stypy_kwargs_param_name = None
    _stypy_temp_lambda_12.stypy_call_defaults = defaults
    _stypy_temp_lambda_12.stypy_call_varargs = varargs
    _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_12', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to f2(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'x' (line 13)
    x_7334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 31), 'x', False)
    # Processing the call keyword arguments (line 13)
    kwargs_7335 = {}
    # Getting the type of 'f2' (line 13)
    f2_7333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'f2', False)
    # Calling f2(args, kwargs) (line 13)
    f2_call_result_7336 = invoke(stypy.reporting.localization.Localization(__file__, 13, 28), f2_7333, *[x_7334], **kwargs_7335)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'stypy_return_type', f2_call_result_7336)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_12' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_7337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7337)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_12'
    return stypy_return_type_7337

# Assigning a type to the variable '_stypy_temp_lambda_12' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
# Getting the type of '_stypy_temp_lambda_12' (line 13)
_stypy_temp_lambda_12_7338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), '_stypy_temp_lambda_12')
# Getting the type of 'l3' (line 13)
l3_7339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'l3', False)
# Processing the call keyword arguments (line 13)
kwargs_7340 = {}
# Getting the type of 'filter' (line 13)
filter_7332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'filter', False)
# Calling filter(args, kwargs) (line 13)
filter_call_result_7341 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), filter_7332, *[_stypy_temp_lambda_12_7338, l3_7339], **kwargs_7340)

# Assigning a type to the variable 'other_l2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'other_l2', filter_call_result_7341)

# Assigning a BinOp to a Name (line 14):

# Obtaining the type of the subscript
int_7342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
# Getting the type of 'other_l2' (line 14)
other_l2_7343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'other_l2')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___7344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), other_l2_7343, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_7345 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), getitem___7344, int_7342)

int_7346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
# Applying the binary operator '+' (line 14)
result_add_7347 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 5), '+', subscript_call_result_7345, int_7346)

# Assigning a type to the variable 'r2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r2', result_add_7347)

# Assigning a Call to a Name (line 16):

# Call to filter(...): (line 16)
# Processing the call arguments (line 16)

@norecursion
def _stypy_temp_lambda_13(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_13'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_13', 16, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_13.stypy_localization = localization
    _stypy_temp_lambda_13.stypy_type_of_self = None
    _stypy_temp_lambda_13.stypy_type_store = module_type_store
    _stypy_temp_lambda_13.stypy_function_name = '_stypy_temp_lambda_13'
    _stypy_temp_lambda_13.stypy_param_names_list = ['x', 'y']
    _stypy_temp_lambda_13.stypy_varargs_param_name = None
    _stypy_temp_lambda_13.stypy_kwargs_param_name = None
    _stypy_temp_lambda_13.stypy_call_defaults = defaults
    _stypy_temp_lambda_13.stypy_call_varargs = varargs
    _stypy_temp_lambda_13.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_13', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_13', ['x', 'y'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to f2(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'x' (line 16)
    x_7350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'x', False)
    # Processing the call keyword arguments (line 16)
    kwargs_7351 = {}
    # Getting the type of 'f2' (line 16)
    f2_7349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 31), 'f2', False)
    # Calling f2(args, kwargs) (line 16)
    f2_call_result_7352 = invoke(stypy.reporting.localization.Localization(__file__, 16, 31), f2_7349, *[x_7350], **kwargs_7351)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'stypy_return_type', f2_call_result_7352)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_13' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_7353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7353)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_13'
    return stypy_return_type_7353

# Assigning a type to the variable '_stypy_temp_lambda_13' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), '_stypy_temp_lambda_13', _stypy_temp_lambda_13)
# Getting the type of '_stypy_temp_lambda_13' (line 16)
_stypy_temp_lambda_13_7354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), '_stypy_temp_lambda_13')
# Getting the type of 'l2' (line 16)
l2_7355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 38), 'l2', False)
# Processing the call keyword arguments (line 16)
kwargs_7356 = {}
# Getting the type of 'filter' (line 16)
filter_7348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'filter', False)
# Calling filter(args, kwargs) (line 16)
filter_call_result_7357 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), filter_7348, *[_stypy_temp_lambda_13_7354, l2_7355], **kwargs_7356)

# Assigning a type to the variable 'other_l3' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'other_l3', filter_call_result_7357)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

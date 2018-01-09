
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: l = [1,2,3,4]
3: 
4: other_l = map(lambda x: str(x), l)
5: 
6: l2 = [False, 1, "string"]
7: other_l2 = map(lambda x: str(x), l)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 2):

# Obtaining an instance of the builtin type 'list' (line 2)
list_6348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
int_6349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_6348, int_6349)
# Adding element type (line 2)
int_6350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_6348, int_6350)
# Adding element type (line 2)
int_6351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_6348, int_6351)
# Adding element type (line 2)
int_6352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_6348, int_6352)

# Assigning a type to the variable 'l' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'l', list_6348)

# Assigning a Call to a Name (line 4):

# Call to map(...): (line 4)
# Processing the call arguments (line 4)

@norecursion
def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_9'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 4, 14, True)
    # Passed parameters checking function
    _stypy_temp_lambda_9.stypy_localization = localization
    _stypy_temp_lambda_9.stypy_type_of_self = None
    _stypy_temp_lambda_9.stypy_type_store = module_type_store
    _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
    _stypy_temp_lambda_9.stypy_param_names_list = ['x']
    _stypy_temp_lambda_9.stypy_varargs_param_name = None
    _stypy_temp_lambda_9.stypy_kwargs_param_name = None
    _stypy_temp_lambda_9.stypy_call_defaults = defaults
    _stypy_temp_lambda_9.stypy_call_varargs = varargs
    _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_9', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to str(...): (line 4)
    # Processing the call arguments (line 4)
    # Getting the type of 'x' (line 4)
    x_6355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 28), 'x', False)
    # Processing the call keyword arguments (line 4)
    kwargs_6356 = {}
    # Getting the type of 'str' (line 4)
    str_6354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 24), 'str', False)
    # Calling str(args, kwargs) (line 4)
    str_call_result_6357 = invoke(stypy.reporting.localization.Localization(__file__, 4, 24), str_6354, *[x_6355], **kwargs_6356)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), 'stypy_return_type', str_call_result_6357)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_9' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_6358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6358)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_9'
    return stypy_return_type_6358

# Assigning a type to the variable '_stypy_temp_lambda_9' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
# Getting the type of '_stypy_temp_lambda_9' (line 4)
_stypy_temp_lambda_9_6359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), '_stypy_temp_lambda_9')
# Getting the type of 'l' (line 4)
l_6360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 32), 'l', False)
# Processing the call keyword arguments (line 4)
kwargs_6361 = {}
# Getting the type of 'map' (line 4)
map_6353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 10), 'map', False)
# Calling map(args, kwargs) (line 4)
map_call_result_6362 = invoke(stypy.reporting.localization.Localization(__file__, 4, 10), map_6353, *[_stypy_temp_lambda_9_6359, l_6360], **kwargs_6361)

# Assigning a type to the variable 'other_l' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'other_l', map_call_result_6362)

# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_6363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'False' (line 6)
False_6364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 5), list_6363, False_6364)
# Adding element type (line 6)
int_6365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 5), list_6363, int_6365)
# Adding element type (line 6)
str_6366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 5), list_6363, str_6366)

# Assigning a type to the variable 'l2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'l2', list_6363)

# Assigning a Call to a Name (line 7):

# Call to map(...): (line 7)
# Processing the call arguments (line 7)

@norecursion
def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_10'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 7, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_10.stypy_localization = localization
    _stypy_temp_lambda_10.stypy_type_of_self = None
    _stypy_temp_lambda_10.stypy_type_store = module_type_store
    _stypy_temp_lambda_10.stypy_function_name = '_stypy_temp_lambda_10'
    _stypy_temp_lambda_10.stypy_param_names_list = ['x']
    _stypy_temp_lambda_10.stypy_varargs_param_name = None
    _stypy_temp_lambda_10.stypy_kwargs_param_name = None
    _stypy_temp_lambda_10.stypy_call_defaults = defaults
    _stypy_temp_lambda_10.stypy_call_varargs = varargs
    _stypy_temp_lambda_10.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_10', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_10', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to str(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'x' (line 7)
    x_6369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 29), 'x', False)
    # Processing the call keyword arguments (line 7)
    kwargs_6370 = {}
    # Getting the type of 'str' (line 7)
    str_6368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', False)
    # Calling str(args, kwargs) (line 7)
    str_call_result_6371 = invoke(stypy.reporting.localization.Localization(__file__, 7, 25), str_6368, *[x_6369], **kwargs_6370)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'stypy_return_type', str_call_result_6371)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_10' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_6372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_10'
    return stypy_return_type_6372

# Assigning a type to the variable '_stypy_temp_lambda_10' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
# Getting the type of '_stypy_temp_lambda_10' (line 7)
_stypy_temp_lambda_10_6373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), '_stypy_temp_lambda_10')
# Getting the type of 'l' (line 7)
l_6374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 33), 'l', False)
# Processing the call keyword arguments (line 7)
kwargs_6375 = {}
# Getting the type of 'map' (line 7)
map_6367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'map', False)
# Calling map(args, kwargs) (line 7)
map_call_result_6376 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), map_6367, *[_stypy_temp_lambda_10_6373, l_6374], **kwargs_6375)

# Assigning a type to the variable 'other_l2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'other_l2', map_call_result_6376)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

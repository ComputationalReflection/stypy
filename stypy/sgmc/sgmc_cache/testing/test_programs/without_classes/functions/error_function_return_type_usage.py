
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def function_1(x):
2:     return x / 2
3: 
4: def function_2(x):
5:     return x / 2
6: 
7: def function_3(x):
8:     return x / 2
9: 
10: r1 = function_1("a")  # Reported on call site (not on the function code, no stack trace)
11: r2 = function_2(range(5))  # Reported on call site (not on the function code, no stack trace)
12: r3 = function_3(4)  # Nothing is reported, as the call is valid
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def function_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function_1'
    module_type_store = module_type_store.open_function_context('function_1', 1, 0, False)
    
    # Passed parameters checking function
    function_1.stypy_localization = localization
    function_1.stypy_type_of_self = None
    function_1.stypy_type_store = module_type_store
    function_1.stypy_function_name = 'function_1'
    function_1.stypy_param_names_list = ['x']
    function_1.stypy_varargs_param_name = None
    function_1.stypy_kwargs_param_name = None
    function_1.stypy_call_defaults = defaults
    function_1.stypy_call_varargs = varargs
    function_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function_1', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function_1', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function_1(...)' code ##################

    # Getting the type of 'x' (line 2)
    x_7632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 11), 'x')
    int_7633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'int')
    # Applying the binary operator 'div' (line 2)
    result_div_7634 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), 'div', x_7632, int_7633)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', result_div_7634)
    
    # ################# End of 'function_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function_1' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7635)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function_1'
    return stypy_return_type_7635

# Assigning a type to the variable 'function_1' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'function_1', function_1)

@norecursion
def function_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function_2'
    module_type_store = module_type_store.open_function_context('function_2', 4, 0, False)
    
    # Passed parameters checking function
    function_2.stypy_localization = localization
    function_2.stypy_type_of_self = None
    function_2.stypy_type_store = module_type_store
    function_2.stypy_function_name = 'function_2'
    function_2.stypy_param_names_list = ['x']
    function_2.stypy_varargs_param_name = None
    function_2.stypy_kwargs_param_name = None
    function_2.stypy_call_defaults = defaults
    function_2.stypy_call_varargs = varargs
    function_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function_2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function_2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function_2(...)' code ##################

    # Getting the type of 'x' (line 5)
    x_7636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 11), 'x')
    int_7637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
    # Applying the binary operator 'div' (line 5)
    result_div_7638 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 11), 'div', x_7636, int_7637)
    
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type', result_div_7638)
    
    # ################# End of 'function_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function_2' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_7639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7639)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function_2'
    return stypy_return_type_7639

# Assigning a type to the variable 'function_2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'function_2', function_2)

@norecursion
def function_3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function_3'
    module_type_store = module_type_store.open_function_context('function_3', 7, 0, False)
    
    # Passed parameters checking function
    function_3.stypy_localization = localization
    function_3.stypy_type_of_self = None
    function_3.stypy_type_store = module_type_store
    function_3.stypy_function_name = 'function_3'
    function_3.stypy_param_names_list = ['x']
    function_3.stypy_varargs_param_name = None
    function_3.stypy_kwargs_param_name = None
    function_3.stypy_call_defaults = defaults
    function_3.stypy_call_varargs = varargs
    function_3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function_3', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function_3', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function_3(...)' code ##################

    # Getting the type of 'x' (line 8)
    x_7640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'x')
    int_7641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
    # Applying the binary operator 'div' (line 8)
    result_div_7642 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 11), 'div', x_7640, int_7641)
    
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', result_div_7642)
    
    # ################# End of 'function_3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function_3' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_7643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7643)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function_3'
    return stypy_return_type_7643

# Assigning a type to the variable 'function_3' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'function_3', function_3)

# Assigning a Call to a Name (line 10):

# Call to function_1(...): (line 10)
# Processing the call arguments (line 10)
str_7645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'str', 'a')
# Processing the call keyword arguments (line 10)
kwargs_7646 = {}
# Getting the type of 'function_1' (line 10)
function_1_7644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'function_1', False)
# Calling function_1(args, kwargs) (line 10)
function_1_call_result_7647 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), function_1_7644, *[str_7645], **kwargs_7646)

# Assigning a type to the variable 'r1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r1', function_1_call_result_7647)

# Assigning a Call to a Name (line 11):

# Call to function_2(...): (line 11)
# Processing the call arguments (line 11)

# Call to range(...): (line 11)
# Processing the call arguments (line 11)
int_7650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
# Processing the call keyword arguments (line 11)
kwargs_7651 = {}
# Getting the type of 'range' (line 11)
range_7649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'range', False)
# Calling range(args, kwargs) (line 11)
range_call_result_7652 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), range_7649, *[int_7650], **kwargs_7651)

# Processing the call keyword arguments (line 11)
kwargs_7653 = {}
# Getting the type of 'function_2' (line 11)
function_2_7648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'function_2', False)
# Calling function_2(args, kwargs) (line 11)
function_2_call_result_7654 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), function_2_7648, *[range_call_result_7652], **kwargs_7653)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', function_2_call_result_7654)

# Assigning a Call to a Name (line 12):

# Call to function_3(...): (line 12)
# Processing the call arguments (line 12)
int_7656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
# Processing the call keyword arguments (line 12)
kwargs_7657 = {}
# Getting the type of 'function_3' (line 12)
function_3_7655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'function_3', False)
# Calling function_3(args, kwargs) (line 12)
function_3_call_result_7658 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), function_3_7655, *[int_7656], **kwargs_7657)

# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', function_3_call_result_7658)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

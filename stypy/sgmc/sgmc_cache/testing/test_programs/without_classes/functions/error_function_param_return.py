
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: 
4: def function(x, **kwargs):
5:     return x
6: 
7: 
8: y = function(3)
9: r1 = y * 2  # Correct and nothing reported
10: r2 = y[12]  # Unreported
11: r3 = math.pow(function("a"), 3)  # Unreported
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import math' statement (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)


@norecursion
def function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function'
    module_type_store = module_type_store.open_function_context('function', 4, 0, False)
    
    # Passed parameters checking function
    function.stypy_localization = localization
    function.stypy_type_of_self = None
    function.stypy_type_store = module_type_store
    function.stypy_function_name = 'function'
    function.stypy_param_names_list = ['x']
    function.stypy_varargs_param_name = None
    function.stypy_kwargs_param_name = 'kwargs'
    function.stypy_call_defaults = defaults
    function.stypy_call_varargs = varargs
    function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function', ['x'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function(...)' code ##################

    # Getting the type of 'x' (line 5)
    x_7598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type', x_7598)
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_7599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7599)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_7599

# Assigning a type to the variable 'function' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'function', function)

# Assigning a Call to a Name (line 8):

# Call to function(...): (line 8)
# Processing the call arguments (line 8)
int_7601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'int')
# Processing the call keyword arguments (line 8)
kwargs_7602 = {}
# Getting the type of 'function' (line 8)
function_7600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'function', False)
# Calling function(args, kwargs) (line 8)
function_call_result_7603 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), function_7600, *[int_7601], **kwargs_7602)

# Assigning a type to the variable 'y' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'y', function_call_result_7603)

# Assigning a BinOp to a Name (line 9):
# Getting the type of 'y' (line 9)
y_7604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'y')
int_7605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'int')
# Applying the binary operator '*' (line 9)
result_mul_7606 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 5), '*', y_7604, int_7605)

# Assigning a type to the variable 'r1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r1', result_mul_7606)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_7607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 7), 'int')
# Getting the type of 'y' (line 10)
y_7608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'y')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___7609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), y_7608, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_7610 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), getitem___7609, int_7607)

# Assigning a type to the variable 'r2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r2', subscript_call_result_7610)

# Assigning a Call to a Name (line 11):

# Call to pow(...): (line 11)
# Processing the call arguments (line 11)

# Call to function(...): (line 11)
# Processing the call arguments (line 11)
str_7614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'str', 'a')
# Processing the call keyword arguments (line 11)
kwargs_7615 = {}
# Getting the type of 'function' (line 11)
function_7613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'function', False)
# Calling function(args, kwargs) (line 11)
function_call_result_7616 = invoke(stypy.reporting.localization.Localization(__file__, 11, 14), function_7613, *[str_7614], **kwargs_7615)

int_7617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'int')
# Processing the call keyword arguments (line 11)
kwargs_7618 = {}
# Getting the type of 'math' (line 11)
math_7611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 11)
pow_7612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), math_7611, 'pow')
# Calling pow(args, kwargs) (line 11)
pow_call_result_7619 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), pow_7612, *[function_call_result_7616, int_7617], **kwargs_7618)

# Assigning a type to the variable 'r3' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r3', pow_call_result_7619)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

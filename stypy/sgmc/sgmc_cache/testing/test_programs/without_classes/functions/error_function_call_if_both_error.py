
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def function_1(x):
2:     a = 0
3:     if a > 0:
4:         x /= 2
5:     else:
6:         x -= 2
7: 
8:     return 3
9: 
10: r1 = function_1("a")  # Not Reported on call site
11: r2 = function_1(range(5))  # Not Reported on call site
12: 
13: def function_2(x):
14:     a = 0
15:     if a > 0:
16:         x /= 2
17:         return x
18:     else:
19:         x -= 2
20:         return x
21: 
22: r3 = function_2("a")  # Not Reported on call site
23: r4 = function_2(range(5))  # Not Reported on call site

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

    
    # Assigning a Num to a Name (line 2):
    int_7425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
    # Assigning a type to the variable 'a' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'a', int_7425)
    
    
    # Getting the type of 'a' (line 3)
    a_7426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'a')
    int_7427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
    # Applying the binary operator '>' (line 3)
    result_gt_7428 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 7), '>', a_7426, int_7427)
    
    # Testing the type of an if condition (line 3)
    if_condition_7429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 4), result_gt_7428)
    # Assigning a type to the variable 'if_condition_7429' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'if_condition_7429', if_condition_7429)
    # SSA begins for if statement (line 3)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 4)
    x_7430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'x')
    int_7431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 13), 'int')
    # Applying the binary operator 'div=' (line 4)
    result_div_7432 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 8), 'div=', x_7430, int_7431)
    # Assigning a type to the variable 'x' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'x', result_div_7432)
    
    # SSA branch for the else part of an if statement (line 3)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'x' (line 6)
    x_7433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'x')
    int_7434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'int')
    # Applying the binary operator '-=' (line 6)
    result_isub_7435 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 8), '-=', x_7433, int_7434)
    # Assigning a type to the variable 'x' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'x', result_isub_7435)
    
    # SSA join for if statement (line 3)
    module_type_store = module_type_store.join_ssa_context()
    
    int_7436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', int_7436)
    
    # ################# End of 'function_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function_1' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7437)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function_1'
    return stypy_return_type_7437

# Assigning a type to the variable 'function_1' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'function_1', function_1)

# Assigning a Call to a Name (line 10):

# Call to function_1(...): (line 10)
# Processing the call arguments (line 10)
str_7439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'str', 'a')
# Processing the call keyword arguments (line 10)
kwargs_7440 = {}
# Getting the type of 'function_1' (line 10)
function_1_7438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'function_1', False)
# Calling function_1(args, kwargs) (line 10)
function_1_call_result_7441 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), function_1_7438, *[str_7439], **kwargs_7440)

# Assigning a type to the variable 'r1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r1', function_1_call_result_7441)

# Assigning a Call to a Name (line 11):

# Call to function_1(...): (line 11)
# Processing the call arguments (line 11)

# Call to range(...): (line 11)
# Processing the call arguments (line 11)
int_7444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
# Processing the call keyword arguments (line 11)
kwargs_7445 = {}
# Getting the type of 'range' (line 11)
range_7443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'range', False)
# Calling range(args, kwargs) (line 11)
range_call_result_7446 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), range_7443, *[int_7444], **kwargs_7445)

# Processing the call keyword arguments (line 11)
kwargs_7447 = {}
# Getting the type of 'function_1' (line 11)
function_1_7442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'function_1', False)
# Calling function_1(args, kwargs) (line 11)
function_1_call_result_7448 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), function_1_7442, *[range_call_result_7446], **kwargs_7447)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', function_1_call_result_7448)

@norecursion
def function_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function_2'
    module_type_store = module_type_store.open_function_context('function_2', 13, 0, False)
    
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

    
    # Assigning a Num to a Name (line 14):
    int_7449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'int')
    # Assigning a type to the variable 'a' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'a', int_7449)
    
    
    # Getting the type of 'a' (line 15)
    a_7450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'a')
    int_7451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'int')
    # Applying the binary operator '>' (line 15)
    result_gt_7452 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 7), '>', a_7450, int_7451)
    
    # Testing the type of an if condition (line 15)
    if_condition_7453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), result_gt_7452)
    # Assigning a type to the variable 'if_condition_7453' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_7453', if_condition_7453)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 16)
    x_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'x')
    int_7455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div=' (line 16)
    result_div_7456 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 8), 'div=', x_7454, int_7455)
    # Assigning a type to the variable 'x' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'x', result_div_7456)
    
    # Getting the type of 'x' (line 17)
    x_7457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', x_7457)
    # SSA branch for the else part of an if statement (line 15)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'x' (line 19)
    x_7458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'x')
    int_7459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'int')
    # Applying the binary operator '-=' (line 19)
    result_isub_7460 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 8), '-=', x_7458, int_7459)
    # Assigning a type to the variable 'x' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'x', result_isub_7460)
    
    # Getting the type of 'x' (line 20)
    x_7461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', x_7461)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'function_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function_2' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_7462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7462)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function_2'
    return stypy_return_type_7462

# Assigning a type to the variable 'function_2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'function_2', function_2)

# Assigning a Call to a Name (line 22):

# Call to function_2(...): (line 22)
# Processing the call arguments (line 22)
str_7464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', 'a')
# Processing the call keyword arguments (line 22)
kwargs_7465 = {}
# Getting the type of 'function_2' (line 22)
function_2_7463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'function_2', False)
# Calling function_2(args, kwargs) (line 22)
function_2_call_result_7466 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), function_2_7463, *[str_7464], **kwargs_7465)

# Assigning a type to the variable 'r3' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r3', function_2_call_result_7466)

# Assigning a Call to a Name (line 23):

# Call to function_2(...): (line 23)
# Processing the call arguments (line 23)

# Call to range(...): (line 23)
# Processing the call arguments (line 23)
int_7469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'int')
# Processing the call keyword arguments (line 23)
kwargs_7470 = {}
# Getting the type of 'range' (line 23)
range_7468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'range', False)
# Calling range(args, kwargs) (line 23)
range_call_result_7471 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), range_7468, *[int_7469], **kwargs_7470)

# Processing the call keyword arguments (line 23)
kwargs_7472 = {}
# Getting the type of 'function_2' (line 23)
function_2_7467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'function_2', False)
# Calling function_2(args, kwargs) (line 23)
function_2_call_result_7473 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), function_2_7467, *[range_call_result_7471], **kwargs_7472)

# Assigning a type to the variable 'r4' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r4', function_2_call_result_7473)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def function(x, **kwargs):
2:     a = 0
3:     if a > 0:
4:         return int(x)
5:     else:
6:         return kwargs[0]  # Should warn about None
7: 
8: 
9: y = function(3, val="hi")
10: 
11: y2 = y.thisdonotexist()  # Unreported
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function'
    module_type_store = module_type_store.open_function_context('function', 1, 0, False)
    
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

    
    # Assigning a Num to a Name (line 2):
    int_7553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
    # Assigning a type to the variable 'a' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'a', int_7553)
    
    
    # Getting the type of 'a' (line 3)
    a_7554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'a')
    int_7555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
    # Applying the binary operator '>' (line 3)
    result_gt_7556 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 7), '>', a_7554, int_7555)
    
    # Testing the type of an if condition (line 3)
    if_condition_7557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 4), result_gt_7556)
    # Assigning a type to the variable 'if_condition_7557' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'if_condition_7557', if_condition_7557)
    # SSA begins for if statement (line 3)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to int(...): (line 4)
    # Processing the call arguments (line 4)
    # Getting the type of 'x' (line 4)
    x_7559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'x', False)
    # Processing the call keyword arguments (line 4)
    kwargs_7560 = {}
    # Getting the type of 'int' (line 4)
    int_7558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), 'int', False)
    # Calling int(args, kwargs) (line 4)
    int_call_result_7561 = invoke(stypy.reporting.localization.Localization(__file__, 4, 15), int_7558, *[x_7559], **kwargs_7560)
    
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'stypy_return_type', int_call_result_7561)
    # SSA branch for the else part of an if statement (line 3)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    int_7562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')
    # Getting the type of 'kwargs' (line 6)
    kwargs_7563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 6)
    getitem___7564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 15), kwargs_7563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 6)
    subscript_call_result_7565 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), getitem___7564, int_7562)
    
    # Assigning a type to the variable 'stypy_return_type' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', subscript_call_result_7565)
    # SSA join for if statement (line 3)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7566)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_7566

# Assigning a type to the variable 'function' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'function', function)

# Assigning a Call to a Name (line 9):

# Call to function(...): (line 9)
# Processing the call arguments (line 9)
int_7568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
# Processing the call keyword arguments (line 9)
str_7569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'hi')
keyword_7570 = str_7569
kwargs_7571 = {'val': keyword_7570}
# Getting the type of 'function' (line 9)
function_7567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'function', False)
# Calling function(args, kwargs) (line 9)
function_call_result_7572 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), function_7567, *[int_7568], **kwargs_7571)

# Assigning a type to the variable 'y' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'y', function_call_result_7572)

# Assigning a Call to a Name (line 11):

# Call to thisdonotexist(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_7575 = {}
# Getting the type of 'y' (line 11)
y_7573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'y', False)
# Obtaining the member 'thisdonotexist' of a type (line 11)
thisdonotexist_7574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), y_7573, 'thisdonotexist')
# Calling thisdonotexist(args, kwargs) (line 11)
thisdonotexist_call_result_7576 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), thisdonotexist_7574, *[], **kwargs_7575)

# Assigning a type to the variable 'y2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'y2', thisdonotexist_call_result_7576)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

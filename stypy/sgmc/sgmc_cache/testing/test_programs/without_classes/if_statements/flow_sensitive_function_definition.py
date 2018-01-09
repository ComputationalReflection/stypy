
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #import random
2: a = 3
3: #condition = random.randint(0, 1) == 0
4: condition = a > 0
5: if condition:
6:     f = lambda x: x
7: else:
8:     f = lambda x,y: x+y
9: f(1)
10: f()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 2):
int_4893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'int')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', int_4893)

# Assigning a Compare to a Name (line 4):

# Getting the type of 'a' (line 4)
a_4894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 12), 'a')
int_4895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 16), 'int')
# Applying the binary operator '>' (line 4)
result_gt_4896 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 12), '>', a_4894, int_4895)

# Assigning a type to the variable 'condition' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'condition', result_gt_4896)

# Getting the type of 'condition' (line 5)
condition_4897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'condition')
# Testing the type of an if condition (line 5)
if_condition_4898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 0), condition_4897)
# Assigning a type to the variable 'if_condition_4898' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'if_condition_4898', if_condition_4898)
# SSA begins for if statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Lambda to a Name (line 6):

@norecursion
def _stypy_temp_lambda_6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_6'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_6', 6, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_6.stypy_localization = localization
    _stypy_temp_lambda_6.stypy_type_of_self = None
    _stypy_temp_lambda_6.stypy_type_store = module_type_store
    _stypy_temp_lambda_6.stypy_function_name = '_stypy_temp_lambda_6'
    _stypy_temp_lambda_6.stypy_param_names_list = ['x']
    _stypy_temp_lambda_6.stypy_varargs_param_name = None
    _stypy_temp_lambda_6.stypy_kwargs_param_name = None
    _stypy_temp_lambda_6.stypy_call_defaults = defaults
    _stypy_temp_lambda_6.stypy_call_varargs = varargs
    _stypy_temp_lambda_6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_6', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_6', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 6)
    x_4899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 18), 'x')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', x_4899)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_6' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_4900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4900)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_6'
    return stypy_return_type_4900

# Assigning a type to the variable '_stypy_temp_lambda_6' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), '_stypy_temp_lambda_6', _stypy_temp_lambda_6)
# Getting the type of '_stypy_temp_lambda_6' (line 6)
_stypy_temp_lambda_6_4901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), '_stypy_temp_lambda_6')
# Assigning a type to the variable 'f' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'f', _stypy_temp_lambda_6_4901)
# SSA branch for the else part of an if statement (line 5)
module_type_store.open_ssa_branch('else')

# Assigning a Lambda to a Name (line 8):

@norecursion
def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_7'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 8, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_7.stypy_localization = localization
    _stypy_temp_lambda_7.stypy_type_of_self = None
    _stypy_temp_lambda_7.stypy_type_store = module_type_store
    _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
    _stypy_temp_lambda_7.stypy_param_names_list = ['x', 'y']
    _stypy_temp_lambda_7.stypy_varargs_param_name = None
    _stypy_temp_lambda_7.stypy_kwargs_param_name = None
    _stypy_temp_lambda_7.stypy_call_defaults = defaults
    _stypy_temp_lambda_7.stypy_call_varargs = varargs
    _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_7', ['x', 'y'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 8)
    x_4902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 20), 'x')
    # Getting the type of 'y' (line 8)
    y_4903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 22), 'y')
    # Applying the binary operator '+' (line 8)
    result_add_4904 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 20), '+', x_4902, y_4903)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', result_add_4904)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_7' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_4905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_7'
    return stypy_return_type_4905

# Assigning a type to the variable '_stypy_temp_lambda_7' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
# Getting the type of '_stypy_temp_lambda_7' (line 8)
_stypy_temp_lambda_7_4906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), '_stypy_temp_lambda_7')
# Assigning a type to the variable 'f' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'f', _stypy_temp_lambda_7_4906)
# SSA join for if statement (line 5)
module_type_store = module_type_store.join_ssa_context()


# Call to f(...): (line 9)
# Processing the call arguments (line 9)
int_4908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 2), 'int')
# Processing the call keyword arguments (line 9)
kwargs_4909 = {}
# Getting the type of 'f' (line 9)
f_4907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'f', False)
# Calling f(args, kwargs) (line 9)
f_call_result_4910 = invoke(stypy.reporting.localization.Localization(__file__, 9, 0), f_4907, *[int_4908], **kwargs_4909)


# Call to f(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_4912 = {}
# Getting the type of 'f' (line 10)
f_4911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'f', False)
# Calling f(args, kwargs) (line 10)
f_call_result_4913 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), f_4911, *[], **kwargs_4912)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def functionb(x):
2:     for i in range(5):
3:         x /= 2
4: 
5:     return x
6: r1 = functionb("a")  # Not Reported on call site
7: r2 = functionb(range(5))  # Not Reported on call site
8: 
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def functionb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'functionb'
    module_type_store = module_type_store.open_function_context('functionb', 1, 0, False)
    
    # Passed parameters checking function
    functionb.stypy_localization = localization
    functionb.stypy_type_of_self = None
    functionb.stypy_type_store = module_type_store
    functionb.stypy_function_name = 'functionb'
    functionb.stypy_param_names_list = ['x']
    functionb.stypy_varargs_param_name = None
    functionb.stypy_kwargs_param_name = None
    functionb.stypy_call_defaults = defaults
    functionb.stypy_call_varargs = varargs
    functionb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'functionb', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'functionb', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'functionb(...)' code ##################

    
    
    # Call to range(...): (line 2)
    # Processing the call arguments (line 2)
    int_7578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 19), 'int')
    # Processing the call keyword arguments (line 2)
    kwargs_7579 = {}
    # Getting the type of 'range' (line 2)
    range_7577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 13), 'range', False)
    # Calling range(args, kwargs) (line 2)
    range_call_result_7580 = invoke(stypy.reporting.localization.Localization(__file__, 2, 13), range_7577, *[int_7578], **kwargs_7579)
    
    # Testing the type of a for loop iterable (line 2)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 2, 4), range_call_result_7580)
    # Getting the type of the for loop variable (line 2)
    for_loop_var_7581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 2, 4), range_call_result_7580)
    # Assigning a type to the variable 'i' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'i', for_loop_var_7581)
    # SSA begins for a for statement (line 2)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'x' (line 3)
    x_7582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'x')
    int_7583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 13), 'int')
    # Applying the binary operator 'div=' (line 3)
    result_div_7584 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 8), 'div=', x_7582, int_7583)
    # Assigning a type to the variable 'x' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'x', result_div_7584)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 5)
    x_7585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type', x_7585)
    
    # ################# End of 'functionb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'functionb' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7586)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'functionb'
    return stypy_return_type_7586

# Assigning a type to the variable 'functionb' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'functionb', functionb)

# Assigning a Call to a Name (line 6):

# Call to functionb(...): (line 6)
# Processing the call arguments (line 6)
str_7588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', 'a')
# Processing the call keyword arguments (line 6)
kwargs_7589 = {}
# Getting the type of 'functionb' (line 6)
functionb_7587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'functionb', False)
# Calling functionb(args, kwargs) (line 6)
functionb_call_result_7590 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), functionb_7587, *[str_7588], **kwargs_7589)

# Assigning a type to the variable 'r1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r1', functionb_call_result_7590)

# Assigning a Call to a Name (line 7):

# Call to functionb(...): (line 7)
# Processing the call arguments (line 7)

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
int_7593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'int')
# Processing the call keyword arguments (line 7)
kwargs_7594 = {}
# Getting the type of 'range' (line 7)
range_7592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_7595 = invoke(stypy.reporting.localization.Localization(__file__, 7, 15), range_7592, *[int_7593], **kwargs_7594)

# Processing the call keyword arguments (line 7)
kwargs_7596 = {}
# Getting the type of 'functionb' (line 7)
functionb_7591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'functionb', False)
# Calling functionb(args, kwargs) (line 7)
functionb_call_result_7597 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), functionb_7591, *[range_call_result_7595], **kwargs_7596)

# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r2', functionb_call_result_7597)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

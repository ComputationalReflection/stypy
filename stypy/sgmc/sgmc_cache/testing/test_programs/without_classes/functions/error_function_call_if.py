
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def functionb(x):
2:     a = 0
3:     if a > 0:
4:         x = x / 2
5:     return x
6: 
7: 
8: r1 = functionb("a")  # Nothing is reported on call site
9: r2 = functionb(range(5))  # Nothing is reported on call site
10: 

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

    
    # Assigning a Num to a Name (line 2):
    int_7404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
    # Assigning a type to the variable 'a' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'a', int_7404)
    
    
    # Getting the type of 'a' (line 3)
    a_7405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'a')
    int_7406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
    # Applying the binary operator '>' (line 3)
    result_gt_7407 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 7), '>', a_7405, int_7406)
    
    # Testing the type of an if condition (line 3)
    if_condition_7408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 4), result_gt_7407)
    # Assigning a type to the variable 'if_condition_7408' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'if_condition_7408', if_condition_7408)
    # SSA begins for if statement (line 3)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 4):
    # Getting the type of 'x' (line 4)
    x_7409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 12), 'x')
    int_7410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 16), 'int')
    # Applying the binary operator 'div' (line 4)
    result_div_7411 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 12), 'div', x_7409, int_7410)
    
    # Assigning a type to the variable 'x' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'x', result_div_7411)
    # SSA join for if statement (line 3)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 5)
    x_7412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type', x_7412)
    
    # ################# End of 'functionb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'functionb' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7413)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'functionb'
    return stypy_return_type_7413

# Assigning a type to the variable 'functionb' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'functionb', functionb)

# Assigning a Call to a Name (line 8):

# Call to functionb(...): (line 8)
# Processing the call arguments (line 8)
str_7415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', 'a')
# Processing the call keyword arguments (line 8)
kwargs_7416 = {}
# Getting the type of 'functionb' (line 8)
functionb_7414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'functionb', False)
# Calling functionb(args, kwargs) (line 8)
functionb_call_result_7417 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), functionb_7414, *[str_7415], **kwargs_7416)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', functionb_call_result_7417)

# Assigning a Call to a Name (line 9):

# Call to functionb(...): (line 9)
# Processing the call arguments (line 9)

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_7420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'int')
# Processing the call keyword arguments (line 9)
kwargs_7421 = {}
# Getting the type of 'range' (line 9)
range_7419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_7422 = invoke(stypy.reporting.localization.Localization(__file__, 9, 15), range_7419, *[int_7420], **kwargs_7421)

# Processing the call keyword arguments (line 9)
kwargs_7423 = {}
# Getting the type of 'functionb' (line 9)
functionb_7418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'functionb', False)
# Calling functionb(args, kwargs) (line 9)
functionb_call_result_7424 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), functionb_7418, *[range_call_result_7422], **kwargs_7423)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', functionb_call_result_7424)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

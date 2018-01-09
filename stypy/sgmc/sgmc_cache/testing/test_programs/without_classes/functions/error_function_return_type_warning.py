
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def get_str():
2:     if True:
3:         return "hi"
4:     else:
5:         return 2
6: 
7: 
8: a = 4 + get_str()  # Not detected
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def get_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_str'
    module_type_store = module_type_store.open_function_context('get_str', 1, 0, False)
    
    # Passed parameters checking function
    get_str.stypy_localization = localization
    get_str.stypy_type_of_self = None
    get_str.stypy_type_store = module_type_store
    get_str.stypy_function_name = 'get_str'
    get_str.stypy_param_names_list = []
    get_str.stypy_varargs_param_name = None
    get_str.stypy_kwargs_param_name = None
    get_str.stypy_call_defaults = defaults
    get_str.stypy_call_varargs = varargs
    get_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_str', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_str', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_str(...)' code ##################

    
    # Getting the type of 'True' (line 2)
    True_7659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 7), 'True')
    # Testing the type of an if condition (line 2)
    if_condition_7660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2, 4), True_7659)
    # Assigning a type to the variable 'if_condition_7660' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'if_condition_7660', if_condition_7660)
    # SSA begins for if statement (line 2)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_7661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'str', 'hi')
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', str_7661)
    # SSA branch for the else part of an if statement (line 2)
    module_type_store.open_ssa_branch('else')
    int_7662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', int_7662)
    # SSA join for if statement (line 2)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_str' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7663)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_str'
    return stypy_return_type_7663

# Assigning a type to the variable 'get_str' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'get_str', get_str)

# Assigning a BinOp to a Name (line 8):
int_7664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'int')

# Call to get_str(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_7666 = {}
# Getting the type of 'get_str' (line 8)
get_str_7665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'get_str', False)
# Calling get_str(args, kwargs) (line 8)
get_str_call_result_7667 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), get_str_7665, *[], **kwargs_7666)

# Applying the binary operator '+' (line 8)
result_add_7668 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), '+', int_7664, get_str_call_result_7667)

# Assigning a type to the variable 'a' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'a', result_add_7668)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

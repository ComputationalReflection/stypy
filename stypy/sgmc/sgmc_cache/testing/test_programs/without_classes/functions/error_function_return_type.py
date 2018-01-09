
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def problematic_get():
2:     if True:
3:         return "hi"
4:     else:
5:         return [1, 2]
6: 
7: 
8: x = problematic_get() / 3
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def problematic_get(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'problematic_get'
    module_type_store = module_type_store.open_function_context('problematic_get', 1, 0, False)
    
    # Passed parameters checking function
    problematic_get.stypy_localization = localization
    problematic_get.stypy_type_of_self = None
    problematic_get.stypy_type_store = module_type_store
    problematic_get.stypy_function_name = 'problematic_get'
    problematic_get.stypy_param_names_list = []
    problematic_get.stypy_varargs_param_name = None
    problematic_get.stypy_kwargs_param_name = None
    problematic_get.stypy_call_defaults = defaults
    problematic_get.stypy_call_varargs = varargs
    problematic_get.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'problematic_get', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'problematic_get', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'problematic_get(...)' code ##################

    
    # Getting the type of 'True' (line 2)
    True_7620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 7), 'True')
    # Testing the type of an if condition (line 2)
    if_condition_7621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2, 4), True_7620)
    # Assigning a type to the variable 'if_condition_7621' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'if_condition_7621', if_condition_7621)
    # SSA begins for if statement (line 2)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_7622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'str', 'hi')
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', str_7622)
    # SSA branch for the else part of an if statement (line 2)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'list' (line 5)
    list_7623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 5)
    # Adding element type (line 5)
    int_7624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), list_7623, int_7624)
    # Adding element type (line 5)
    int_7625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), list_7623, int_7625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', list_7623)
    # SSA join for if statement (line 2)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'problematic_get(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'problematic_get' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'problematic_get'
    return stypy_return_type_7626

# Assigning a type to the variable 'problematic_get' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'problematic_get', problematic_get)

# Assigning a BinOp to a Name (line 8):

# Call to problematic_get(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_7628 = {}
# Getting the type of 'problematic_get' (line 8)
problematic_get_7627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'problematic_get', False)
# Calling problematic_get(args, kwargs) (line 8)
problematic_get_call_result_7629 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), problematic_get_7627, *[], **kwargs_7628)

int_7630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'int')
# Applying the binary operator 'div' (line 8)
result_div_7631 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), 'div', problematic_get_call_result_7629, int_7630)

# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x', result_div_7631)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

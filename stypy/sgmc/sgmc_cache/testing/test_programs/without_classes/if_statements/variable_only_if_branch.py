
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: def f():
3:     a = 3
4:     condition = a > 0
5:     if condition:
6:         return a.x
7:     else:
8:         return a
9: 
10: x = f()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 2, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = []
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    
    # Assigning a Num to a Name (line 3):
    int_4934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
    # Assigning a type to the variable 'a' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'a', int_4934)
    
    # Assigning a Compare to a Name (line 4):
    
    # Getting the type of 'a' (line 4)
    a_4935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 16), 'a')
    int_4936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 20), 'int')
    # Applying the binary operator '>' (line 4)
    result_gt_4937 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 16), '>', a_4935, int_4936)
    
    # Assigning a type to the variable 'condition' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'condition', result_gt_4937)
    
    # Getting the type of 'condition' (line 5)
    condition_4938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'condition')
    # Testing the type of an if condition (line 5)
    if_condition_4939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 4), condition_4938)
    # Assigning a type to the variable 'if_condition_4939' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'if_condition_4939', if_condition_4939)
    # SSA begins for if statement (line 5)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'a' (line 6)
    a_4940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'a')
    # Obtaining the member 'x' of a type (line 6)
    x_4941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 15), a_4940, 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', x_4941)
    # SSA branch for the else part of an if statement (line 5)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'a' (line 8)
    a_4942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', a_4942)
    # SSA join for if statement (line 5)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 2)
    stypy_return_type_4943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_4943

# Assigning a type to the variable 'f' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'f', f)

# Assigning a Call to a Name (line 10):

# Call to f(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_4945 = {}
# Getting the type of 'f' (line 10)
f_4944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'f', False)
# Calling f(args, kwargs) (line 10)
f_call_result_4946 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), f_4944, *[], **kwargs_4945)

# Assigning a type to the variable 'x' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'x', f_call_result_4946)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

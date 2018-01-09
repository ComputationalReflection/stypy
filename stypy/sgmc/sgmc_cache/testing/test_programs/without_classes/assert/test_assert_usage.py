
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def f():
2:     return 3 > 2
3: 
4: 
5: assert True
6: assert False
7: assert f()
8: 

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
    module_type_store = module_type_store.open_function_context('f', 1, 0, False)
    
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

    
    int_6213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'int')
    int_6214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'int')
    # Applying the binary operator '>' (line 2)
    result_gt_6215 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), '>', int_6213, int_6214)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', result_gt_6215)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_6216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_6216

# Assigning a type to the variable 'f' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'f', f)
# Evaluating assert statement condition
# Getting the type of 'True' (line 5)
True_6217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'True')
# Evaluating assert statement condition
# Getting the type of 'False' (line 6)
False_6218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'False')
# Evaluating assert statement condition

# Call to f(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_6220 = {}
# Getting the type of 'f' (line 7)
f_6219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'f', False)
# Calling f(args, kwargs) (line 7)
f_call_result_6221 = invoke(stypy.reporting.localization.Localization(__file__, 7, 7), f_6219, *[], **kwargs_6220)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

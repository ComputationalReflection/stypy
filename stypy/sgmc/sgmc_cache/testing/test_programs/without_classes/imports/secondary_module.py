
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def secondary_function():
2:     return "foo"
3: 
4: number = 4.5

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def secondary_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'secondary_function'
    module_type_store = module_type_store.open_function_context('secondary_function', 1, 0, False)
    
    # Passed parameters checking function
    secondary_function.stypy_localization = localization
    secondary_function.stypy_type_of_self = None
    secondary_function.stypy_type_store = module_type_store
    secondary_function.stypy_function_name = 'secondary_function'
    secondary_function.stypy_param_names_list = []
    secondary_function.stypy_varargs_param_name = None
    secondary_function.stypy_kwargs_param_name = None
    secondary_function.stypy_call_defaults = defaults
    secondary_function.stypy_call_varargs = varargs
    secondary_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'secondary_function', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'secondary_function', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'secondary_function(...)' code ##################

    str_5166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'foo')
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', str_5166)
    
    # ################# End of 'secondary_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'secondary_function' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_5167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5167)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'secondary_function'
    return stypy_return_type_5167

# Assigning a type to the variable 'secondary_function' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_function', secondary_function)

# Assigning a Num to a Name (line 4):
float_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 9), 'float')
# Assigning a type to the variable 'number' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'number', float_5168)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

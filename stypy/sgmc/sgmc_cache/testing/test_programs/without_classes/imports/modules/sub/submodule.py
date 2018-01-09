
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: submodule_var = 3
2: 
3: def submodule_func():
4:     return "submodule"
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_5194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 16), 'int')
# Assigning a type to the variable 'submodule_var' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'submodule_var', int_5194)

@norecursion
def submodule_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'submodule_func'
    module_type_store = module_type_store.open_function_context('submodule_func', 3, 0, False)
    
    # Passed parameters checking function
    submodule_func.stypy_localization = localization
    submodule_func.stypy_type_of_self = None
    submodule_func.stypy_type_store = module_type_store
    submodule_func.stypy_function_name = 'submodule_func'
    submodule_func.stypy_param_names_list = []
    submodule_func.stypy_varargs_param_name = None
    submodule_func.stypy_kwargs_param_name = None
    submodule_func.stypy_call_defaults = defaults
    submodule_func.stypy_call_varargs = varargs
    submodule_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'submodule_func', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'submodule_func', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'submodule_func(...)' code ##################

    str_5195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', 'submodule')
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type', str_5195)
    
    # ################# End of 'submodule_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'submodule_func' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'submodule_func'
    return stypy_return_type_5196

# Assigning a type to the variable 'submodule_func' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'submodule_func', submodule_func)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

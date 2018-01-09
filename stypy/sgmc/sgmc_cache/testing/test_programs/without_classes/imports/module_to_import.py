
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: global_a = 1
2: 
3: 
4: def f_parent():
5:     local_a = 2
6: 
7:     return local_a
8: 
9: 
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_5141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'int')
# Assigning a type to the variable 'global_a' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'global_a', int_5141)

@norecursion
def f_parent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f_parent'
    module_type_store = module_type_store.open_function_context('f_parent', 4, 0, False)
    
    # Passed parameters checking function
    f_parent.stypy_localization = localization
    f_parent.stypy_type_of_self = None
    f_parent.stypy_type_store = module_type_store
    f_parent.stypy_function_name = 'f_parent'
    f_parent.stypy_param_names_list = []
    f_parent.stypy_varargs_param_name = None
    f_parent.stypy_kwargs_param_name = None
    f_parent.stypy_call_defaults = defaults
    f_parent.stypy_call_varargs = varargs
    f_parent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f_parent', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f_parent', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f_parent(...)' code ##################

    
    # Assigning a Num to a Name (line 5):
    int_5142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
    # Assigning a type to the variable 'local_a' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'local_a', int_5142)
    # Getting the type of 'local_a' (line 7)
    local_a_5143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'local_a')
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', local_a_5143)
    
    # ################# End of 'f_parent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f_parent' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_5144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5144)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f_parent'
    return stypy_return_type_5144

# Assigning a type to the variable 'f_parent' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'f_parent', f_parent)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

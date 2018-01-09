
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: def identity(x):
5:     return x
6: 
7: y = identity(3)
8: z = identity("3")
9: w = identity(3.4)
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def identity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'identity'
    module_type_store = module_type_store.open_function_context('identity', 4, 0, False)
    
    # Passed parameters checking function
    identity.stypy_localization = localization
    identity.stypy_type_of_self = None
    identity.stypy_type_store = module_type_store
    identity.stypy_function_name = 'identity'
    identity.stypy_param_names_list = ['x']
    identity.stypy_varargs_param_name = None
    identity.stypy_kwargs_param_name = None
    identity.stypy_call_defaults = defaults
    identity.stypy_call_varargs = varargs
    identity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'identity', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'identity', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'identity(...)' code ##################

    # Getting the type of 'x' (line 5)
    x_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type', x_675)
    
    # ################# End of 'identity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'identity' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_676)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'identity'
    return stypy_return_type_676

# Assigning a type to the variable 'identity' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'identity', identity)

# Assigning a Call to a Name (line 7):

# Call to identity(...): (line 7)
# Processing the call arguments (line 7)
int_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
# Processing the call keyword arguments (line 7)
kwargs_679 = {}
# Getting the type of 'identity' (line 7)
identity_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'identity', False)
# Calling identity(args, kwargs) (line 7)
identity_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), identity_677, *[int_678], **kwargs_679)

# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'y', identity_call_result_680)

# Assigning a Call to a Name (line 8):

# Call to identity(...): (line 8)
# Processing the call arguments (line 8)
str_682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'str', '3')
# Processing the call keyword arguments (line 8)
kwargs_683 = {}
# Getting the type of 'identity' (line 8)
identity_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'identity', False)
# Calling identity(args, kwargs) (line 8)
identity_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), identity_681, *[str_682], **kwargs_683)

# Assigning a type to the variable 'z' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'z', identity_call_result_684)

# Assigning a Call to a Name (line 9):

# Call to identity(...): (line 9)
# Processing the call arguments (line 9)
float_686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'float')
# Processing the call keyword arguments (line 9)
kwargs_687 = {}
# Getting the type of 'identity' (line 9)
identity_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'identity', False)
# Calling identity(args, kwargs) (line 9)
identity_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), identity_685, *[float_686], **kwargs_687)

# Assigning a type to the variable 'w' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'w', identity_call_result_688)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

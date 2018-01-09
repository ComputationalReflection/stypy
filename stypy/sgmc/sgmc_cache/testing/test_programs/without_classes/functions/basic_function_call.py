
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def function(x):
2:     return x
3: 
4: 
5: y = function(3)
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function'
    module_type_store = module_type_store.open_function_context('function', 1, 0, False)
    
    # Passed parameters checking function
    function.stypy_localization = localization
    function.stypy_type_of_self = None
    function.stypy_type_store = module_type_store
    function.stypy_function_name = 'function'
    function.stypy_param_names_list = ['x']
    function.stypy_varargs_param_name = None
    function.stypy_kwargs_param_name = None
    function.stypy_call_defaults = defaults
    function.stypy_call_varargs = varargs
    function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function(...)' code ##################

    # Getting the type of 'x' (line 2)
    x_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', x_669)
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_670)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_670

# Assigning a type to the variable 'function' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'function', function)

# Assigning a Call to a Name (line 5):

# Call to function(...): (line 5)
# Processing the call arguments (line 5)
int_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Processing the call keyword arguments (line 5)
kwargs_673 = {}
# Getting the type of 'function' (line 5)
function_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'function', False)
# Calling function(args, kwargs) (line 5)
function_call_result_674 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), function_671, *[int_672], **kwargs_673)

# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'y', function_call_result_674)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

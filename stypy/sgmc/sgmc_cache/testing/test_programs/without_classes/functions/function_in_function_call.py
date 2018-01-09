
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: def function(x):
4:     def another_function(z):
5:         return str(z)
6: 
7:     return another_function(x)
8: 
9: ret = function(3)
10: 

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
    module_type_store = module_type_store.open_function_context('function', 3, 0, False)
    
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


    @norecursion
    def another_function(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'another_function'
        module_type_store = module_type_store.open_function_context('another_function', 4, 4, False)
        
        # Passed parameters checking function
        another_function.stypy_localization = localization
        another_function.stypy_type_of_self = None
        another_function.stypy_type_store = module_type_store
        another_function.stypy_function_name = 'another_function'
        another_function.stypy_param_names_list = ['z']
        another_function.stypy_varargs_param_name = None
        another_function.stypy_kwargs_param_name = None
        another_function.stypy_call_defaults = defaults
        another_function.stypy_call_varargs = varargs
        another_function.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'another_function', ['z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'another_function', localization, ['z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'another_function(...)' code ##################

        
        # Call to str(...): (line 5)
        # Processing the call arguments (line 5)
        # Getting the type of 'z' (line 5)
        z_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 19), 'z', False)
        # Processing the call keyword arguments (line 5)
        kwargs_1006 = {}
        # Getting the type of 'str' (line 5)
        str_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', False)
        # Calling str(args, kwargs) (line 5)
        str_call_result_1007 = invoke(stypy.reporting.localization.Localization(__file__, 5, 15), str_1004, *[z_1005], **kwargs_1006)
        
        # Assigning a type to the variable 'stypy_return_type' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', str_call_result_1007)
        
        # ################# End of 'another_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'another_function' in the type store
        # Getting the type of 'stypy_return_type' (line 4)
        stypy_return_type_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1008)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'another_function'
        return stypy_return_type_1008

    # Assigning a type to the variable 'another_function' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'another_function', another_function)
    
    # Call to another_function(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'x' (line 7)
    x_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 28), 'x', False)
    # Processing the call keyword arguments (line 7)
    kwargs_1011 = {}
    # Getting the type of 'another_function' (line 7)
    another_function_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'another_function', False)
    # Calling another_function(args, kwargs) (line 7)
    another_function_call_result_1012 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), another_function_1009, *[x_1010], **kwargs_1011)
    
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', another_function_call_result_1012)
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1013)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_1013

# Assigning a type to the variable 'function' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'function', function)

# Assigning a Call to a Name (line 9):

# Call to function(...): (line 9)
# Processing the call arguments (line 9)
int_1015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
# Processing the call keyword arguments (line 9)
kwargs_1016 = {}
# Getting the type of 'function' (line 9)
function_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 6), 'function', False)
# Calling function(args, kwargs) (line 9)
function_call_result_1017 = invoke(stypy.reporting.localization.Localization(__file__, 9, 6), function_1014, *[int_1015], **kwargs_1016)

# Assigning a type to the variable 'ret' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'ret', function_call_result_1017)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

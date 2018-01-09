
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Returning a parameter from a function"
4: 
5: if __name__ == '__main__':
6:     import math
7: 
8: 
9:     def function(x):
10:         return x
11: 
12: 
13:     y = function(3)
14:     print y * 2
15:     # Type error
16:     print y[12]
17:     # Type error
18:     print math.pow(function("a"), 3)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Returning a parameter from a function')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'import math' statement (line 6)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'math', math, module_type_store)
    

    @norecursion
    def function(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'function'
        module_type_store = module_type_store.open_function_context('function', 9, 4, False)
        
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

        # Getting the type of 'x' (line 10)
        x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', x_2)
        
        # ################# End of 'function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'function' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'function'
        return stypy_return_type_3

    # Assigning a type to the variable 'function' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'function', function)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to function(...): (line 13)
    # Processing the call arguments (line 13)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_6 = {}
    # Getting the type of 'function' (line 13)
    function_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'function', False)
    # Calling function(args, kwargs) (line 13)
    function_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), function_4, *[int_5], **kwargs_6)
    
    # Assigning a type to the variable 'y' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'y', function_call_result_7)
    # Getting the type of 'y' (line 14)
    y_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'y')
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
    # Applying the binary operator '*' (line 14)
    result_mul_10 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 10), '*', y_8, int_9)
    
    
    # Obtaining the type of the subscript
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
    # Getting the type of 'y' (line 16)
    y_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'y')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 10), y_12, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), getitem___13, int_11)
    
    
    # Call to pow(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to function(...): (line 18)
    # Processing the call arguments (line 18)
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'str', 'a')
    # Processing the call keyword arguments (line 18)
    kwargs_19 = {}
    # Getting the type of 'function' (line 18)
    function_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'function', False)
    # Calling function(args, kwargs) (line 18)
    function_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), function_17, *[str_18], **kwargs_19)
    
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_22 = {}
    # Getting the type of 'math' (line 18)
    math_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'math', False)
    # Obtaining the member 'pow' of a type (line 18)
    pow_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 10), math_15, 'pow')
    # Calling pow(args, kwargs) (line 18)
    pow_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), pow_16, *[function_call_result_20, int_21], **kwargs_22)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

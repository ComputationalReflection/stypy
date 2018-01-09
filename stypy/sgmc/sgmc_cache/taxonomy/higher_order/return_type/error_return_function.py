
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Infer the return type when returning a function"
3: 
4: if __name__ == '__main__':
5:     def ret_func():
6:         return lambda x, y: x + y
7: 
8: 
9:     f = ret_func()
10: 
11:     # Type error
12:     print f(3)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Infer the return type when returning a function')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def ret_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ret_func'
        module_type_store = module_type_store.open_function_context('ret_func', 5, 4, False)
        
        # Passed parameters checking function
        ret_func.stypy_localization = localization
        ret_func.stypy_type_of_self = None
        ret_func.stypy_type_store = module_type_store
        ret_func.stypy_function_name = 'ret_func'
        ret_func.stypy_param_names_list = []
        ret_func.stypy_varargs_param_name = None
        ret_func.stypy_kwargs_param_name = None
        ret_func.stypy_call_defaults = defaults
        ret_func.stypy_call_varargs = varargs
        ret_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'ret_func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ret_func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ret_func(...)' code ##################


        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 6, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 6)
            x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 28), 'x')
            # Getting the type of 'y' (line 6)
            y_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 32), 'y')
            # Applying the binary operator '+' (line 6)
            result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 28), '+', x_2, y_3)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'stypy_return_type', result_add_4)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_5

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 6)
        _stypy_temp_lambda_1_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), '_stypy_temp_lambda_1')
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', _stypy_temp_lambda_1_6)
        
        # ################# End of 'ret_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ret_func' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ret_func'
        return stypy_return_type_7

    # Assigning a type to the variable 'ret_func' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'ret_func', ret_func)
    
    # Assigning a Call to a Name (line 9):
    
    # Call to ret_func(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_9 = {}
    # Getting the type of 'ret_func' (line 9)
    ret_func_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'ret_func', False)
    # Calling ret_func(args, kwargs) (line 9)
    ret_func_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), ret_func_8, *[], **kwargs_9)
    
    # Assigning a type to the variable 'f' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'f', ret_func_call_result_10)
    
    # Call to f(...): (line 12)
    # Processing the call arguments (line 12)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_13 = {}
    # Getting the type of 'f' (line 12)
    f_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'f', False)
    # Calling f(args, kwargs) (line 12)
    f_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), f_11, *[int_12], **kwargs_13)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

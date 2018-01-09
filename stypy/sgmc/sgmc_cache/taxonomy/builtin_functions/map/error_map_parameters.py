
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "map method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, IterableObject) -> <type 'list'>
7:     # (Has__call__, IterableObject, IterableObject) -> <type 'list'>
8:     # (Has__call__, IterableObject, IterableObject, IterableObject) -> <type 'list'>
9:     # (Has__call__, Str) -> <type 'list'>
10:     # (Has__call__, Str, IterableObject) -> <type 'list'>
11:     # (Has__call__, IterableObject, Str) -> <type 'list'>
12:     # (Has__call__, Str, Str) -> <type 'list'>
13:     # (Has__call__, Str, IterableObject, IterableObject) -> <type 'list'>
14:     # (Has__call__, IterableObject, Str, IterableObject) -> <type 'list'>
15:     # (Has__call__, IterableObject, IterableObject, Str) -> <type 'list'>
16:     # (Has__call__, Str, Str, IterableObject) -> <type 'list'>
17:     # (Has__call__, IterableObject, Str, Str) -> <type 'list'>
18:     # (Has__call__, Str, IterableObject, Str) -> <type 'list'>
19:     # (Has__call__, Str, Str, Str) -> <type 'list'>
20:     # (Has__call__, IterableObject, VarArgs) -> <type 'list'>
21: 
22: 
23:     # Call the builtin with incorrect number of parameters
24:     l = range(5)
25: 
26: 
27:     def f(x):
28:         return str(x)
29: 
30: 
31:     # Type error
32:     other_l4 = map(lambda x, y: f(x), l)
33: 
34:     # Type error
35:     other_l4 = map(lambda x, y: f(x))
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'map method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to range(...): (line 24)
    # Processing the call arguments (line 24)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_4 = {}
    # Getting the type of 'range' (line 24)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'range', False)
    # Calling range(args, kwargs) (line 24)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'l' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'l', range_call_result_5)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 27, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to str(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'x' (line 28)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'x', False)
        # Processing the call keyword arguments (line 28)
        kwargs_8 = {}
        # Getting the type of 'str' (line 28)
        str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'str', False)
        # Calling str(args, kwargs) (line 28)
        str_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), str_6, *[x_7], **kwargs_8)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', str_call_result_9)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_10

    # Assigning a type to the variable 'f' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'f', f)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to map(...): (line 32)
    # Processing the call arguments (line 32)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 32, 19, True)
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

        
        # Call to f(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'x' (line 32)
        x_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'x', False)
        # Processing the call keyword arguments (line 32)
        kwargs_14 = {}
        # Getting the type of 'f' (line 32)
        f_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'f', False)
        # Calling f(args, kwargs) (line 32)
        f_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 32, 32), f_12, *[x_13], **kwargs_14)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'stypy_return_type', f_call_result_15)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_16

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 32)
    _stypy_temp_lambda_1_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), '_stypy_temp_lambda_1')
    # Getting the type of 'l' (line 32)
    l_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'l', False)
    # Processing the call keyword arguments (line 32)
    kwargs_19 = {}
    # Getting the type of 'map' (line 32)
    map_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'map', False)
    # Calling map(args, kwargs) (line 32)
    map_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), map_11, *[_stypy_temp_lambda_1_17, l_18], **kwargs_19)
    
    # Assigning a type to the variable 'other_l4' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'other_l4', map_call_result_20)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to map(...): (line 35)
    # Processing the call arguments (line 35)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 35, 19, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to f(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'x' (line 35)
        x_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'x', False)
        # Processing the call keyword arguments (line 35)
        kwargs_24 = {}
        # Getting the type of 'f' (line 35)
        f_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'f', False)
        # Calling f(args, kwargs) (line 35)
        f_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 35, 32), f_22, *[x_23], **kwargs_24)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'stypy_return_type', f_call_result_25)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_26

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 35)
    _stypy_temp_lambda_2_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), '_stypy_temp_lambda_2')
    # Processing the call keyword arguments (line 35)
    kwargs_28 = {}
    # Getting the type of 'map' (line 35)
    map_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'map', False)
    # Calling map(args, kwargs) (line 35)
    map_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), map_21, *[_stypy_temp_lambda_2_27], **kwargs_28)
    
    # Assigning a type to the variable 'other_l4' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'other_l4', map_call_result_29)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

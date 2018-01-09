
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sorted builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'list'>
7:     # (IterableObject, Has__call__) -> <type 'list'>
8:     # (IterableObject, Has__call__, Has__call__) -> <type 'list'>
9:     # (IterableObject, Has__call__, Has__call__, <type bool>) -> <type 'list'>
10:     # (Str) -> <type 'list'>
11:     # (Str, Has__call__) -> <type 'list'>
12:     # (Str, Has__call__, Has__call__) -> <type 'list'>
13:     # (Str, Has__call__, Has__call__, <type bool>) -> <type 'list'>
14: 
15: 
16:     # Call the builtin with correct parameters
17:     ret = sorted([1, 2], lambda x, y: str(x) == str(y))
18:     ret = sorted("str", lambda x, y: str(x) == str(y))
19: 
20:     # Call the builtin with incorrect types of parameters
21:     # Type error
22:     ret = sorted([1, 2], lambda x: str(x))
23:     # Type error
24:     ret = sorted(4)
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sorted builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 17):
    
    # Call to sorted(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), list_3, int_4)
    # Adding element type (line 17)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), list_3, int_5)
    

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 17, 25, True)
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

        
        
        # Call to str(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'x' (line 17)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 42), 'x', False)
        # Processing the call keyword arguments (line 17)
        kwargs_8 = {}
        # Getting the type of 'str' (line 17)
        str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 38), 'str', False)
        # Calling str(args, kwargs) (line 17)
        str_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 17, 38), str_6, *[x_7], **kwargs_8)
        
        
        # Call to str(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'y' (line 17)
        y_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 52), 'y', False)
        # Processing the call keyword arguments (line 17)
        kwargs_12 = {}
        # Getting the type of 'str' (line 17)
        str_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 48), 'str', False)
        # Calling str(args, kwargs) (line 17)
        str_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 17, 48), str_10, *[y_11], **kwargs_12)
        
        # Applying the binary operator '==' (line 17)
        result_eq_14 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 38), '==', str_call_result_9, str_call_result_13)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'stypy_return_type', result_eq_14)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_15

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 17)
    _stypy_temp_lambda_1_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), '_stypy_temp_lambda_1')
    # Processing the call keyword arguments (line 17)
    kwargs_17 = {}
    # Getting the type of 'sorted' (line 17)
    sorted_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 17)
    sorted_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), sorted_2, *[list_3, _stypy_temp_lambda_1_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', sorted_call_result_18)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to sorted(...): (line 18)
    # Processing the call arguments (line 18)
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'str', 'str')

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 18, 24, True)
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

        
        
        # Call to str(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'x' (line 18)
        x_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 41), 'x', False)
        # Processing the call keyword arguments (line 18)
        kwargs_23 = {}
        # Getting the type of 'str' (line 18)
        str_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 37), 'str', False)
        # Calling str(args, kwargs) (line 18)
        str_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 18, 37), str_21, *[x_22], **kwargs_23)
        
        
        # Call to str(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'y' (line 18)
        y_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 51), 'y', False)
        # Processing the call keyword arguments (line 18)
        kwargs_27 = {}
        # Getting the type of 'str' (line 18)
        str_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', False)
        # Calling str(args, kwargs) (line 18)
        str_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 18, 47), str_25, *[y_26], **kwargs_27)
        
        # Applying the binary operator '==' (line 18)
        result_eq_29 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 37), '==', str_call_result_24, str_call_result_28)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'stypy_return_type', result_eq_29)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_30

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 18)
    _stypy_temp_lambda_2_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), '_stypy_temp_lambda_2')
    # Processing the call keyword arguments (line 18)
    kwargs_32 = {}
    # Getting the type of 'sorted' (line 18)
    sorted_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 18)
    sorted_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), sorted_19, *[str_20, _stypy_temp_lambda_2_31], **kwargs_32)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', sorted_call_result_33)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to sorted(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), list_35, int_36)
    # Adding element type (line 22)
    int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), list_35, int_37)
    

    @norecursion
    def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_3'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 22, 25, True)
        # Passed parameters checking function
        _stypy_temp_lambda_3.stypy_localization = localization
        _stypy_temp_lambda_3.stypy_type_of_self = None
        _stypy_temp_lambda_3.stypy_type_store = module_type_store
        _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
        _stypy_temp_lambda_3.stypy_param_names_list = ['x']
        _stypy_temp_lambda_3.stypy_varargs_param_name = None
        _stypy_temp_lambda_3.stypy_kwargs_param_name = None
        _stypy_temp_lambda_3.stypy_call_defaults = defaults
        _stypy_temp_lambda_3.stypy_call_varargs = varargs
        _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_3', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to str(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'x' (line 22)
        x_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'x', False)
        # Processing the call keyword arguments (line 22)
        kwargs_40 = {}
        # Getting the type of 'str' (line 22)
        str_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'str', False)
        # Calling str(args, kwargs) (line 22)
        str_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 22, 35), str_38, *[x_39], **kwargs_40)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'stypy_return_type', str_call_result_41)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_3' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_3'
        return stypy_return_type_42

    # Assigning a type to the variable '_stypy_temp_lambda_3' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
    # Getting the type of '_stypy_temp_lambda_3' (line 22)
    _stypy_temp_lambda_3_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), '_stypy_temp_lambda_3')
    # Processing the call keyword arguments (line 22)
    kwargs_44 = {}
    # Getting the type of 'sorted' (line 22)
    sorted_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 22)
    sorted_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), sorted_34, *[list_35, _stypy_temp_lambda_3_43], **kwargs_44)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', sorted_call_result_45)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to sorted(...): (line 24)
    # Processing the call arguments (line 24)
    int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_48 = {}
    # Getting the type of 'sorted' (line 24)
    sorted_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 24)
    sorted_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), sorted_46, *[int_47], **kwargs_48)
    
    # Assigning a type to the variable 'ret' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'ret', sorted_call_result_49)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

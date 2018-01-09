
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "map builtin is invoked, but incorrect parameter types are passed"
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
23:     # Call the builtin with correct parameters
24: 
25:     l = [1, 2, 3, 4]
26: 
27:     other_l = map(lambda x: str(x), l)
28: 
29:     l2 = [False, 1, "string"]
30:     other_l2 = map(lambda x: str(x), l)
31: 
32:     # Call the builtin with incorrect types of parameters
33:     # Type error
34:     ret = map(3, [1, 2])
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'map builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a List to a Name (line 25):
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_2, int_3)
    # Adding element type (line 25)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_2, int_4)
    # Adding element type (line 25)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_2, int_5)
    # Adding element type (line 25)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_2, int_6)
    
    # Assigning a type to the variable 'l' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'l', list_2)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to map(...): (line 27)
    # Processing the call arguments (line 27)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 27, 18, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to str(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'x' (line 27)
        x_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'x', False)
        # Processing the call keyword arguments (line 27)
        kwargs_10 = {}
        # Getting the type of 'str' (line 27)
        str_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'str', False)
        # Calling str(args, kwargs) (line 27)
        str_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 27, 28), str_8, *[x_9], **kwargs_10)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'stypy_return_type', str_call_result_11)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_12

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 27)
    _stypy_temp_lambda_1_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), '_stypy_temp_lambda_1')
    # Getting the type of 'l' (line 27)
    l_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 36), 'l', False)
    # Processing the call keyword arguments (line 27)
    kwargs_15 = {}
    # Getting the type of 'map' (line 27)
    map_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'map', False)
    # Calling map(args, kwargs) (line 27)
    map_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 27, 14), map_7, *[_stypy_temp_lambda_1_13, l_14], **kwargs_15)
    
    # Assigning a type to the variable 'other_l' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'other_l', map_call_result_16)
    
    # Assigning a List to a Name (line 29):
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    # Getting the type of 'False' (line 29)
    False_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), list_17, False_18)
    # Adding element type (line 29)
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), list_17, int_19)
    # Adding element type (line 29)
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), list_17, str_20)
    
    # Assigning a type to the variable 'l2' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'l2', list_17)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to map(...): (line 30)
    # Processing the call arguments (line 30)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 30, 19, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['x']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to str(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'x' (line 30)
        x_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'x', False)
        # Processing the call keyword arguments (line 30)
        kwargs_24 = {}
        # Getting the type of 'str' (line 30)
        str_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'str', False)
        # Calling str(args, kwargs) (line 30)
        str_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 30, 29), str_22, *[x_23], **kwargs_24)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'stypy_return_type', str_call_result_25)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_26

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 30)
    _stypy_temp_lambda_2_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), '_stypy_temp_lambda_2')
    # Getting the type of 'l' (line 30)
    l_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), 'l', False)
    # Processing the call keyword arguments (line 30)
    kwargs_29 = {}
    # Getting the type of 'map' (line 30)
    map_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'map', False)
    # Calling map(args, kwargs) (line 30)
    map_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), map_21, *[_stypy_temp_lambda_2_27, l_28], **kwargs_29)
    
    # Assigning a type to the variable 'other_l2' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'other_l2', map_call_result_30)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to map(...): (line 34)
    # Processing the call arguments (line 34)
    int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_33, int_34)
    # Adding element type (line 34)
    int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_33, int_35)
    
    # Processing the call keyword arguments (line 34)
    kwargs_36 = {}
    # Getting the type of 'map' (line 34)
    map_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'map', False)
    # Calling map(args, kwargs) (line 34)
    map_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), map_31, *[int_32, list_33], **kwargs_36)
    
    # Assigning a type to the variable 'ret' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'ret', map_call_result_37)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reduce method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, Str) -> DynamicType
7:     # (Has__call__, IterableObject) -> DynamicType
8:     # (Has__call__, Str, AnyType) -> DynamicType
9:     # (Has__call__, IterableObject, AnyType) -> DynamicType
10: 
11: 
12:     # Call the builtin with incorrect number of parameters
13:     l = [1, 2, 3, 4]
14:     l3 = ["False", "1", "string"]
15: 
16:     # Type error
17:     other_l4 = reduce(lambda x, y, z: x + y, l, 0)
18:     # Type error
19:     other_l4 = reduce(lambda x, y, z: x + y, l, 0, 0)
20:     # Type error
21:     other_l4 = reduce(lambda x, y, z: x + y)
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reduce method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a List to a Name (line 13):
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), list_2, int_3)
    # Adding element type (line 13)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), list_2, int_4)
    # Adding element type (line 13)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), list_2, int_5)
    # Adding element type (line 13)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), list_2, int_6)
    
    # Assigning a type to the variable 'l' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'l', list_2)
    
    # Assigning a List to a Name (line 14):
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'str', 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 9), list_7, str_8)
    # Adding element type (line 14)
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 9), list_7, str_9)
    # Adding element type (line 14)
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 9), list_7, str_10)
    
    # Assigning a type to the variable 'l3' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'l3', list_7)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to reduce(...): (line 17)
    # Processing the call arguments (line 17)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 17, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x', 'y', 'z']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x', 'y', 'z'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 17)
        x_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 38), 'x', False)
        # Getting the type of 'y' (line 17)
        y_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 42), 'y', False)
        # Applying the binary operator '+' (line 17)
        result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 38), '+', x_12, y_13)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'stypy_return_type', result_add_14)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_15

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 17)
    _stypy_temp_lambda_1_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), '_stypy_temp_lambda_1')
    # Getting the type of 'l' (line 17)
    l_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 45), 'l', False)
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 48), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_19 = {}
    # Getting the type of 'reduce' (line 17)
    reduce_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'reduce', False)
    # Calling reduce(args, kwargs) (line 17)
    reduce_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), reduce_11, *[_stypy_temp_lambda_1_16, l_17, int_18], **kwargs_19)
    
    # Assigning a type to the variable 'other_l4' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'other_l4', reduce_call_result_20)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to reduce(...): (line 19)
    # Processing the call arguments (line 19)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 19, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['x', 'y', 'z']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['x', 'y', 'z'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 19)
        x_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'x', False)
        # Getting the type of 'y' (line 19)
        y_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'y', False)
        # Applying the binary operator '+' (line 19)
        result_add_24 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 38), '+', x_22, y_23)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'stypy_return_type', result_add_24)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_25

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 19)
    _stypy_temp_lambda_2_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), '_stypy_temp_lambda_2')
    # Getting the type of 'l' (line 19)
    l_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 45), 'l', False)
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 48), 'int')
    int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 51), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_30 = {}
    # Getting the type of 'reduce' (line 19)
    reduce_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'reduce', False)
    # Calling reduce(args, kwargs) (line 19)
    reduce_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), reduce_21, *[_stypy_temp_lambda_2_26, l_27, int_28, int_29], **kwargs_30)
    
    # Assigning a type to the variable 'other_l4' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'other_l4', reduce_call_result_31)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to reduce(...): (line 21)
    # Processing the call arguments (line 21)

    @norecursion
    def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_3'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 21, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_3.stypy_localization = localization
        _stypy_temp_lambda_3.stypy_type_of_self = None
        _stypy_temp_lambda_3.stypy_type_store = module_type_store
        _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
        _stypy_temp_lambda_3.stypy_param_names_list = ['x', 'y', 'z']
        _stypy_temp_lambda_3.stypy_varargs_param_name = None
        _stypy_temp_lambda_3.stypy_kwargs_param_name = None
        _stypy_temp_lambda_3.stypy_call_defaults = defaults
        _stypy_temp_lambda_3.stypy_call_varargs = varargs
        _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_3', ['x', 'y', 'z'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 21)
        x_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 38), 'x', False)
        # Getting the type of 'y' (line 21)
        y_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 42), 'y', False)
        # Applying the binary operator '+' (line 21)
        result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 38), '+', x_33, y_34)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'stypy_return_type', result_add_35)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_3' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_3'
        return stypy_return_type_36

    # Assigning a type to the variable '_stypy_temp_lambda_3' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
    # Getting the type of '_stypy_temp_lambda_3' (line 21)
    _stypy_temp_lambda_3_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), '_stypy_temp_lambda_3')
    # Processing the call keyword arguments (line 21)
    kwargs_38 = {}
    # Getting the type of 'reduce' (line 21)
    reduce_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'reduce', False)
    # Calling reduce(args, kwargs) (line 21)
    reduce_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), reduce_32, *[_stypy_temp_lambda_3_37], **kwargs_38)
    
    # Assigning a type to the variable 'other_l4' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'other_l4', reduce_call_result_39)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reduce builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, Str) -> DynamicType
7:     # (Has__call__, IterableObject) -> DynamicType
8:     # (Has__call__, Str, AnyType) -> DynamicType
9:     # (Has__call__, IterableObject, AnyType) -> DynamicType
10: 
11: 
12:     # Call the builtin
13:     l = [1, 2, 3, 4]
14:     l3 = ["False", "1", "string"]
15: 
16:     other_l3 = reduce(lambda x, y: x + y, l, 0)
17:     # Type error
18:     r3 = other_l3[5]
19:     # Type error
20:     r4 = other_l3.capitalize()
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reduce builtin is invoked and its return type is used to call an non existing method')
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
    
    # Assigning a Call to a Name (line 16):
    
    # Call to reduce(...): (line 16)
    # Processing the call arguments (line 16)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 16, 22, True)
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

        # Getting the type of 'x' (line 16)
        x_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 35), 'x', False)
        # Getting the type of 'y' (line 16)
        y_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 39), 'y', False)
        # Applying the binary operator '+' (line 16)
        result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 35), '+', x_12, y_13)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'stypy_return_type', result_add_14)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_15

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 16)
    _stypy_temp_lambda_1_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), '_stypy_temp_lambda_1')
    # Getting the type of 'l' (line 16)
    l_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 42), 'l', False)
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 45), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_19 = {}
    # Getting the type of 'reduce' (line 16)
    reduce_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'reduce', False)
    # Calling reduce(args, kwargs) (line 16)
    reduce_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), reduce_11, *[_stypy_temp_lambda_1_16, l_17, int_18], **kwargs_19)
    
    # Assigning a type to the variable 'other_l3' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'other_l3', reduce_call_result_20)
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
    # Getting the type of 'other_l3' (line 18)
    other_l3_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'other_l3')
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 9), other_l3_22, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 18, 9), getitem___23, int_21)
    
    # Assigning a type to the variable 'r3' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r3', subscript_call_result_24)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to capitalize(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_27 = {}
    # Getting the type of 'other_l3' (line 20)
    other_l3_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'other_l3', False)
    # Obtaining the member 'capitalize' of a type (line 20)
    capitalize_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), other_l3_25, 'capitalize')
    # Calling capitalize(args, kwargs) (line 20)
    capitalize_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), capitalize_26, *[], **kwargs_27)
    
    # Assigning a type to the variable 'r4' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'r4', capitalize_call_result_28)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

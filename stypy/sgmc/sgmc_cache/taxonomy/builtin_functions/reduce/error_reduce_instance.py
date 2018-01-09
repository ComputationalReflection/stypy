
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reduce builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, Str) -> DynamicType
7:     # (Has__call__, IterableObject) -> DynamicType
8:     # (Has__call__, Str, AnyType) -> DynamicType
9:     # (Has__call__, IterableObject, AnyType) -> DynamicType
10:     import types
11: 
12:     # Type error
13:     other_l3 = reduce(lambda x, y: x + y, list, 0)
14: 
15:     # Type error
16:     other_l = reduce(types.FunctionType, list, 0)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reduce builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))
    
    # 'import types' statement (line 10)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'types', types, module_type_store)
    
    
    # Assigning a Call to a Name (line 13):
    
    # Call to reduce(...): (line 13)
    # Processing the call arguments (line 13)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 13, 22, True)
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

        # Getting the type of 'x' (line 13)
        x_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'x', False)
        # Getting the type of 'y' (line 13)
        y_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 39), 'y', False)
        # Applying the binary operator '+' (line 13)
        result_add_5 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 35), '+', x_3, y_4)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'stypy_return_type', result_add_5)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_6

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 13)
    _stypy_temp_lambda_1_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), '_stypy_temp_lambda_1')
    # Getting the type of 'list' (line 13)
    list_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 42), 'list', False)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 48), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_10 = {}
    # Getting the type of 'reduce' (line 13)
    reduce_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'reduce', False)
    # Calling reduce(args, kwargs) (line 13)
    reduce_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 13, 15), reduce_2, *[_stypy_temp_lambda_1_7, list_8, int_9], **kwargs_10)
    
    # Assigning a type to the variable 'other_l3' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'other_l3', reduce_call_result_11)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to reduce(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'types' (line 16)
    types_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 21), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 16)
    FunctionType_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 21), types_13, 'FunctionType')
    # Getting the type of 'list' (line 16)
    list_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 41), 'list', False)
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 47), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_17 = {}
    # Getting the type of 'reduce' (line 16)
    reduce_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'reduce', False)
    # Calling reduce(args, kwargs) (line 16)
    reduce_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), reduce_12, *[FunctionType_14, list_15, int_16], **kwargs_17)
    
    # Assigning a type to the variable 'other_l' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'other_l', reduce_call_result_18)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

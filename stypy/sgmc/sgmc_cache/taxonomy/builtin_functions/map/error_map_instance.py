
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "map builtin is invoked, but a class is used instead of an instance"
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
21:     import types
22: 
23:     # Type error
24:     other_l = map(lambda x: str(x), list)
25:     # Type error
26:     other_l = map(types.FunctionType, list)
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'map builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))
    
    # 'import types' statement (line 21)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'types', types, module_type_store)
    
    
    # Assigning a Call to a Name (line 24):
    
    # Call to map(...): (line 24)
    # Processing the call arguments (line 24)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 24, 18, True)
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

        
        # Call to str(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'x' (line 24)
        x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 32), 'x', False)
        # Processing the call keyword arguments (line 24)
        kwargs_5 = {}
        # Getting the type of 'str' (line 24)
        str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 28), 'str', False)
        # Calling str(args, kwargs) (line 24)
        str_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 24, 28), str_3, *[x_4], **kwargs_5)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'stypy_return_type', str_call_result_6)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_7

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 24)
    _stypy_temp_lambda_1_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), '_stypy_temp_lambda_1')
    # Getting the type of 'list' (line 24)
    list_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 36), 'list', False)
    # Processing the call keyword arguments (line 24)
    kwargs_10 = {}
    # Getting the type of 'map' (line 24)
    map_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'map', False)
    # Calling map(args, kwargs) (line 24)
    map_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), map_2, *[_stypy_temp_lambda_1_8, list_9], **kwargs_10)
    
    # Assigning a type to the variable 'other_l' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'other_l', map_call_result_11)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to map(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'types' (line 26)
    types_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 26)
    FunctionType_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 18), types_13, 'FunctionType')
    # Getting the type of 'list' (line 26)
    list_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'list', False)
    # Processing the call keyword arguments (line 26)
    kwargs_16 = {}
    # Getting the type of 'map' (line 26)
    map_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'map', False)
    # Calling map(args, kwargs) (line 26)
    map_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), map_12, *[FunctionType_14, list_15], **kwargs_16)
    
    # Assigning a type to the variable 'other_l' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'other_l', map_call_result_17)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

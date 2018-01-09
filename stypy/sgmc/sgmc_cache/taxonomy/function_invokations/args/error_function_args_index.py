
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Wrong indexing of args parameter"
4: 
5: if __name__ == '__main__':
6:     def functionargs2(*args):
7:         # Type error
8:         return args["str"]
9: 
10: 
11:     y = functionargs2("hi")
12:     y = y.thisdonotexist()
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Wrong indexing of args parameter')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def functionargs2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionargs2'
        module_type_store = module_type_store.open_function_context('functionargs2', 6, 4, False)
        
        # Passed parameters checking function
        functionargs2.stypy_localization = localization
        functionargs2.stypy_type_of_self = None
        functionargs2.stypy_type_store = module_type_store
        functionargs2.stypy_function_name = 'functionargs2'
        functionargs2.stypy_param_names_list = []
        functionargs2.stypy_varargs_param_name = 'args'
        functionargs2.stypy_kwargs_param_name = None
        functionargs2.stypy_call_defaults = defaults
        functionargs2.stypy_call_varargs = varargs
        functionargs2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionargs2', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionargs2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionargs2(...)' code ##################

        
        # Obtaining the type of the subscript
        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'str', 'str')
        # Getting the type of 'args' (line 8)
        args_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'args')
        # Obtaining the member '__getitem__' of a type (line 8)
        getitem___4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 15), args_3, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 8)
        subscript_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 8, 15), getitem___4, str_2)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', subscript_call_result_5)
        
        # ################# End of 'functionargs2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionargs2' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionargs2'
        return stypy_return_type_6

    # Assigning a type to the variable 'functionargs2' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'functionargs2', functionargs2)
    
    # Assigning a Call to a Name (line 11):
    
    # Call to functionargs2(...): (line 11)
    # Processing the call arguments (line 11)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', 'hi')
    # Processing the call keyword arguments (line 11)
    kwargs_9 = {}
    # Getting the type of 'functionargs2' (line 11)
    functionargs2_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'functionargs2', False)
    # Calling functionargs2(args, kwargs) (line 11)
    functionargs2_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), functionargs2_7, *[str_8], **kwargs_9)
    
    # Assigning a type to the variable 'y' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'y', functionargs2_call_result_10)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to thisdonotexist(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_13 = {}
    # Getting the type of 'y' (line 12)
    y_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'y', False)
    # Obtaining the member 'thisdonotexist' of a type (line 12)
    thisdonotexist_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), y_11, 'thisdonotexist')
    # Calling thisdonotexist(args, kwargs) (line 12)
    thisdonotexist_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), thisdonotexist_12, *[], **kwargs_13)
    
    # Assigning a type to the variable 'y' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'y', thisdonotexist_call_result_14)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

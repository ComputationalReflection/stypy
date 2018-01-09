
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Incorrect index for a kwargs argument"
4: 
5: if __name__ == '__main__':
6:     def functionkw(**kwargs):
7:         # Type error
8:         return kwargs[0]
9: 
10: 
11:     y = functionkw(val="hi")
12:     y = y.thisdonotexist()
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Incorrect index for a kwargs argument')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def functionkw(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionkw'
        module_type_store = module_type_store.open_function_context('functionkw', 6, 4, False)
        
        # Passed parameters checking function
        functionkw.stypy_localization = localization
        functionkw.stypy_type_of_self = None
        functionkw.stypy_type_store = module_type_store
        functionkw.stypy_function_name = 'functionkw'
        functionkw.stypy_param_names_list = []
        functionkw.stypy_varargs_param_name = None
        functionkw.stypy_kwargs_param_name = 'kwargs'
        functionkw.stypy_call_defaults = defaults
        functionkw.stypy_call_varargs = varargs
        functionkw.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionkw', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionkw', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionkw(...)' code ##################

        
        # Obtaining the type of the subscript
        int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 22), 'int')
        # Getting the type of 'kwargs' (line 8)
        kwargs_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'kwargs')
        # Obtaining the member '__getitem__' of a type (line 8)
        getitem___4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 15), kwargs_3, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 8)
        subscript_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 8, 15), getitem___4, int_2)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', subscript_call_result_5)
        
        # ################# End of 'functionkw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionkw' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionkw'
        return stypy_return_type_6

    # Assigning a type to the variable 'functionkw' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'functionkw', functionkw)
    
    # Assigning a Call to a Name (line 11):
    
    # Call to functionkw(...): (line 11)
    # Processing the call keyword arguments (line 11)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'str', 'hi')
    keyword_9 = str_8
    kwargs_10 = {'val': keyword_9}
    # Getting the type of 'functionkw' (line 11)
    functionkw_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'functionkw', False)
    # Calling functionkw(args, kwargs) (line 11)
    functionkw_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), functionkw_7, *[], **kwargs_10)
    
    # Assigning a type to the variable 'y' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'y', functionkw_call_result_11)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to thisdonotexist(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_14 = {}
    # Getting the type of 'y' (line 12)
    y_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'y', False)
    # Obtaining the member 'thisdonotexist' of a type (line 12)
    thisdonotexist_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), y_12, 'thisdonotexist')
    # Calling thisdonotexist(args, kwargs) (line 12)
    thisdonotexist_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), thisdonotexist_13, *[], **kwargs_14)
    
    # Assigning a type to the variable 'y' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'y', thisdonotexist_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

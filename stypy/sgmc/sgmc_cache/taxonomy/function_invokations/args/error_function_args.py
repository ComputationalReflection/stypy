
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Wrong handling of args parameter after call"
4: 
5: if __name__ == '__main__':
6:     def functionargs(*args):
7:         return args[0]
8: 
9: 
10:     y = functionargs("hi")
11:     # Type error
12:     y = y.thisdonotexist()
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Wrong handling of args parameter after call')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def functionargs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionargs'
        module_type_store = module_type_store.open_function_context('functionargs', 6, 4, False)
        
        # Passed parameters checking function
        functionargs.stypy_localization = localization
        functionargs.stypy_type_of_self = None
        functionargs.stypy_type_store = module_type_store
        functionargs.stypy_function_name = 'functionargs'
        functionargs.stypy_param_names_list = []
        functionargs.stypy_varargs_param_name = 'args'
        functionargs.stypy_kwargs_param_name = None
        functionargs.stypy_call_defaults = defaults
        functionargs.stypy_call_varargs = varargs
        functionargs.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionargs', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionargs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionargs(...)' code ##################

        
        # Obtaining the type of the subscript
        int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
        # Getting the type of 'args' (line 7)
        args_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'args')
        # Obtaining the member '__getitem__' of a type (line 7)
        getitem___4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 15), args_3, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 7)
        subscript_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 7, 15), getitem___4, int_2)
        
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', subscript_call_result_5)
        
        # ################# End of 'functionargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionargs' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionargs'
        return stypy_return_type_6

    # Assigning a type to the variable 'functionargs' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'functionargs', functionargs)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to functionargs(...): (line 10)
    # Processing the call arguments (line 10)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', 'hi')
    # Processing the call keyword arguments (line 10)
    kwargs_9 = {}
    # Getting the type of 'functionargs' (line 10)
    functionargs_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'functionargs', False)
    # Calling functionargs(args, kwargs) (line 10)
    functionargs_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), functionargs_7, *[str_8], **kwargs_9)
    
    # Assigning a type to the variable 'y' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'y', functionargs_call_result_10)
    
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

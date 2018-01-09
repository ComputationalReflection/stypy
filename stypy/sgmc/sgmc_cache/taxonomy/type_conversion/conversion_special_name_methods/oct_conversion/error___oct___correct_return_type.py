
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: 
4: __doc__ = "Special __oct__ method is defined, but with incorrect arity"
5: 
6: if __name__ == '__main__':
7:     class DefinesMethod:
8:         def __oct__(self, param):
9:             return str(param)
10: 
11: 
12:     # Type error #
13:     print oct(DefinesMethod())
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 4):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Special __oct__ method is defined, but with incorrect arity')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DefinesMethod' class

    class DefinesMethod:

        @norecursion
        def __oct__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__oct__'
            module_type_store = module_type_store.open_function_context('__oct__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_localization', localization)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_function_name', 'DefinesMethod.__oct__')
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_param_names_list', ['param'])
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesMethod.__oct__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__oct__', ['param'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__oct__', localization, ['param'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__oct__(...)' code ##################

            
            # Call to str(...): (line 9)
            # Processing the call arguments (line 9)
            # Getting the type of 'param' (line 9)
            param_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'param', False)
            # Processing the call keyword arguments (line 9)
            kwargs_4 = {}
            # Getting the type of 'str' (line 9)
            str_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'str', False)
            # Calling str(args, kwargs) (line 9)
            str_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 9, 19), str_2, *[param_3], **kwargs_4)
            
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', str_call_result_5)
            
            # ################# End of '__oct__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__oct__' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__oct__'
            return stypy_return_type_6


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__init__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'DefinesMethod' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'DefinesMethod', DefinesMethod)
    
    # Call to oct(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to DefinesMethod(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_9 = {}
    # Getting the type of 'DefinesMethod' (line 13)
    DefinesMethod_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'DefinesMethod', False)
    # Calling DefinesMethod(args, kwargs) (line 13)
    DefinesMethod_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), DefinesMethod_8, *[], **kwargs_9)
    
    # Processing the call keyword arguments (line 13)
    kwargs_11 = {}
    # Getting the type of 'oct' (line 13)
    oct_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'oct', False)
    # Calling oct(args, kwargs) (line 13)
    oct_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), oct_7, *[DefinesMethod_call_result_10], **kwargs_11)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

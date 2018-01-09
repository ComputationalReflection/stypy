
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: 
4: __doc__ = "Special __int__ method is defined, but with incorrect arity"
5: 
6: if __name__ == '__main__':
7:     class DefinesMethod:
8:         def __int__(self, param):
9:             return 1 + param
10: 
11: 
12:     # Type error #
13:     print int(DefinesMethod())
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 4):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Special __int__ method is defined, but with incorrect arity')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DefinesMethod' class

    class DefinesMethod:

        @norecursion
        def __int__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__int__'
            module_type_store = module_type_store.open_function_context('__int__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesMethod.__int__.__dict__.__setitem__('stypy_localization', localization)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_function_name', 'DefinesMethod.__int__')
            DefinesMethod.__int__.__dict__.__setitem__('stypy_param_names_list', ['param'])
            DefinesMethod.__int__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesMethod.__int__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__int__', ['param'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__int__', localization, ['param'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__int__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
            # Getting the type of 'param' (line 9)
            param_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'param')
            # Applying the binary operator '+' (line 9)
            result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 19), '+', int_2, param_3)
            
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', result_add_4)
            
            # ################# End of '__int__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__int__' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__int__'
            return stypy_return_type_5


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
    
    # Call to int(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to DefinesMethod(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_8 = {}
    # Getting the type of 'DefinesMethod' (line 13)
    DefinesMethod_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'DefinesMethod', False)
    # Calling DefinesMethod(args, kwargs) (line 13)
    DefinesMethod_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), DefinesMethod_7, *[], **kwargs_8)
    
    # Processing the call keyword arguments (line 13)
    kwargs_10 = {}
    # Getting the type of 'int' (line 13)
    int_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'int', False)
    # Calling int(args, kwargs) (line 13)
    int_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), int_6, *[DefinesMethod_call_result_9], **kwargs_10)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

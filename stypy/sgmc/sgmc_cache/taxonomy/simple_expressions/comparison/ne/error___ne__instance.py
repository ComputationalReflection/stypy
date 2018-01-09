
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__ne__ method is invoked without using an instance "
3: 
4: if __name__ == '__main__':
5:     class Eq:
6:         def __ne__(self, other):
7:             return True
8: 
9: 
10:     print Eq != 3
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__ne__ method is invoked without using an instance ')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Eq' class

    class Eq:

        @norecursion
        def __ne__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__ne__'
            module_type_store = module_type_store.open_function_context('__ne__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Eq.__ne__.__dict__.__setitem__('stypy_localization', localization)
            Eq.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq.__ne__.__dict__.__setitem__('stypy_function_name', 'Eq.__ne__')
            Eq.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Eq.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq.__ne__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__ne__(...)' code ##################

            # Getting the type of 'True' (line 7)
            True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', True_2)
            
            # ################# End of '__ne__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__ne__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__ne__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
            # Assigning a type to the variable 'self' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Eq' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Eq', Eq)
    
    # Getting the type of 'Eq' (line 10)
    Eq_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'Eq')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'int')
    # Applying the binary operator '!=' (line 10)
    result_ne_6 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '!=', Eq_4, int_5)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

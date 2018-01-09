
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__gt__ method is not explicitly declared"
3: 
4: if __name__ == '__main__':
5:     class Empty:
6:         pass
7: 
8: 
9:     class Eq:
10:         def __gt__(self, other):
11:             return True
12: 
13: 
14:     print Eq() > 3
15: 
16:     print Empty() > 3
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__gt__ method is not explicitly declared')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Empty' class

    class Empty:
        pass

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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Empty.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Empty' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Empty', Empty)
    # Declaration of the 'Eq' class

    class Eq:

        @norecursion
        def __gt__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__gt__'
            module_type_store = module_type_store.open_function_context('__gt__', 10, 8, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Eq.__gt__.__dict__.__setitem__('stypy_localization', localization)
            Eq.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq.__gt__.__dict__.__setitem__('stypy_function_name', 'Eq.__gt__')
            Eq.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Eq.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq.__gt__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__gt__(...)' code ##################

            # Getting the type of 'True' (line 11)
            True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'stypy_return_type', True_2)
            
            # ################# End of '__gt__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__gt__' in the type store
            # Getting the type of 'stypy_return_type' (line 10)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__gt__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Eq' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'Eq', Eq)
    
    
    # Call to Eq(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_5 = {}
    # Getting the type of 'Eq' (line 14)
    Eq_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'Eq', False)
    # Calling Eq(args, kwargs) (line 14)
    Eq_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), Eq_4, *[], **kwargs_5)
    
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'int')
    # Applying the binary operator '>' (line 14)
    result_gt_8 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 10), '>', Eq_call_result_6, int_7)
    
    
    
    # Call to Empty(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_10 = {}
    # Getting the type of 'Empty' (line 16)
    Empty_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'Empty', False)
    # Calling Empty(args, kwargs) (line 16)
    Empty_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), Empty_9, *[], **kwargs_10)
    
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    # Applying the binary operator '>' (line 16)
    result_gt_13 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 10), '>', Empty_call_result_11, int_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

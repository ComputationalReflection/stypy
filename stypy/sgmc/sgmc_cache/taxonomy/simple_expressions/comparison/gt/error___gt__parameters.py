
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__gt__ method is called with wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     class Eq3:
6:         def __gt__(self, other, another):
7:             return "str"
8: 
9: 
10:     class Eq4:
11:         def __gt__(self):
12:             return "str"
13: 
14: 
15:     # Type error
16:     print Eq3() > 3
17:     # Type error
18:     print Eq4() > 3
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__gt__ method is called with wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Eq3' class

    class Eq3:

        @norecursion
        def __gt__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__gt__'
            module_type_store = module_type_store.open_function_context('__gt__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Eq3.__gt__.__dict__.__setitem__('stypy_localization', localization)
            Eq3.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq3.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq3.__gt__.__dict__.__setitem__('stypy_function_name', 'Eq3.__gt__')
            Eq3.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other', 'another'])
            Eq3.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq3.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq3.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq3.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq3.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq3.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq3.__gt__', ['other', 'another'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__gt__', localization, ['other', 'another'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__gt__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', str_2)
            
            # ################# End of '__gt__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__gt__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
            # Assigning a type to the variable 'self' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq3.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Eq3' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Eq3', Eq3)
    # Declaration of the 'Eq4' class

    class Eq4:

        @norecursion
        def __gt__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__gt__'
            module_type_store = module_type_store.open_function_context('__gt__', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Eq4.__gt__.__dict__.__setitem__('stypy_localization', localization)
            Eq4.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq4.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq4.__gt__.__dict__.__setitem__('stypy_function_name', 'Eq4.__gt__')
            Eq4.__gt__.__dict__.__setitem__('stypy_param_names_list', [])
            Eq4.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq4.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq4.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq4.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq4.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq4.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq4.__gt__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__gt__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__gt__(...)' code ##################

            str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', str_4)
            
            # ################# End of '__gt__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__gt__' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__gt__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq4.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Eq4' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Eq4', Eq4)
    
    
    # Call to Eq3(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_7 = {}
    # Getting the type of 'Eq3' (line 16)
    Eq3_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'Eq3', False)
    # Calling Eq3(args, kwargs) (line 16)
    Eq3_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), Eq3_6, *[], **kwargs_7)
    
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'int')
    # Applying the binary operator '>' (line 16)
    result_gt_10 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 10), '>', Eq3_call_result_8, int_9)
    
    
    
    # Call to Eq4(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_12 = {}
    # Getting the type of 'Eq4' (line 18)
    Eq4_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'Eq4', False)
    # Calling Eq4(args, kwargs) (line 18)
    Eq4_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), Eq4_11, *[], **kwargs_12)
    
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
    # Applying the binary operator '>' (line 18)
    result_gt_15 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 10), '>', Eq4_call_result_13, int_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the correct implementation of __enter__"
3: 
4: if __name__ == '__main__':
5:     class controlled_execution2:
6:         def __enter__(self):
7:             return 0
8: 
9:         def __exit__(self, exc_type, exc_val, exc_tb):
10:             pass
11: 
12: 
13:     a = 3
14: 
15:     with controlled_execution2() as thing:
16:         # Type error
17:         print thing[0]
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check the correct implementation of __enter__')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'controlled_execution2' class

    class controlled_execution2:

        @norecursion
        def __enter__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__enter__'
            module_type_store = module_type_store.open_function_context('__enter__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_localization', localization)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_function_name', 'controlled_execution2.__enter__')
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            controlled_execution2.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'controlled_execution2.__enter__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__enter__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__enter__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__enter__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__enter__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__enter__'
            return stypy_return_type_3


        @norecursion
        def __exit__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__exit__'
            module_type_store = module_type_store.open_function_context('__exit__', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_localization', localization)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_function_name', 'controlled_execution2.__exit__')
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_param_names_list', ['exc_type', 'exc_val', 'exc_tb'])
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            controlled_execution2.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'controlled_execution2.__exit__', ['exc_type', 'exc_val', 'exc_tb'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__exit__', localization, ['exc_type', 'exc_val', 'exc_tb'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__exit__(...)' code ##################

            pass
            
            # ################# End of '__exit__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__exit__' in the type store
            # Getting the type of 'stypy_return_type' (line 9)
            stypy_return_type_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_4)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__exit__'
            return stypy_return_type_4


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'controlled_execution2.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'controlled_execution2' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'controlled_execution2', controlled_execution2)
    
    # Assigning a Num to a Name (line 13):
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'int')
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'a', int_5)
    
    # Call to controlled_execution2(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_7 = {}
    # Getting the type of 'controlled_execution2' (line 15)
    controlled_execution2_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'controlled_execution2', False)
    # Calling controlled_execution2(args, kwargs) (line 15)
    controlled_execution2_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), controlled_execution2_6, *[], **kwargs_7)
    
    with_9 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 15, 9), controlled_execution2_call_result_8, 'with parameter', '__enter__', '__exit__')

    if with_9:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 15)
        enter___10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), controlled_execution2_call_result_8, '__enter__')
        with_enter_11 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), enter___10)
        # Assigning a type to the variable 'thing' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'thing', with_enter_11)
        
        # Obtaining the type of the subscript
        int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
        # Getting the type of 'thing' (line 17)
        thing_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'thing')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), thing_13, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), getitem___14, int_12)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 15)
        exit___16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), controlled_execution2_call_result_8, '__exit__')
        with_exit_17 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), exit___16, None, None, None)



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

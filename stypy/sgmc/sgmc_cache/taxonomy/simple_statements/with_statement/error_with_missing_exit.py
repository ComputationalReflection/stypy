
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the correct implementation of __enter__"
3: 
4: if __name__ == '__main__':
5:     class controlled_execution2:
6:         def __exit__(self, exc_type, exc_val, exc_tb):
7:             pass
8: 
9: 
10:     a = 3
11:     # Type error
12:     with controlled_execution2() as thing:
13:         a = a + 1
14:         print thing
15: 

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
        def __exit__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__exit__'
            module_type_store = module_type_store.open_function_context('__exit__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
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
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__exit__'
            return stypy_return_type_2


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
    
    # Assigning a Num to a Name (line 10):
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')
    # Assigning a type to the variable 'a' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a', int_3)
    
    # Call to controlled_execution2(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_5 = {}
    # Getting the type of 'controlled_execution2' (line 12)
    controlled_execution2_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'controlled_execution2', False)
    # Calling controlled_execution2(args, kwargs) (line 12)
    controlled_execution2_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), controlled_execution2_4, *[], **kwargs_5)
    
    with_7 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 12, 9), controlled_execution2_call_result_6, 'with parameter', '__enter__', '__exit__')

    if with_7:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 12)
        enter___8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), controlled_execution2_call_result_6, '__enter__')
        with_enter_9 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), enter___8)
        # Assigning a type to the variable 'thing' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'thing', with_enter_9)
        
        # Assigning a BinOp to a Name (line 13):
        # Getting the type of 'a' (line 13)
        a_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'a')
        int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
        # Applying the binary operator '+' (line 13)
        result_add_12 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), '+', a_10, int_11)
        
        # Assigning a type to the variable 'a' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'a', result_add_12)
        # Getting the type of 'thing' (line 14)
        thing_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'thing')
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 12)
        exit___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), controlled_execution2_call_result_6, '__exit__')
        with_exit_15 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), exit___14, None, None, None)



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

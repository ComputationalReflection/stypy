
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__ne__ method is called with incompatible type parameters"
3: 
4: if __name__ == '__main__':
5:     class Eq5:
6:         def __ne__(self, other):
7:             # Type error
8:             return str(other) + other
9: 
10: 
11:     print Eq5() != 3
12: 
13: 
14:     class Eq6:
15:         def __ne__(self, other):
16:             # Type error
17:             return other[other]
18: 
19: 
20:     print Eq6() != 3
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__ne__ method is called with incompatible type parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Eq5' class

    class Eq5:

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
            Eq5.__ne__.__dict__.__setitem__('stypy_localization', localization)
            Eq5.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq5.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq5.__ne__.__dict__.__setitem__('stypy_function_name', 'Eq5.__ne__')
            Eq5.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Eq5.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq5.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq5.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq5.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq5.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq5.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq5.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

            
            # Call to str(...): (line 8)
            # Processing the call arguments (line 8)
            # Getting the type of 'other' (line 8)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'other', False)
            # Processing the call keyword arguments (line 8)
            kwargs_4 = {}
            # Getting the type of 'str' (line 8)
            str_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', False)
            # Calling str(args, kwargs) (line 8)
            str_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 8, 19), str_2, *[other_3], **kwargs_4)
            
            # Getting the type of 'other' (line 8)
            other_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 32), 'other')
            # Applying the binary operator '+' (line 8)
            result_add_7 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 19), '+', str_call_result_5, other_6)
            
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', result_add_7)
            
            # ################# End of '__ne__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__ne__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_8)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__ne__'
            return stypy_return_type_8


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq5.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Eq5' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Eq5', Eq5)
    
    
    # Call to Eq5(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_10 = {}
    # Getting the type of 'Eq5' (line 11)
    Eq5_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'Eq5', False)
    # Calling Eq5(args, kwargs) (line 11)
    Eq5_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), Eq5_9, *[], **kwargs_10)
    
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
    # Applying the binary operator '!=' (line 11)
    result_ne_13 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), '!=', Eq5_call_result_11, int_12)
    
    # Declaration of the 'Eq6' class

    class Eq6:

        @norecursion
        def __ne__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__ne__'
            module_type_store = module_type_store.open_function_context('__ne__', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Eq6.__ne__.__dict__.__setitem__('stypy_localization', localization)
            Eq6.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq6.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq6.__ne__.__dict__.__setitem__('stypy_function_name', 'Eq6.__ne__')
            Eq6.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Eq6.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq6.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq6.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq6.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq6.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq6.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq6.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

            
            # Obtaining the type of the subscript
            # Getting the type of 'other' (line 17)
            other_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'other')
            # Getting the type of 'other' (line 17)
            other_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'other')
            # Obtaining the member '__getitem__' of a type (line 17)
            getitem___16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), other_15, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 17)
            subscript_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), getitem___16, other_14)
            
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', subscript_call_result_17)
            
            # ################# End of '__ne__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__ne__' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_18)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__ne__'
            return stypy_return_type_18


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
            # Assigning a type to the variable 'self' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq6.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Eq6' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'Eq6', Eq6)
    
    
    # Call to Eq6(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_20 = {}
    # Getting the type of 'Eq6' (line 20)
    Eq6_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'Eq6', False)
    # Calling Eq6(args, kwargs) (line 20)
    Eq6_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), Eq6_19, *[], **kwargs_20)
    
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'int')
    # Applying the binary operator '!=' (line 20)
    result_ne_23 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 10), '!=', Eq6_call_result_21, int_22)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

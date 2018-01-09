
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No __ixor__ method is present"
3: 
4: if __name__ == '__main__':
5:     class Correct:
6:         def __ixor__(self, other):
7:             other ^= other
8:             return other
9: 
10: 
11:     x = Correct()
12:     x ^= 4
13: 
14: 
15:     class Sample:
16:         pass
17: 
18: 
19:     x = Sample()
20:     # Type error
21:     x ^= 1
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'No __ixor__ method is present')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Correct' class

    class Correct:

        @norecursion
        def __ixor__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__ixor__'
            module_type_store = module_type_store.open_function_context('__ixor__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Correct.__ixor__.__dict__.__setitem__('stypy_localization', localization)
            Correct.__ixor__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Correct.__ixor__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Correct.__ixor__.__dict__.__setitem__('stypy_function_name', 'Correct.__ixor__')
            Correct.__ixor__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Correct.__ixor__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Correct.__ixor__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Correct.__ixor__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Correct.__ixor__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Correct.__ixor__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Correct.__ixor__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Correct.__ixor__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__ixor__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__ixor__(...)' code ##################

            
            # Getting the type of 'other' (line 7)
            other_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'other')
            # Getting the type of 'other' (line 7)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 21), 'other')
            # Applying the binary operator '^=' (line 7)
            result_ixor_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '^=', other_2, other_3)
            # Assigning a type to the variable 'other' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'other', result_ixor_4)
            
            # Getting the type of 'other' (line 8)
            other_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'other')
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', other_5)
            
            # ################# End of '__ixor__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__ixor__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__ixor__'
            return stypy_return_type_6


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Correct.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Correct' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Correct', Correct)
    
    # Assigning a Call to a Name (line 11):
    
    # Call to Correct(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_8 = {}
    # Getting the type of 'Correct' (line 11)
    Correct_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'Correct', False)
    # Calling Correct(args, kwargs) (line 11)
    Correct_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), Correct_7, *[], **kwargs_8)
    
    # Assigning a type to the variable 'x' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'x', Correct_call_result_9)
    
    # Getting the type of 'x' (line 12)
    x_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x')
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 9), 'int')
    # Applying the binary operator '^=' (line 12)
    result_ixor_12 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 4), '^=', x_10, int_11)
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', result_ixor_12)
    
    # Declaration of the 'Sample' class

    class Sample:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Sample' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to Sample(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_14 = {}
    # Getting the type of 'Sample' (line 19)
    Sample_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'Sample', False)
    # Calling Sample(args, kwargs) (line 19)
    Sample_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), Sample_13, *[], **kwargs_14)
    
    # Assigning a type to the variable 'x' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'x', Sample_call_result_15)
    
    # Getting the type of 'x' (line 21)
    x_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'x')
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
    # Applying the binary operator '^=' (line 21)
    result_ixor_18 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 4), '^=', x_16, int_17)
    # Assigning a type to the variable 'x' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'x', result_ixor_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

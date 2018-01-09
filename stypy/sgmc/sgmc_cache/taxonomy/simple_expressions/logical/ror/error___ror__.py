
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No __ror__ method is present"
3: 
4: if __name__ == '__main__':
5:     class Correct:
6:         def __ror__(self, other):
7:             return other | other
8: 
9: 
10:     print 4 | Correct()
11: 
12: 
13:     class Sample:
14:         pass
15: 
16: 
17:     # Type error
18:     print 1 | Sample()
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'No __ror__ method is present')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Correct' class

    class Correct:

        @norecursion
        def __ror__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__ror__'
            module_type_store = module_type_store.open_function_context('__ror__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Correct.__ror__.__dict__.__setitem__('stypy_localization', localization)
            Correct.__ror__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Correct.__ror__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Correct.__ror__.__dict__.__setitem__('stypy_function_name', 'Correct.__ror__')
            Correct.__ror__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Correct.__ror__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Correct.__ror__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Correct.__ror__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Correct.__ror__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Correct.__ror__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Correct.__ror__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Correct.__ror__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__ror__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__ror__(...)' code ##################

            # Getting the type of 'other' (line 7)
            other_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'other')
            # Getting the type of 'other' (line 7)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 27), 'other')
            # Applying the binary operator '|' (line 7)
            result_or__4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 19), '|', other_2, other_3)
            
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', result_or__4)
            
            # ################# End of '__ror__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__ror__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__ror__'
            return stypy_return_type_5


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
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'int')
    
    # Call to Correct(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_8 = {}
    # Getting the type of 'Correct' (line 10)
    Correct_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'Correct', False)
    # Calling Correct(args, kwargs) (line 10)
    Correct_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 10, 14), Correct_7, *[], **kwargs_8)
    
    # Applying the binary operator '|' (line 10)
    result_or__10 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '|', int_6, Correct_call_result_9)
    
    # Declaration of the 'Sample' class

    class Sample:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Sample', Sample)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'int')
    
    # Call to Sample(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 18)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'Sample', False)
    # Calling Sample(args, kwargs) (line 18)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), Sample_12, *[], **kwargs_13)
    
    # Applying the binary operator '|' (line 18)
    result_or__15 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 10), '|', int_11, Sample_call_result_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Usage of or keyword"
3: 
4: if __name__ == '__main__':
5:     class Correct:
6:         def __or__(self, other):
7:             print "launch"
8:             return other | other
9: 
10: 
11:     print Correct() or 4
12: 
13: 
14:     class Sample:
15:         pass
16: 
17:     print Sample() or 1
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Usage of or keyword')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Correct' class

    class Correct:

        @norecursion
        def __or__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__or__'
            module_type_store = module_type_store.open_function_context('__or__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Correct.__or__.__dict__.__setitem__('stypy_localization', localization)
            Correct.__or__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Correct.__or__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Correct.__or__.__dict__.__setitem__('stypy_function_name', 'Correct.__or__')
            Correct.__or__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Correct.__or__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Correct.__or__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Correct.__or__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Correct.__or__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Correct.__or__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Correct.__or__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Correct.__or__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__or__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__or__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'str', 'launch')
            # Getting the type of 'other' (line 8)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'other')
            # Getting the type of 'other' (line 8)
            other_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 27), 'other')
            # Applying the binary operator '|' (line 8)
            result_or__5 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 19), '|', other_3, other_4)
            
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', result_or__5)
            
            # ################# End of '__or__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__or__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__or__'
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
    
    # Evaluating a boolean operation
    
    # Call to Correct(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_8 = {}
    # Getting the type of 'Correct' (line 11)
    Correct_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'Correct', False)
    # Calling Correct(args, kwargs) (line 11)
    Correct_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), Correct_7, *[], **kwargs_8)
    
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
    # Applying the binary operator 'or' (line 11)
    result_or_keyword_11 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), 'or', Correct_call_result_9, int_10)
    
    # Declaration of the 'Sample' class

    class Sample:
        pass

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

    
    # Assigning a type to the variable 'Sample' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'Sample', Sample)
    
    # Evaluating a boolean operation
    
    # Call to Sample(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 17)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'Sample', False)
    # Calling Sample(args, kwargs) (line 17)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), Sample_12, *[], **kwargs_13)
    
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'int')
    # Applying the binary operator 'or' (line 17)
    result_or_keyword_16 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 10), 'or', Sample_call_result_14, int_15)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

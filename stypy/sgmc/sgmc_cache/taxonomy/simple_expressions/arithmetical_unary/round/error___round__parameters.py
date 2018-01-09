
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__round__ method is present, but is declared with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __float__(self, other, another):
7:             return 1.0
8: 
9: 
10:     # Type error
11:     print round(Sample())
12: 
13: 
14:     class OtherSample:
15:         def __float__(self, other):
16:             return 1.0
17: 
18: 
19:     # Type error
20:     print round(OtherSample())
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__round__ method is present, but is declared with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__float__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__float__.__dict__.__setitem__('stypy_function_name', 'Sample.__float__')
            Sample.__float__.__dict__.__setitem__('stypy_param_names_list', ['other', 'another'])
            Sample.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__float__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__float__', ['other', 'another'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__float__', localization, ['other', 'another'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__float__(...)' code ##################

            float_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', float_2)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__float__'
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

    
    # Assigning a type to the variable 'Sample' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Sample', Sample)
    
    # Call to round(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to Sample(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 11)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'Sample', False)
    # Calling Sample(args, kwargs) (line 11)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), Sample_5, *[], **kwargs_6)
    
    # Processing the call keyword arguments (line 11)
    kwargs_8 = {}
    # Getting the type of 'round' (line 11)
    round_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'round', False)
    # Calling round(args, kwargs) (line 11)
    round_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), round_4, *[Sample_call_result_7], **kwargs_8)
    
    # Declaration of the 'OtherSample' class

    class OtherSample:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            OtherSample.__float__.__dict__.__setitem__('stypy_localization', localization)
            OtherSample.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            OtherSample.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            OtherSample.__float__.__dict__.__setitem__('stypy_function_name', 'OtherSample.__float__')
            OtherSample.__float__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            OtherSample.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            OtherSample.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            OtherSample.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            OtherSample.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            OtherSample.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            OtherSample.__float__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'OtherSample.__float__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__float__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__float__(...)' code ##################

            float_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'stypy_return_type', float_10)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__float__'
            return stypy_return_type_11


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'OtherSample.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'OtherSample' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'OtherSample', OtherSample)
    
    # Call to round(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to OtherSample(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_14 = {}
    # Getting the type of 'OtherSample' (line 20)
    OtherSample_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'OtherSample', False)
    # Calling OtherSample(args, kwargs) (line 20)
    OtherSample_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), OtherSample_13, *[], **kwargs_14)
    
    # Processing the call keyword arguments (line 20)
    kwargs_16 = {}
    # Getting the type of 'round' (line 20)
    round_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'round', False)
    # Calling round(args, kwargs) (line 20)
    round_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), round_12, *[OtherSample_call_result_15], **kwargs_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

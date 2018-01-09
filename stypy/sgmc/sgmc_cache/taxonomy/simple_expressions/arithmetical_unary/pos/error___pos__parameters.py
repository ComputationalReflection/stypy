
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__pos__ method is present, but is declared with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __pos__(self, other, another):
7:             return 1 + other + another
8: 
9: 
10:     # Type error
11:     print +Sample()
12: 
13: 
14:     class OtherSample:
15:         def __pos__(self, other):
16:             return 1
17: 
18: 
19:     # Type error
20:     print +OtherSample()
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__pos__ method is present, but is declared with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __pos__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__pos__'
            module_type_store = module_type_store.open_function_context('__pos__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__pos__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__pos__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__pos__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__pos__.__dict__.__setitem__('stypy_function_name', 'Sample.__pos__')
            Sample.__pos__.__dict__.__setitem__('stypy_param_names_list', ['other', 'another'])
            Sample.__pos__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__pos__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__pos__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__pos__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__pos__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__pos__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__pos__', ['other', 'another'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__pos__', localization, ['other', 'another'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__pos__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
            # Getting the type of 'other' (line 7)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 23), 'other')
            # Applying the binary operator '+' (line 7)
            result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 19), '+', int_2, other_3)
            
            # Getting the type of 'another' (line 7)
            another_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 31), 'another')
            # Applying the binary operator '+' (line 7)
            result_add_6 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 29), '+', result_add_4, another_5)
            
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', result_add_6)
            
            # ################# End of '__pos__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__pos__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__pos__'
            return stypy_return_type_7


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
    
    
    # Call to Sample(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_9 = {}
    # Getting the type of 'Sample' (line 11)
    Sample_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'Sample', False)
    # Calling Sample(args, kwargs) (line 11)
    Sample_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), Sample_8, *[], **kwargs_9)
    
    # Applying the 'uadd' unary operator (line 11)
    result___pos___11 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), 'uadd', Sample_call_result_10)
    
    # Declaration of the 'OtherSample' class

    class OtherSample:

        @norecursion
        def __pos__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__pos__'
            module_type_store = module_type_store.open_function_context('__pos__', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            OtherSample.__pos__.__dict__.__setitem__('stypy_localization', localization)
            OtherSample.__pos__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            OtherSample.__pos__.__dict__.__setitem__('stypy_type_store', module_type_store)
            OtherSample.__pos__.__dict__.__setitem__('stypy_function_name', 'OtherSample.__pos__')
            OtherSample.__pos__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            OtherSample.__pos__.__dict__.__setitem__('stypy_varargs_param_name', None)
            OtherSample.__pos__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            OtherSample.__pos__.__dict__.__setitem__('stypy_call_defaults', defaults)
            OtherSample.__pos__.__dict__.__setitem__('stypy_call_varargs', varargs)
            OtherSample.__pos__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            OtherSample.__pos__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'OtherSample.__pos__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__pos__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__pos__(...)' code ##################

            int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'stypy_return_type', int_12)
            
            # ################# End of '__pos__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__pos__' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__pos__'
            return stypy_return_type_13


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
    
    
    # Call to OtherSample(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_15 = {}
    # Getting the type of 'OtherSample' (line 20)
    OtherSample_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'OtherSample', False)
    # Calling OtherSample(args, kwargs) (line 20)
    OtherSample_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), OtherSample_14, *[], **kwargs_15)
    
    # Applying the 'uadd' unary operator (line 20)
    result___pos___17 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 10), 'uadd', OtherSample_call_result_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

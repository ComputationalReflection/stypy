
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__imul__ method is present, but is declared with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __imul__(self, other, another):
7:             other *= another
8:             return other
9: 
10: 
11:     x = Sample()
12:     # Type error
13:     x *= 1
14: 
15: 
16:     class OtherSample:
17:         def __imul__(self):
18:             return 1
19: 
20: 
21:     x = OtherSample()
22:     # Type error
23:     x *= 1
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__imul__ method is present, but is declared with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __imul__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__imul__'
            module_type_store = module_type_store.open_function_context('__imul__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__imul__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__imul__.__dict__.__setitem__('stypy_function_name', 'Sample.__imul__')
            Sample.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other', 'another'])
            Sample.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__imul__', ['other', 'another'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__imul__', localization, ['other', 'another'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__imul__(...)' code ##################

            
            # Getting the type of 'other' (line 7)
            other_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'other')
            # Getting the type of 'another' (line 7)
            another_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 21), 'another')
            # Applying the binary operator '*=' (line 7)
            result_imul_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '*=', other_2, another_3)
            # Assigning a type to the variable 'other' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'other', result_imul_4)
            
            # Getting the type of 'other' (line 8)
            other_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'other')
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', other_5)
            
            # ################# End of '__imul__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__imul__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__imul__'
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
    
    # Assigning a Call to a Name (line 11):
    
    # Call to Sample(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_8 = {}
    # Getting the type of 'Sample' (line 11)
    Sample_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'Sample', False)
    # Calling Sample(args, kwargs) (line 11)
    Sample_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), Sample_7, *[], **kwargs_8)
    
    # Assigning a type to the variable 'x' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'x', Sample_call_result_9)
    
    # Getting the type of 'x' (line 13)
    x_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x')
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
    # Applying the binary operator '*=' (line 13)
    result_imul_12 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 4), '*=', x_10, int_11)
    # Assigning a type to the variable 'x' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x', result_imul_12)
    
    # Declaration of the 'OtherSample' class

    class OtherSample:

        @norecursion
        def __imul__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__imul__'
            module_type_store = module_type_store.open_function_context('__imul__', 17, 8, False)
            # Assigning a type to the variable 'self' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            OtherSample.__imul__.__dict__.__setitem__('stypy_localization', localization)
            OtherSample.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            OtherSample.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
            OtherSample.__imul__.__dict__.__setitem__('stypy_function_name', 'OtherSample.__imul__')
            OtherSample.__imul__.__dict__.__setitem__('stypy_param_names_list', [])
            OtherSample.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
            OtherSample.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            OtherSample.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
            OtherSample.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
            OtherSample.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            OtherSample.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'OtherSample.__imul__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__imul__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__imul__(...)' code ##################

            int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'stypy_return_type', int_13)
            
            # ################# End of '__imul__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__imul__' in the type store
            # Getting the type of 'stypy_return_type' (line 17)
            stypy_return_type_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_14)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__imul__'
            return stypy_return_type_14


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'OtherSample' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'OtherSample', OtherSample)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to OtherSample(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_16 = {}
    # Getting the type of 'OtherSample' (line 21)
    OtherSample_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'OtherSample', False)
    # Calling OtherSample(args, kwargs) (line 21)
    OtherSample_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), OtherSample_15, *[], **kwargs_16)
    
    # Assigning a type to the variable 'x' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'x', OtherSample_call_result_17)
    
    # Getting the type of 'x' (line 23)
    x_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'x')
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'int')
    # Applying the binary operator '*=' (line 23)
    result_imul_20 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 4), '*=', x_18, int_19)
    # Assigning a type to the variable 'x' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'x', result_imul_20)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "next builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__next) -> DynamicType
7:     # (Has__next, AnyType) -> DynamicType
8: 
9: 
10:     class Sample:
11:         def next(self):
12:             return 4
13: 
14: 
15:     class Wrong1:
16:         def next(self, x):
17:             return x
18: 
19: 
20:     # Call the builtin with correct parameters
21:     ret = next(Sample())
22:     ret = next(Sample(), 4)
23: 
24:     # Call the builtin with incorrect types of parameters
25:     # Type error
26:     ret = next(Wrong1())
27:     # Type error
28:     ret = next(3)
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'next builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def next(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'next'
            module_type_store = module_type_store.open_function_context('next', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.next.__dict__.__setitem__('stypy_localization', localization)
            Sample.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.next.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.next.__dict__.__setitem__('stypy_function_name', 'Sample.next')
            Sample.next.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.next.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.next.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.next.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.next.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.next', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'next', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'next(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', int_2)
            
            # ################# End of 'next(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'next' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'next'
            return stypy_return_type_3


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

    
    # Assigning a type to the variable 'Sample' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def next(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'next'
            module_type_store = module_type_store.open_function_context('next', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.next.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.next.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.next.__dict__.__setitem__('stypy_function_name', 'Wrong1.next')
            Wrong1.next.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.next.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.next.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.next.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.next.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.next', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'next', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'next(...)' code ##################

            # Getting the type of 'x' (line 17)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', x_4)
            
            # ################# End of 'next(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'next' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'next'
            return stypy_return_type_5


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Wrong1' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Wrong1', Wrong1)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to next(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to Sample(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_8 = {}
    # Getting the type of 'Sample' (line 21)
    Sample_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'Sample', False)
    # Calling Sample(args, kwargs) (line 21)
    Sample_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), Sample_7, *[], **kwargs_8)
    
    # Processing the call keyword arguments (line 21)
    kwargs_10 = {}
    # Getting the type of 'next' (line 21)
    next_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'next', False)
    # Calling next(args, kwargs) (line 21)
    next_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), next_6, *[Sample_call_result_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', next_call_result_11)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to next(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to Sample(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_14 = {}
    # Getting the type of 'Sample' (line 22)
    Sample_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'Sample', False)
    # Calling Sample(args, kwargs) (line 22)
    Sample_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), Sample_13, *[], **kwargs_14)
    
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_17 = {}
    # Getting the type of 'next' (line 22)
    next_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'next', False)
    # Calling next(args, kwargs) (line 22)
    next_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), next_12, *[Sample_call_result_15, int_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', next_call_result_18)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to next(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to Wrong1(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_21 = {}
    # Getting the type of 'Wrong1' (line 26)
    Wrong1_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 26)
    Wrong1_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), Wrong1_20, *[], **kwargs_21)
    
    # Processing the call keyword arguments (line 26)
    kwargs_23 = {}
    # Getting the type of 'next' (line 26)
    next_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'next', False)
    # Calling next(args, kwargs) (line 26)
    next_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), next_19, *[Wrong1_call_result_22], **kwargs_23)
    
    # Assigning a type to the variable 'ret' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'ret', next_call_result_24)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to next(...): (line 28)
    # Processing the call arguments (line 28)
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_27 = {}
    # Getting the type of 'next' (line 28)
    next_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'next', False)
    # Calling next(args, kwargs) (line 28)
    next_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), next_25, *[int_26], **kwargs_27)
    
    # Assigning a type to the variable 'ret' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ret', next_call_result_28)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "round builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (RealNumber) -> <type 'float'>
7:     # (RealNumber, Integer) -> <type 'float'>
8:     # (RealNumber, CastsToIndex) -> <type 'float'>
9:     # (CastsToFloat) -> <type 'float'>
10:     # (CastsToFloat, Integer) -> <type 'float'>
11:     # (CastsToFloat, CastsToIndex) -> <type 'float'>
12: 
13:     class Empty:
14:         pass
15: 
16: 
17:     class Sample:
18:         def __float__(self):
19:             return 4.0
20: 
21: 
22:     class Wrong1:
23:         def __float__(self, x):
24:             return x
25: 
26: 
27:     class Wrong2:
28:         def __float__(self):
29:             return "str"
30: 
31: 
32:     # Call the builtin with correct parameters
33: 
34:     ret = round(Sample(), 10)
35: 
36:     # Call the builtin with incorrect types of parameters
37:     # Type error
38:     ret = round(Wrong1())
39:     # Type error
40:     ret = round(Wrong2())
41:     # Type error
42:     ret = round(Empty())
43: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'round builtin is invoked, but classes and instances with special name methods are passed ')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Empty' class

    class Empty:
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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Empty.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Empty' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 18, 8, False)
            # Assigning a type to the variable 'self' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__float__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__float__.__dict__.__setitem__('stypy_function_name', 'Sample.__float__')
            Sample.__float__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__float__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__float__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__float__(...)' code ##################

            float_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type', float_2)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 17, 4, False)
            # Assigning a type to the variable 'self' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 23, 8, False)
            # Assigning a type to the variable 'self' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__float__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__float__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__float__')
            Wrong1.__float__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__float__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__float__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__float__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__float__(...)' code ##################

            # Getting the type of 'x' (line 24)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 23)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__float__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 22, 4, False)
            # Assigning a type to the variable 'self' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 28, 8, False)
            # Assigning a type to the variable 'self' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__float__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__float__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__float__')
            Wrong2.__float__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__float__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__float__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__float__(...)' code ##################

            str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'stypy_return_type', str_6)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 28)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__float__'
            return stypy_return_type_7


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 27, 4, False)
            # Assigning a type to the variable 'self' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Wrong2' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to round(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to Sample(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_10 = {}
    # Getting the type of 'Sample' (line 34)
    Sample_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'Sample', False)
    # Calling Sample(args, kwargs) (line 34)
    Sample_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), Sample_9, *[], **kwargs_10)
    
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_13 = {}
    # Getting the type of 'round' (line 34)
    round_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'round', False)
    # Calling round(args, kwargs) (line 34)
    round_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), round_8, *[Sample_call_result_11, int_12], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'ret', round_call_result_14)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to round(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Call to Wrong1(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_17 = {}
    # Getting the type of 'Wrong1' (line 38)
    Wrong1_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 38)
    Wrong1_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), Wrong1_16, *[], **kwargs_17)
    
    # Processing the call keyword arguments (line 38)
    kwargs_19 = {}
    # Getting the type of 'round' (line 38)
    round_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'round', False)
    # Calling round(args, kwargs) (line 38)
    round_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), round_15, *[Wrong1_call_result_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'ret', round_call_result_20)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to round(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to Wrong2(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_23 = {}
    # Getting the type of 'Wrong2' (line 40)
    Wrong2_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 40)
    Wrong2_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), Wrong2_22, *[], **kwargs_23)
    
    # Processing the call keyword arguments (line 40)
    kwargs_25 = {}
    # Getting the type of 'round' (line 40)
    round_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'round', False)
    # Calling round(args, kwargs) (line 40)
    round_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), round_21, *[Wrong2_call_result_24], **kwargs_25)
    
    # Assigning a type to the variable 'ret' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'ret', round_call_result_26)
    
    # Assigning a Call to a Name (line 42):
    
    # Call to round(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to Empty(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_29 = {}
    # Getting the type of 'Empty' (line 42)
    Empty_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'Empty', False)
    # Calling Empty(args, kwargs) (line 42)
    Empty_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), Empty_28, *[], **kwargs_29)
    
    # Processing the call keyword arguments (line 42)
    kwargs_31 = {}
    # Getting the type of 'round' (line 42)
    round_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'round', False)
    # Calling round(args, kwargs) (line 42)
    round_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), round_27, *[Empty_call_result_30], **kwargs_31)
    
    # Assigning a type to the variable 'ret' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'ret', round_call_result_32)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

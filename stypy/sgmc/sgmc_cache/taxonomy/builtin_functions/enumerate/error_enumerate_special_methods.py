
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "enumerate builtin is invoked, but classes and instances with special name methods are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'enumerate'>
7:     # (IterableObject) -> <type 'enumerate'>
8:     # (Has__iter__) -> <type 'enumerate'>
9:     # (IterableObject, Integer) -> <type 'enumerate'>
10:     # (Has__iter__, Integer) -> <type 'enumerate'>
11: 
12: 
13:     class Empty:
14:         pass
15: 
16: 
17:     class Sample:
18:         def __iter__(self):
19:             return iter(list())
20: 
21: 
22:     class Wrong1:
23:         def __iter__(self):
24:             return 3
25: 
26: 
27:     class Wrong2:
28:         def __iter__(self, x):
29:             return "str"
30: 
31: 
32:     # Call the builtin with correct parameters
33:     # No error
34:     ret = enumerate(Sample())
35:     # No error
36:     ret = enumerate(Sample(), 3)
37: 
38:     # Call the builtin with incorrect types of parameters
39: 
40:     # Type error
41:     ret = enumerate(Wrong1())
42:     # Type error
43:     ret = enumerate(Wrong2())
44:     # Type error
45:     ret = enumerate(Empty())
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'enumerate builtin is invoked, but classes and instances with special name methods are passed')
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
        def __iter__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__iter__'
            module_type_store = module_type_store.open_function_context('__iter__', 18, 8, False)
            # Assigning a type to the variable 'self' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__iter__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__iter__.__dict__.__setitem__('stypy_function_name', 'Sample.__iter__')
            Sample.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__iter__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__iter__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__iter__(...)' code ##################

            
            # Call to iter(...): (line 19)
            # Processing the call arguments (line 19)
            
            # Call to list(...): (line 19)
            # Processing the call keyword arguments (line 19)
            kwargs_4 = {}
            # Getting the type of 'list' (line 19)
            list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'list', False)
            # Calling list(args, kwargs) (line 19)
            list_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 19, 24), list_3, *[], **kwargs_4)
            
            # Processing the call keyword arguments (line 19)
            kwargs_6 = {}
            # Getting the type of 'iter' (line 19)
            iter_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'iter', False)
            # Calling iter(args, kwargs) (line 19)
            iter_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), iter_2, *[list_call_result_5], **kwargs_6)
            
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type', iter_call_result_7)
            
            # ################# End of '__iter__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__iter__' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_8)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__iter__'
            return stypy_return_type_8


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
        def __iter__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__iter__'
            module_type_store = module_type_store.open_function_context('__iter__', 23, 8, False)
            # Assigning a type to the variable 'self' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__iter__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__iter__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__iter__')
            Wrong1.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong1.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__iter__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__iter__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__iter__(...)' code ##################

            int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'stypy_return_type', int_9)
            
            # ################# End of '__iter__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__iter__' in the type store
            # Getting the type of 'stypy_return_type' (line 23)
            stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_10)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__iter__'
            return stypy_return_type_10


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
        def __iter__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__iter__'
            module_type_store = module_type_store.open_function_context('__iter__', 28, 8, False)
            # Assigning a type to the variable 'self' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__iter__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__iter__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__iter__')
            Wrong2.__iter__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong2.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__iter__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__iter__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__iter__(...)' code ##################

            str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'stypy_return_type', str_11)
            
            # ################# End of '__iter__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__iter__' in the type store
            # Getting the type of 'stypy_return_type' (line 28)
            stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_12)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__iter__'
            return stypy_return_type_12


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
    
    # Call to enumerate(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to Sample(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_15 = {}
    # Getting the type of 'Sample' (line 34)
    Sample_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'Sample', False)
    # Calling Sample(args, kwargs) (line 34)
    Sample_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 34, 20), Sample_14, *[], **kwargs_15)
    
    # Processing the call keyword arguments (line 34)
    kwargs_17 = {}
    # Getting the type of 'enumerate' (line 34)
    enumerate_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 34)
    enumerate_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), enumerate_13, *[Sample_call_result_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'ret', enumerate_call_result_18)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to enumerate(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to Sample(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_21 = {}
    # Getting the type of 'Sample' (line 36)
    Sample_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'Sample', False)
    # Calling Sample(args, kwargs) (line 36)
    Sample_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 36, 20), Sample_20, *[], **kwargs_21)
    
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_24 = {}
    # Getting the type of 'enumerate' (line 36)
    enumerate_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 36)
    enumerate_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), enumerate_19, *[Sample_call_result_22, int_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'ret', enumerate_call_result_25)
    
    # Assigning a Call to a Name (line 41):
    
    # Call to enumerate(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to Wrong1(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_28 = {}
    # Getting the type of 'Wrong1' (line 41)
    Wrong1_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 41)
    Wrong1_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 41, 20), Wrong1_27, *[], **kwargs_28)
    
    # Processing the call keyword arguments (line 41)
    kwargs_30 = {}
    # Getting the type of 'enumerate' (line 41)
    enumerate_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 41)
    enumerate_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 41, 10), enumerate_26, *[Wrong1_call_result_29], **kwargs_30)
    
    # Assigning a type to the variable 'ret' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'ret', enumerate_call_result_31)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to enumerate(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Call to Wrong2(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_34 = {}
    # Getting the type of 'Wrong2' (line 43)
    Wrong2_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 43)
    Wrong2_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), Wrong2_33, *[], **kwargs_34)
    
    # Processing the call keyword arguments (line 43)
    kwargs_36 = {}
    # Getting the type of 'enumerate' (line 43)
    enumerate_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 43)
    enumerate_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), enumerate_32, *[Wrong2_call_result_35], **kwargs_36)
    
    # Assigning a type to the variable 'ret' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'ret', enumerate_call_result_37)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to enumerate(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to Empty(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_40 = {}
    # Getting the type of 'Empty' (line 45)
    Empty_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'Empty', False)
    # Calling Empty(args, kwargs) (line 45)
    Empty_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 45, 20), Empty_39, *[], **kwargs_40)
    
    # Processing the call keyword arguments (line 45)
    kwargs_42 = {}
    # Getting the type of 'enumerate' (line 45)
    enumerate_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 45)
    enumerate_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 45, 10), enumerate_38, *[Empty_call_result_41], **kwargs_42)
    
    # Assigning a type to the variable 'ret' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'ret', enumerate_call_result_43)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

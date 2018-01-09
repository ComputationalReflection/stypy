
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "len builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'int'>
7:     # (Str) -> <type 'int'>
8:     # (Has__len__) -> <type 'int'>
9: 
10:     class Empty:
11:         pass
12: 
13: 
14:     class Sample:
15:         def __len__(self):
16:             return 4
17: 
18: 
19:     class Wrong1:
20:         def __len__(self, x):
21:             return x
22: 
23: 
24:     class Wrong2:
25:         def __len__(self):
26:             return "str"
27: 
28: 
29:     # Call the builtin with correct parameters
30:     # No error
31:     ret = len(Sample())
32: 
33:     # Call the builtin with incorrect types of parameters
34:     # Type error
35:     ret = len(Wrong1())
36:     # Type error
37:     ret = len(Wrong2())
38:     # Type error
39:     ret = len(Empty())
40: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'len builtin is invoked, but classes and instances with special name methods are passed ')
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
            module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __len__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__len__'
            module_type_store = module_type_store.open_function_context('__len__', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__len__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__len__.__dict__.__setitem__('stypy_function_name', 'Sample.__len__')
            Sample.__len__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__len__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__len__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__len__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__len__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__len__' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__len__'
            return stypy_return_type_3


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
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __len__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__len__'
            module_type_store = module_type_store.open_function_context('__len__', 20, 8, False)
            # Assigning a type to the variable 'self' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__len__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__len__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__len__')
            Wrong1.__len__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__len__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__len__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__len__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__len__(...)' code ##################

            # Getting the type of 'x' (line 21)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__len__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__len__' in the type store
            # Getting the type of 'stypy_return_type' (line 20)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__len__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
            # Assigning a type to the variable 'self' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __len__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__len__'
            module_type_store = module_type_store.open_function_context('__len__', 25, 8, False)
            # Assigning a type to the variable 'self' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__len__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__len__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__len__')
            Wrong2.__len__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__len__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__len__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__len__(...)' code ##################

            str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'stypy_return_type', str_6)
            
            # ################# End of '__len__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__len__' in the type store
            # Getting the type of 'stypy_return_type' (line 25)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__len__'
            return stypy_return_type_7


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 24, 4, False)
            # Assigning a type to the variable 'self' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to len(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to Sample(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_10 = {}
    # Getting the type of 'Sample' (line 31)
    Sample_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'Sample', False)
    # Calling Sample(args, kwargs) (line 31)
    Sample_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), Sample_9, *[], **kwargs_10)
    
    # Processing the call keyword arguments (line 31)
    kwargs_12 = {}
    # Getting the type of 'len' (line 31)
    len_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'len', False)
    # Calling len(args, kwargs) (line 31)
    len_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), len_8, *[Sample_call_result_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ret', len_call_result_13)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to len(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to Wrong1(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_16 = {}
    # Getting the type of 'Wrong1' (line 35)
    Wrong1_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 35)
    Wrong1_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), Wrong1_15, *[], **kwargs_16)
    
    # Processing the call keyword arguments (line 35)
    kwargs_18 = {}
    # Getting the type of 'len' (line 35)
    len_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 10), 'len', False)
    # Calling len(args, kwargs) (line 35)
    len_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 35, 10), len_14, *[Wrong1_call_result_17], **kwargs_18)
    
    # Assigning a type to the variable 'ret' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'ret', len_call_result_19)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to len(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to Wrong2(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_22 = {}
    # Getting the type of 'Wrong2' (line 37)
    Wrong2_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 37)
    Wrong2_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), Wrong2_21, *[], **kwargs_22)
    
    # Processing the call keyword arguments (line 37)
    kwargs_24 = {}
    # Getting the type of 'len' (line 37)
    len_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'len', False)
    # Calling len(args, kwargs) (line 37)
    len_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), len_20, *[Wrong2_call_result_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'ret', len_call_result_25)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to len(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to Empty(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_28 = {}
    # Getting the type of 'Empty' (line 39)
    Empty_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'Empty', False)
    # Calling Empty(args, kwargs) (line 39)
    Empty_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), Empty_27, *[], **kwargs_28)
    
    # Processing the call keyword arguments (line 39)
    kwargs_30 = {}
    # Getting the type of 'len' (line 39)
    len_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'len', False)
    # Calling len(args, kwargs) (line 39)
    len_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), len_26, *[Empty_call_result_29], **kwargs_30)
    
    # Assigning a type to the variable 'ret' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'ret', len_call_result_31)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

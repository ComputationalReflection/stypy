
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "unichr builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <type 'unicode'>
7:     # (Overloads__trunc__) -> <type 'unicode'>
8:     class Empty:
9:         pass
10: 
11: 
12:     class Sample:
13:         def __trunc__(self):
14:             return 4
15: 
16: 
17:     class Wrong1:
18:         def __trunc__(self, x):
19:             return x
20: 
21: 
22:     class Wrong2:
23:         def __trunc__(self):
24:             return "str"
25: 
26: 
27:     # Call the builtin with correct parameters
28:     ret = unichr(Sample())
29: 
30:     # Call the builtin with incorrect types of parameters
31:     # Type error
32:     ret = unichr(Wrong1())
33:     # Type error
34:     ret = unichr(Wrong2())
35:     # Type error
36:     ret = unichr(Empty())
37: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'unichr builtin is invoked, but classes and instances with special name methods are passed ')
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
            module_type_store = module_type_store.open_function_context('__init__', 8, 4, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 13, 8, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__trunc__.__dict__.__setitem__('stypy_function_name', 'Sample.__trunc__')
            Sample.__trunc__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__trunc__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 13)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 12, 4, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 18, 8, False)
            # Assigning a type to the variable 'self' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__trunc__')
            Wrong1.__trunc__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__trunc__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            # Getting the type of 'x' (line 19)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_5


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

    
    # Assigning a type to the variable 'Wrong1' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 23, 8, False)
            # Assigning a type to the variable 'self' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__trunc__')
            Wrong2.__trunc__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__trunc__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'stypy_return_type', str_6)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 23)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_7


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

    
    # Assigning a type to the variable 'Wrong2' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to unichr(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to Sample(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_10 = {}
    # Getting the type of 'Sample' (line 28)
    Sample_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'Sample', False)
    # Calling Sample(args, kwargs) (line 28)
    Sample_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), Sample_9, *[], **kwargs_10)
    
    # Processing the call keyword arguments (line 28)
    kwargs_12 = {}
    # Getting the type of 'unichr' (line 28)
    unichr_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'unichr', False)
    # Calling unichr(args, kwargs) (line 28)
    unichr_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), unichr_8, *[Sample_call_result_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ret', unichr_call_result_13)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to unichr(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to Wrong1(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_16 = {}
    # Getting the type of 'Wrong1' (line 32)
    Wrong1_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 32)
    Wrong1_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), Wrong1_15, *[], **kwargs_16)
    
    # Processing the call keyword arguments (line 32)
    kwargs_18 = {}
    # Getting the type of 'unichr' (line 32)
    unichr_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'unichr', False)
    # Calling unichr(args, kwargs) (line 32)
    unichr_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), unichr_14, *[Wrong1_call_result_17], **kwargs_18)
    
    # Assigning a type to the variable 'ret' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'ret', unichr_call_result_19)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to unichr(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to Wrong2(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_22 = {}
    # Getting the type of 'Wrong2' (line 34)
    Wrong2_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 34)
    Wrong2_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), Wrong2_21, *[], **kwargs_22)
    
    # Processing the call keyword arguments (line 34)
    kwargs_24 = {}
    # Getting the type of 'unichr' (line 34)
    unichr_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'unichr', False)
    # Calling unichr(args, kwargs) (line 34)
    unichr_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), unichr_20, *[Wrong2_call_result_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'ret', unichr_call_result_25)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to unichr(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to Empty(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_28 = {}
    # Getting the type of 'Empty' (line 36)
    Empty_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'Empty', False)
    # Calling Empty(args, kwargs) (line 36)
    Empty_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), Empty_27, *[], **kwargs_28)
    
    # Processing the call keyword arguments (line 36)
    kwargs_30 = {}
    # Getting the type of 'unichr' (line 36)
    unichr_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'unichr', False)
    # Calling unichr(args, kwargs) (line 36)
    unichr_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), unichr_26, *[Empty_call_result_29], **kwargs_30)
    
    # Assigning a type to the variable 'ret' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'ret', unichr_call_result_31)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

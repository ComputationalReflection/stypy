
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "No special __complex__ method is defined"
4: 
5: if __name__ == '__main__':
6:     class DefinesMethod:
7:         def __complex__(self):
8:             return 1 + 2j
9: 
10: 
11:     class DefinesFloatMethod:
12:         def __float__(self):
13:             return 1.2
14: 
15: 
16:     class DefinesComplexReturnFloatMethod:
17:         def __complex__(self):
18:             return 1.2
19: 
20: 
21:     class Empty:
22:         pass
23: 
24: 
25:     print complex(DefinesMethod())
26: 
27:     print complex(DefinesFloatMethod())
28: 
29:     print complex(DefinesComplexReturnFloatMethod())
30: 
31:     # Type error #
32:     print complex(Empty())
33: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'No special __complex__ method is defined')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DefinesMethod' class

    class DefinesMethod:

        @norecursion
        def __complex__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__complex__'
            module_type_store = module_type_store.open_function_context('__complex__', 7, 8, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_localization', localization)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_function_name', 'DefinesMethod.__complex__')
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_param_names_list', [])
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesMethod.__complex__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__complex__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__complex__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__complex__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'int')
            complex_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 23), 'complex')
            # Applying the binary operator '+' (line 8)
            result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 19), '+', int_2, complex_3)
            
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', result_add_4)
            
            # ################# End of '__complex__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__complex__' in the type store
            # Getting the type of 'stypy_return_type' (line 7)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__complex__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 6, 4, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'DefinesMethod' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'DefinesMethod', DefinesMethod)
    # Declaration of the 'DefinesFloatMethod' class

    class DefinesFloatMethod:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_localization', localization)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_function_name', 'DefinesFloatMethod.__float__')
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_param_names_list', [])
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesFloatMethod.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesFloatMethod.__float__', [], None, None, defaults, varargs, kwargs)

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

            float_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', float_6)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 12)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 11, 4, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesFloatMethod.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'DefinesFloatMethod' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'DefinesFloatMethod', DefinesFloatMethod)
    # Declaration of the 'DefinesComplexReturnFloatMethod' class

    class DefinesComplexReturnFloatMethod:

        @norecursion
        def __complex__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__complex__'
            module_type_store = module_type_store.open_function_context('__complex__', 17, 8, False)
            # Assigning a type to the variable 'self' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_localization', localization)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_function_name', 'DefinesComplexReturnFloatMethod.__complex__')
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_param_names_list', [])
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesComplexReturnFloatMethod.__complex__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesComplexReturnFloatMethod.__complex__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__complex__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__complex__(...)' code ##################

            float_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'stypy_return_type', float_8)
            
            # ################# End of '__complex__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__complex__' in the type store
            # Getting the type of 'stypy_return_type' (line 17)
            stypy_return_type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_9)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__complex__'
            return stypy_return_type_9


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesComplexReturnFloatMethod.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'DefinesComplexReturnFloatMethod' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'DefinesComplexReturnFloatMethod', DefinesComplexReturnFloatMethod)
    # Declaration of the 'Empty' class

    class Empty:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 21, 4, False)
            # Assigning a type to the variable 'self' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'Empty', Empty)
    
    # Call to complex(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to DefinesMethod(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_12 = {}
    # Getting the type of 'DefinesMethod' (line 25)
    DefinesMethod_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'DefinesMethod', False)
    # Calling DefinesMethod(args, kwargs) (line 25)
    DefinesMethod_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), DefinesMethod_11, *[], **kwargs_12)
    
    # Processing the call keyword arguments (line 25)
    kwargs_14 = {}
    # Getting the type of 'complex' (line 25)
    complex_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 25)
    complex_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), complex_10, *[DefinesMethod_call_result_13], **kwargs_14)
    
    
    # Call to complex(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to DefinesFloatMethod(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_18 = {}
    # Getting the type of 'DefinesFloatMethod' (line 27)
    DefinesFloatMethod_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'DefinesFloatMethod', False)
    # Calling DefinesFloatMethod(args, kwargs) (line 27)
    DefinesFloatMethod_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 27, 18), DefinesFloatMethod_17, *[], **kwargs_18)
    
    # Processing the call keyword arguments (line 27)
    kwargs_20 = {}
    # Getting the type of 'complex' (line 27)
    complex_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 27)
    complex_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), complex_16, *[DefinesFloatMethod_call_result_19], **kwargs_20)
    
    
    # Call to complex(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Call to DefinesComplexReturnFloatMethod(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_24 = {}
    # Getting the type of 'DefinesComplexReturnFloatMethod' (line 29)
    DefinesComplexReturnFloatMethod_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'DefinesComplexReturnFloatMethod', False)
    # Calling DefinesComplexReturnFloatMethod(args, kwargs) (line 29)
    DefinesComplexReturnFloatMethod_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), DefinesComplexReturnFloatMethod_23, *[], **kwargs_24)
    
    # Processing the call keyword arguments (line 29)
    kwargs_26 = {}
    # Getting the type of 'complex' (line 29)
    complex_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 29)
    complex_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), complex_22, *[DefinesComplexReturnFloatMethod_call_result_25], **kwargs_26)
    
    
    # Call to complex(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to Empty(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_30 = {}
    # Getting the type of 'Empty' (line 32)
    Empty_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'Empty', False)
    # Calling Empty(args, kwargs) (line 32)
    Empty_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 32, 18), Empty_29, *[], **kwargs_30)
    
    # Processing the call keyword arguments (line 32)
    kwargs_32 = {}
    # Getting the type of 'complex' (line 32)
    complex_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 32)
    complex_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), complex_28, *[Empty_call_result_31], **kwargs_32)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

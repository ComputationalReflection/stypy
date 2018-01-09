
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "iter builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> DynamicType
7:     # (IterableObject) -> DynamicType
8:     # (IterableObject, AnyType) -> DynamicType
9:     # (Has__call__, AnyType) -> DynamicType
10: 
11:     class Empty:
12:         pass
13: 
14: 
15:     class Sample:
16:         def __call__(self):
17:             return 4
18: 
19: 
20:     class Wrong1:
21:         def __call__(self, x):
22:             return x
23: 
24: 
25:     # Call the builtin with correct parameters
26:     # No error
27:     ret = iter(Sample(), 4)
28:     # No error
29:     ret = iter(Wrong1(), 0)
30: 
31:     # Call the builtin with incorrect types of parameters
32: 
33:     # Type error
34:     ret = iter(Empty(), 0)
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'iter builtin is invoked, but classes and instances with special name methods are passed ')
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
            module_type_store = module_type_store.open_function_context('__init__', 11, 4, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __call__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__call__'
            module_type_store = module_type_store.open_function_context('__call__', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__call__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__call__.__dict__.__setitem__('stypy_function_name', 'Sample.__call__')
            Sample.__call__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__call__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__call__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__call__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__call__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__call__' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__call__'
            return stypy_return_type_3


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

    
    # Assigning a type to the variable 'Sample' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __call__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__call__'
            module_type_store = module_type_store.open_function_context('__call__', 21, 8, False)
            # Assigning a type to the variable 'self' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__call__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__call__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__call__')
            Wrong1.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__call__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__call__(...)' code ##################

            # Getting the type of 'x' (line 22)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__call__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__call__' in the type store
            # Getting the type of 'stypy_return_type' (line 21)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__call__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
            # Assigning a type to the variable 'self' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'Wrong1', Wrong1)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to iter(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to Sample(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_8 = {}
    # Getting the type of 'Sample' (line 27)
    Sample_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'Sample', False)
    # Calling Sample(args, kwargs) (line 27)
    Sample_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), Sample_7, *[], **kwargs_8)
    
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_11 = {}
    # Getting the type of 'iter' (line 27)
    iter_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 27)
    iter_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), iter_6, *[Sample_call_result_9, int_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', iter_call_result_12)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to iter(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Call to Wrong1(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_15 = {}
    # Getting the type of 'Wrong1' (line 29)
    Wrong1_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 29)
    Wrong1_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), Wrong1_14, *[], **kwargs_15)
    
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_18 = {}
    # Getting the type of 'iter' (line 29)
    iter_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 29)
    iter_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), iter_13, *[Wrong1_call_result_16, int_17], **kwargs_18)
    
    # Assigning a type to the variable 'ret' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'ret', iter_call_result_19)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to iter(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to Empty(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_22 = {}
    # Getting the type of 'Empty' (line 34)
    Empty_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'Empty', False)
    # Calling Empty(args, kwargs) (line 34)
    Empty_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), Empty_21, *[], **kwargs_22)
    
    int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_25 = {}
    # Getting the type of 'iter' (line 34)
    iter_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 34)
    iter_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), iter_20, *[Empty_call_result_23, int_24], **kwargs_25)
    
    # Assigning a type to the variable 'ret' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'ret', iter_call_result_26)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

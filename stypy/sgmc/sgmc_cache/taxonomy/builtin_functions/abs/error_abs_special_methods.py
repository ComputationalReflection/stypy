
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "abs builtin is invoked, but classes and instances with special name methods are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type bool>) -> <type 'int'>
7:     # (<type complex>) -> <type 'float'>
8:     # (Number) -> TypeOfParam(1)
9:     # (Overloads__abs__) -> <type 'int'>
10: 
11:     class Empty:
12:         pass
13: 
14: 
15:     class Sample:
16:         def __abs__(self):
17:             return 4
18: 
19: 
20:     class Wrong1:
21:         def __abs__(self, x):
22:             return x
23: 
24: 
25:     # Call the builtin with correct parameters
26:     # No error
27:     ret = abs(Sample())
28: 
29:     # Call the builtin with incorrect parameters
30:     # Type error
31:     ret = abs(Wrong1())
32:     # Type error
33:     ret = abs(Empty())
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'abs builtin is invoked, but classes and instances with special name methods are passed')
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
        def __abs__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__abs__'
            module_type_store = module_type_store.open_function_context('__abs__', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__abs__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__abs__.__dict__.__setitem__('stypy_function_name', 'Sample.__abs__')
            Sample.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__abs__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__abs__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__abs__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__abs__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__abs__' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__abs__'
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
        def __abs__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__abs__'
            module_type_store = module_type_store.open_function_context('__abs__', 21, 8, False)
            # Assigning a type to the variable 'self' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__abs__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__abs__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__abs__')
            Wrong1.__abs__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__abs__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__abs__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__abs__(...)' code ##################

            # Getting the type of 'x' (line 22)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__abs__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__abs__' in the type store
            # Getting the type of 'stypy_return_type' (line 21)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__abs__'
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
    
    # Call to abs(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to Sample(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_8 = {}
    # Getting the type of 'Sample' (line 27)
    Sample_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'Sample', False)
    # Calling Sample(args, kwargs) (line 27)
    Sample_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 27, 14), Sample_7, *[], **kwargs_8)
    
    # Processing the call keyword arguments (line 27)
    kwargs_10 = {}
    # Getting the type of 'abs' (line 27)
    abs_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 27)
    abs_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), abs_6, *[Sample_call_result_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', abs_call_result_11)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to abs(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to Wrong1(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_14 = {}
    # Getting the type of 'Wrong1' (line 31)
    Wrong1_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 31)
    Wrong1_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), Wrong1_13, *[], **kwargs_14)
    
    # Processing the call keyword arguments (line 31)
    kwargs_16 = {}
    # Getting the type of 'abs' (line 31)
    abs_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 31)
    abs_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), abs_12, *[Wrong1_call_result_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ret', abs_call_result_17)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to abs(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to Empty(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_20 = {}
    # Getting the type of 'Empty' (line 33)
    Empty_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'Empty', False)
    # Calling Empty(args, kwargs) (line 33)
    Empty_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 33, 14), Empty_19, *[], **kwargs_20)
    
    # Processing the call keyword arguments (line 33)
    kwargs_22 = {}
    # Getting the type of 'abs' (line 33)
    abs_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 33)
    abs_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), abs_18, *[Empty_call_result_21], **kwargs_22)
    
    # Assigning a type to the variable 'ret' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'ret', abs_call_result_23)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "bytearray builtin is invoked, but classes and instances with special name methods are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'bytearray'>
7:     # (IterableDataStructureWithTypedElements(Integer, Overloads__trunc__)) -> <type 'bytearray'>
8:     # (Integer) -> <type 'bytearray'>
9:     # (Str) -> <type 'bytearray'>
10: 
11:     class Empty:
12:         pass
13: 
14: 
15:     class Sample:
16:         def __trunc__(self):
17:             return 4
18: 
19: 
20:     # Call the builtin with correct parameters
21:     list_trunc = [Sample(), Sample()]
22:     # No error
23:     ret = bytearray(list_trunc)
24: 
25:     # Call the builtin with incorrect types of parameters
26:     # Type error
27:     ret = bytearray(list())
28: 
29: 
30:     class Wrong1:
31:         def __trunc__(self, x):
32:             return x
33: 
34: 
35:     list_trunc2 = [Wrong1(), Wrong1()]
36:     # Type error
37:     ret = bytearray(list_trunc2)
38: 
39: 
40:     class Wrong2:
41:         def __trunc__(self):
42:             return "str"
43: 
44: 
45:     list_trunc3 = [Wrong2(), Wrong2()]
46:     # Type error
47:     ret = bytearray(list_trunc3)
48:     # Type error
49:     ret = bytearray([Empty()])
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'bytearray builtin is invoked, but classes and instances with special name methods are passed')
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
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
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

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type')
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
    
    # Assigning a List to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Call to Sample(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 21)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 21)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), Sample_5, *[], **kwargs_6)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 17), list_4, Sample_call_result_7)
    # Adding element type (line 21)
    
    # Call to Sample(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_9 = {}
    # Getting the type of 'Sample' (line 21)
    Sample_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'Sample', False)
    # Calling Sample(args, kwargs) (line 21)
    Sample_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 21, 28), Sample_8, *[], **kwargs_9)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 17), list_4, Sample_call_result_10)
    
    # Assigning a type to the variable 'list_trunc' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'list_trunc', list_4)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to bytearray(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'list_trunc' (line 23)
    list_trunc_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'list_trunc', False)
    # Processing the call keyword arguments (line 23)
    kwargs_13 = {}
    # Getting the type of 'bytearray' (line 23)
    bytearray_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 23)
    bytearray_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), bytearray_11, *[list_trunc_12], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', bytearray_call_result_14)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to bytearray(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to list(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_17 = {}
    # Getting the type of 'list' (line 27)
    list_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'list', False)
    # Calling list(args, kwargs) (line 27)
    list_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 27, 20), list_16, *[], **kwargs_17)
    
    # Processing the call keyword arguments (line 27)
    kwargs_19 = {}
    # Getting the type of 'bytearray' (line 27)
    bytearray_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 27)
    bytearray_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), bytearray_15, *[list_call_result_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', bytearray_call_result_20)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 31, 8, False)
            # Assigning a type to the variable 'self' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self', type_of_self)
            
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

            # Getting the type of 'x' (line 32)
            x_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', x_21)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 31)
            stypy_return_type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_22)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_22


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 30, 4, False)
            # Assigning a type to the variable 'self' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'Wrong1', Wrong1)
    
    # Assigning a List to a Name (line 35):
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    
    # Call to Wrong1(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_25 = {}
    # Getting the type of 'Wrong1' (line 35)
    Wrong1_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 35)
    Wrong1_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), Wrong1_24, *[], **kwargs_25)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), list_23, Wrong1_call_result_26)
    # Adding element type (line 35)
    
    # Call to Wrong1(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_28 = {}
    # Getting the type of 'Wrong1' (line 35)
    Wrong1_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 35)
    Wrong1_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 35, 29), Wrong1_27, *[], **kwargs_28)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), list_23, Wrong1_call_result_29)
    
    # Assigning a type to the variable 'list_trunc2' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'list_trunc2', list_23)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to bytearray(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'list_trunc2' (line 37)
    list_trunc2_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'list_trunc2', False)
    # Processing the call keyword arguments (line 37)
    kwargs_32 = {}
    # Getting the type of 'bytearray' (line 37)
    bytearray_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 37)
    bytearray_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), bytearray_30, *[list_trunc2_31], **kwargs_32)
    
    # Assigning a type to the variable 'ret' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'ret', bytearray_call_result_33)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 41, 8, False)
            # Assigning a type to the variable 'self' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', type_of_self)
            
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

            str_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'stypy_return_type', str_34)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 41)
            stypy_return_type_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_35)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_35


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
            # Assigning a type to the variable 'self' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'Wrong2', Wrong2)
    
    # Assigning a List to a Name (line 45):
    
    # Obtaining an instance of the builtin type 'list' (line 45)
    list_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 45)
    # Adding element type (line 45)
    
    # Call to Wrong2(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_38 = {}
    # Getting the type of 'Wrong2' (line 45)
    Wrong2_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 45)
    Wrong2_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), Wrong2_37, *[], **kwargs_38)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), list_36, Wrong2_call_result_39)
    # Adding element type (line 45)
    
    # Call to Wrong2(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_41 = {}
    # Getting the type of 'Wrong2' (line 45)
    Wrong2_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 45)
    Wrong2_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 45, 29), Wrong2_40, *[], **kwargs_41)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), list_36, Wrong2_call_result_42)
    
    # Assigning a type to the variable 'list_trunc3' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'list_trunc3', list_36)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to bytearray(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'list_trunc3' (line 47)
    list_trunc3_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'list_trunc3', False)
    # Processing the call keyword arguments (line 47)
    kwargs_45 = {}
    # Getting the type of 'bytearray' (line 47)
    bytearray_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 47)
    bytearray_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 47, 10), bytearray_43, *[list_trunc3_44], **kwargs_45)
    
    # Assigning a type to the variable 'ret' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'ret', bytearray_call_result_46)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to bytearray(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    
    # Call to Empty(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_50 = {}
    # Getting the type of 'Empty' (line 49)
    Empty_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'Empty', False)
    # Calling Empty(args, kwargs) (line 49)
    Empty_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), Empty_49, *[], **kwargs_50)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), list_48, Empty_call_result_51)
    
    # Processing the call keyword arguments (line 49)
    kwargs_52 = {}
    # Getting the type of 'bytearray' (line 49)
    bytearray_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 49)
    bytearray_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), bytearray_47, *[list_48], **kwargs_52)
    
    # Assigning a type to the variable 'ret' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'ret', bytearray_call_result_53)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

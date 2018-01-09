
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "range builtin is invoked, but classes and instances with special name methods are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <built-in function range>
7:     # (Overloads__trunc__) -> <built-in function range>
8:     # (Integer, Integer) -> <built-in function range>
9:     # (Overloads__trunc__, Integer) -> <built-in function range>
10:     # (Integer, Overloads__trunc__) -> <built-in function range>
11:     # (Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
12:     # (Integer, Integer, Integer) -> <built-in function range>
13:     # (Overloads__trunc__, Integer, Integer) -> <built-in function range>
14:     # (Integer, Overloads__trunc__, Integer) -> <built-in function range>
15:     # (Integer, Integer, Overloads__trunc__) -> <built-in function range>
16:     # (Integer, Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
17:     # (Overloads__trunc__, Overloads__trunc__, Integer) -> <built-in function range>
18:     # (Overloads__trunc__, Integer, Overloads__trunc__) -> <built-in function range>
19:     # (Overloads__trunc__, Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
20: 
21: 
22:     class Empty:
23:         pass
24: 
25: 
26:     class Sample:
27:         def __trunc__(self):
28:             return 4
29: 
30: 
31:     class Wrong1:
32:         def __trunc__(self, x):
33:             return x
34: 
35: 
36:     class Wrong2:
37:         def __trunc__(self):
38:             return "str"
39: 
40: 
41:     # Call the builtin with correct parameters
42:     ret = xrange(Sample(), Sample())
43:     ret = xrange(Sample(), Sample(), 4)
44: 
45:     # Call the builtin with incorrect types of parameters
46:     # Type error
47:     ret = xrange(Wrong2(), Sample())
48:     # Type error
49:     ret = xrange(Wrong1(), Sample())
50:     # Type error
51:     ret = xrange(Empty(), Empty())
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'range builtin is invoked, but classes and instances with special name methods are passed')
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
            module_type_store = module_type_store.open_function_context('__init__', 22, 4, False)
            # Assigning a type to the variable 'self' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 27, 8, False)
            # Assigning a type to the variable 'self' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', type_of_self)
            
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

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 27)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 26, 4, False)
            # Assigning a type to the variable 'self' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 32, 8, False)
            # Assigning a type to the variable 'self' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', type_of_self)
            
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

            # Getting the type of 'x' (line 33)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 32)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 31, 4, False)
            # Assigning a type to the variable 'self' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 37, 8, False)
            # Assigning a type to the variable 'self' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', type_of_self)
            
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

            str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'stypy_return_type', str_6)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 37)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 36, 4, False)
            # Assigning a type to the variable 'self' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 42):
    
    # Call to xrange(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to Sample(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_10 = {}
    # Getting the type of 'Sample' (line 42)
    Sample_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'Sample', False)
    # Calling Sample(args, kwargs) (line 42)
    Sample_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 42, 17), Sample_9, *[], **kwargs_10)
    
    
    # Call to Sample(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 42)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 42)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), Sample_12, *[], **kwargs_13)
    
    # Processing the call keyword arguments (line 42)
    kwargs_15 = {}
    # Getting the type of 'xrange' (line 42)
    xrange_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 42)
    xrange_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), xrange_8, *[Sample_call_result_11, Sample_call_result_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'ret', xrange_call_result_16)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to xrange(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Call to Sample(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_19 = {}
    # Getting the type of 'Sample' (line 43)
    Sample_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'Sample', False)
    # Calling Sample(args, kwargs) (line 43)
    Sample_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), Sample_18, *[], **kwargs_19)
    
    
    # Call to Sample(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_22 = {}
    # Getting the type of 'Sample' (line 43)
    Sample_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 43)
    Sample_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 43, 27), Sample_21, *[], **kwargs_22)
    
    int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 37), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_25 = {}
    # Getting the type of 'xrange' (line 43)
    xrange_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 43)
    xrange_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), xrange_17, *[Sample_call_result_20, Sample_call_result_23, int_24], **kwargs_25)
    
    # Assigning a type to the variable 'ret' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'ret', xrange_call_result_26)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to xrange(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to Wrong2(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_29 = {}
    # Getting the type of 'Wrong2' (line 47)
    Wrong2_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 47)
    Wrong2_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), Wrong2_28, *[], **kwargs_29)
    
    
    # Call to Sample(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_32 = {}
    # Getting the type of 'Sample' (line 47)
    Sample_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 47)
    Sample_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 47, 27), Sample_31, *[], **kwargs_32)
    
    # Processing the call keyword arguments (line 47)
    kwargs_34 = {}
    # Getting the type of 'xrange' (line 47)
    xrange_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 47)
    xrange_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 47, 10), xrange_27, *[Wrong2_call_result_30, Sample_call_result_33], **kwargs_34)
    
    # Assigning a type to the variable 'ret' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'ret', xrange_call_result_35)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to xrange(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Call to Wrong1(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_38 = {}
    # Getting the type of 'Wrong1' (line 49)
    Wrong1_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 49)
    Wrong1_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), Wrong1_37, *[], **kwargs_38)
    
    
    # Call to Sample(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_41 = {}
    # Getting the type of 'Sample' (line 49)
    Sample_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 49)
    Sample_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 49, 27), Sample_40, *[], **kwargs_41)
    
    # Processing the call keyword arguments (line 49)
    kwargs_43 = {}
    # Getting the type of 'xrange' (line 49)
    xrange_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 49)
    xrange_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), xrange_36, *[Wrong1_call_result_39, Sample_call_result_42], **kwargs_43)
    
    # Assigning a type to the variable 'ret' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'ret', xrange_call_result_44)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to xrange(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to Empty(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_47 = {}
    # Getting the type of 'Empty' (line 51)
    Empty_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'Empty', False)
    # Calling Empty(args, kwargs) (line 51)
    Empty_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), Empty_46, *[], **kwargs_47)
    
    
    # Call to Empty(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_50 = {}
    # Getting the type of 'Empty' (line 51)
    Empty_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'Empty', False)
    # Calling Empty(args, kwargs) (line 51)
    Empty_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 51, 26), Empty_49, *[], **kwargs_50)
    
    # Processing the call keyword arguments (line 51)
    kwargs_52 = {}
    # Getting the type of 'xrange' (line 51)
    xrange_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 51)
    xrange_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), xrange_45, *[Empty_call_result_48, Empty_call_result_51], **kwargs_52)
    
    # Assigning a type to the variable 'ret' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'ret', xrange_call_result_53)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "xrange builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <type 'xrange'>
7:     # (Overloads__trunc__) -> <type 'xrange'>
8:     # (Integer, Integer) -> <type 'xrange'>
9:     # (Overloads__trunc__, Integer) -> <type 'xrange'>
10:     # (Integer, Overloads__trunc__) -> <type 'xrange'>
11:     # (Overloads__trunc__, Overloads__trunc__) -> <type 'xrange'>
12:     # (Integer, Integer, Integer) -> <type 'xrange'>
13:     # (Overloads__trunc__, Integer, Integer) -> <type 'xrange'>
14:     # (Integer, Overloads__trunc__, Integer) -> <type 'xrange'>
15:     # (Integer, Integer, Overloads__trunc__) -> <type 'xrange'>
16:     # (Integer, Overloads__trunc__, Overloads__trunc__) -> <type 'xrange'>
17:     # (Overloads__trunc__, Overloads__trunc__, Integer) -> <type 'xrange'>
18:     # (Overloads__trunc__, Integer, Overloads__trunc__) -> <type 'xrange'>
19:     # (Overloads__trunc__, Overloads__trunc__, Overloads__trunc__) -> <type 'xrange'>
20: 
21: 
22:     class Sample:
23:         def __trunc__(self):
24:             return 4
25: 
26: 
27:     class Wrong1:
28:         def __trunc__(self, x):
29:             return x
30: 
31: 
32:     class Wrong2:
33:         def __trunc__(self):
34:             return "str"
35: 
36: 
37:     # Call the builtin with correct parameters
38:     ret = xrange(3)
39:     ret = xrange(3, 6)
40:     ret = xrange(Sample(), Sample())
41:     ret = xrange(Sample(), Sample(), 4)
42: 
43:     # Call the builtin with incorrect types of parameters
44:     # Type error
45:     ret = xrange(Wrong2(), Sample())
46:     # Type error
47:     ret = xrange(Wrong1(), Sample())
48:     # Type error
49:     ret = xrange()
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'xrange builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

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

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 23)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 22, 4, False)
            # Assigning a type to the variable 'self' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 28, 8, False)
            # Assigning a type to the variable 'self' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', type_of_self)
            
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

            # Getting the type of 'x' (line 29)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 28)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 27, 4, False)
            # Assigning a type to the variable 'self' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 33, 8, False)
            # Assigning a type to the variable 'self' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', type_of_self)
            
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

            str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', str_6)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 33)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 32, 4, False)
            # Assigning a type to the variable 'self' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to xrange(...): (line 38)
    # Processing the call arguments (line 38)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_10 = {}
    # Getting the type of 'xrange' (line 38)
    xrange_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 38)
    xrange_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), xrange_8, *[int_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'ret', xrange_call_result_11)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to xrange(...): (line 39)
    # Processing the call arguments (line 39)
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'int')
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
    # Processing the call keyword arguments (line 39)
    kwargs_15 = {}
    # Getting the type of 'xrange' (line 39)
    xrange_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 39)
    xrange_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), xrange_12, *[int_13, int_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'ret', xrange_call_result_16)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to xrange(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to Sample(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_19 = {}
    # Getting the type of 'Sample' (line 40)
    Sample_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'Sample', False)
    # Calling Sample(args, kwargs) (line 40)
    Sample_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), Sample_18, *[], **kwargs_19)
    
    
    # Call to Sample(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_22 = {}
    # Getting the type of 'Sample' (line 40)
    Sample_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 40)
    Sample_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), Sample_21, *[], **kwargs_22)
    
    # Processing the call keyword arguments (line 40)
    kwargs_24 = {}
    # Getting the type of 'xrange' (line 40)
    xrange_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 40)
    xrange_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), xrange_17, *[Sample_call_result_20, Sample_call_result_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'ret', xrange_call_result_25)
    
    # Assigning a Call to a Name (line 41):
    
    # Call to xrange(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to Sample(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_28 = {}
    # Getting the type of 'Sample' (line 41)
    Sample_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'Sample', False)
    # Calling Sample(args, kwargs) (line 41)
    Sample_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), Sample_27, *[], **kwargs_28)
    
    
    # Call to Sample(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_31 = {}
    # Getting the type of 'Sample' (line 41)
    Sample_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 41)
    Sample_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), Sample_30, *[], **kwargs_31)
    
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 37), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_34 = {}
    # Getting the type of 'xrange' (line 41)
    xrange_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 41)
    xrange_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 41, 10), xrange_26, *[Sample_call_result_29, Sample_call_result_32, int_33], **kwargs_34)
    
    # Assigning a type to the variable 'ret' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'ret', xrange_call_result_35)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to xrange(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to Wrong2(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_38 = {}
    # Getting the type of 'Wrong2' (line 45)
    Wrong2_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 45)
    Wrong2_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), Wrong2_37, *[], **kwargs_38)
    
    
    # Call to Sample(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_41 = {}
    # Getting the type of 'Sample' (line 45)
    Sample_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 45)
    Sample_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), Sample_40, *[], **kwargs_41)
    
    # Processing the call keyword arguments (line 45)
    kwargs_43 = {}
    # Getting the type of 'xrange' (line 45)
    xrange_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 45)
    xrange_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 45, 10), xrange_36, *[Wrong2_call_result_39, Sample_call_result_42], **kwargs_43)
    
    # Assigning a type to the variable 'ret' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'ret', xrange_call_result_44)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to xrange(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to Wrong1(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_47 = {}
    # Getting the type of 'Wrong1' (line 47)
    Wrong1_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 47)
    Wrong1_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), Wrong1_46, *[], **kwargs_47)
    
    
    # Call to Sample(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_50 = {}
    # Getting the type of 'Sample' (line 47)
    Sample_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'Sample', False)
    # Calling Sample(args, kwargs) (line 47)
    Sample_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 47, 27), Sample_49, *[], **kwargs_50)
    
    # Processing the call keyword arguments (line 47)
    kwargs_52 = {}
    # Getting the type of 'xrange' (line 47)
    xrange_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 47)
    xrange_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 47, 10), xrange_45, *[Wrong1_call_result_48, Sample_call_result_51], **kwargs_52)
    
    # Assigning a type to the variable 'ret' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'ret', xrange_call_result_53)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to xrange(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_55 = {}
    # Getting the type of 'xrange' (line 49)
    xrange_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 49)
    xrange_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), xrange_54, *[], **kwargs_55)
    
    # Assigning a type to the variable 'ret' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'ret', xrange_call_result_56)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

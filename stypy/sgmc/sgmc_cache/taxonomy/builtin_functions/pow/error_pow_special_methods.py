
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "pow builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type int>, Number) -> TypeOfParam(2)
7:     # (<type bool>, <type bool>) -> <type 'int'>
8:     # (<type bool>, Number) -> TypeOfParam(2)
9:     # (<type complex>, Number) -> <type 'complex'>
10:     # (<type long>, Integer) -> <type 'long'>
11:     # (<type long>, <type complex>) -> <type 'complex'>
12:     # (<type long>, <type float>) -> <type 'float'>
13:     # (<type int>, <type bool>) -> <type 'int'>
14:     # (<type float>, RealNumber) -> <type 'float'>
15:     # (<type float>, <type complex>) -> <type 'complex'>
16:     # (<type bool>, <type bool>, <type bool>) -> <type 'int'>
17:     # (<type bool>, <type bool>, types.NoneType) -> <type 'int'>
18:     # (<type bool>, <type bool>, Integer) -> TypeOfParam(2)
19:     # (<type bool>, <type complex>, types.NoneType) -> <type 'complex'>
20:     # (<type bool>, <type long>, <type bool>) -> <type 'long'>
21:     # (<type bool>, <type long>, types.NoneType) -> <type 'long'>
22:     # (<type bool>, <type long>, Integer) -> TypeOfParam(1)
23:     # (<type bool>, <type int>, <type bool>) -> <type 'int'>
24:     # (<type bool>, <type int>, types.NoneType) -> <type 'int'>
25:     # (<type bool>, <type int>, Integer) -> TypeOfParam(2)
26:     # (<type bool>, <type float>, types.NoneType) -> <type 'float'>
27:     # (<type complex>, Number, types.NoneType) -> <type 'complex'>
28:     # (<type long>, <type bool>, Integer) -> <type 'long'>
29:     # (<type long>, <type bool>, types.NoneType) -> <type 'long'>
30:     # (<type long>, <type complex>, types.NoneType) -> <type 'complex'>
31:     # (<type long>, <type long>, Integer) -> <type 'long'>
32:     # (<type long>, <type long>, types.NoneType) -> <type 'long'>
33:     # (<type long>, <type int>, Integer) -> <type 'long'>
34:     # (<type long>, <type int>, types.NoneType) -> <type 'long'>
35:     # (<type long>, <type float>, types.NoneType) -> <type 'float'>
36:     # (<type int>, <type bool>, <type bool>) -> <type 'int'>
37:     # (<type int>, <type bool>, types.NoneType) -> <type 'int'>
38:     # (<type int>, <type bool>, Integer) -> TypeOfParam(3)
39:     # (<type int>, <type complex>, types.NoneType) -> <type 'complex'>
40:     # (<type int>, <type long>, Integer) -> <type 'long'>
41:     # (<type int>, <type long>, types.NoneType) -> <type 'long'>
42:     # (<type int>, <type int>, <type bool>) -> <type 'int'>
43:     # (<type int>, <type int>, types.NoneType) -> <type 'int'>
44:     # (<type int>, <type int>, Integer) -> TypeOfParam(3)
45:     # (<type int>, RealNumber, types.NoneType) -> <type 'float'>
46:     # (<type float>, <type complex>, types.NoneType) -> <type 'complex'>
47:     # (Overloads__pow__, AnyType, AnyType) -> DynamicType
48:     # (Overloads__pow__, AnyType, AnyType, AnyType) -> DynamicType
49: 
50:     class Empty:
51:         pass
52: 
53: 
54:     class Sample:
55:         def __pow__(self, other):
56:             return 4
57: 
58: 
59:     class Wrong1:
60:         def __pow__(self):
61:             return 4
62: 
63: 
64:     class Wrong2:
65:         def __pow__(self, other):
66:             # Type error
67:             return list() + other
68: 
69: 
70:     # Call the builtin with correct parameters
71:     ret = pow(Sample(), 4)
72: 
73:     # Call the builtin with incorrect types of parameters
74:     ret = pow(Wrong2(), 4)
75:     # Type error
76:     ret = pow(Wrong1(), 4)
77:     # Type error
78:     ret = pow(Empty(), 4)
79: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'pow builtin is invoked, but classes and instances with special name methods are passed ')
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
            module_type_store = module_type_store.open_function_context('__init__', 50, 4, False)
            # Assigning a type to the variable 'self' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __pow__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__pow__'
            module_type_store = module_type_store.open_function_context('__pow__', 55, 8, False)
            # Assigning a type to the variable 'self' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__pow__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__pow__.__dict__.__setitem__('stypy_function_name', 'Sample.__pow__')
            Sample.__pow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__pow__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__pow__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__pow__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__pow__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__pow__' in the type store
            # Getting the type of 'stypy_return_type' (line 55)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__pow__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 54, 4, False)
            # Assigning a type to the variable 'self' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __pow__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__pow__'
            module_type_store = module_type_store.open_function_context('__pow__', 60, 8, False)
            # Assigning a type to the variable 'self' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__pow__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__pow__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__pow__')
            Wrong1.__pow__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong1.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__pow__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__pow__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__pow__(...)' code ##################

            int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'stypy_return_type', int_4)
            
            # ################# End of '__pow__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__pow__' in the type store
            # Getting the type of 'stypy_return_type' (line 60)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__pow__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 59, 4, False)
            # Assigning a type to the variable 'self' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __pow__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__pow__'
            module_type_store = module_type_store.open_function_context('__pow__', 65, 8, False)
            # Assigning a type to the variable 'self' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__pow__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__pow__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__pow__')
            Wrong2.__pow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Wrong2.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__pow__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__pow__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__pow__(...)' code ##################

            
            # Call to list(...): (line 67)
            # Processing the call keyword arguments (line 67)
            kwargs_7 = {}
            # Getting the type of 'list' (line 67)
            list_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'list', False)
            # Calling list(args, kwargs) (line 67)
            list_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), list_6, *[], **kwargs_7)
            
            # Getting the type of 'other' (line 67)
            other_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'other')
            # Applying the binary operator '+' (line 67)
            result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 19), '+', list_call_result_8, other_9)
            
            # Assigning a type to the variable 'stypy_return_type' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type', result_add_10)
            
            # ################# End of '__pow__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__pow__' in the type store
            # Getting the type of 'stypy_return_type' (line 65)
            stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__pow__'
            return stypy_return_type_11


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 64, 4, False)
            # Assigning a type to the variable 'self' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 71):
    
    # Call to pow(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Call to Sample(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_14 = {}
    # Getting the type of 'Sample' (line 71)
    Sample_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'Sample', False)
    # Calling Sample(args, kwargs) (line 71)
    Sample_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), Sample_13, *[], **kwargs_14)
    
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_17 = {}
    # Getting the type of 'pow' (line 71)
    pow_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 71)
    pow_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), pow_12, *[Sample_call_result_15, int_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'ret', pow_call_result_18)
    
    # Assigning a Call to a Name (line 74):
    
    # Call to pow(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to Wrong2(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_21 = {}
    # Getting the type of 'Wrong2' (line 74)
    Wrong2_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 74)
    Wrong2_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 74, 14), Wrong2_20, *[], **kwargs_21)
    
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
    # Processing the call keyword arguments (line 74)
    kwargs_24 = {}
    # Getting the type of 'pow' (line 74)
    pow_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 74)
    pow_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 74, 10), pow_19, *[Wrong2_call_result_22, int_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'ret', pow_call_result_25)
    
    # Assigning a Call to a Name (line 76):
    
    # Call to pow(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Call to Wrong1(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_28 = {}
    # Getting the type of 'Wrong1' (line 76)
    Wrong1_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 76)
    Wrong1_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), Wrong1_27, *[], **kwargs_28)
    
    int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'int')
    # Processing the call keyword arguments (line 76)
    kwargs_31 = {}
    # Getting the type of 'pow' (line 76)
    pow_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 76)
    pow_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 76, 10), pow_26, *[Wrong1_call_result_29, int_30], **kwargs_31)
    
    # Assigning a type to the variable 'ret' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'ret', pow_call_result_32)
    
    # Assigning a Call to a Name (line 78):
    
    # Call to pow(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to Empty(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_35 = {}
    # Getting the type of 'Empty' (line 78)
    Empty_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'Empty', False)
    # Calling Empty(args, kwargs) (line 78)
    Empty_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 78, 14), Empty_34, *[], **kwargs_35)
    
    int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
    # Processing the call keyword arguments (line 78)
    kwargs_38 = {}
    # Getting the type of 'pow' (line 78)
    pow_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 78)
    pow_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 78, 10), pow_33, *[Empty_call_result_36, int_37], **kwargs_38)
    
    # Assigning a type to the variable 'ret' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'ret', pow_call_result_39)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

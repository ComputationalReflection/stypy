
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "pow builtin is invoked, but a class is used instead of an instance"
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
50: 
51:     class Sample:
52:         def __pow__(self, other):
53:             return 4
54: 
55: 
56:     # Type error
57:     ret = pow(Sample, 4)
58:     # Type error
59:     ret = pow(int, 4)
60:     # Type error
61:     ret = pow(int, int)
62: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'pow builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __pow__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__pow__'
            module_type_store = module_type_store.open_function_context('__pow__', 52, 8, False)
            # Assigning a type to the variable 'self' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self', type_of_self)
            
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

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__pow__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__pow__' in the type store
            # Getting the type of 'stypy_return_type' (line 52)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 51, 4, False)
            # Assigning a type to the variable 'self' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 57):
    
    # Call to pow(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'Sample' (line 57)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'Sample', False)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    # Processing the call keyword arguments (line 57)
    kwargs_7 = {}
    # Getting the type of 'pow' (line 57)
    pow_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 57)
    pow_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 57, 10), pow_4, *[Sample_5, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'ret', pow_call_result_8)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to pow(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'int' (line 59)
    int_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'int', False)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_12 = {}
    # Getting the type of 'pow' (line 59)
    pow_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 59)
    pow_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 59, 10), pow_9, *[int_10, int_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'ret', pow_call_result_13)
    
    # Assigning a Call to a Name (line 61):
    
    # Call to pow(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'int' (line 61)
    int_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'int', False)
    # Getting the type of 'int' (line 61)
    int_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'int', False)
    # Processing the call keyword arguments (line 61)
    kwargs_17 = {}
    # Getting the type of 'pow' (line 61)
    pow_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 61)
    pow_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 61, 10), pow_14, *[int_15, int_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'ret', pow_call_result_18)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

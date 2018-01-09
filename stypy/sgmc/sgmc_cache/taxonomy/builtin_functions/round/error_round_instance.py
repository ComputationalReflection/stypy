
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "round builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (RealNumber) -> <type 'float'>
7:     # (RealNumber, Integer) -> <type 'float'>
8:     # (RealNumber, CastsToIndex) -> <type 'float'>
9:     # (CastsToFloat) -> <type 'float'>
10:     # (CastsToFloat, Integer) -> <type 'float'>
11:     # (CastsToFloat, CastsToIndex) -> <type 'float'>
12: 
13: 
14:     class Sample:
15:         def __float__(self):
16:             return 4.0
17: 
18: 
19:     # Type error
20:     ret = round(Sample, 10)
21:     # Type error
22:     ret = round(float, 10)
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'round builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__float__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__float__.__dict__.__setitem__('stypy_function_name', 'Sample.__float__')
            Sample.__float__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__float__', [], None, None, defaults, varargs, kwargs)

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

            float_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'stypy_return_type', float_2)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__float__'
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
    
    # Assigning a Call to a Name (line 20):
    
    # Call to round(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'Sample' (line 20)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'Sample', False)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_7 = {}
    # Getting the type of 'round' (line 20)
    round_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'round', False)
    # Calling round(args, kwargs) (line 20)
    round_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), round_4, *[Sample_5, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', round_call_result_8)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to round(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'float' (line 22)
    float_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'float', False)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_12 = {}
    # Getting the type of 'round' (line 22)
    round_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'round', False)
    # Calling round(args, kwargs) (line 22)
    round_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), round_9, *[float_10, int_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', round_call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

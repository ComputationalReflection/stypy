
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "enumerate builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'enumerate'>
7:     # (IterableObject) -> <type 'enumerate'>
8:     # (Has__iter__) -> <type 'enumerate'>
9:     # (IterableObject, Integer) -> <type 'enumerate'>
10:     # (Has__iter__, Integer) -> <type 'enumerate'>
11:     class Sample:
12:         def __iter__(self):
13:             return iter(list())
14: 
15: 
16:     # Type error
17:     ret = enumerate(Sample)
18:     # Type error
19:     ret = enumerate(str)
20:     # Type error
21:     ret = enumerate(list)
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'enumerate builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __iter__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__iter__'
            module_type_store = module_type_store.open_function_context('__iter__', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__iter__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__iter__.__dict__.__setitem__('stypy_function_name', 'Sample.__iter__')
            Sample.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__iter__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__iter__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__iter__(...)' code ##################

            
            # Call to iter(...): (line 13)
            # Processing the call arguments (line 13)
            
            # Call to list(...): (line 13)
            # Processing the call keyword arguments (line 13)
            kwargs_4 = {}
            # Getting the type of 'list' (line 13)
            list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'list', False)
            # Calling list(args, kwargs) (line 13)
            list_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 13, 24), list_3, *[], **kwargs_4)
            
            # Processing the call keyword arguments (line 13)
            kwargs_6 = {}
            # Getting the type of 'iter' (line 13)
            iter_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'iter', False)
            # Calling iter(args, kwargs) (line 13)
            iter_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 13, 19), iter_2, *[list_call_result_5], **kwargs_6)
            
            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', iter_call_result_7)
            
            # ################# End of '__iter__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__iter__' in the type store
            # Getting the type of 'stypy_return_type' (line 12)
            stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_8)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__iter__'
            return stypy_return_type_8


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

    
    # Assigning a type to the variable 'Sample' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to enumerate(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'Sample' (line 17)
    Sample_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'Sample', False)
    # Processing the call keyword arguments (line 17)
    kwargs_11 = {}
    # Getting the type of 'enumerate' (line 17)
    enumerate_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 17)
    enumerate_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), enumerate_9, *[Sample_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', enumerate_call_result_12)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to enumerate(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'str' (line 19)
    str_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'str', False)
    # Processing the call keyword arguments (line 19)
    kwargs_15 = {}
    # Getting the type of 'enumerate' (line 19)
    enumerate_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 19)
    enumerate_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), enumerate_13, *[str_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', enumerate_call_result_16)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to enumerate(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'list' (line 21)
    list_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'list', False)
    # Processing the call keyword arguments (line 21)
    kwargs_19 = {}
    # Getting the type of 'enumerate' (line 21)
    enumerate_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 21)
    enumerate_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), enumerate_17, *[list_18], **kwargs_19)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', enumerate_call_result_20)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

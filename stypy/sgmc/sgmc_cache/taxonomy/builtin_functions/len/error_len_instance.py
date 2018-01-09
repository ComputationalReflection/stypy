
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "len builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'int'>
7:     # (Str) -> <type 'int'>
8:     # (Has__len__) -> <type 'int'>
9:     class Sample:
10:         def __len__(self):
11:             return 4
12: 
13: 
14:     # Type error
15:     ret = len(Sample)
16:     # Type error
17:     ret = len(str)
18:     # Type error
19:     ret = len(list)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'len builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __len__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__len__'
            module_type_store = module_type_store.open_function_context('__len__', 10, 8, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__len__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__len__.__dict__.__setitem__('stypy_function_name', 'Sample.__len__')
            Sample.__len__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__len__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__len__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__len__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__len__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__len__' in the type store
            # Getting the type of 'stypy_return_type' (line 10)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__len__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to len(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'Sample' (line 15)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'Sample', False)
    # Processing the call keyword arguments (line 15)
    kwargs_6 = {}
    # Getting the type of 'len' (line 15)
    len_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'len', False)
    # Calling len(args, kwargs) (line 15)
    len_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), len_4, *[Sample_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', len_call_result_7)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to len(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'str' (line 17)
    str_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'str', False)
    # Processing the call keyword arguments (line 17)
    kwargs_10 = {}
    # Getting the type of 'len' (line 17)
    len_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'len', False)
    # Calling len(args, kwargs) (line 17)
    len_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), len_8, *[str_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', len_call_result_11)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to len(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'list' (line 19)
    list_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'list', False)
    # Processing the call keyword arguments (line 19)
    kwargs_14 = {}
    # Getting the type of 'len' (line 19)
    len_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'len', False)
    # Calling len(args, kwargs) (line 19)
    len_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), len_12, *[list_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', len_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

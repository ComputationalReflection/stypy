
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "next builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__next) -> DynamicType
7:     # (Has__next, AnyType) -> DynamicType
8: 
9: 
10:     class Sample:
11:         def next(self):
12:             return 4
13: 
14: 
15:     # Call the builtin with correct parameters
16:     ret = next(Sample())
17: 
18:     # Type error
19:     ret.unexisting_method()
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'next builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def next(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'next'
            module_type_store = module_type_store.open_function_context('next', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.next.__dict__.__setitem__('stypy_localization', localization)
            Sample.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.next.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.next.__dict__.__setitem__('stypy_function_name', 'Sample.next')
            Sample.next.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.next.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.next.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.next.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.next.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.next', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'next', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'next(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', int_2)
            
            # ################# End of 'next(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'next' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'next'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to next(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to Sample(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 16)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'Sample', False)
    # Calling Sample(args, kwargs) (line 16)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), Sample_5, *[], **kwargs_6)
    
    # Processing the call keyword arguments (line 16)
    kwargs_8 = {}
    # Getting the type of 'next' (line 16)
    next_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'next', False)
    # Calling next(args, kwargs) (line 16)
    next_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), next_4, *[Sample_call_result_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', next_call_result_9)
    
    # Call to unexisting_method(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_12 = {}
    # Getting the type of 'ret' (line 19)
    ret_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 19)
    unexisting_method_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), ret_10, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 19)
    unexisting_method_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), unexisting_method_11, *[], **kwargs_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

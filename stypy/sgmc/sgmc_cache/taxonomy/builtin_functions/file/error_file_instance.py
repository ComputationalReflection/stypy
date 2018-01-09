
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "file builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'file'>
7:     # (Str, Str) -> <type 'file'>
8:     # (Str, Str, Integer) -> <type 'file'>
9:     # (Str, Str, Overloads__trunc__) -> <type 'file'>
10: 
11:     class Sample:
12:         def __trunc__(self):
13:             return 4
14: 
15: 
16:     # Type error
17:     ret = file("f.py", "r", Sample)
18:     # Type error
19:     ret = file(str)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'file builtin is invoked, but a class is used instead of an instance')
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
            module_type_store = module_type_store.open_function_context('__trunc__', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
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

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 12)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
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
    
    # Call to file(...): (line 17)
    # Processing the call arguments (line 17)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'str', 'f.py')
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'str', 'r')
    # Getting the type of 'Sample' (line 17)
    Sample_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 28), 'Sample', False)
    # Processing the call keyword arguments (line 17)
    kwargs_8 = {}
    # Getting the type of 'file' (line 17)
    file_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'file', False)
    # Calling file(args, kwargs) (line 17)
    file_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), file_4, *[str_5, str_6, Sample_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', file_call_result_9)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to file(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'str' (line 19)
    str_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'str', False)
    # Processing the call keyword arguments (line 19)
    kwargs_12 = {}
    # Getting the type of 'file' (line 19)
    file_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'file', False)
    # Calling file(args, kwargs) (line 19)
    file_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), file_10, *[str_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', file_call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__xor__ method is present, but return a different type from the operands"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __xor__(self, other):
7:             # Type error
8:             return str(other) ^ str(other)
9: 
10: 
11:     print Sample() ^ 1 ^ 3
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__xor__ method is present, but return a different type from the operands')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __xor__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__xor__'
            module_type_store = module_type_store.open_function_context('__xor__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__xor__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__xor__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__xor__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__xor__.__dict__.__setitem__('stypy_function_name', 'Sample.__xor__')
            Sample.__xor__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__xor__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__xor__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__xor__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__xor__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__xor__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__xor__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__xor__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__xor__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__xor__(...)' code ##################

            
            # Call to str(...): (line 8)
            # Processing the call arguments (line 8)
            # Getting the type of 'other' (line 8)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'other', False)
            # Processing the call keyword arguments (line 8)
            kwargs_4 = {}
            # Getting the type of 'str' (line 8)
            str_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', False)
            # Calling str(args, kwargs) (line 8)
            str_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 8, 19), str_2, *[other_3], **kwargs_4)
            
            
            # Call to str(...): (line 8)
            # Processing the call arguments (line 8)
            # Getting the type of 'other' (line 8)
            other_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 36), 'other', False)
            # Processing the call keyword arguments (line 8)
            kwargs_8 = {}
            # Getting the type of 'str' (line 8)
            str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 32), 'str', False)
            # Calling str(args, kwargs) (line 8)
            str_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 8, 32), str_6, *[other_7], **kwargs_8)
            
            # Applying the binary operator '^' (line 8)
            result_xor_10 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 19), '^', str_call_result_5, str_call_result_9)
            
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', result_xor_10)
            
            # ################# End of '__xor__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__xor__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__xor__'
            return stypy_return_type_11


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
            # Assigning a type to the variable 'self' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Sample', Sample)
    
    # Call to Sample(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 11)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'Sample', False)
    # Calling Sample(args, kwargs) (line 11)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), Sample_12, *[], **kwargs_13)
    
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
    # Applying the binary operator '^' (line 11)
    result_xor_16 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), '^', Sample_call_result_14, int_15)
    
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 25), 'int')
    # Applying the binary operator '^' (line 11)
    result_xor_18 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 23), '^', result_xor_16, int_17)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

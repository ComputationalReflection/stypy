
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__rlshift__ method is present, but return a different type from the operands"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __rlshift__(self, other):
7:             # Type error
8:             return str(other) << str(other)
9: 
10: 
11:     print 3 << 1 << Sample()
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__rlshift__ method is present, but return a different type from the operands')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __rlshift__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__rlshift__'
            module_type_store = module_type_store.open_function_context('__rlshift__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__rlshift__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__rlshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__rlshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__rlshift__.__dict__.__setitem__('stypy_function_name', 'Sample.__rlshift__')
            Sample.__rlshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__rlshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__rlshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__rlshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__rlshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__rlshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__rlshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__rlshift__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__rlshift__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__rlshift__(...)' code ##################

            
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
            other_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 37), 'other', False)
            # Processing the call keyword arguments (line 8)
            kwargs_8 = {}
            # Getting the type of 'str' (line 8)
            str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 33), 'str', False)
            # Calling str(args, kwargs) (line 8)
            str_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 8, 33), str_6, *[other_7], **kwargs_8)
            
            # Applying the binary operator '<<' (line 8)
            result_lshift_10 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 19), '<<', str_call_result_5, str_call_result_9)
            
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', result_lshift_10)
            
            # ################# End of '__rlshift__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__rlshift__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__rlshift__'
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
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'int')
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'int')
    # Applying the binary operator '<<' (line 11)
    result_lshift_14 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), '<<', int_12, int_13)
    
    
    # Call to Sample(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_16 = {}
    # Getting the type of 'Sample' (line 11)
    Sample_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'Sample', False)
    # Calling Sample(args, kwargs) (line 11)
    Sample_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 11, 20), Sample_15, *[], **kwargs_16)
    
    # Applying the binary operator '<<' (line 11)
    result_lshift_18 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 17), '<<', result_lshift_14, Sample_call_result_17)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

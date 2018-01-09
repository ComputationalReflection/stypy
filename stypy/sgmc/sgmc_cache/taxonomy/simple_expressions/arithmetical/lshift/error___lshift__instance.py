
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__lshift__ method is present, but a class is used to invoke the operation"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __lshift__(self, other):
7:             return 1 << other
8: 
9: 
10:     # Type error
11:     print Sample << 1
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__lshift__ method is present, but a class is used to invoke the operation')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __lshift__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__lshift__'
            module_type_store = module_type_store.open_function_context('__lshift__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__lshift__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__lshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__lshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__lshift__.__dict__.__setitem__('stypy_function_name', 'Sample.__lshift__')
            Sample.__lshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__lshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__lshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__lshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__lshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__lshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__lshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__lshift__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__lshift__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__lshift__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
            # Getting the type of 'other' (line 7)
            other_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 24), 'other')
            # Applying the binary operator '<<' (line 7)
            result_lshift_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 19), '<<', int_2, other_3)
            
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', result_lshift_4)
            
            # ################# End of '__lshift__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__lshift__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__lshift__'
            return stypy_return_type_5


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
    # Getting the type of 'Sample' (line 11)
    Sample_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'Sample')
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'int')
    # Applying the binary operator '<<' (line 11)
    result_lshift_8 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), '<<', Sample_6, int_7)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

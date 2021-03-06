
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__irshift__ method is present, but incorrect types are passed"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __irshift__(self, other):
7:             # Type error
8:             other >>= None
9:             return other
10: 
11: 
12:     x = Sample()
13: 
14:     x >>= 1
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__irshift__ method is present, but incorrect types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __irshift__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__irshift__'
            module_type_store = module_type_store.open_function_context('__irshift__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__irshift__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__irshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__irshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__irshift__.__dict__.__setitem__('stypy_function_name', 'Sample.__irshift__')
            Sample.__irshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__irshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__irshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__irshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__irshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__irshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__irshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__irshift__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__irshift__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__irshift__(...)' code ##################

            
            # Getting the type of 'other' (line 8)
            other_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'other')
            # Getting the type of 'None' (line 8)
            None_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 22), 'None')
            # Applying the binary operator '>>=' (line 8)
            result_irshift_4 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 12), '>>=', other_2, None_3)
            # Assigning a type to the variable 'other' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'other', result_irshift_4)
            
            # Getting the type of 'other' (line 9)
            other_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'other')
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', other_5)
            
            # ################# End of '__irshift__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__irshift__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__irshift__'
            return stypy_return_type_6


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
    
    # Assigning a Call to a Name (line 12):
    
    # Call to Sample(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_8 = {}
    # Getting the type of 'Sample' (line 12)
    Sample_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'Sample', False)
    # Calling Sample(args, kwargs) (line 12)
    Sample_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), Sample_7, *[], **kwargs_8)
    
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', Sample_call_result_9)
    
    # Getting the type of 'x' (line 14)
    x_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'x')
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'int')
    # Applying the binary operator '>>=' (line 14)
    result_irshift_12 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 4), '>>=', x_10, int_11)
    # Assigning a type to the variable 'x' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'x', result_irshift_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

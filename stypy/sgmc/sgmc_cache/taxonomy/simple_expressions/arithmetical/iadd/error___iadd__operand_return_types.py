
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__iadd__ method is present, but return a different type from the operands"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         def __iadd__(self, other):
7:             s = ""
8:             s += str(other)
9:             return s
10: 
11: 
12:     x = Sample()
13:     x += 1
14:     # Type error
15:     print x + 1
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__iadd__ method is present, but return a different type from the operands')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __iadd__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__iadd__'
            module_type_store = module_type_store.open_function_context('__iadd__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__iadd__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__iadd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__iadd__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__iadd__.__dict__.__setitem__('stypy_function_name', 'Sample.__iadd__')
            Sample.__iadd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__iadd__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__iadd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__iadd__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__iadd__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__iadd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__iadd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__iadd__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__iadd__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__iadd__(...)' code ##################

            
            # Assigning a Str to a Name (line 7):
            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'str', '')
            # Assigning a type to the variable 's' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 's', str_2)
            
            # Getting the type of 's' (line 8)
            s_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 's')
            
            # Call to str(...): (line 8)
            # Processing the call arguments (line 8)
            # Getting the type of 'other' (line 8)
            other_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 21), 'other', False)
            # Processing the call keyword arguments (line 8)
            kwargs_6 = {}
            # Getting the type of 'str' (line 8)
            str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'str', False)
            # Calling str(args, kwargs) (line 8)
            str_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 8, 17), str_4, *[other_5], **kwargs_6)
            
            # Applying the binary operator '+=' (line 8)
            result_iadd_8 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 12), '+=', s_3, str_call_result_7)
            # Assigning a type to the variable 's' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 's', result_iadd_8)
            
            # Getting the type of 's' (line 9)
            s_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 's')
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', s_9)
            
            # ################# End of '__iadd__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__iadd__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_10)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__iadd__'
            return stypy_return_type_10


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
    kwargs_12 = {}
    # Getting the type of 'Sample' (line 12)
    Sample_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'Sample', False)
    # Calling Sample(args, kwargs) (line 12)
    Sample_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), Sample_11, *[], **kwargs_12)
    
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', Sample_call_result_13)
    
    # Getting the type of 'x' (line 13)
    x_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x')
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
    # Applying the binary operator '+=' (line 13)
    result_iadd_16 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 4), '+=', x_14, int_15)
    # Assigning a type to the variable 'x' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x', result_iadd_16)
    
    # Getting the type of 'x' (line 15)
    x_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'x')
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'int')
    # Applying the binary operator '+' (line 15)
    result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 10), '+', x_17, int_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__eq__ return type is incorrectly used after call"
3: 
4: if __name__ == '__main__':
5:     class Eq2:
6:         def __eq__(self, other):
7:             return "str"
8: 
9: 
10:     r = Eq2() == 3
11: 
12:     print r[0]
13:     # Type error
14:     r.non_existing_method()
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__eq__ return type is incorrectly used after call')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Eq2' class

    class Eq2:

        @norecursion
        def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__eq__'
            module_type_store = module_type_store.open_function_context('__eq__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Eq2.__eq__')
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Eq2.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq2.__eq__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__eq__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', str_2)
            
            # ################# End of '__eq__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__eq__' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__eq__'
            return stypy_return_type_3


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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq2.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Eq2' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Eq2', Eq2)
    
    # Assigning a Compare to a Name (line 10):
    
    
    # Call to Eq2(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_5 = {}
    # Getting the type of 'Eq2' (line 10)
    Eq2_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'Eq2', False)
    # Calling Eq2(args, kwargs) (line 10)
    Eq2_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), Eq2_4, *[], **kwargs_5)
    
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'int')
    # Applying the binary operator '==' (line 10)
    result_eq_8 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 8), '==', Eq2_call_result_6, int_7)
    
    # Assigning a type to the variable 'r' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'r', result_eq_8)
    
    # Obtaining the type of the subscript
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'int')
    # Getting the type of 'r' (line 12)
    r_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'r')
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), r_10, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), getitem___11, int_9)
    
    
    # Call to non_existing_method(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_15 = {}
    # Getting the type of 'r' (line 14)
    r_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'r', False)
    # Obtaining the member 'non_existing_method' of a type (line 14)
    non_existing_method_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), r_13, 'non_existing_method')
    # Calling non_existing_method(args, kwargs) (line 14)
    non_existing_method_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), non_existing_method_14, *[], **kwargs_15)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

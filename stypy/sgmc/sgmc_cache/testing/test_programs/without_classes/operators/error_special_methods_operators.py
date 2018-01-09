
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: class Eq5:
3:     def __gt__(self, other):
4:         return str(other) + other
5: 
6: 
7: r1 = Eq5() > 3
8: 
9: 
10: class Eq6:
11:     def __gt__(self, other):
12:         return other[other]
13: 
14: r2 = Eq6() > 3

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Eq5' class

class Eq5:

    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 3, 4, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Eq5.__gt__.__dict__.__setitem__('stypy_localization', localization)
        Eq5.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Eq5.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Eq5.__gt__.__dict__.__setitem__('stypy_function_name', 'Eq5.__gt__')
        Eq5.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Eq5.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Eq5.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Eq5.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Eq5.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Eq5.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Eq5.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq5.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        
        # Call to str(...): (line 4)
        # Processing the call arguments (line 4)
        # Getting the type of 'other' (line 4)
        other_5520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'other', False)
        # Processing the call keyword arguments (line 4)
        kwargs_5521 = {}
        # Getting the type of 'str' (line 4)
        str_5519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), 'str', False)
        # Calling str(args, kwargs) (line 4)
        str_call_result_5522 = invoke(stypy.reporting.localization.Localization(__file__, 4, 15), str_5519, *[other_5520], **kwargs_5521)
        
        # Getting the type of 'other' (line 4)
        other_5523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 28), 'other')
        # Applying the binary operator '+' (line 4)
        result_add_5524 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 15), '+', str_call_result_5522, other_5523)
        
        # Assigning a type to the variable 'stypy_return_type' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'stypy_return_type', result_add_5524)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 3)
        stypy_return_type_5525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5525)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_5525


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 2, 0, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq5.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Eq5' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'Eq5', Eq5)

# Assigning a Compare to a Name (line 7):


# Call to Eq5(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_5527 = {}
# Getting the type of 'Eq5' (line 7)
Eq5_5526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'Eq5', False)
# Calling Eq5(args, kwargs) (line 7)
Eq5_call_result_5528 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), Eq5_5526, *[], **kwargs_5527)

int_5529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
# Applying the binary operator '>' (line 7)
result_gt_5530 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 5), '>', Eq5_call_result_5528, int_5529)

# Assigning a type to the variable 'r1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r1', result_gt_5530)
# Declaration of the 'Eq6' class

class Eq6:

    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Eq6.__gt__.__dict__.__setitem__('stypy_localization', localization)
        Eq6.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Eq6.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Eq6.__gt__.__dict__.__setitem__('stypy_function_name', 'Eq6.__gt__')
        Eq6.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Eq6.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Eq6.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Eq6.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Eq6.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Eq6.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Eq6.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq6.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'other' (line 12)
        other_5531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'other')
        # Getting the type of 'other' (line 12)
        other_5532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'other')
        # Obtaining the member '__getitem__' of a type (line 12)
        getitem___5533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 15), other_5532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 12)
        subscript_call_result_5534 = invoke(stypy.reporting.localization.Localization(__file__, 12, 15), getitem___5533, other_5531)
        
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', subscript_call_result_5534)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_5535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_5535


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Eq6.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Eq6' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Eq6', Eq6)

# Assigning a Compare to a Name (line 14):


# Call to Eq6(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_5537 = {}
# Getting the type of 'Eq6' (line 14)
Eq6_5536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'Eq6', False)
# Calling Eq6(args, kwargs) (line 14)
Eq6_call_result_5538 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), Eq6_5536, *[], **kwargs_5537)

int_5539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'int')
# Applying the binary operator '>' (line 14)
result_gt_5540 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 5), '>', Eq6_call_result_5538, int_5539)

# Assigning a type to the variable 'r2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r2', result_gt_5540)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

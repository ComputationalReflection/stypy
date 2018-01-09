
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Small modules to cope with python 2 vs 3 incompatibilities inside
2: numpy.distutils
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: import sys
8: 
9: def get_exception():
10:     return sys.exc_info()[1]
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_28134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'Small modules to cope with python 2 vs 3 incompatibilities inside\nnumpy.distutils\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)


@norecursion
def get_exception(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_exception'
    module_type_store = module_type_store.open_function_context('get_exception', 9, 0, False)
    
    # Passed parameters checking function
    get_exception.stypy_localization = localization
    get_exception.stypy_type_of_self = None
    get_exception.stypy_type_store = module_type_store
    get_exception.stypy_function_name = 'get_exception'
    get_exception.stypy_param_names_list = []
    get_exception.stypy_varargs_param_name = None
    get_exception.stypy_kwargs_param_name = None
    get_exception.stypy_call_defaults = defaults
    get_exception.stypy_call_varargs = varargs
    get_exception.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_exception', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_exception', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_exception(...)' code ##################

    
    # Obtaining the type of the subscript
    int_28135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'int')
    
    # Call to exc_info(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_28138 = {}
    # Getting the type of 'sys' (line 10)
    sys_28136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'sys', False)
    # Obtaining the member 'exc_info' of a type (line 10)
    exc_info_28137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), sys_28136, 'exc_info')
    # Calling exc_info(args, kwargs) (line 10)
    exc_info_call_result_28139 = invoke(stypy.reporting.localization.Localization(__file__, 10, 11), exc_info_28137, *[], **kwargs_28138)
    
    # Obtaining the member '__getitem__' of a type (line 10)
    getitem___28140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), exc_info_call_result_28139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 10)
    subscript_call_result_28141 = invoke(stypy.reporting.localization.Localization(__file__, 10, 11), getitem___28140, int_28135)
    
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', subscript_call_result_28141)
    
    # ################# End of 'get_exception(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_exception' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_28142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28142)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_exception'
    return stypy_return_type_28142

# Assigning a type to the variable 'get_exception' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'get_exception', get_exception)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

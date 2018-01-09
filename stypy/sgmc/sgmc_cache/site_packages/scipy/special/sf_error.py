
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Warnings and Exceptions that can be raised by special functions.'''
2: import warnings
3: 
4: 
5: class SpecialFunctionWarning(Warning):
6:     '''Warning that can be emitted by special functions.'''
7:     pass
8: warnings.simplefilter("always", category=SpecialFunctionWarning)
9: 
10: 
11: class SpecialFunctionError(Exception):
12:     '''Exception that can be raised by special functions.'''
13:     pass
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_503702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Warnings and Exceptions that can be raised by special functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import warnings' statement (line 2)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'warnings', warnings, module_type_store)

# Declaration of the 'SpecialFunctionWarning' class
# Getting the type of 'Warning' (line 5)
Warning_503703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 29), 'Warning')

class SpecialFunctionWarning(Warning_503703, ):
    str_503704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'Warning that can be emitted by special functions.')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpecialFunctionWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SpecialFunctionWarning' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'SpecialFunctionWarning', SpecialFunctionWarning)

# Call to simplefilter(...): (line 8)
# Processing the call arguments (line 8)
str_503707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 22), 'str', 'always')
# Processing the call keyword arguments (line 8)
# Getting the type of 'SpecialFunctionWarning' (line 8)
SpecialFunctionWarning_503708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 41), 'SpecialFunctionWarning', False)
keyword_503709 = SpecialFunctionWarning_503708
kwargs_503710 = {'category': keyword_503709}
# Getting the type of 'warnings' (line 8)
warnings_503705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', False)
# Obtaining the member 'simplefilter' of a type (line 8)
simplefilter_503706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 0), warnings_503705, 'simplefilter')
# Calling simplefilter(args, kwargs) (line 8)
simplefilter_call_result_503711 = invoke(stypy.reporting.localization.Localization(__file__, 8, 0), simplefilter_503706, *[str_503707], **kwargs_503710)

# Declaration of the 'SpecialFunctionError' class
# Getting the type of 'Exception' (line 11)
Exception_503712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'Exception')

class SpecialFunctionError(Exception_503712, ):
    str_503713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', 'Exception that can be raised by special functions.')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpecialFunctionError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SpecialFunctionError' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'SpecialFunctionError', SpecialFunctionError)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

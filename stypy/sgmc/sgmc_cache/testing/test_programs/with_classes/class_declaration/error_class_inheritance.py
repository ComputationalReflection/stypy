
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import exceptions
2: 
3: 
4: class MiException(exceptions.ArithmeticError):
5:     pass
6: 
7: 
8: class MiExceptionWrong(exceptions.foo):
9:     pass
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import exceptions' statement (line 1)
import exceptions

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'exceptions', exceptions, module_type_store)

# Declaration of the 'MiException' class
# Getting the type of 'exceptions' (line 4)
exceptions_2515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 18), 'exceptions')
# Obtaining the member 'ArithmeticError' of a type (line 4)
ArithmeticError_2516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 18), exceptions_2515, 'ArithmeticError')

class MiException(ArithmeticError_2516, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 0, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MiException.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MiException' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'MiException', MiException)
# Declaration of the 'MiExceptionWrong' class
# Getting the type of 'exceptions' (line 8)
exceptions_2517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'exceptions')
# Obtaining the member 'foo' of a type (line 8)
foo_2518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 23), exceptions_2517, 'foo')

class MiExceptionWrong(foo_2518, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MiExceptionWrong.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MiExceptionWrong' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'MiExceptionWrong', MiExceptionWrong)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

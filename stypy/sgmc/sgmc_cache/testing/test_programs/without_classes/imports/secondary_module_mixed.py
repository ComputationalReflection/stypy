
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import time
2: 
3: def secondary_function():
4:     return "foo"
5: 
6: number = 4.5
7: 
8: object = time.clock()
9: clock_func = time.clock

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import time' statement (line 1)
import time

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'time', time, module_type_store)


@norecursion
def secondary_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'secondary_function'
    module_type_store = module_type_store.open_function_context('secondary_function', 3, 0, False)
    
    # Passed parameters checking function
    secondary_function.stypy_localization = localization
    secondary_function.stypy_type_of_self = None
    secondary_function.stypy_type_store = module_type_store
    secondary_function.stypy_function_name = 'secondary_function'
    secondary_function.stypy_param_names_list = []
    secondary_function.stypy_varargs_param_name = None
    secondary_function.stypy_kwargs_param_name = None
    secondary_function.stypy_call_defaults = defaults
    secondary_function.stypy_call_varargs = varargs
    secondary_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'secondary_function', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'secondary_function', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'secondary_function(...)' code ##################

    str_5169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', 'foo')
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type', str_5169)
    
    # ################# End of 'secondary_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'secondary_function' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_5170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'secondary_function'
    return stypy_return_type_5170

# Assigning a type to the variable 'secondary_function' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'secondary_function', secondary_function)

# Assigning a Num to a Name (line 6):
float_5171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'float')
# Assigning a type to the variable 'number' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'number', float_5171)

# Assigning a Call to a Name (line 8):

# Call to clock(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_5174 = {}
# Getting the type of 'time' (line 8)
time_5172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'time', False)
# Obtaining the member 'clock' of a type (line 8)
clock_5173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 9), time_5172, 'clock')
# Calling clock(args, kwargs) (line 8)
clock_call_result_5175 = invoke(stypy.reporting.localization.Localization(__file__, 8, 9), clock_5173, *[], **kwargs_5174)

# Assigning a type to the variable 'object' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'object', clock_call_result_5175)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'time' (line 9)
time_5176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'time')
# Obtaining the member 'clock' of a type (line 9)
clock_5177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 13), time_5176, 'clock')
# Assigning a type to the variable 'clock_func' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'clock_func', clock_5177)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

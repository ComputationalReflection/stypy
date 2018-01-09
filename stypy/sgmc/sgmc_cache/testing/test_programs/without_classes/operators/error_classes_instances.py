
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: 
4: class RightOps:
5:     def __float__(self):
6:         return 3.0
7: 
8: 
9: r1 = math.pow(RightOps, 4)  # Not reported
10: r2 = math.cos(RightOps)  # Not reported
11: 
12: r3 = math.pow(RightOps(), 4)
13: r4 = math.cos(RightOps())
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import math' statement (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)

# Declaration of the 'RightOps' class

class RightOps:

    @norecursion
    def __float__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__float__'
        module_type_store = module_type_store.open_function_context('__float__', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RightOps.__float__.__dict__.__setitem__('stypy_localization', localization)
        RightOps.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RightOps.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RightOps.__float__.__dict__.__setitem__('stypy_function_name', 'RightOps.__float__')
        RightOps.__float__.__dict__.__setitem__('stypy_param_names_list', [])
        RightOps.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RightOps.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RightOps.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RightOps.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RightOps.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RightOps.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RightOps.__float__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__float__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__float__(...)' code ##################

        float_7133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', float_7133)
        
        # ################# End of '__float__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__float__' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_7134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__float__'
        return stypy_return_type_7134


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RightOps.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RightOps' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'RightOps', RightOps)

# Assigning a Call to a Name (line 9):

# Call to pow(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'RightOps' (line 9)
RightOps_7137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'RightOps', False)
int_7138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
# Processing the call keyword arguments (line 9)
kwargs_7139 = {}
# Getting the type of 'math' (line 9)
math_7135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 9)
pow_7136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), math_7135, 'pow')
# Calling pow(args, kwargs) (line 9)
pow_call_result_7140 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), pow_7136, *[RightOps_7137, int_7138], **kwargs_7139)

# Assigning a type to the variable 'r1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r1', pow_call_result_7140)

# Assigning a Call to a Name (line 10):

# Call to cos(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'RightOps' (line 10)
RightOps_7143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'RightOps', False)
# Processing the call keyword arguments (line 10)
kwargs_7144 = {}
# Getting the type of 'math' (line 10)
math_7141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'math', False)
# Obtaining the member 'cos' of a type (line 10)
cos_7142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), math_7141, 'cos')
# Calling cos(args, kwargs) (line 10)
cos_call_result_7145 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), cos_7142, *[RightOps_7143], **kwargs_7144)

# Assigning a type to the variable 'r2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r2', cos_call_result_7145)

# Assigning a Call to a Name (line 12):

# Call to pow(...): (line 12)
# Processing the call arguments (line 12)

# Call to RightOps(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_7149 = {}
# Getting the type of 'RightOps' (line 12)
RightOps_7148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'RightOps', False)
# Calling RightOps(args, kwargs) (line 12)
RightOps_call_result_7150 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), RightOps_7148, *[], **kwargs_7149)

int_7151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'int')
# Processing the call keyword arguments (line 12)
kwargs_7152 = {}
# Getting the type of 'math' (line 12)
math_7146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 12)
pow_7147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), math_7146, 'pow')
# Calling pow(args, kwargs) (line 12)
pow_call_result_7153 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), pow_7147, *[RightOps_call_result_7150, int_7151], **kwargs_7152)

# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', pow_call_result_7153)

# Assigning a Call to a Name (line 13):

# Call to cos(...): (line 13)
# Processing the call arguments (line 13)

# Call to RightOps(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_7157 = {}
# Getting the type of 'RightOps' (line 13)
RightOps_7156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'RightOps', False)
# Calling RightOps(args, kwargs) (line 13)
RightOps_call_result_7158 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), RightOps_7156, *[], **kwargs_7157)

# Processing the call keyword arguments (line 13)
kwargs_7159 = {}
# Getting the type of 'math' (line 13)
math_7154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'math', False)
# Obtaining the member 'cos' of a type (line 13)
cos_7155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), math_7154, 'cos')
# Calling cos(args, kwargs) (line 13)
cos_call_result_7160 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), cos_7155, *[RightOps_call_result_7158], **kwargs_7159)

# Assigning a type to the variable 'r4' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r4', cos_call_result_7160)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: r1 = math.pow("a", 4)
4: 
5: 
6: def get_str():
7:     if r1 > 0:
8:         return "hi"
9:     else:
10:         return 2
11: 
12: r2 = get_str()
13: r3 = math.pow(r2, 3)
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


# Assigning a Call to a Name (line 3):

# Call to pow(...): (line 3)
# Processing the call arguments (line 3)
str_7743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 14), 'str', 'a')
int_7744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 19), 'int')
# Processing the call keyword arguments (line 3)
kwargs_7745 = {}
# Getting the type of 'math' (line 3)
math_7741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 3)
pow_7742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 5), math_7741, 'pow')
# Calling pow(args, kwargs) (line 3)
pow_call_result_7746 = invoke(stypy.reporting.localization.Localization(__file__, 3, 5), pow_7742, *[str_7743, int_7744], **kwargs_7745)

# Assigning a type to the variable 'r1' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'r1', pow_call_result_7746)

@norecursion
def get_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_str'
    module_type_store = module_type_store.open_function_context('get_str', 6, 0, False)
    
    # Passed parameters checking function
    get_str.stypy_localization = localization
    get_str.stypy_type_of_self = None
    get_str.stypy_type_store = module_type_store
    get_str.stypy_function_name = 'get_str'
    get_str.stypy_param_names_list = []
    get_str.stypy_varargs_param_name = None
    get_str.stypy_kwargs_param_name = None
    get_str.stypy_call_defaults = defaults
    get_str.stypy_call_varargs = varargs
    get_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_str', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_str', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_str(...)' code ##################

    
    
    # Getting the type of 'r1' (line 7)
    r1_7747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'r1')
    int_7748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
    # Applying the binary operator '>' (line 7)
    result_gt_7749 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 7), '>', r1_7747, int_7748)
    
    # Testing the type of an if condition (line 7)
    if_condition_7750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 4), result_gt_7749)
    # Assigning a type to the variable 'if_condition_7750' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'if_condition_7750', if_condition_7750)
    # SSA begins for if statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_7751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', 'hi')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', str_7751)
    # SSA branch for the else part of an if statement (line 7)
    module_type_store.open_ssa_branch('else')
    int_7752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', int_7752)
    # SSA join for if statement (line 7)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_str' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_7753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7753)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_str'
    return stypy_return_type_7753

# Assigning a type to the variable 'get_str' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'get_str', get_str)

# Assigning a Call to a Name (line 12):

# Call to get_str(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_7755 = {}
# Getting the type of 'get_str' (line 12)
get_str_7754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'get_str', False)
# Calling get_str(args, kwargs) (line 12)
get_str_call_result_7756 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), get_str_7754, *[], **kwargs_7755)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', get_str_call_result_7756)

# Assigning a Call to a Name (line 13):

# Call to pow(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'r2' (line 13)
r2_7759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'r2', False)
int_7760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'int')
# Processing the call keyword arguments (line 13)
kwargs_7761 = {}
# Getting the type of 'math' (line 13)
math_7757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 13)
pow_7758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), math_7757, 'pow')
# Calling pow(args, kwargs) (line 13)
pow_call_result_7762 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), pow_7758, *[r2_7759, int_7760], **kwargs_7761)

# Assigning a type to the variable 'r3' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r3', pow_call_result_7762)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

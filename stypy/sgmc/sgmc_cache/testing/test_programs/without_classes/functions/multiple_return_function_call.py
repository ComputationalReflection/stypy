
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: def function(a):
4:     if a > 0:
5:         return "Positive"
6:     if a < 0:
7:         return a
8:     if a == 0:
9:         return False
10: 
11: x = function(3)
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function'
    module_type_store = module_type_store.open_function_context('function', 3, 0, False)
    
    # Passed parameters checking function
    function.stypy_localization = localization
    function.stypy_type_of_self = None
    function.stypy_type_store = module_type_store
    function.stypy_function_name = 'function'
    function.stypy_param_names_list = ['a']
    function.stypy_varargs_param_name = None
    function.stypy_kwargs_param_name = None
    function.stypy_call_defaults = defaults
    function.stypy_call_varargs = varargs
    function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function(...)' code ##################

    
    
    # Getting the type of 'a' (line 4)
    a_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 7), 'a')
    int_1019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
    # Applying the binary operator '>' (line 4)
    result_gt_1020 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 7), '>', a_1018, int_1019)
    
    # Testing the type of an if condition (line 4)
    if_condition_1021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 4), result_gt_1020)
    # Assigning a type to the variable 'if_condition_1021' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'if_condition_1021', if_condition_1021)
    # SSA begins for if statement (line 4)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_1022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', 'Positive')
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', str_1022)
    # SSA join for if statement (line 4)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 6)
    a_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'a')
    int_1024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'int')
    # Applying the binary operator '<' (line 6)
    result_lt_1025 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 7), '<', a_1023, int_1024)
    
    # Testing the type of an if condition (line 6)
    if_condition_1026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 4), result_lt_1025)
    # Assigning a type to the variable 'if_condition_1026' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'if_condition_1026', if_condition_1026)
    # SSA begins for if statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'a' (line 7)
    a_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', a_1027)
    # SSA join for if statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 8)
    a_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'a')
    int_1029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'int')
    # Applying the binary operator '==' (line 8)
    result_eq_1030 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 7), '==', a_1028, int_1029)
    
    # Testing the type of an if condition (line 8)
    if_condition_1031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 4), result_eq_1030)
    # Assigning a type to the variable 'if_condition_1031' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'if_condition_1031', if_condition_1031)
    # SSA begins for if statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 9)
    False_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', False_1032)
    # SSA join for if statement (line 8)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1033)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_1033

# Assigning a type to the variable 'function' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'function', function)

# Assigning a Call to a Name (line 11):

# Call to function(...): (line 11)
# Processing the call arguments (line 11)
int_1035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'int')
# Processing the call keyword arguments (line 11)
kwargs_1036 = {}
# Getting the type of 'function' (line 11)
function_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'function', False)
# Calling function(args, kwargs) (line 11)
function_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), function_1034, *[int_1035], **kwargs_1036)

# Assigning a type to the variable 'x' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'x', function_call_result_1037)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

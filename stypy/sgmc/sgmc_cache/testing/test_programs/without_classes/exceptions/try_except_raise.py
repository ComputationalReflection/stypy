
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: def fun():
3:     return Exception()
4: 
5: try:
6:     a = 3
7:     raise fun()
8: except KeyError as k:
9:     a = "3"
10: except Exception as k2:
11:     a = k2
12: 
13: z = None

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun'
    module_type_store = module_type_store.open_function_context('fun', 2, 0, False)
    
    # Passed parameters checking function
    fun.stypy_localization = localization
    fun.stypy_type_of_self = None
    fun.stypy_type_store = module_type_store
    fun.stypy_function_name = 'fun'
    fun.stypy_param_names_list = []
    fun.stypy_varargs_param_name = None
    fun.stypy_kwargs_param_name = None
    fun.stypy_call_defaults = defaults
    fun.stypy_call_varargs = varargs
    fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun(...)' code ##################

    
    # Call to Exception(...): (line 3)
    # Processing the call keyword arguments (line 3)
    kwargs_2715 = {}
    # Getting the type of 'Exception' (line 3)
    Exception_2714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 11), 'Exception', False)
    # Calling Exception(args, kwargs) (line 3)
    Exception_call_result_2716 = invoke(stypy.reporting.localization.Localization(__file__, 3, 11), Exception_2714, *[], **kwargs_2715)
    
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'stypy_return_type', Exception_call_result_2716)
    
    # ################# End of 'fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun' in the type store
    # Getting the type of 'stypy_return_type' (line 2)
    stypy_return_type_2717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2717)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun'
    return stypy_return_type_2717

# Assigning a type to the variable 'fun' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'fun', fun)


# SSA begins for try-except statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Num to a Name (line 6):
int_2718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', int_2718)

# Call to fun(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_2720 = {}
# Getting the type of 'fun' (line 7)
fun_2719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'fun', False)
# Calling fun(args, kwargs) (line 7)
fun_call_result_2721 = invoke(stypy.reporting.localization.Localization(__file__, 7, 10), fun_2719, *[], **kwargs_2720)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 7, 4), fun_call_result_2721, 'raise parameter', BaseException)
# SSA branch for the except part of a try statement (line 5)
# SSA branch for the except 'KeyError' branch of a try statement (line 5)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'KeyError' (line 8)
KeyError_2722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'KeyError')
# Assigning a type to the variable 'k' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'k', KeyError_2722)

# Assigning a Str to a Name (line 9):
str_2723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', '3')
# Assigning a type to the variable 'a' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'a', str_2723)
# SSA branch for the except 'Exception' branch of a try statement (line 5)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 10)
Exception_2724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 7), 'Exception')
# Assigning a type to the variable 'k2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'k2', Exception_2724)

# Assigning a Name to a Name (line 11):
# Getting the type of 'k2' (line 11)
k2_2725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'k2')
# Assigning a type to the variable 'a' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', k2_2725)
# SSA join for try-except statement (line 5)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 13):
# Getting the type of 'None' (line 13)
None_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'None')
# Assigning a type to the variable 'z' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'z', None_2726)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

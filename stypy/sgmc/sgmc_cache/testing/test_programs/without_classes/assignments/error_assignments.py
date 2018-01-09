
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def fun():
2:     if True:
3:         return 3
4:     else:
5:         return [3, 4]
6: 
7: 
8: fun()[0] = 4  # Not Reported
9: 

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
    module_type_store = module_type_store.open_function_context('fun', 1, 0, False)
    
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

    
    # Getting the type of 'True' (line 2)
    True_6925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 7), 'True')
    # Testing the type of an if condition (line 2)
    if_condition_6926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2, 4), True_6925)
    # Assigning a type to the variable 'if_condition_6926' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'if_condition_6926', if_condition_6926)
    # SSA begins for if statement (line 2)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_6927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', int_6927)
    # SSA branch for the else part of an if statement (line 2)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'list' (line 5)
    list_6928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 5)
    # Adding element type (line 5)
    int_6929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), list_6928, int_6929)
    # Adding element type (line 5)
    int_6930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), list_6928, int_6930)
    
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', list_6928)
    # SSA join for if statement (line 2)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_6931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun'
    return stypy_return_type_6931

# Assigning a type to the variable 'fun' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'fun', fun)

# Assigning a Num to a Subscript (line 8):
int_6932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'int')

# Call to fun(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_6934 = {}
# Getting the type of 'fun' (line 8)
fun_6933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'fun', False)
# Calling fun(args, kwargs) (line 8)
fun_call_result_6935 = invoke(stypy.reporting.localization.Localization(__file__, 8, 0), fun_6933, *[], **kwargs_6934)

int_6936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 6), 'int')
# Storing an element on a container (line 8)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 0), fun_call_result_6935, (int_6936, int_6932))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

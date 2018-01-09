
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def fun1(l):
2:     return "aaa" + l[0]
3: 
4: 
5: r1 = fun1(list) # No error reported
6: r2 = fun1(tuple) # No error reported
7: 
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def fun1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun1'
    module_type_store = module_type_store.open_function_context('fun1', 1, 0, False)
    
    # Passed parameters checking function
    fun1.stypy_localization = localization
    fun1.stypy_type_of_self = None
    fun1.stypy_type_store = module_type_store
    fun1.stypy_function_name = 'fun1'
    fun1.stypy_param_names_list = ['l']
    fun1.stypy_varargs_param_name = None
    fun1.stypy_kwargs_param_name = None
    fun1.stypy_call_defaults = defaults
    fun1.stypy_call_varargs = varargs
    fun1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun1', ['l'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun1', localization, ['l'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun1(...)' code ##################

    str_7538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'aaa')
    
    # Obtaining the type of the subscript
    int_7539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 21), 'int')
    # Getting the type of 'l' (line 2)
    l_7540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 19), 'l')
    # Obtaining the member '__getitem__' of a type (line 2)
    getitem___7541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2, 19), l_7540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 2)
    subscript_call_result_7542 = invoke(stypy.reporting.localization.Localization(__file__, 2, 19), getitem___7541, int_7539)
    
    # Applying the binary operator '+' (line 2)
    result_add_7543 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), '+', str_7538, subscript_call_result_7542)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', result_add_7543)
    
    # ################# End of 'fun1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun1' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7544)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun1'
    return stypy_return_type_7544

# Assigning a type to the variable 'fun1' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'fun1', fun1)

# Assigning a Call to a Name (line 5):

# Call to fun1(...): (line 5)
# Processing the call arguments (line 5)
# Getting the type of 'list' (line 5)
list_7546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 10), 'list', False)
# Processing the call keyword arguments (line 5)
kwargs_7547 = {}
# Getting the type of 'fun1' (line 5)
fun1_7545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'fun1', False)
# Calling fun1(args, kwargs) (line 5)
fun1_call_result_7548 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), fun1_7545, *[list_7546], **kwargs_7547)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', fun1_call_result_7548)

# Assigning a Call to a Name (line 6):

# Call to fun1(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'tuple' (line 6)
tuple_7550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 10), 'tuple', False)
# Processing the call keyword arguments (line 6)
kwargs_7551 = {}
# Getting the type of 'fun1' (line 6)
fun1_7549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'fun1', False)
# Calling fun1(args, kwargs) (line 6)
fun1_call_result_7552 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), fun1_7549, *[tuple_7550], **kwargs_7551)

# Assigning a type to the variable 'r2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r2', fun1_call_result_7552)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

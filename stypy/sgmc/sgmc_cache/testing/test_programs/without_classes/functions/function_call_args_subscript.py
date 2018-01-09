
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: dic = {"a": 1, "b": 2}
2: tup = ([3], dic)
3: 
4: 
5: def function(*args, **kwargs):
6:     print args
7:     print kwargs
8: 
9: function(*[3], **dic)
10: function(*tup[0], **tup[1])
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Dict to a Name (line 1):

# Obtaining an instance of the builtin type 'dict' (line 1)
dict_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 6), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1)
# Adding element type (key, value) (line 1)
str_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 7), 'str', 'a')
int_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 12), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 6), dict_713, (str_714, int_715))
# Adding element type (key, value) (line 1)
str_716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 15), 'str', 'b')
int_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 20), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 6), dict_713, (str_716, int_717))

# Assigning a type to the variable 'dic' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'dic', dict_713)

# Assigning a Tuple to a Name (line 2):

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 7), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'list' (line 2)
list_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
int_720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 7), list_719, int_720)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 7), tuple_718, list_719)
# Adding element type (line 2)
# Getting the type of 'dic' (line 2)
dic_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 12), 'dic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 7), tuple_718, dic_721)

# Assigning a type to the variable 'tup' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'tup', tuple_718)

@norecursion
def function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function'
    module_type_store = module_type_store.open_function_context('function', 5, 0, False)
    
    # Passed parameters checking function
    function.stypy_localization = localization
    function.stypy_type_of_self = None
    function.stypy_type_store = module_type_store
    function.stypy_function_name = 'function'
    function.stypy_param_names_list = []
    function.stypy_varargs_param_name = 'args'
    function.stypy_kwargs_param_name = 'kwargs'
    function.stypy_call_defaults = defaults
    function.stypy_call_varargs = varargs
    function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function(...)' code ##################

    # Getting the type of 'args' (line 6)
    args_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 10), 'args')
    # Getting the type of 'kwargs' (line 7)
    kwargs_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'kwargs')
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_724)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_724

# Assigning a type to the variable 'function' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'function', function)

# Call to function(...): (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_726, int_727)

# Processing the call keyword arguments (line 9)
# Getting the type of 'dic' (line 9)
dic_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'dic', False)
kwargs_729 = {'dic_728': dic_728}
# Getting the type of 'function' (line 9)
function_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'function', False)
# Calling function(args, kwargs) (line 9)
function_call_result_730 = invoke(stypy.reporting.localization.Localization(__file__, 9, 0), function_725, *[list_726], **kwargs_729)


# Call to function(...): (line 10)

# Obtaining the type of the subscript
int_732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
# Getting the type of 'tup' (line 10)
tup_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'tup', False)
# Obtaining the member '__getitem__' of a type (line 10)
getitem___734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), tup_733, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_735 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), getitem___734, int_732)

# Processing the call keyword arguments (line 10)

# Obtaining the type of the subscript
int_736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
# Getting the type of 'tup' (line 10)
tup_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'tup', False)
# Obtaining the member '__getitem__' of a type (line 10)
getitem___738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 20), tup_737, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_739 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), getitem___738, int_736)

kwargs_740 = {'subscript_call_result_739': subscript_call_result_739}
# Getting the type of 'function' (line 10)
function_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'function', False)
# Calling function(args, kwargs) (line 10)
function_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), function_731, *[subscript_call_result_735], **kwargs_740)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

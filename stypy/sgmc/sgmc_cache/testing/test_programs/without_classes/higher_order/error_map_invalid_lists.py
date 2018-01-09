
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: l = range(5)
3: l2 = [False, 1, "string"]
4: other_l2 = map(lambda elem: elem / 2, l2)  # Unreported
5: r1 = other_l2[0].capitalize()  # PyCharm assumes that the return type is list[string] (?)
6: 
7: l3 = [[], {}, "string"]
8: other_l3 = map(lambda elem: elem / 2, l3)  # Unreported
9: r2 = other_l3[0].capitalize()  # PyCharm assumes that the return type is list[string] (?)
10: 
11: l4 = ["False", "1", "string"]
12: other_l4 = map(lambda elem: elem / 2, l3)  # Unreported
13: r3 = other_l4[0].capitalize()  # PyCharm assumes that the return type is list[string] (?)
14: 
15: 
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 2):

# Call to range(...): (line 2)
# Processing the call arguments (line 2)
int_7864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Processing the call keyword arguments (line 2)
kwargs_7865 = {}
# Getting the type of 'range' (line 2)
range_7863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'range', False)
# Calling range(args, kwargs) (line 2)
range_call_result_7866 = invoke(stypy.reporting.localization.Localization(__file__, 2, 4), range_7863, *[int_7864], **kwargs_7865)

# Assigning a type to the variable 'l' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'l', range_call_result_7866)

# Assigning a List to a Name (line 3):

# Obtaining an instance of the builtin type 'list' (line 3)
list_7867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
# Getting the type of 'False' (line 3)
False_7868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 6), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7867, False_7868)
# Adding element type (line 3)
int_7869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7867, int_7869)
# Adding element type (line 3)
str_7870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7867, str_7870)

# Assigning a type to the variable 'l2' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'l2', list_7867)

# Assigning a Call to a Name (line 4):

# Call to map(...): (line 4)
# Processing the call arguments (line 4)

@norecursion
def _stypy_temp_lambda_18(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_18'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_18', 4, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_18.stypy_localization = localization
    _stypy_temp_lambda_18.stypy_type_of_self = None
    _stypy_temp_lambda_18.stypy_type_store = module_type_store
    _stypy_temp_lambda_18.stypy_function_name = '_stypy_temp_lambda_18'
    _stypy_temp_lambda_18.stypy_param_names_list = ['elem']
    _stypy_temp_lambda_18.stypy_varargs_param_name = None
    _stypy_temp_lambda_18.stypy_kwargs_param_name = None
    _stypy_temp_lambda_18.stypy_call_defaults = defaults
    _stypy_temp_lambda_18.stypy_call_varargs = varargs
    _stypy_temp_lambda_18.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_18', ['elem'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_18', ['elem'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'elem' (line 4)
    elem_7872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 28), 'elem', False)
    int_7873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 35), 'int')
    # Applying the binary operator 'div' (line 4)
    result_div_7874 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 28), 'div', elem_7872, int_7873)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), 'stypy_return_type', result_div_7874)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_18' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_7875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7875)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_18'
    return stypy_return_type_7875

# Assigning a type to the variable '_stypy_temp_lambda_18' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), '_stypy_temp_lambda_18', _stypy_temp_lambda_18)
# Getting the type of '_stypy_temp_lambda_18' (line 4)
_stypy_temp_lambda_18_7876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), '_stypy_temp_lambda_18')
# Getting the type of 'l2' (line 4)
l2_7877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 38), 'l2', False)
# Processing the call keyword arguments (line 4)
kwargs_7878 = {}
# Getting the type of 'map' (line 4)
map_7871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 11), 'map', False)
# Calling map(args, kwargs) (line 4)
map_call_result_7879 = invoke(stypy.reporting.localization.Localization(__file__, 4, 11), map_7871, *[_stypy_temp_lambda_18_7876, l2_7877], **kwargs_7878)

# Assigning a type to the variable 'other_l2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'other_l2', map_call_result_7879)

# Assigning a Call to a Name (line 5):

# Call to capitalize(...): (line 5)
# Processing the call keyword arguments (line 5)
kwargs_7885 = {}

# Obtaining the type of the subscript
int_7880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Getting the type of 'other_l2' (line 5)
other_l2_7881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'other_l2', False)
# Obtaining the member '__getitem__' of a type (line 5)
getitem___7882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), other_l2_7881, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_7883 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), getitem___7882, int_7880)

# Obtaining the member 'capitalize' of a type (line 5)
capitalize_7884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), subscript_call_result_7883, 'capitalize')
# Calling capitalize(args, kwargs) (line 5)
capitalize_call_result_7886 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), capitalize_7884, *[], **kwargs_7885)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', capitalize_call_result_7886)

# Assigning a List to a Name (line 7):

# Obtaining an instance of the builtin type 'list' (line 7)
list_7887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_7888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_7887, list_7888)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'dict' (line 7)
dict_7889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 7)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_7887, dict_7889)
# Adding element type (line 7)
str_7890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_7887, str_7890)

# Assigning a type to the variable 'l3' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'l3', list_7887)

# Assigning a Call to a Name (line 8):

# Call to map(...): (line 8)
# Processing the call arguments (line 8)

@norecursion
def _stypy_temp_lambda_19(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_19'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_19', 8, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_19.stypy_localization = localization
    _stypy_temp_lambda_19.stypy_type_of_self = None
    _stypy_temp_lambda_19.stypy_type_store = module_type_store
    _stypy_temp_lambda_19.stypy_function_name = '_stypy_temp_lambda_19'
    _stypy_temp_lambda_19.stypy_param_names_list = ['elem']
    _stypy_temp_lambda_19.stypy_varargs_param_name = None
    _stypy_temp_lambda_19.stypy_kwargs_param_name = None
    _stypy_temp_lambda_19.stypy_call_defaults = defaults
    _stypy_temp_lambda_19.stypy_call_varargs = varargs
    _stypy_temp_lambda_19.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_19', ['elem'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_19', ['elem'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'elem' (line 8)
    elem_7892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 28), 'elem', False)
    int_7893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 35), 'int')
    # Applying the binary operator 'div' (line 8)
    result_div_7894 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 28), 'div', elem_7892, int_7893)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'stypy_return_type', result_div_7894)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_19' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_7895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7895)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_19'
    return stypy_return_type_7895

# Assigning a type to the variable '_stypy_temp_lambda_19' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), '_stypy_temp_lambda_19', _stypy_temp_lambda_19)
# Getting the type of '_stypy_temp_lambda_19' (line 8)
_stypy_temp_lambda_19_7896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), '_stypy_temp_lambda_19')
# Getting the type of 'l3' (line 8)
l3_7897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 38), 'l3', False)
# Processing the call keyword arguments (line 8)
kwargs_7898 = {}
# Getting the type of 'map' (line 8)
map_7891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'map', False)
# Calling map(args, kwargs) (line 8)
map_call_result_7899 = invoke(stypy.reporting.localization.Localization(__file__, 8, 11), map_7891, *[_stypy_temp_lambda_19_7896, l3_7897], **kwargs_7898)

# Assigning a type to the variable 'other_l3' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'other_l3', map_call_result_7899)

# Assigning a Call to a Name (line 9):

# Call to capitalize(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_7905 = {}

# Obtaining the type of the subscript
int_7900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'int')
# Getting the type of 'other_l3' (line 9)
other_l3_7901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'other_l3', False)
# Obtaining the member '__getitem__' of a type (line 9)
getitem___7902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), other_l3_7901, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_7903 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), getitem___7902, int_7900)

# Obtaining the member 'capitalize' of a type (line 9)
capitalize_7904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), subscript_call_result_7903, 'capitalize')
# Calling capitalize(args, kwargs) (line 9)
capitalize_call_result_7906 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), capitalize_7904, *[], **kwargs_7905)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', capitalize_call_result_7906)

# Assigning a List to a Name (line 11):

# Obtaining an instance of the builtin type 'list' (line 11)
list_7907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_7908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 6), 'str', 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 5), list_7907, str_7908)
# Adding element type (line 11)
str_7909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', '1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 5), list_7907, str_7909)
# Adding element type (line 11)
str_7910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 5), list_7907, str_7910)

# Assigning a type to the variable 'l4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'l4', list_7907)

# Assigning a Call to a Name (line 12):

# Call to map(...): (line 12)
# Processing the call arguments (line 12)

@norecursion
def _stypy_temp_lambda_20(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_20'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_20', 12, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_20.stypy_localization = localization
    _stypy_temp_lambda_20.stypy_type_of_self = None
    _stypy_temp_lambda_20.stypy_type_store = module_type_store
    _stypy_temp_lambda_20.stypy_function_name = '_stypy_temp_lambda_20'
    _stypy_temp_lambda_20.stypy_param_names_list = ['elem']
    _stypy_temp_lambda_20.stypy_varargs_param_name = None
    _stypy_temp_lambda_20.stypy_kwargs_param_name = None
    _stypy_temp_lambda_20.stypy_call_defaults = defaults
    _stypy_temp_lambda_20.stypy_call_varargs = varargs
    _stypy_temp_lambda_20.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_20', ['elem'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_20', ['elem'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'elem' (line 12)
    elem_7912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), 'elem', False)
    int_7913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'int')
    # Applying the binary operator 'div' (line 12)
    result_div_7914 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 28), 'div', elem_7912, int_7913)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'stypy_return_type', result_div_7914)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_20' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_7915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_20'
    return stypy_return_type_7915

# Assigning a type to the variable '_stypy_temp_lambda_20' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), '_stypy_temp_lambda_20', _stypy_temp_lambda_20)
# Getting the type of '_stypy_temp_lambda_20' (line 12)
_stypy_temp_lambda_20_7916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), '_stypy_temp_lambda_20')
# Getting the type of 'l3' (line 12)
l3_7917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 38), 'l3', False)
# Processing the call keyword arguments (line 12)
kwargs_7918 = {}
# Getting the type of 'map' (line 12)
map_7911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'map', False)
# Calling map(args, kwargs) (line 12)
map_call_result_7919 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), map_7911, *[_stypy_temp_lambda_20_7916, l3_7917], **kwargs_7918)

# Assigning a type to the variable 'other_l4' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'other_l4', map_call_result_7919)

# Assigning a Call to a Name (line 13):

# Call to capitalize(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_7925 = {}

# Obtaining the type of the subscript
int_7920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'int')
# Getting the type of 'other_l4' (line 13)
other_l4_7921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'other_l4', False)
# Obtaining the member '__getitem__' of a type (line 13)
getitem___7922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), other_l4_7921, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_7923 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), getitem___7922, int_7920)

# Obtaining the member 'capitalize' of a type (line 13)
capitalize_7924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), subscript_call_result_7923, 'capitalize')
# Calling capitalize(args, kwargs) (line 13)
capitalize_call_result_7926 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), capitalize_7924, *[], **kwargs_7925)

# Assigning a type to the variable 'r3' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r3', capitalize_call_result_7926)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: l = [1, 2, 3, 4]
2: l3 = ["False", "1", "string"]
3: 
4: other_l = reduce(lambda x, y: x + str(y), l, 0)  # Unreported. Runtime Crash
5: r1 = other_l.nothing()  # PyCharm Intellisense shows int type methods, but no error is reported here.
6: 
7: other_l2 = reduce(lambda x, y: x / y, l3, "")
8: r2 = other_l.nothing()  # Same error.
9: 
10: other_l3 = reduce(lambda x, y: x + y, l, 0)
11: r3 = other_l[5]  # Nothing is reported. Unchecked reduce return type. Intellisense shows int methods
12: r4 = other_l.capitalize()  # Nothing is reported
13: 
14: other_l4 = reduce(lambda x, y, z: x + y, l, 0)  # No error report
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):

# Obtaining an instance of the builtin type 'list' (line 1)
list_7934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
int_7935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_7934, int_7935)
# Adding element type (line 1)
int_7936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_7934, int_7936)
# Adding element type (line 1)
int_7937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_7934, int_7937)
# Adding element type (line 1)
int_7938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_7934, int_7938)

# Assigning a type to the variable 'l' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'l', list_7934)

# Assigning a List to a Name (line 2):

# Obtaining an instance of the builtin type 'list' (line 2)
list_7939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_7940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 6), 'str', 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 5), list_7939, str_7940)
# Adding element type (line 2)
str_7941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'str', '1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 5), list_7939, str_7941)
# Adding element type (line 2)
str_7942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 20), 'str', 'string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 5), list_7939, str_7942)

# Assigning a type to the variable 'l3' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'l3', list_7939)

# Assigning a Call to a Name (line 4):

# Call to reduce(...): (line 4)
# Processing the call arguments (line 4)

@norecursion
def _stypy_temp_lambda_21(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_21'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_21', 4, 17, True)
    # Passed parameters checking function
    _stypy_temp_lambda_21.stypy_localization = localization
    _stypy_temp_lambda_21.stypy_type_of_self = None
    _stypy_temp_lambda_21.stypy_type_store = module_type_store
    _stypy_temp_lambda_21.stypy_function_name = '_stypy_temp_lambda_21'
    _stypy_temp_lambda_21.stypy_param_names_list = ['x', 'y']
    _stypy_temp_lambda_21.stypy_varargs_param_name = None
    _stypy_temp_lambda_21.stypy_kwargs_param_name = None
    _stypy_temp_lambda_21.stypy_call_defaults = defaults
    _stypy_temp_lambda_21.stypy_call_varargs = varargs
    _stypy_temp_lambda_21.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_21', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_21', ['x', 'y'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 4)
    x_7944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 30), 'x', False)
    
    # Call to str(...): (line 4)
    # Processing the call arguments (line 4)
    # Getting the type of 'y' (line 4)
    y_7946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 38), 'y', False)
    # Processing the call keyword arguments (line 4)
    kwargs_7947 = {}
    # Getting the type of 'str' (line 4)
    str_7945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 34), 'str', False)
    # Calling str(args, kwargs) (line 4)
    str_call_result_7948 = invoke(stypy.reporting.localization.Localization(__file__, 4, 34), str_7945, *[y_7946], **kwargs_7947)
    
    # Applying the binary operator '+' (line 4)
    result_add_7949 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 30), '+', x_7944, str_call_result_7948)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 17), 'stypy_return_type', result_add_7949)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_21' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_7950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 17), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7950)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_21'
    return stypy_return_type_7950

# Assigning a type to the variable '_stypy_temp_lambda_21' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 17), '_stypy_temp_lambda_21', _stypy_temp_lambda_21)
# Getting the type of '_stypy_temp_lambda_21' (line 4)
_stypy_temp_lambda_21_7951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 17), '_stypy_temp_lambda_21')
# Getting the type of 'l' (line 4)
l_7952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 42), 'l', False)
int_7953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 45), 'int')
# Processing the call keyword arguments (line 4)
kwargs_7954 = {}
# Getting the type of 'reduce' (line 4)
reduce_7943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 10), 'reduce', False)
# Calling reduce(args, kwargs) (line 4)
reduce_call_result_7955 = invoke(stypy.reporting.localization.Localization(__file__, 4, 10), reduce_7943, *[_stypy_temp_lambda_21_7951, l_7952, int_7953], **kwargs_7954)

# Assigning a type to the variable 'other_l' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'other_l', reduce_call_result_7955)

# Assigning a Call to a Name (line 5):

# Call to nothing(...): (line 5)
# Processing the call keyword arguments (line 5)
kwargs_7958 = {}
# Getting the type of 'other_l' (line 5)
other_l_7956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'other_l', False)
# Obtaining the member 'nothing' of a type (line 5)
nothing_7957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), other_l_7956, 'nothing')
# Calling nothing(args, kwargs) (line 5)
nothing_call_result_7959 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), nothing_7957, *[], **kwargs_7958)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', nothing_call_result_7959)

# Assigning a Call to a Name (line 7):

# Call to reduce(...): (line 7)
# Processing the call arguments (line 7)

@norecursion
def _stypy_temp_lambda_22(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_22'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_22', 7, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_22.stypy_localization = localization
    _stypy_temp_lambda_22.stypy_type_of_self = None
    _stypy_temp_lambda_22.stypy_type_store = module_type_store
    _stypy_temp_lambda_22.stypy_function_name = '_stypy_temp_lambda_22'
    _stypy_temp_lambda_22.stypy_param_names_list = ['x', 'y']
    _stypy_temp_lambda_22.stypy_varargs_param_name = None
    _stypy_temp_lambda_22.stypy_kwargs_param_name = None
    _stypy_temp_lambda_22.stypy_call_defaults = defaults
    _stypy_temp_lambda_22.stypy_call_varargs = varargs
    _stypy_temp_lambda_22.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_22', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_22', ['x', 'y'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 7)
    x_7961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 31), 'x', False)
    # Getting the type of 'y' (line 7)
    y_7962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 35), 'y', False)
    # Applying the binary operator 'div' (line 7)
    result_div_7963 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 31), 'div', x_7961, y_7962)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'stypy_return_type', result_div_7963)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_22' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_7964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7964)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_22'
    return stypy_return_type_7964

# Assigning a type to the variable '_stypy_temp_lambda_22' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), '_stypy_temp_lambda_22', _stypy_temp_lambda_22)
# Getting the type of '_stypy_temp_lambda_22' (line 7)
_stypy_temp_lambda_22_7965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), '_stypy_temp_lambda_22')
# Getting the type of 'l3' (line 7)
l3_7966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 38), 'l3', False)
str_7967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 42), 'str', '')
# Processing the call keyword arguments (line 7)
kwargs_7968 = {}
# Getting the type of 'reduce' (line 7)
reduce_7960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'reduce', False)
# Calling reduce(args, kwargs) (line 7)
reduce_call_result_7969 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), reduce_7960, *[_stypy_temp_lambda_22_7965, l3_7966, str_7967], **kwargs_7968)

# Assigning a type to the variable 'other_l2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'other_l2', reduce_call_result_7969)

# Assigning a Call to a Name (line 8):

# Call to nothing(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_7972 = {}
# Getting the type of 'other_l' (line 8)
other_l_7970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'other_l', False)
# Obtaining the member 'nothing' of a type (line 8)
nothing_7971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), other_l_7970, 'nothing')
# Calling nothing(args, kwargs) (line 8)
nothing_call_result_7973 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), nothing_7971, *[], **kwargs_7972)

# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', nothing_call_result_7973)

# Assigning a Call to a Name (line 10):

# Call to reduce(...): (line 10)
# Processing the call arguments (line 10)

@norecursion
def _stypy_temp_lambda_23(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_23'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_23', 10, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_23.stypy_localization = localization
    _stypy_temp_lambda_23.stypy_type_of_self = None
    _stypy_temp_lambda_23.stypy_type_store = module_type_store
    _stypy_temp_lambda_23.stypy_function_name = '_stypy_temp_lambda_23'
    _stypy_temp_lambda_23.stypy_param_names_list = ['x', 'y']
    _stypy_temp_lambda_23.stypy_varargs_param_name = None
    _stypy_temp_lambda_23.stypy_kwargs_param_name = None
    _stypy_temp_lambda_23.stypy_call_defaults = defaults
    _stypy_temp_lambda_23.stypy_call_varargs = varargs
    _stypy_temp_lambda_23.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_23', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_23', ['x', 'y'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 10)
    x_7975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 31), 'x', False)
    # Getting the type of 'y' (line 10)
    y_7976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 35), 'y', False)
    # Applying the binary operator '+' (line 10)
    result_add_7977 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 31), '+', x_7975, y_7976)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'stypy_return_type', result_add_7977)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_23' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_7978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7978)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_23'
    return stypy_return_type_7978

# Assigning a type to the variable '_stypy_temp_lambda_23' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), '_stypy_temp_lambda_23', _stypy_temp_lambda_23)
# Getting the type of '_stypy_temp_lambda_23' (line 10)
_stypy_temp_lambda_23_7979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), '_stypy_temp_lambda_23')
# Getting the type of 'l' (line 10)
l_7980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 38), 'l', False)
int_7981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 41), 'int')
# Processing the call keyword arguments (line 10)
kwargs_7982 = {}
# Getting the type of 'reduce' (line 10)
reduce_7974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'reduce', False)
# Calling reduce(args, kwargs) (line 10)
reduce_call_result_7983 = invoke(stypy.reporting.localization.Localization(__file__, 10, 11), reduce_7974, *[_stypy_temp_lambda_23_7979, l_7980, int_7981], **kwargs_7982)

# Assigning a type to the variable 'other_l3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'other_l3', reduce_call_result_7983)

# Assigning a Subscript to a Name (line 11):

# Obtaining the type of the subscript
int_7984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'int')
# Getting the type of 'other_l' (line 11)
other_l_7985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'other_l')
# Obtaining the member '__getitem__' of a type (line 11)
getitem___7986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), other_l_7985, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_7987 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), getitem___7986, int_7984)

# Assigning a type to the variable 'r3' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r3', subscript_call_result_7987)

# Assigning a Call to a Name (line 12):

# Call to capitalize(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_7990 = {}
# Getting the type of 'other_l' (line 12)
other_l_7988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'other_l', False)
# Obtaining the member 'capitalize' of a type (line 12)
capitalize_7989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), other_l_7988, 'capitalize')
# Calling capitalize(args, kwargs) (line 12)
capitalize_call_result_7991 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), capitalize_7989, *[], **kwargs_7990)

# Assigning a type to the variable 'r4' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r4', capitalize_call_result_7991)

# Assigning a Call to a Name (line 14):

# Call to reduce(...): (line 14)
# Processing the call arguments (line 14)

@norecursion
def _stypy_temp_lambda_24(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_24'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_24', 14, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_24.stypy_localization = localization
    _stypy_temp_lambda_24.stypy_type_of_self = None
    _stypy_temp_lambda_24.stypy_type_store = module_type_store
    _stypy_temp_lambda_24.stypy_function_name = '_stypy_temp_lambda_24'
    _stypy_temp_lambda_24.stypy_param_names_list = ['x', 'y', 'z']
    _stypy_temp_lambda_24.stypy_varargs_param_name = None
    _stypy_temp_lambda_24.stypy_kwargs_param_name = None
    _stypy_temp_lambda_24.stypy_call_defaults = defaults
    _stypy_temp_lambda_24.stypy_call_varargs = varargs
    _stypy_temp_lambda_24.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_24', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_24', ['x', 'y', 'z'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 14)
    x_7993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 34), 'x', False)
    # Getting the type of 'y' (line 14)
    y_7994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'y', False)
    # Applying the binary operator '+' (line 14)
    result_add_7995 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 34), '+', x_7993, y_7994)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'stypy_return_type', result_add_7995)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_24' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_7996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7996)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_24'
    return stypy_return_type_7996

# Assigning a type to the variable '_stypy_temp_lambda_24' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), '_stypy_temp_lambda_24', _stypy_temp_lambda_24)
# Getting the type of '_stypy_temp_lambda_24' (line 14)
_stypy_temp_lambda_24_7997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), '_stypy_temp_lambda_24')
# Getting the type of 'l' (line 14)
l_7998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 41), 'l', False)
int_7999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 44), 'int')
# Processing the call keyword arguments (line 14)
kwargs_8000 = {}
# Getting the type of 'reduce' (line 14)
reduce_7992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'reduce', False)
# Calling reduce(args, kwargs) (line 14)
reduce_call_result_8001 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), reduce_7992, *[_stypy_temp_lambda_24_7997, l_7998, int_7999], **kwargs_8000)

# Assigning a type to the variable 'other_l4' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'other_l4', reduce_call_result_8001)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

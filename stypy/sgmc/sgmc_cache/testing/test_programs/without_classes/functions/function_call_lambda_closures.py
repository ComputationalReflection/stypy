
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: d = dict()
5: 
6: d[1] = "one"
7: d[2] = "two"
8: 
9: cast = dict()
10: 
11: for key in d.keys():
12:     cast[key] = lambda x, k=key: str(k) + str(x)
13:     r = cast[key](10)
14:     print r
15: 
16: for key, val in d.items():
17:     if val not in cast:
18:          cast[val] = key
19: 
20: print d
21: print cast

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 4):

# Call to dict(...): (line 4)
# Processing the call keyword arguments (line 4)
kwargs_743 = {}
# Getting the type of 'dict' (line 4)
dict_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'dict', False)
# Calling dict(args, kwargs) (line 4)
dict_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), dict_742, *[], **kwargs_743)

# Assigning a type to the variable 'd' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'd', dict_call_result_744)

# Assigning a Str to a Subscript (line 6):
str_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 7), 'str', 'one')
# Getting the type of 'd' (line 6)
d_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'd')
int_747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 2), 'int')
# Storing an element on a container (line 6)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 0), d_746, (int_747, str_745))

# Assigning a Str to a Subscript (line 7):
str_748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 7), 'str', 'two')
# Getting the type of 'd' (line 7)
d_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'd')
int_750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 2), 'int')
# Storing an element on a container (line 7)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), d_749, (int_750, str_748))

# Assigning a Call to a Name (line 9):

# Call to dict(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_752 = {}
# Getting the type of 'dict' (line 9)
dict_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 7), 'dict', False)
# Calling dict(args, kwargs) (line 9)
dict_call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 9, 7), dict_751, *[], **kwargs_752)

# Assigning a type to the variable 'cast' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'cast', dict_call_result_753)


# Call to keys(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_756 = {}
# Getting the type of 'd' (line 11)
d_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'd', False)
# Obtaining the member 'keys' of a type (line 11)
keys_755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), d_754, 'keys')
# Calling keys(args, kwargs) (line 11)
keys_call_result_757 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), keys_755, *[], **kwargs_756)

# Testing the type of a for loop iterable (line 11)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 11, 0), keys_call_result_757)
# Getting the type of the for loop variable (line 11)
for_loop_var_758 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 11, 0), keys_call_result_757)
# Assigning a type to the variable 'key' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'key', for_loop_var_758)
# SSA begins for a for statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Lambda to a Subscript (line 12):

@norecursion
def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
    global module_type_store
    # Getting the type of 'key' (line 12)
    key_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), 'key')
    # Assign values to the parameters with defaults
    defaults = [key_759]
    # Create a new context for function '_stypy_temp_lambda_2'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 12, 16, True)
    # Passed parameters checking function
    _stypy_temp_lambda_2.stypy_localization = localization
    _stypy_temp_lambda_2.stypy_type_of_self = None
    _stypy_temp_lambda_2.stypy_type_store = module_type_store
    _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
    _stypy_temp_lambda_2.stypy_param_names_list = ['x', 'k']
    _stypy_temp_lambda_2.stypy_varargs_param_name = None
    _stypy_temp_lambda_2.stypy_kwargs_param_name = None
    _stypy_temp_lambda_2.stypy_call_defaults = defaults
    _stypy_temp_lambda_2.stypy_call_varargs = varargs
    _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_2', ['x', 'k'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to str(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'k' (line 12)
    k_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 37), 'k', False)
    # Processing the call keyword arguments (line 12)
    kwargs_762 = {}
    # Getting the type of 'str' (line 12)
    str_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), 'str', False)
    # Calling str(args, kwargs) (line 12)
    str_call_result_763 = invoke(stypy.reporting.localization.Localization(__file__, 12, 33), str_760, *[k_761], **kwargs_762)
    
    
    # Call to str(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'x' (line 12)
    x_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 46), 'x', False)
    # Processing the call keyword arguments (line 12)
    kwargs_766 = {}
    # Getting the type of 'str' (line 12)
    str_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 42), 'str', False)
    # Calling str(args, kwargs) (line 12)
    str_call_result_767 = invoke(stypy.reporting.localization.Localization(__file__, 12, 42), str_764, *[x_765], **kwargs_766)
    
    # Applying the binary operator '+' (line 12)
    result_add_768 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 33), '+', str_call_result_763, str_call_result_767)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'stypy_return_type', result_add_768)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_2' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_769)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_2'
    return stypy_return_type_769

# Assigning a type to the variable '_stypy_temp_lambda_2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
# Getting the type of '_stypy_temp_lambda_2' (line 12)
_stypy_temp_lambda_2_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), '_stypy_temp_lambda_2')
# Getting the type of 'cast' (line 12)
cast_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'cast')
# Getting the type of 'key' (line 12)
key_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'key')
# Storing an element on a container (line 12)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), cast_771, (key_772, _stypy_temp_lambda_2_770))

# Assigning a Call to a Name (line 13):

# Call to (...): (line 13)
# Processing the call arguments (line 13)
int_777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'int')
# Processing the call keyword arguments (line 13)
kwargs_778 = {}

# Obtaining the type of the subscript
# Getting the type of 'key' (line 13)
key_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'key', False)
# Getting the type of 'cast' (line 13)
cast_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'cast', False)
# Obtaining the member '__getitem__' of a type (line 13)
getitem___775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), cast_774, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), getitem___775, key_773)

# Calling (args, kwargs) (line 13)
_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), subscript_call_result_776, *[int_777], **kwargs_778)

# Assigning a type to the variable 'r' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'r', _call_result_779)
# Getting the type of 'r' (line 14)
r_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'r')
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()



# Call to items(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_783 = {}
# Getting the type of 'd' (line 16)
d_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'd', False)
# Obtaining the member 'items' of a type (line 16)
items_782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), d_781, 'items')
# Calling items(args, kwargs) (line 16)
items_call_result_784 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), items_782, *[], **kwargs_783)

# Testing the type of a for loop iterable (line 16)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 0), items_call_result_784)
# Getting the type of the for loop variable (line 16)
for_loop_var_785 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 0), items_call_result_784)
# Assigning a type to the variable 'key' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 0), for_loop_var_785))
# Assigning a type to the variable 'val' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 0), for_loop_var_785))
# SSA begins for a for statement (line 16)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Getting the type of 'val' (line 17)
val_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'val')
# Getting the type of 'cast' (line 17)
cast_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'cast')
# Applying the binary operator 'notin' (line 17)
result_contains_788 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 7), 'notin', val_786, cast_787)

# Testing the type of an if condition (line 17)
if_condition_789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), result_contains_788)
# Assigning a type to the variable 'if_condition_789' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_789', if_condition_789)
# SSA begins for if statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Subscript (line 18):
# Getting the type of 'key' (line 18)
key_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'key')
# Getting the type of 'cast' (line 18)
cast_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'cast')
# Getting the type of 'val' (line 18)
val_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'val')
# Storing an element on a container (line 18)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), cast_791, (val_792, key_790))
# SSA join for if statement (line 17)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Getting the type of 'd' (line 20)
d_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'd')
# Getting the type of 'cast' (line 21)
cast_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'cast')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

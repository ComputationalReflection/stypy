
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: l = [1, 2, 4, 5]
2: l[0] = l[0] - "a"  # Error detected
3: l[0] -= "a"  # Not detected
4: 
5: s = 3
6: s = s + str(3)  # Error detected
7: 
8: s += str(3)  # Not detected
9: s += str(5)
10: s += str(7)
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):

# Obtaining an instance of the builtin type 'list' (line 1)
list_6937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
int_6938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_6937, int_6938)
# Adding element type (line 1)
int_6939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_6937, int_6939)
# Adding element type (line 1)
int_6940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_6937, int_6940)
# Adding element type (line 1)
int_6941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 4), list_6937, int_6941)

# Assigning a type to the variable 'l' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'l', list_6937)

# Assigning a BinOp to a Subscript (line 2):

# Obtaining the type of the subscript
int_6942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'int')
# Getting the type of 'l' (line 2)
l_6943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 7), 'l')
# Obtaining the member '__getitem__' of a type (line 2)
getitem___6944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2, 7), l_6943, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 2)
subscript_call_result_6945 = invoke(stypy.reporting.localization.Localization(__file__, 2, 7), getitem___6944, int_6942)

str_6946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 14), 'str', 'a')
# Applying the binary operator '-' (line 2)
result_sub_6947 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 7), '-', subscript_call_result_6945, str_6946)

# Getting the type of 'l' (line 2)
l_6948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'l')
int_6949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 2), 'int')
# Storing an element on a container (line 2)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 0), l_6948, (int_6949, result_sub_6947))

# Getting the type of 'l' (line 3)
l_6950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'l')

# Obtaining the type of the subscript
int_6951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 2), 'int')
# Getting the type of 'l' (line 3)
l_6952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'l')
# Obtaining the member '__getitem__' of a type (line 3)
getitem___6953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 0), l_6952, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 3)
subscript_call_result_6954 = invoke(stypy.reporting.localization.Localization(__file__, 3, 0), getitem___6953, int_6951)

str_6955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'str', 'a')
# Applying the binary operator '-=' (line 3)
result_isub_6956 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 0), '-=', subscript_call_result_6954, str_6955)
# Getting the type of 'l' (line 3)
l_6957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'l')
int_6958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 2), 'int')
# Storing an element on a container (line 3)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 0), l_6957, (int_6958, result_isub_6956))


# Assigning a Num to a Name (line 5):
int_6959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 's' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 's', int_6959)

# Assigning a BinOp to a Name (line 6):
# Getting the type of 's' (line 6)
s_6960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 's')

# Call to str(...): (line 6)
# Processing the call arguments (line 6)
int_6962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
# Processing the call keyword arguments (line 6)
kwargs_6963 = {}
# Getting the type of 'str' (line 6)
str_6961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'str', False)
# Calling str(args, kwargs) (line 6)
str_call_result_6964 = invoke(stypy.reporting.localization.Localization(__file__, 6, 8), str_6961, *[int_6962], **kwargs_6963)

# Applying the binary operator '+' (line 6)
result_add_6965 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), '+', s_6960, str_call_result_6964)

# Assigning a type to the variable 's' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 's', result_add_6965)

# Getting the type of 's' (line 8)
s_6966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 's')

# Call to str(...): (line 8)
# Processing the call arguments (line 8)
int_6968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 9), 'int')
# Processing the call keyword arguments (line 8)
kwargs_6969 = {}
# Getting the type of 'str' (line 8)
str_6967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'str', False)
# Calling str(args, kwargs) (line 8)
str_call_result_6970 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), str_6967, *[int_6968], **kwargs_6969)

# Applying the binary operator '+=' (line 8)
result_iadd_6971 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 0), '+=', s_6966, str_call_result_6970)
# Assigning a type to the variable 's' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 's', result_iadd_6971)


# Getting the type of 's' (line 9)
s_6972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 's')

# Call to str(...): (line 9)
# Processing the call arguments (line 9)
int_6974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'int')
# Processing the call keyword arguments (line 9)
kwargs_6975 = {}
# Getting the type of 'str' (line 9)
str_6973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'str', False)
# Calling str(args, kwargs) (line 9)
str_call_result_6976 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), str_6973, *[int_6974], **kwargs_6975)

# Applying the binary operator '+=' (line 9)
result_iadd_6977 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 0), '+=', s_6972, str_call_result_6976)
# Assigning a type to the variable 's' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 's', result_iadd_6977)


# Getting the type of 's' (line 10)
s_6978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 's')

# Call to str(...): (line 10)
# Processing the call arguments (line 10)
int_6980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'int')
# Processing the call keyword arguments (line 10)
kwargs_6981 = {}
# Getting the type of 'str' (line 10)
str_6979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'str', False)
# Calling str(args, kwargs) (line 10)
str_call_result_6982 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), str_6979, *[int_6980], **kwargs_6981)

# Applying the binary operator '+=' (line 10)
result_iadd_6983 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 0), '+=', s_6978, str_call_result_6982)
# Assigning a type to the variable 's' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 's', result_iadd_6983)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

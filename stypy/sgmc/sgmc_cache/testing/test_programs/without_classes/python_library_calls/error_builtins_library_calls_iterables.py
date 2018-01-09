
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
4: normal_list = range(5)
5: 
6: r1 = math.fsum(words)  # Reported
7: r2 = math.fsum(list) # Not reported
8: r3 = len(3)  # Reported
9: r4 = len(list)  # Not reported
10: 
11: r5 = math.fsum(normal_list)
12: 
13: 

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


# Assigning a List to a Name (line 3):

# Obtaining an instance of the builtin type 'list' (line 3)
list_7096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_7097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 9), 'str', 'The')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7097)
# Adding element type (line 3)
str_7098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'str', 'quick')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7098)
# Adding element type (line 3)
str_7099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'str', 'brown')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7099)
# Adding element type (line 3)
str_7100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 34), 'str', 'fox')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7100)
# Adding element type (line 3)
str_7101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 41), 'str', 'jumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7101)
# Adding element type (line 3)
str_7102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 50), 'str', 'over')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7102)
# Adding element type (line 3)
str_7103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 58), 'str', 'the')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7103)
# Adding element type (line 3)
str_7104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 65), 'str', 'lazy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7104)
# Adding element type (line 3)
str_7105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 73), 'str', 'dog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_7096, str_7105)

# Assigning a type to the variable 'words' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'words', list_7096)

# Assigning a Call to a Name (line 4):

# Call to range(...): (line 4)
# Processing the call arguments (line 4)
int_7107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 20), 'int')
# Processing the call keyword arguments (line 4)
kwargs_7108 = {}
# Getting the type of 'range' (line 4)
range_7106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), 'range', False)
# Calling range(args, kwargs) (line 4)
range_call_result_7109 = invoke(stypy.reporting.localization.Localization(__file__, 4, 14), range_7106, *[int_7107], **kwargs_7108)

# Assigning a type to the variable 'normal_list' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'normal_list', range_call_result_7109)

# Assigning a Call to a Name (line 6):

# Call to fsum(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'words' (line 6)
words_7112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'words', False)
# Processing the call keyword arguments (line 6)
kwargs_7113 = {}
# Getting the type of 'math' (line 6)
math_7110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'math', False)
# Obtaining the member 'fsum' of a type (line 6)
fsum_7111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), math_7110, 'fsum')
# Calling fsum(args, kwargs) (line 6)
fsum_call_result_7114 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), fsum_7111, *[words_7112], **kwargs_7113)

# Assigning a type to the variable 'r1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r1', fsum_call_result_7114)

# Assigning a Call to a Name (line 7):

# Call to fsum(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'list' (line 7)
list_7117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'list', False)
# Processing the call keyword arguments (line 7)
kwargs_7118 = {}
# Getting the type of 'math' (line 7)
math_7115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'math', False)
# Obtaining the member 'fsum' of a type (line 7)
fsum_7116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), math_7115, 'fsum')
# Calling fsum(args, kwargs) (line 7)
fsum_call_result_7119 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), fsum_7116, *[list_7117], **kwargs_7118)

# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r2', fsum_call_result_7119)

# Assigning a Call to a Name (line 8):

# Call to len(...): (line 8)
# Processing the call arguments (line 8)
int_7121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 9), 'int')
# Processing the call keyword arguments (line 8)
kwargs_7122 = {}
# Getting the type of 'len' (line 8)
len_7120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'len', False)
# Calling len(args, kwargs) (line 8)
len_call_result_7123 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), len_7120, *[int_7121], **kwargs_7122)

# Assigning a type to the variable 'r3' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r3', len_call_result_7123)

# Assigning a Call to a Name (line 9):

# Call to len(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'list' (line 9)
list_7125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 9), 'list', False)
# Processing the call keyword arguments (line 9)
kwargs_7126 = {}
# Getting the type of 'len' (line 9)
len_7124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'len', False)
# Calling len(args, kwargs) (line 9)
len_call_result_7127 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), len_7124, *[list_7125], **kwargs_7126)

# Assigning a type to the variable 'r4' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r4', len_call_result_7127)

# Assigning a Call to a Name (line 11):

# Call to fsum(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'normal_list' (line 11)
normal_list_7130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'normal_list', False)
# Processing the call keyword arguments (line 11)
kwargs_7131 = {}
# Getting the type of 'math' (line 11)
math_7128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'math', False)
# Obtaining the member 'fsum' of a type (line 11)
fsum_7129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), math_7128, 'fsum')
# Calling fsum(args, kwargs) (line 11)
fsum_call_result_7132 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), fsum_7129, *[normal_list_7130], **kwargs_7131)

# Assigning a type to the variable 'r5' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r5', fsum_call_result_7132)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator
2: 
3: a = 5
4: b = 6
5: 
6: x = 1 if a > b else -1
7: y = 1 if a > b else -1 if a < b else 0
8: 
9: z = 1 if a > b else "foo"
10: 
11: print x
12: print y
13: print z
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_2734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Assigning a type to the variable 'a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'a', int_2734)

# Assigning a Num to a Name (line 4):
int_2735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'b', int_2735)

# Assigning a IfExp to a Name (line 6):


# Getting the type of 'a' (line 6)
a_2736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'a')
# Getting the type of 'b' (line 6)
b_2737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'b')
# Applying the binary operator '>' (line 6)
result_gt_2738 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 9), '>', a_2736, b_2737)

# Testing the type of an if expression (line 6)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 4), result_gt_2738)
# SSA begins for if expression (line 6)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
int_2739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')
# SSA branch for the else part of an if expression (line 6)
module_type_store.open_ssa_branch('if expression else')
int_2740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
# SSA join for if expression (line 6)
module_type_store = module_type_store.join_ssa_context()
if_exp_2741 = union_type.UnionType.add(int_2739, int_2740)

# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', if_exp_2741)

# Assigning a IfExp to a Name (line 7):


# Getting the type of 'a' (line 7)
a_2742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'a')
# Getting the type of 'b' (line 7)
b_2743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'b')
# Applying the binary operator '>' (line 7)
result_gt_2744 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 9), '>', a_2742, b_2743)

# Testing the type of an if expression (line 7)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 4), result_gt_2744)
# SSA begins for if expression (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
int_2745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'int')
# SSA branch for the else part of an if expression (line 7)
module_type_store.open_ssa_branch('if expression else')


# Getting the type of 'a' (line 7)
a_2746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'a')
# Getting the type of 'b' (line 7)
b_2747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 30), 'b')
# Applying the binary operator '<' (line 7)
result_lt_2748 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 26), '<', a_2746, b_2747)

# Testing the type of an if expression (line 7)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 20), result_lt_2748)
# SSA begins for if expression (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
int_2749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
# SSA branch for the else part of an if expression (line 7)
module_type_store.open_ssa_branch('if expression else')
int_2750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 37), 'int')
# SSA join for if expression (line 7)
module_type_store = module_type_store.join_ssa_context()
if_exp_2751 = union_type.UnionType.add(int_2749, int_2750)

# SSA join for if expression (line 7)
module_type_store = module_type_store.join_ssa_context()
if_exp_2752 = union_type.UnionType.add(int_2745, if_exp_2751)

# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'y', if_exp_2752)

# Assigning a IfExp to a Name (line 9):


# Getting the type of 'a' (line 9)
a_2753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 9), 'a')
# Getting the type of 'b' (line 9)
b_2754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'b')
# Applying the binary operator '>' (line 9)
result_gt_2755 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 9), '>', a_2753, b_2754)

# Testing the type of an if expression (line 9)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 4), result_gt_2755)
# SSA begins for if expression (line 9)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
int_2756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'int')
# SSA branch for the else part of an if expression (line 9)
module_type_store.open_ssa_branch('if expression else')
str_2757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'foo')
# SSA join for if expression (line 9)
module_type_store = module_type_store.join_ssa_context()
if_exp_2758 = union_type.UnionType.add(int_2756, str_2757)

# Assigning a type to the variable 'z' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'z', if_exp_2758)
# Getting the type of 'x' (line 11)
x_2759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'x')
# Getting the type of 'y' (line 12)
y_2760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'y')
# Getting the type of 'z' (line 13)
z_2761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 6), 'z')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

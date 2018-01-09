
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: 
5: ref_dir = os.path.join(os.path.dirname(__file__))
6: 
7: __all__ = sorted(f[:-3] for f in os.listdir(ref_dir) if f.endswith('.py') and
8:            not f.startswith('__'))
9: 
10: for f in __all__:
11:     __import__(__name__ + '.' + f)
12: 
13: del f, ref_dir
14: 
15: __doc__ = '''\
16: Topical documentation
17: =====================
18: 
19: The following topics are available:
20: %s
21: 
22: You can view them by
23: 
24: >>> help(np.doc.TOPIC)                                      #doctest: +SKIP
25: 
26: ''' % '\n- '.join([''] + __all__)
27: 
28: __all__.extend(['__doc__'])
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)


# Assigning a Call to a Name (line 5):

# Call to join(...): (line 5)
# Processing the call arguments (line 5)

# Call to dirname(...): (line 5)
# Processing the call arguments (line 5)
# Getting the type of '__file__' (line 5)
file___66618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 39), '__file__', False)
# Processing the call keyword arguments (line 5)
kwargs_66619 = {}
# Getting the type of 'os' (line 5)
os_66615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 23), 'os', False)
# Obtaining the member 'path' of a type (line 5)
path_66616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 23), os_66615, 'path')
# Obtaining the member 'dirname' of a type (line 5)
dirname_66617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 23), path_66616, 'dirname')
# Calling dirname(args, kwargs) (line 5)
dirname_call_result_66620 = invoke(stypy.reporting.localization.Localization(__file__, 5, 23), dirname_66617, *[file___66618], **kwargs_66619)

# Processing the call keyword arguments (line 5)
kwargs_66621 = {}
# Getting the type of 'os' (line 5)
os_66612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 10), 'os', False)
# Obtaining the member 'path' of a type (line 5)
path_66613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 10), os_66612, 'path')
# Obtaining the member 'join' of a type (line 5)
join_66614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 10), path_66613, 'join')
# Calling join(args, kwargs) (line 5)
join_call_result_66622 = invoke(stypy.reporting.localization.Localization(__file__, 5, 10), join_66614, *[dirname_call_result_66620], **kwargs_66621)

# Assigning a type to the variable 'ref_dir' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'ref_dir', join_call_result_66622)

# Assigning a Call to a Name (line 7):

# Call to sorted(...): (line 7)
# Processing the call arguments (line 7)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 7, 17, True)
# Calculating comprehension expression

# Call to listdir(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'ref_dir' (line 7)
ref_dir_66643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 44), 'ref_dir', False)
# Processing the call keyword arguments (line 7)
kwargs_66644 = {}
# Getting the type of 'os' (line 7)
os_66641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 33), 'os', False)
# Obtaining the member 'listdir' of a type (line 7)
listdir_66642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 33), os_66641, 'listdir')
# Calling listdir(args, kwargs) (line 7)
listdir_call_result_66645 = invoke(stypy.reporting.localization.Localization(__file__, 7, 33), listdir_66642, *[ref_dir_66643], **kwargs_66644)

comprehension_66646 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 17), listdir_call_result_66645)
# Assigning a type to the variable 'f' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 17), 'f', comprehension_66646)

# Evaluating a boolean operation

# Call to endswith(...): (line 7)
# Processing the call arguments (line 7)
str_66631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 67), 'str', '.py')
# Processing the call keyword arguments (line 7)
kwargs_66632 = {}
# Getting the type of 'f' (line 7)
f_66629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 56), 'f', False)
# Obtaining the member 'endswith' of a type (line 7)
endswith_66630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 56), f_66629, 'endswith')
# Calling endswith(args, kwargs) (line 7)
endswith_call_result_66633 = invoke(stypy.reporting.localization.Localization(__file__, 7, 56), endswith_66630, *[str_66631], **kwargs_66632)



# Call to startswith(...): (line 8)
# Processing the call arguments (line 8)
str_66636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 28), 'str', '__')
# Processing the call keyword arguments (line 8)
kwargs_66637 = {}
# Getting the type of 'f' (line 8)
f_66634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'f', False)
# Obtaining the member 'startswith' of a type (line 8)
startswith_66635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 15), f_66634, 'startswith')
# Calling startswith(args, kwargs) (line 8)
startswith_call_result_66638 = invoke(stypy.reporting.localization.Localization(__file__, 8, 15), startswith_66635, *[str_66636], **kwargs_66637)

# Applying the 'not' unary operator (line 8)
result_not__66639 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 11), 'not', startswith_call_result_66638)

# Applying the binary operator 'and' (line 7)
result_and_keyword_66640 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 56), 'and', endswith_call_result_66633, result_not__66639)


# Obtaining the type of the subscript
int_66624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
slice_66625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 17), None, int_66624, None)
# Getting the type of 'f' (line 7)
f_66626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 17), 'f', False)
# Obtaining the member '__getitem__' of a type (line 7)
getitem___66627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 17), f_66626, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_66628 = invoke(stypy.reporting.localization.Localization(__file__, 7, 17), getitem___66627, slice_66625)

list_66647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 17), list_66647, subscript_call_result_66628)
# Processing the call keyword arguments (line 7)
kwargs_66648 = {}
# Getting the type of 'sorted' (line 7)
sorted_66623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'sorted', False)
# Calling sorted(args, kwargs) (line 7)
sorted_call_result_66649 = invoke(stypy.reporting.localization.Localization(__file__, 7, 10), sorted_66623, *[list_66647], **kwargs_66648)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', sorted_call_result_66649)

# Getting the type of '__all__' (line 10)
all___66650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), '__all__')
# Testing the type of a for loop iterable (line 10)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 10, 0), all___66650)
# Getting the type of the for loop variable (line 10)
for_loop_var_66651 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 10, 0), all___66650)
# Assigning a type to the variable 'f' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'f', for_loop_var_66651)
# SSA begins for a for statement (line 10)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to __import__(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of '__name__' (line 11)
name___66653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), '__name__', False)
str_66654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', '.')
# Applying the binary operator '+' (line 11)
result_add_66655 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 15), '+', name___66653, str_66654)

# Getting the type of 'f' (line 11)
f_66656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 32), 'f', False)
# Applying the binary operator '+' (line 11)
result_add_66657 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 30), '+', result_add_66655, f_66656)

# Processing the call keyword arguments (line 11)
kwargs_66658 = {}
# Getting the type of '__import__' (line 11)
import___66652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), '__import__', False)
# Calling __import__(args, kwargs) (line 11)
import___call_result_66659 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), import___66652, *[result_add_66657], **kwargs_66658)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 13, 0), module_type_store, 'f')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 13, 0), module_type_store, 'ref_dir')

# Assigning a BinOp to a Name (line 15):
str_66660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', 'Topical documentation\n=====================\n\nThe following topics are available:\n%s\n\nYou can view them by\n\n>>> help(np.doc.TOPIC)                                      #doctest: +SKIP\n\n')

# Call to join(...): (line 26)
# Processing the call arguments (line 26)

# Obtaining an instance of the builtin type 'list' (line 26)
list_66663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_66664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', '')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 18), list_66663, str_66664)

# Getting the type of '__all__' (line 26)
all___66665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), '__all__', False)
# Applying the binary operator '+' (line 26)
result_add_66666 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 18), '+', list_66663, all___66665)

# Processing the call keyword arguments (line 26)
kwargs_66667 = {}
str_66661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 6), 'str', '\n- ')
# Obtaining the member 'join' of a type (line 26)
join_66662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), str_66661, 'join')
# Calling join(args, kwargs) (line 26)
join_call_result_66668 = invoke(stypy.reporting.localization.Localization(__file__, 26, 6), join_66662, *[result_add_66666], **kwargs_66667)

# Applying the binary operator '%' (line 26)
result_mod_66669 = python_operator(stypy.reporting.localization.Localization(__file__, 26, (-1)), '%', str_66660, join_call_result_66668)

# Assigning a type to the variable '__doc__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__doc__', result_mod_66669)

# Call to extend(...): (line 28)
# Processing the call arguments (line 28)

# Obtaining an instance of the builtin type 'list' (line 28)
list_66672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_66673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'str', '__doc__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_66672, str_66673)

# Processing the call keyword arguments (line 28)
kwargs_66674 = {}
# Getting the type of '__all__' (line 28)
all___66670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '__all__', False)
# Obtaining the member 'extend' of a type (line 28)
extend_66671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 0), all___66670, 'extend')
# Calling extend(args, kwargs) (line 28)
extend_call_result_66675 = invoke(stypy.reporting.localization.Localization(__file__, 28, 0), extend_66671, *[list_66672], **kwargs_66674)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

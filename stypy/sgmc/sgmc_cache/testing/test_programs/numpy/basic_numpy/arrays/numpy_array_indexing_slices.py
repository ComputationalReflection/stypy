
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: # Create the following rank 2 array with shape (3, 4)
6: # [[ 1  2  3  4]
7: #  [ 5  6  7  8]
8: #  [ 9 10 11 12]]
9: a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
10: 
11: # Two ways of accessing the data in the middle row of the array.
12: # Mixing integer indexing with slices yields an array of lower rank,
13: # while using only slices yields an array of the same rank as the
14: # original array:
15: row_r1 = a[1, :]  # Rank 1 view of the second row of a
16: row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
17: r1, r2 = row_r1, row_r1.shape  # Prints "[5 6 7 8] (4,)"
18: r3, r4 = row_r2, row_r2.shape  # Prints "[[5 6 7 8]] (1, 4)"
19: 
20: # We can make the same distinction when accessing columns of an array:
21: col_r1 = a[:, 1]
22: col_r2 = a[:, 1:2]
23: r5, r6 = col_r1, col_r1.shape  # Prints "[ 2  6 10] (3,)"
24: r7, r8 = col_r2, col_r2.shape  # Prints "[[ 2]
25: #          [ 6]
26: #          [10]] (3, 1)"
27: #
28: # l = globals().copy()
29: # for v in l:
30: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_652 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_652) is not StypyTypeError):

    if (import_652 != 'pyd_module'):
        __import__(import_652)
        sys_modules_653 = sys.modules[import_652]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_653.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_652)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 9):

# Assigning a Call to a Name (line 9):

# Call to array(...): (line 9)
# Processing the call arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_657, int_658)
# Adding element type (line 9)
int_659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_657, int_659)
# Adding element type (line 9)
int_660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_657, int_660)
# Adding element type (line 9)
int_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_657, int_661)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_656, list_657)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_662, int_663)
# Adding element type (line 9)
int_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_662, int_664)
# Adding element type (line 9)
int_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_662, int_665)
# Adding element type (line 9)
int_666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_662, int_666)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_656, list_662)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 42), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_667, int_668)
# Adding element type (line 9)
int_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_667, int_669)
# Adding element type (line 9)
int_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_667, int_670)
# Adding element type (line 9)
int_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_667, int_671)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_656, list_667)

# Processing the call keyword arguments (line 9)
kwargs_672 = {}
# Getting the type of 'np' (line 9)
np_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'array' of a type (line 9)
array_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_654, 'array')
# Calling array(args, kwargs) (line 9)
array_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), array_655, *[list_656], **kwargs_672)

# Assigning a type to the variable 'a' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'a', array_call_result_673)

# Assigning a Subscript to a Name (line 15):

# Assigning a Subscript to a Name (line 15):

# Obtaining the type of the subscript
int_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'int')
slice_675 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 15, 9), None, None, None)
# Getting the type of 'a' (line 15)
a_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), a_676, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), getitem___677, (int_674, slice_675))

# Assigning a type to the variable 'row_r1' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'row_r1', subscript_call_result_678)

# Assigning a Subscript to a Name (line 16):

# Assigning a Subscript to a Name (line 16):

# Obtaining the type of the subscript
int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
int_680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
slice_681 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 16, 9), int_679, int_680, None)
slice_682 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 16, 9), None, None, None)
# Getting the type of 'a' (line 16)
a_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'a')
# Obtaining the member '__getitem__' of a type (line 16)
getitem___684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), a_683, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 16)
subscript_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), getitem___684, (slice_681, slice_682))

# Assigning a type to the variable 'row_r2' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'row_r2', subscript_call_result_685)

# Assigning a Tuple to a Tuple (line 17):

# Assigning a Name to a Name (line 17):
# Getting the type of 'row_r1' (line 17)
row_r1_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 9), 'row_r1')
# Assigning a type to the variable 'tuple_assignment_644' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'tuple_assignment_644', row_r1_686)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'row_r1' (line 17)
row_r1_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'row_r1')
# Obtaining the member 'shape' of a type (line 17)
shape_688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 17), row_r1_687, 'shape')
# Assigning a type to the variable 'tuple_assignment_645' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'tuple_assignment_645', shape_688)

# Assigning a Name to a Name (line 17):
# Getting the type of 'tuple_assignment_644' (line 17)
tuple_assignment_644_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'tuple_assignment_644')
# Assigning a type to the variable 'r1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r1', tuple_assignment_644_689)

# Assigning a Name to a Name (line 17):
# Getting the type of 'tuple_assignment_645' (line 17)
tuple_assignment_645_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'tuple_assignment_645')
# Assigning a type to the variable 'r2' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'r2', tuple_assignment_645_690)

# Assigning a Tuple to a Tuple (line 18):

# Assigning a Name to a Name (line 18):
# Getting the type of 'row_r2' (line 18)
row_r2_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'row_r2')
# Assigning a type to the variable 'tuple_assignment_646' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'tuple_assignment_646', row_r2_691)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'row_r2' (line 18)
row_r2_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'row_r2')
# Obtaining the member 'shape' of a type (line 18)
shape_693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), row_r2_692, 'shape')
# Assigning a type to the variable 'tuple_assignment_647' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'tuple_assignment_647', shape_693)

# Assigning a Name to a Name (line 18):
# Getting the type of 'tuple_assignment_646' (line 18)
tuple_assignment_646_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'tuple_assignment_646')
# Assigning a type to the variable 'r3' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r3', tuple_assignment_646_694)

# Assigning a Name to a Name (line 18):
# Getting the type of 'tuple_assignment_647' (line 18)
tuple_assignment_647_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'tuple_assignment_647')
# Assigning a type to the variable 'r4' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r4', tuple_assignment_647_695)

# Assigning a Subscript to a Name (line 21):

# Assigning a Subscript to a Name (line 21):

# Obtaining the type of the subscript
slice_696 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 21, 9), None, None, None)
int_697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'int')
# Getting the type of 'a' (line 21)
a_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'a')
# Obtaining the member '__getitem__' of a type (line 21)
getitem___699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 9), a_698, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 21)
subscript_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 21, 9), getitem___699, (slice_696, int_697))

# Assigning a type to the variable 'col_r1' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'col_r1', subscript_call_result_700)

# Assigning a Subscript to a Name (line 22):

# Assigning a Subscript to a Name (line 22):

# Obtaining the type of the subscript
slice_701 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 22, 9), None, None, None)
int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'int')
int_703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
slice_704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 22, 9), int_702, int_703, None)
# Getting the type of 'a' (line 22)
a_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'a')
# Obtaining the member '__getitem__' of a type (line 22)
getitem___706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 9), a_705, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 22, 9), getitem___706, (slice_701, slice_704))

# Assigning a type to the variable 'col_r2' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'col_r2', subscript_call_result_707)

# Assigning a Tuple to a Tuple (line 23):

# Assigning a Name to a Name (line 23):
# Getting the type of 'col_r1' (line 23)
col_r1_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 9), 'col_r1')
# Assigning a type to the variable 'tuple_assignment_648' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'tuple_assignment_648', col_r1_708)

# Assigning a Attribute to a Name (line 23):
# Getting the type of 'col_r1' (line 23)
col_r1_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'col_r1')
# Obtaining the member 'shape' of a type (line 23)
shape_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 17), col_r1_709, 'shape')
# Assigning a type to the variable 'tuple_assignment_649' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'tuple_assignment_649', shape_710)

# Assigning a Name to a Name (line 23):
# Getting the type of 'tuple_assignment_648' (line 23)
tuple_assignment_648_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'tuple_assignment_648')
# Assigning a type to the variable 'r5' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r5', tuple_assignment_648_711)

# Assigning a Name to a Name (line 23):
# Getting the type of 'tuple_assignment_649' (line 23)
tuple_assignment_649_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'tuple_assignment_649')
# Assigning a type to the variable 'r6' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'r6', tuple_assignment_649_712)

# Assigning a Tuple to a Tuple (line 24):

# Assigning a Name to a Name (line 24):
# Getting the type of 'col_r2' (line 24)
col_r2_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 9), 'col_r2')
# Assigning a type to the variable 'tuple_assignment_650' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'tuple_assignment_650', col_r2_713)

# Assigning a Attribute to a Name (line 24):
# Getting the type of 'col_r2' (line 24)
col_r2_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'col_r2')
# Obtaining the member 'shape' of a type (line 24)
shape_715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 17), col_r2_714, 'shape')
# Assigning a type to the variable 'tuple_assignment_651' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'tuple_assignment_651', shape_715)

# Assigning a Name to a Name (line 24):
# Getting the type of 'tuple_assignment_650' (line 24)
tuple_assignment_650_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'tuple_assignment_650')
# Assigning a type to the variable 'r7' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r7', tuple_assignment_650_716)

# Assigning a Name to a Name (line 24):
# Getting the type of 'tuple_assignment_651' (line 24)
tuple_assignment_651_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'tuple_assignment_651')
# Assigning a type to the variable 'r8' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'r8', tuple_assignment_651_717)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

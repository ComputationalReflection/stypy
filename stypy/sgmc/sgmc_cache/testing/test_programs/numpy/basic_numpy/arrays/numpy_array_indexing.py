
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
11: # Use slicing to pull out the subarray consisting of the first 2 rows
12: # and columns 1 and 2; b is the following array of shape (2, 2):
13: # [[2 3]
14: #  [6 7]]
15: b = a[:2, 1:3]
16: 
17: # A slice of an array is a view into the same data, so modifying it
18: # will modify the original array.
19: r = a[0, 1]  # Prints "2"
20: b[0, 0] = 77  # b[0, 0] is the same piece of data as a[0, 1]
21: r2 = a[0, 1]  # Prints "77"
22: 
23: # l = globals().copy()
24: #
25: # for v in l:
26: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_597 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_597) is not StypyTypeError):

    if (import_597 != 'pyd_module'):
        __import__(import_597)
        sys_modules_598 = sys.modules[import_597]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_598.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_597)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 9):

# Call to array(...): (line 9)
# Processing the call arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_602, int_603)
# Adding element type (line 9)
int_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_602, int_604)
# Adding element type (line 9)
int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_602, int_605)
# Adding element type (line 9)
int_606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_602, int_606)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_601, list_602)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_607, int_608)
# Adding element type (line 9)
int_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_607, int_609)
# Adding element type (line 9)
int_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_607, int_610)
# Adding element type (line 9)
int_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 28), list_607, int_611)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_601, list_607)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 42), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_612, int_613)
# Adding element type (line 9)
int_614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_612, int_614)
# Adding element type (line 9)
int_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_612, int_615)
# Adding element type (line 9)
int_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 42), list_612, int_616)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_601, list_612)

# Processing the call keyword arguments (line 9)
kwargs_617 = {}
# Getting the type of 'np' (line 9)
np_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'array' of a type (line 9)
array_600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_599, 'array')
# Calling array(args, kwargs) (line 9)
array_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), array_600, *[list_601], **kwargs_617)

# Assigning a type to the variable 'a' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'a', array_call_result_618)

# Assigning a Subscript to a Name (line 15):

# Obtaining the type of the subscript
int_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 7), 'int')
slice_620 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 15, 4), None, int_619, None)
int_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
slice_623 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 15, 4), int_621, int_622, None)
# Getting the type of 'a' (line 15)
a_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'a')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), a_624, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), getitem___625, (slice_620, slice_623))

# Assigning a type to the variable 'b' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'b', subscript_call_result_626)

# Assigning a Subscript to a Name (line 19):

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
int_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 6), tuple_627, int_628)
# Adding element type (line 19)
int_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 6), tuple_627, int_629)

# Getting the type of 'a' (line 19)
a_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'a')
# Obtaining the member '__getitem__' of a type (line 19)
getitem___631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), a_630, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), getitem___631, tuple_627)

# Assigning a type to the variable 'r' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r', subscript_call_result_632)

# Assigning a Num to a Subscript (line 20):
int_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'int')
# Getting the type of 'b' (line 20)
b_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'b')

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 2), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
int_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 2), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 2), tuple_635, int_636)
# Adding element type (line 20)
int_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 2), tuple_635, int_637)

# Storing an element on a container (line 20)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 0), b_634, (tuple_635, int_633))

# Assigning a Subscript to a Name (line 21):

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 7), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)
# Adding element type (line 21)
int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 7), tuple_638, int_639)
# Adding element type (line 21)
int_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 7), tuple_638, int_640)

# Getting the type of 'a' (line 21)
a_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'a')
# Obtaining the member '__getitem__' of a type (line 21)
getitem___642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), a_641, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 21)
subscript_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), getitem___642, tuple_638)

# Assigning a type to the variable 'r2' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r2', subscript_call_result_643)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

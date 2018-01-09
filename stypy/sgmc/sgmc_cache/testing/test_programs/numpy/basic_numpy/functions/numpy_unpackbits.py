
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
6: B = ((I.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
7: r = (B[:, ::-1])
8: 
9: # Author: Daniel T. McDonald
10: 
11: I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
12: r2 = (np.unpackbits(I[:, np.newaxis], axis=1))
13: 
14: # l = globals().copy()
15: # for v in l:
16: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2613 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2613) is not StypyTypeError):

    if (import_2613 != 'pyd_module'):
        __import__(import_2613)
        sys_modules_2614 = sys.modules[import_2613]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2614.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2613)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_2618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2618)
# Adding element type (line 5)
int_2619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2619)
# Adding element type (line 5)
int_2620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2620)
# Adding element type (line 5)
int_2621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2621)
# Adding element type (line 5)
int_2622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2622)
# Adding element type (line 5)
int_2623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2623)
# Adding element type (line 5)
int_2624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2624)
# Adding element type (line 5)
int_2625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2625)
# Adding element type (line 5)
int_2626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2617, int_2626)

# Processing the call keyword arguments (line 5)
kwargs_2627 = {}
# Getting the type of 'np' (line 5)
np_2615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_2616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2615, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_2628 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_2616, *[list_2617], **kwargs_2627)

# Assigning a type to the variable 'I' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'I', array_call_result_2628)

# Assigning a Call to a Name (line 6):

# Call to astype(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'int' (line 6)
int_2646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 59), 'int', False)
# Processing the call keyword arguments (line 6)
kwargs_2647 = {}


# Call to reshape(...): (line 6)
# Processing the call arguments (line 6)
int_2631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'int')
int_2632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
# Processing the call keyword arguments (line 6)
kwargs_2633 = {}
# Getting the type of 'I' (line 6)
I_2629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'I', False)
# Obtaining the member 'reshape' of a type (line 6)
reshape_2630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 6), I_2629, 'reshape')
# Calling reshape(args, kwargs) (line 6)
reshape_call_result_2634 = invoke(stypy.reporting.localization.Localization(__file__, 6, 6), reshape_2630, *[int_2631, int_2632], **kwargs_2633)

int_2635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')

# Call to arange(...): (line 6)
# Processing the call arguments (line 6)
int_2638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 41), 'int')
# Processing the call keyword arguments (line 6)
kwargs_2639 = {}
# Getting the type of 'np' (line 6)
np_2636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 31), 'np', False)
# Obtaining the member 'arange' of a type (line 6)
arange_2637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 31), np_2636, 'arange')
# Calling arange(args, kwargs) (line 6)
arange_call_result_2640 = invoke(stypy.reporting.localization.Localization(__file__, 6, 31), arange_2637, *[int_2638], **kwargs_2639)

# Applying the binary operator '**' (line 6)
result_pow_2641 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 26), '**', int_2635, arange_call_result_2640)

# Applying the binary operator '&' (line 6)
result_and__2642 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 6), '&', reshape_call_result_2634, result_pow_2641)

int_2643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 49), 'int')
# Applying the binary operator '!=' (line 6)
result_ne_2644 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 5), '!=', result_and__2642, int_2643)

# Obtaining the member 'astype' of a type (line 6)
astype_2645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), result_ne_2644, 'astype')
# Calling astype(args, kwargs) (line 6)
astype_call_result_2648 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), astype_2645, *[int_2646], **kwargs_2647)

# Assigning a type to the variable 'B' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'B', astype_call_result_2648)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
slice_2649 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 5), None, None, None)
int_2650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
slice_2651 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 5), None, None, int_2650)
# Getting the type of 'B' (line 7)
B_2652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'B')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___2653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), B_2652, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_2654 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), getitem___2653, (slice_2649, slice_2651))

# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r', subscript_call_result_2654)

# Assigning a Call to a Name (line 11):

# Call to array(...): (line 11)
# Processing the call arguments (line 11)

# Obtaining an instance of the builtin type 'list' (line 11)
list_2657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
int_2658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2658)
# Adding element type (line 11)
int_2659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2659)
# Adding element type (line 11)
int_2660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2660)
# Adding element type (line 11)
int_2661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2661)
# Adding element type (line 11)
int_2662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2662)
# Adding element type (line 11)
int_2663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2663)
# Adding element type (line 11)
int_2664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2664)
# Adding element type (line 11)
int_2665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2665)
# Adding element type (line 11)
int_2666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_2657, int_2666)

# Processing the call keyword arguments (line 11)
# Getting the type of 'np' (line 11)
np_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 54), 'np', False)
# Obtaining the member 'uint8' of a type (line 11)
uint8_2668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 54), np_2667, 'uint8')
keyword_2669 = uint8_2668
kwargs_2670 = {'dtype': keyword_2669}
# Getting the type of 'np' (line 11)
np_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'array' of a type (line 11)
array_2656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_2655, 'array')
# Calling array(args, kwargs) (line 11)
array_call_result_2671 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), array_2656, *[list_2657], **kwargs_2670)

# Assigning a type to the variable 'I' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'I', array_call_result_2671)

# Assigning a Call to a Name (line 12):

# Call to unpackbits(...): (line 12)
# Processing the call arguments (line 12)

# Obtaining the type of the subscript
slice_2674 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 20), None, None, None)
# Getting the type of 'np' (line 12)
np_2675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'np', False)
# Obtaining the member 'newaxis' of a type (line 12)
newaxis_2676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 25), np_2675, 'newaxis')
# Getting the type of 'I' (line 12)
I_2677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'I', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___2678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), I_2677, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_2679 = invoke(stypy.reporting.localization.Localization(__file__, 12, 20), getitem___2678, (slice_2674, newaxis_2676))

# Processing the call keyword arguments (line 12)
int_2680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 43), 'int')
keyword_2681 = int_2680
kwargs_2682 = {'axis': keyword_2681}
# Getting the type of 'np' (line 12)
np_2672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'np', False)
# Obtaining the member 'unpackbits' of a type (line 12)
unpackbits_2673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 6), np_2672, 'unpackbits')
# Calling unpackbits(args, kwargs) (line 12)
unpackbits_call_result_2683 = invoke(stypy.reporting.localization.Localization(__file__, 12, 6), unpackbits_2673, *[subscript_call_result_2679], **kwargs_2682)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', unpackbits_call_result_2683)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

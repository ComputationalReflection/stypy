
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: phi = np.arange(0, 10 * np.pi, 0.1)
6: a = 1
7: x = a * phi * np.cos(phi)
8: y = a * phi * np.sin(phi)
9: 
10: dr = (np.diff(x) ** 2 + np.diff(y) ** 2) ** .5  # segment lengths
11: r = np.zeros_like(x)
12: r[1:] = np.cumsum(dr)  # integrate path
13: r_int = np.linspace(0, r.max(), 200)  # regular spaced path
14: x_int = np.interp(r_int, r, x)  # integrate path
15: y_int = np.interp(r_int, r, y)
16: 
17: # l = globals().copy()
18: # for v in l:
19: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2704 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2704) is not StypyTypeError):

    if (import_2704 != 'pyd_module'):
        __import__(import_2704)
        sys_modules_2705 = sys.modules[import_2704]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2705.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2704)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_2708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
int_2709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
# Getting the type of 'np' (line 5)
np_2710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 24), 'np', False)
# Obtaining the member 'pi' of a type (line 5)
pi_2711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 24), np_2710, 'pi')
# Applying the binary operator '*' (line 5)
result_mul_2712 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 19), '*', int_2709, pi_2711)

float_2713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 31), 'float')
# Processing the call keyword arguments (line 5)
kwargs_2714 = {}
# Getting the type of 'np' (line 5)
np_2706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 6), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_2707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 6), np_2706, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_2715 = invoke(stypy.reporting.localization.Localization(__file__, 5, 6), arange_2707, *[int_2708, result_mul_2712, float_2713], **kwargs_2714)

# Assigning a type to the variable 'phi' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'phi', arange_call_result_2715)

# Assigning a Num to a Name (line 6):
int_2716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')
# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'a', int_2716)

# Assigning a BinOp to a Name (line 7):
# Getting the type of 'a' (line 7)
a_2717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'a')
# Getting the type of 'phi' (line 7)
phi_2718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'phi')
# Applying the binary operator '*' (line 7)
result_mul_2719 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 4), '*', a_2717, phi_2718)


# Call to cos(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'phi' (line 7)
phi_2722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 21), 'phi', False)
# Processing the call keyword arguments (line 7)
kwargs_2723 = {}
# Getting the type of 'np' (line 7)
np_2720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'np', False)
# Obtaining the member 'cos' of a type (line 7)
cos_2721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 14), np_2720, 'cos')
# Calling cos(args, kwargs) (line 7)
cos_call_result_2724 = invoke(stypy.reporting.localization.Localization(__file__, 7, 14), cos_2721, *[phi_2722], **kwargs_2723)

# Applying the binary operator '*' (line 7)
result_mul_2725 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '*', result_mul_2719, cos_call_result_2724)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', result_mul_2725)

# Assigning a BinOp to a Name (line 8):
# Getting the type of 'a' (line 8)
a_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'a')
# Getting the type of 'phi' (line 8)
phi_2727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'phi')
# Applying the binary operator '*' (line 8)
result_mul_2728 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), '*', a_2726, phi_2727)


# Call to sin(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'phi' (line 8)
phi_2731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 21), 'phi', False)
# Processing the call keyword arguments (line 8)
kwargs_2732 = {}
# Getting the type of 'np' (line 8)
np_2729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'np', False)
# Obtaining the member 'sin' of a type (line 8)
sin_2730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), np_2729, 'sin')
# Calling sin(args, kwargs) (line 8)
sin_call_result_2733 = invoke(stypy.reporting.localization.Localization(__file__, 8, 14), sin_2730, *[phi_2731], **kwargs_2732)

# Applying the binary operator '*' (line 8)
result_mul_2734 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 12), '*', result_mul_2728, sin_call_result_2733)

# Assigning a type to the variable 'y' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'y', result_mul_2734)

# Assigning a BinOp to a Name (line 10):

# Call to diff(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'x' (line 10)
x_2737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'x', False)
# Processing the call keyword arguments (line 10)
kwargs_2738 = {}
# Getting the type of 'np' (line 10)
np_2735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'np', False)
# Obtaining the member 'diff' of a type (line 10)
diff_2736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 6), np_2735, 'diff')
# Calling diff(args, kwargs) (line 10)
diff_call_result_2739 = invoke(stypy.reporting.localization.Localization(__file__, 10, 6), diff_2736, *[x_2737], **kwargs_2738)

int_2740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'int')
# Applying the binary operator '**' (line 10)
result_pow_2741 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 6), '**', diff_call_result_2739, int_2740)


# Call to diff(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'y' (line 10)
y_2744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 32), 'y', False)
# Processing the call keyword arguments (line 10)
kwargs_2745 = {}
# Getting the type of 'np' (line 10)
np_2742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 24), 'np', False)
# Obtaining the member 'diff' of a type (line 10)
diff_2743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 24), np_2742, 'diff')
# Calling diff(args, kwargs) (line 10)
diff_call_result_2746 = invoke(stypy.reporting.localization.Localization(__file__, 10, 24), diff_2743, *[y_2744], **kwargs_2745)

int_2747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 38), 'int')
# Applying the binary operator '**' (line 10)
result_pow_2748 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 24), '**', diff_call_result_2746, int_2747)

# Applying the binary operator '+' (line 10)
result_add_2749 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 6), '+', result_pow_2741, result_pow_2748)

float_2750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 44), 'float')
# Applying the binary operator '**' (line 10)
result_pow_2751 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '**', result_add_2749, float_2750)

# Assigning a type to the variable 'dr' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'dr', result_pow_2751)

# Assigning a Call to a Name (line 11):

# Call to zeros_like(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x' (line 11)
x_2754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'x', False)
# Processing the call keyword arguments (line 11)
kwargs_2755 = {}
# Getting the type of 'np' (line 11)
np_2752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'zeros_like' of a type (line 11)
zeros_like_2753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_2752, 'zeros_like')
# Calling zeros_like(args, kwargs) (line 11)
zeros_like_call_result_2756 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), zeros_like_2753, *[x_2754], **kwargs_2755)

# Assigning a type to the variable 'r' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r', zeros_like_call_result_2756)

# Assigning a Call to a Subscript (line 12):

# Call to cumsum(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'dr' (line 12)
dr_2759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'dr', False)
# Processing the call keyword arguments (line 12)
kwargs_2760 = {}
# Getting the type of 'np' (line 12)
np_2757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'np', False)
# Obtaining the member 'cumsum' of a type (line 12)
cumsum_2758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), np_2757, 'cumsum')
# Calling cumsum(args, kwargs) (line 12)
cumsum_call_result_2761 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), cumsum_2758, *[dr_2759], **kwargs_2760)

# Getting the type of 'r' (line 12)
r_2762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r')
int_2763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 2), 'int')
slice_2764 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 0), int_2763, None, None)
# Storing an element on a container (line 12)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 0), r_2762, (slice_2764, cumsum_call_result_2761))

# Assigning a Call to a Name (line 13):

# Call to linspace(...): (line 13)
# Processing the call arguments (line 13)
int_2767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')

# Call to max(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_2770 = {}
# Getting the type of 'r' (line 13)
r_2768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'r', False)
# Obtaining the member 'max' of a type (line 13)
max_2769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 23), r_2768, 'max')
# Calling max(args, kwargs) (line 13)
max_call_result_2771 = invoke(stypy.reporting.localization.Localization(__file__, 13, 23), max_2769, *[], **kwargs_2770)

int_2772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'int')
# Processing the call keyword arguments (line 13)
kwargs_2773 = {}
# Getting the type of 'np' (line 13)
np_2765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
# Obtaining the member 'linspace' of a type (line 13)
linspace_2766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_2765, 'linspace')
# Calling linspace(args, kwargs) (line 13)
linspace_call_result_2774 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), linspace_2766, *[int_2767, max_call_result_2771, int_2772], **kwargs_2773)

# Assigning a type to the variable 'r_int' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r_int', linspace_call_result_2774)

# Assigning a Call to a Name (line 14):

# Call to interp(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'r_int' (line 14)
r_int_2777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'r_int', False)
# Getting the type of 'r' (line 14)
r_2778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'r', False)
# Getting the type of 'x' (line 14)
x_2779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'x', False)
# Processing the call keyword arguments (line 14)
kwargs_2780 = {}
# Getting the type of 'np' (line 14)
np_2775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'np', False)
# Obtaining the member 'interp' of a type (line 14)
interp_2776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), np_2775, 'interp')
# Calling interp(args, kwargs) (line 14)
interp_call_result_2781 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), interp_2776, *[r_int_2777, r_2778, x_2779], **kwargs_2780)

# Assigning a type to the variable 'x_int' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'x_int', interp_call_result_2781)

# Assigning a Call to a Name (line 15):

# Call to interp(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'r_int' (line 15)
r_int_2784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'r_int', False)
# Getting the type of 'r' (line 15)
r_2785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'r', False)
# Getting the type of 'y' (line 15)
y_2786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 28), 'y', False)
# Processing the call keyword arguments (line 15)
kwargs_2787 = {}
# Getting the type of 'np' (line 15)
np_2782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'np', False)
# Obtaining the member 'interp' of a type (line 15)
interp_2783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), np_2782, 'interp')
# Calling interp(args, kwargs) (line 15)
interp_call_result_2788 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), interp_2783, *[r_int_2784, r_2785, y_2786], **kwargs_2787)

# Assigning a type to the variable 'y_int' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'y_int', interp_call_result_2788)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: w, h = 16, 16
6: I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
7: F = I[..., 0] * 256 * 256 + I[..., 1] * 256 + I[..., 2]
8: n = len(np.unique(F))
9: r = (np.unique(I))
10: 
11: # l = globals().copy()
12: # for v in l:
13: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2555 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2555) is not StypyTypeError):

    if (import_2555 != 'pyd_module'):
        __import__(import_2555)
        sys_modules_2556 = sys.modules[import_2555]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2556.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2555)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Tuple to a Tuple (line 5):

# Assigning a Num to a Name (line 5):
int_2557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 7), 'int')
# Assigning a type to the variable 'tuple_assignment_2553' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2553', int_2557)

# Assigning a Num to a Name (line 5):
int_2558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'int')
# Assigning a type to the variable 'tuple_assignment_2554' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2554', int_2558)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_2553' (line 5)
tuple_assignment_2553_2559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2553')
# Assigning a type to the variable 'w' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'w', tuple_assignment_2553_2559)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_2554' (line 5)
tuple_assignment_2554_2560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2554')
# Assigning a type to the variable 'h' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'h', tuple_assignment_2554_2560)

# Assigning a Call to a Name (line 6):

# Assigning a Call to a Name (line 6):

# Call to astype(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'np' (line 6)
np_2573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 46), 'np', False)
# Obtaining the member 'ubyte' of a type (line 6)
ubyte_2574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 46), np_2573, 'ubyte')
# Processing the call keyword arguments (line 6)
kwargs_2575 = {}

# Call to randint(...): (line 6)
# Processing the call arguments (line 6)
int_2564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')
int_2565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_2566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'h' (line 6)
h_2567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 29), 'h', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 29), tuple_2566, h_2567)
# Adding element type (line 6)
# Getting the type of 'w' (line 6)
w_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 32), 'w', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 29), tuple_2566, w_2568)
# Adding element type (line 6)
int_2569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 29), tuple_2566, int_2569)

# Processing the call keyword arguments (line 6)
kwargs_2570 = {}
# Getting the type of 'np' (line 6)
np_2561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'random' of a type (line 6)
random_2562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_2561, 'random')
# Obtaining the member 'randint' of a type (line 6)
randint_2563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), random_2562, 'randint')
# Calling randint(args, kwargs) (line 6)
randint_call_result_2571 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), randint_2563, *[int_2564, int_2565, tuple_2566], **kwargs_2570)

# Obtaining the member 'astype' of a type (line 6)
astype_2572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), randint_call_result_2571, 'astype')
# Calling astype(args, kwargs) (line 6)
astype_call_result_2576 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), astype_2572, *[ubyte_2574], **kwargs_2575)

# Assigning a type to the variable 'I' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'I', astype_call_result_2576)

# Assigning a BinOp to a Name (line 7):

# Assigning a BinOp to a Name (line 7):

# Obtaining the type of the subscript
Ellipsis_2577 = Ellipsis
int_2578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'int')
# Getting the type of 'I' (line 7)
I_2579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'I')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___2580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), I_2579, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_2581 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), getitem___2580, (Ellipsis_2577, int_2578))

int_2582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'int')
# Applying the binary operator '*' (line 7)
result_mul_2583 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 4), '*', subscript_call_result_2581, int_2582)

int_2584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'int')
# Applying the binary operator '*' (line 7)
result_mul_2585 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 20), '*', result_mul_2583, int_2584)


# Obtaining the type of the subscript
Ellipsis_2586 = Ellipsis
int_2587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 35), 'int')
# Getting the type of 'I' (line 7)
I_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 28), 'I')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___2589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 28), I_2588, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_2590 = invoke(stypy.reporting.localization.Localization(__file__, 7, 28), getitem___2589, (Ellipsis_2586, int_2587))

int_2591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 40), 'int')
# Applying the binary operator '*' (line 7)
result_mul_2592 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 28), '*', subscript_call_result_2590, int_2591)

# Applying the binary operator '+' (line 7)
result_add_2593 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 4), '+', result_mul_2585, result_mul_2592)


# Obtaining the type of the subscript
Ellipsis_2594 = Ellipsis
int_2595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 53), 'int')
# Getting the type of 'I' (line 7)
I_2596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 46), 'I')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___2597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 46), I_2596, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_2598 = invoke(stypy.reporting.localization.Localization(__file__, 7, 46), getitem___2597, (Ellipsis_2594, int_2595))

# Applying the binary operator '+' (line 7)
result_add_2599 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 44), '+', result_add_2593, subscript_call_result_2598)

# Assigning a type to the variable 'F' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'F', result_add_2599)

# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to len(...): (line 8)
# Processing the call arguments (line 8)

# Call to unique(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'F' (line 8)
F_2603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 'F', False)
# Processing the call keyword arguments (line 8)
kwargs_2604 = {}
# Getting the type of 'np' (line 8)
np_2601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'np', False)
# Obtaining the member 'unique' of a type (line 8)
unique_2602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), np_2601, 'unique')
# Calling unique(args, kwargs) (line 8)
unique_call_result_2605 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), unique_2602, *[F_2603], **kwargs_2604)

# Processing the call keyword arguments (line 8)
kwargs_2606 = {}
# Getting the type of 'len' (line 8)
len_2600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'len', False)
# Calling len(args, kwargs) (line 8)
len_call_result_2607 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), len_2600, *[unique_call_result_2605], **kwargs_2606)

# Assigning a type to the variable 'n' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'n', len_call_result_2607)

# Assigning a Call to a Name (line 9):

# Assigning a Call to a Name (line 9):

# Call to unique(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'I' (line 9)
I_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'I', False)
# Processing the call keyword arguments (line 9)
kwargs_2611 = {}
# Getting the type of 'np' (line 9)
np_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'unique' of a type (line 9)
unique_2609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_2608, 'unique')
# Calling unique(args, kwargs) (line 9)
unique_call_result_2612 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), unique_2609, *[I_2610], **kwargs_2611)

# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r', unique_call_result_2612)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

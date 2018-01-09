
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: p, n = 10, 20
6: M = np.ones((p, n, n))
7: V = np.ones((p, n, 1))
8: S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
9: 
10: # It works, because:
11: # M is (p,n,n)
12: # V is (p,n,1)
13: # Thus, summing over the paired axes 0 and 0 (of M and V independently),
14: # and 2 and 1, to remain with a (n,1) vector.
15: 
16: # l = globals().copy()
17: # for v in l:
18: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2481) is not StypyTypeError):

    if (import_2481 != 'pyd_module'):
        __import__(import_2481)
        sys_modules_2482 = sys.modules[import_2481]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2482.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2481)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Tuple to a Tuple (line 5):

# Assigning a Num to a Name (line 5):
int_2483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 7), 'int')
# Assigning a type to the variable 'tuple_assignment_2479' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2479', int_2483)

# Assigning a Num to a Name (line 5):
int_2484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'int')
# Assigning a type to the variable 'tuple_assignment_2480' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2480', int_2484)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_2479' (line 5)
tuple_assignment_2479_2485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2479')
# Assigning a type to the variable 'p' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'p', tuple_assignment_2479_2485)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_2480' (line 5)
tuple_assignment_2480_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_2480')
# Assigning a type to the variable 'n' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'n', tuple_assignment_2480_2486)

# Assigning a Call to a Name (line 6):

# Assigning a Call to a Name (line 6):

# Call to ones(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_2489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'p' (line 6)
p_2490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'p', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), tuple_2489, p_2490)
# Adding element type (line 6)
# Getting the type of 'n' (line 6)
n_2491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 16), 'n', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), tuple_2489, n_2491)
# Adding element type (line 6)
# Getting the type of 'n' (line 6)
n_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'n', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), tuple_2489, n_2492)

# Processing the call keyword arguments (line 6)
kwargs_2493 = {}
# Getting the type of 'np' (line 6)
np_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 6)
ones_2488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_2487, 'ones')
# Calling ones(args, kwargs) (line 6)
ones_call_result_2494 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), ones_2488, *[tuple_2489], **kwargs_2493)

# Assigning a type to the variable 'M' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'M', ones_call_result_2494)

# Assigning a Call to a Name (line 7):

# Assigning a Call to a Name (line 7):

# Call to ones(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_2497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'p' (line 7)
p_2498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'p', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_2497, p_2498)
# Adding element type (line 7)
# Getting the type of 'n' (line 7)
n_2499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'n', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_2497, n_2499)
# Adding element type (line 7)
int_2500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_2497, int_2500)

# Processing the call keyword arguments (line 7)
kwargs_2501 = {}
# Getting the type of 'np' (line 7)
np_2495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 7)
ones_2496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_2495, 'ones')
# Calling ones(args, kwargs) (line 7)
ones_call_result_2502 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), ones_2496, *[tuple_2497], **kwargs_2501)

# Assigning a type to the variable 'V' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'V', ones_call_result_2502)

# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to tensordot(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'M' (line 8)
M_2505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'M', False)
# Getting the type of 'V' (line 8)
V_2506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 20), 'V', False)
# Processing the call keyword arguments (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_2507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_2508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_2509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), list_2508, int_2509)
# Adding element type (line 8)
int_2510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), list_2508, int_2510)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 28), list_2507, list_2508)
# Adding element type (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_2511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 37), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_2512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 37), list_2511, int_2512)
# Adding element type (line 8)
int_2513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 37), list_2511, int_2513)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 28), list_2507, list_2511)

keyword_2514 = list_2507
kwargs_2515 = {'axes': keyword_2514}
# Getting the type of 'np' (line 8)
np_2503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'tensordot' of a type (line 8)
tensordot_2504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_2503, 'tensordot')
# Calling tensordot(args, kwargs) (line 8)
tensordot_call_result_2516 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), tensordot_2504, *[M_2505, V_2506], **kwargs_2515)

# Assigning a type to the variable 'S' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'S', tensordot_call_result_2516)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

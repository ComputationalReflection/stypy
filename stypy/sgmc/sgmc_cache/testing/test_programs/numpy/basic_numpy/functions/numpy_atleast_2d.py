
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.random((10, 2))
6: X, Y = np.atleast_2d(Z[:, 0]), np.atleast_2d(Z[:, 1])
7: D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
8: 
9: #
10: # l = globals().copy()
11: # for v in l:
12: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_432 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_432) is not StypyTypeError):

    if (import_432 != 'pyd_module'):
        __import__(import_432)
        sys_modules_433 = sys.modules[import_432]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_433.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_432)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Assigning a Call to a Name (line 5):

# Call to random(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_437, int_438)
# Adding element type (line 5)
int_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_437, int_439)

# Processing the call keyword arguments (line 5)
kwargs_440 = {}
# Getting the type of 'np' (line 5)
np_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_434, 'random')
# Obtaining the member 'random' of a type (line 5)
random_436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_435, 'random')
# Calling random(args, kwargs) (line 5)
random_call_result_441 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), random_436, *[tuple_437], **kwargs_440)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', random_call_result_441)

# Assigning a Tuple to a Tuple (line 6):

# Assigning a Call to a Name (line 6):

# Call to atleast_2d(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining the type of the subscript
slice_444 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 6, 21), None, None, None)
int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')
# Getting the type of 'Z' (line 6)
Z_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 21), 'Z', False)
# Obtaining the member '__getitem__' of a type (line 6)
getitem___447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 21), Z_446, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 6, 21), getitem___447, (slice_444, int_445))

# Processing the call keyword arguments (line 6)
kwargs_449 = {}
# Getting the type of 'np' (line 6)
np_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'np', False)
# Obtaining the member 'atleast_2d' of a type (line 6)
atleast_2d_443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 7), np_442, 'atleast_2d')
# Calling atleast_2d(args, kwargs) (line 6)
atleast_2d_call_result_450 = invoke(stypy.reporting.localization.Localization(__file__, 6, 7), atleast_2d_443, *[subscript_call_result_448], **kwargs_449)

# Assigning a type to the variable 'tuple_assignment_430' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_430', atleast_2d_call_result_450)

# Assigning a Call to a Name (line 6):

# Call to atleast_2d(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining the type of the subscript
slice_453 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 6, 45), None, None, None)
int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 50), 'int')
# Getting the type of 'Z' (line 6)
Z_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'Z', False)
# Obtaining the member '__getitem__' of a type (line 6)
getitem___456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 45), Z_455, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 6, 45), getitem___456, (slice_453, int_454))

# Processing the call keyword arguments (line 6)
kwargs_458 = {}
# Getting the type of 'np' (line 6)
np_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 31), 'np', False)
# Obtaining the member 'atleast_2d' of a type (line 6)
atleast_2d_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 31), np_451, 'atleast_2d')
# Calling atleast_2d(args, kwargs) (line 6)
atleast_2d_call_result_459 = invoke(stypy.reporting.localization.Localization(__file__, 6, 31), atleast_2d_452, *[subscript_call_result_457], **kwargs_458)

# Assigning a type to the variable 'tuple_assignment_431' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_431', atleast_2d_call_result_459)

# Assigning a Name to a Name (line 6):
# Getting the type of 'tuple_assignment_430' (line 6)
tuple_assignment_430_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_430')
# Assigning a type to the variable 'X' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'X', tuple_assignment_430_460)

# Assigning a Name to a Name (line 6):
# Getting the type of 'tuple_assignment_431' (line 6)
tuple_assignment_431_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_431')
# Assigning a type to the variable 'Y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 3), 'Y', tuple_assignment_431_461)

# Assigning a Call to a Name (line 7):

# Assigning a Call to a Name (line 7):

# Call to sqrt(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'X' (line 7)
X_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'X', False)
# Getting the type of 'X' (line 7)
X_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 17), 'X', False)
# Obtaining the member 'T' of a type (line 7)
T_466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 17), X_465, 'T')
# Applying the binary operator '-' (line 7)
result_sub_467 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 13), '-', X_464, T_466)

int_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'int')
# Applying the binary operator '**' (line 7)
result_pow_469 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '**', result_sub_467, int_468)

# Getting the type of 'Y' (line 7)
Y_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 30), 'Y', False)
# Getting the type of 'Y' (line 7)
Y_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 34), 'Y', False)
# Obtaining the member 'T' of a type (line 7)
T_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 34), Y_471, 'T')
# Applying the binary operator '-' (line 7)
result_sub_473 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 30), '-', Y_470, T_472)

int_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 42), 'int')
# Applying the binary operator '**' (line 7)
result_pow_475 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 29), '**', result_sub_473, int_474)

# Applying the binary operator '+' (line 7)
result_add_476 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '+', result_pow_469, result_pow_475)

# Processing the call keyword arguments (line 7)
kwargs_477 = {}
# Getting the type of 'np' (line 7)
np_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'sqrt' of a type (line 7)
sqrt_463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_462, 'sqrt')
# Calling sqrt(args, kwargs) (line 7)
sqrt_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), sqrt_463, *[result_add_476], **kwargs_477)

# Assigning a type to the variable 'D' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'D', sqrt_call_result_478)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

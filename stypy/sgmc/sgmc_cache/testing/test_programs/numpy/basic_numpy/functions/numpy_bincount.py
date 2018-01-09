
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.ones(10)
6: I = np.random.randint(0, len(Z), 20)
7: Z += np.bincount(I, minlength=len(Z))
8: 
9: X = [1, 2, 3, 4, 5, 6]
10: I = [1, 3, 9, 3, 4, 1]
11: F = np.bincount(I, X)
12: 
13: D = np.random.uniform(0, 1, 100)
14: S = np.random.randint(0, 10, 100)
15: D_sums = np.bincount(S, weights=D)
16: D_counts = np.bincount(S)
17: D_means = D_sums / D_counts
18: 
19: # l = globals().copy()
20: # for v in l:
21: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_479 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_479) is not StypyTypeError):

    if (import_479 != 'pyd_module'):
        __import__(import_479)
        sys_modules_480 = sys.modules[import_479]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_480.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_479)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to ones(...): (line 5)
# Processing the call arguments (line 5)
int_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'int')
# Processing the call keyword arguments (line 5)
kwargs_484 = {}
# Getting the type of 'np' (line 5)
np_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 5)
ones_482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_481, 'ones')
# Calling ones(args, kwargs) (line 5)
ones_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), ones_482, *[int_483], **kwargs_484)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', ones_call_result_485)

# Assigning a Call to a Name (line 6):

# Call to randint(...): (line 6)
# Processing the call arguments (line 6)
int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')

# Call to len(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 29), 'Z', False)
# Processing the call keyword arguments (line 6)
kwargs_492 = {}
# Getting the type of 'len' (line 6)
len_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 25), 'len', False)
# Calling len(args, kwargs) (line 6)
len_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 6, 25), len_490, *[Z_491], **kwargs_492)

int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'int')
# Processing the call keyword arguments (line 6)
kwargs_495 = {}
# Getting the type of 'np' (line 6)
np_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'random' of a type (line 6)
random_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_486, 'random')
# Obtaining the member 'randint' of a type (line 6)
randint_488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), random_487, 'randint')
# Calling randint(args, kwargs) (line 6)
randint_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), randint_488, *[int_489, len_call_result_493, int_494], **kwargs_495)

# Assigning a type to the variable 'I' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'I', randint_call_result_496)

# Getting the type of 'Z' (line 7)
Z_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Z')

# Call to bincount(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'I' (line 7)
I_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 17), 'I', False)
# Processing the call keyword arguments (line 7)

# Call to len(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'Z' (line 7)
Z_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 34), 'Z', False)
# Processing the call keyword arguments (line 7)
kwargs_503 = {}
# Getting the type of 'len' (line 7)
len_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 30), 'len', False)
# Calling len(args, kwargs) (line 7)
len_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 7, 30), len_501, *[Z_502], **kwargs_503)

keyword_505 = len_call_result_504
kwargs_506 = {'minlength': keyword_505}
# Getting the type of 'np' (line 7)
np_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'np', False)
# Obtaining the member 'bincount' of a type (line 7)
bincount_499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), np_498, 'bincount')
# Calling bincount(args, kwargs) (line 7)
bincount_call_result_507 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), bincount_499, *[I_500], **kwargs_506)

# Applying the binary operator '+=' (line 7)
result_iadd_508 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 0), '+=', Z_497, bincount_call_result_507)
# Assigning a type to the variable 'Z' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Z', result_iadd_508)


# Assigning a List to a Name (line 9):

# Obtaining an instance of the builtin type 'list' (line 9)
list_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_509, int_510)
# Adding element type (line 9)
int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_509, int_511)
# Adding element type (line 9)
int_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_509, int_512)
# Adding element type (line 9)
int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_509, int_513)
# Adding element type (line 9)
int_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_509, int_514)
# Adding element type (line 9)
int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_509, int_515)

# Assigning a type to the variable 'X' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'X', list_509)

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_516, int_517)
# Adding element type (line 10)
int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_516, int_518)
# Adding element type (line 10)
int_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_516, int_519)
# Adding element type (line 10)
int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_516, int_520)
# Adding element type (line 10)
int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_516, int_521)
# Adding element type (line 10)
int_522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_516, int_522)

# Assigning a type to the variable 'I' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'I', list_516)

# Assigning a Call to a Name (line 11):

# Call to bincount(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'I' (line 11)
I_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'I', False)
# Getting the type of 'X' (line 11)
X_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 19), 'X', False)
# Processing the call keyword arguments (line 11)
kwargs_527 = {}
# Getting the type of 'np' (line 11)
np_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'bincount' of a type (line 11)
bincount_524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_523, 'bincount')
# Calling bincount(args, kwargs) (line 11)
bincount_call_result_528 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), bincount_524, *[I_525, X_526], **kwargs_527)

# Assigning a type to the variable 'F' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'F', bincount_call_result_528)

# Assigning a Call to a Name (line 13):

# Call to uniform(...): (line 13)
# Processing the call arguments (line 13)
int_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
int_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'int')
# Processing the call keyword arguments (line 13)
kwargs_535 = {}
# Getting the type of 'np' (line 13)
np_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'np', False)
# Obtaining the member 'random' of a type (line 13)
random_530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), np_529, 'random')
# Obtaining the member 'uniform' of a type (line 13)
uniform_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), random_530, 'uniform')
# Calling uniform(args, kwargs) (line 13)
uniform_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), uniform_531, *[int_532, int_533, int_534], **kwargs_535)

# Assigning a type to the variable 'D' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'D', uniform_call_result_536)

# Assigning a Call to a Name (line 14):

# Call to randint(...): (line 14)
# Processing the call arguments (line 14)
int_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'int')
int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
# Processing the call keyword arguments (line 14)
kwargs_543 = {}
# Getting the type of 'np' (line 14)
np_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'np', False)
# Obtaining the member 'random' of a type (line 14)
random_538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), np_537, 'random')
# Obtaining the member 'randint' of a type (line 14)
randint_539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), random_538, 'randint')
# Calling randint(args, kwargs) (line 14)
randint_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), randint_539, *[int_540, int_541, int_542], **kwargs_543)

# Assigning a type to the variable 'S' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'S', randint_call_result_544)

# Assigning a Call to a Name (line 15):

# Call to bincount(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'S' (line 15)
S_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'S', False)
# Processing the call keyword arguments (line 15)
# Getting the type of 'D' (line 15)
D_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 32), 'D', False)
keyword_549 = D_548
kwargs_550 = {'weights': keyword_549}
# Getting the type of 'np' (line 15)
np_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'np', False)
# Obtaining the member 'bincount' of a type (line 15)
bincount_546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), np_545, 'bincount')
# Calling bincount(args, kwargs) (line 15)
bincount_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), bincount_546, *[S_547], **kwargs_550)

# Assigning a type to the variable 'D_sums' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'D_sums', bincount_call_result_551)

# Assigning a Call to a Name (line 16):

# Call to bincount(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'S' (line 16)
S_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'S', False)
# Processing the call keyword arguments (line 16)
kwargs_555 = {}
# Getting the type of 'np' (line 16)
np_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'np', False)
# Obtaining the member 'bincount' of a type (line 16)
bincount_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), np_552, 'bincount')
# Calling bincount(args, kwargs) (line 16)
bincount_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), bincount_553, *[S_554], **kwargs_555)

# Assigning a type to the variable 'D_counts' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'D_counts', bincount_call_result_556)

# Assigning a BinOp to a Name (line 17):
# Getting the type of 'D_sums' (line 17)
D_sums_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'D_sums')
# Getting the type of 'D_counts' (line 17)
D_counts_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'D_counts')
# Applying the binary operator 'div' (line 17)
result_div_559 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 10), 'div', D_sums_557, D_counts_558)

# Assigning a type to the variable 'D_means' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'D_means', result_div_559)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

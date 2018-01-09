
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: A = np.array([[3.4, 8.7, 9.9],
6:               [1.1, -7.8, -0.7],
7:               [4.1, 12.3, 4.8]])
8: r = (A.ndim)
9: 
10: B = np.array([[[111, 112], [121, 122]],
11:               [[211, 212], [221, 222]],
12:               [[311, 312], [321, 322]]])
13: 
14: r2 = (B.ndim)
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_491) is not StypyTypeError):

    if (import_491 != 'pyd_module'):
        __import__(import_491)
        sys_modules_492 = sys.modules[import_491]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_492.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_491)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
float_497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_496, float_497)
# Adding element type (line 5)
float_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_496, float_498)
# Adding element type (line 5)
float_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_496, float_499)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_495, list_496)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 6)
list_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
float_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_500, float_501)
# Adding element type (line 6)
float_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_500, float_502)
# Adding element type (line 6)
float_503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_500, float_503)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_495, list_500)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 7)
list_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
float_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_504, float_505)
# Adding element type (line 7)
float_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_504, float_506)
# Adding element type (line 7)
float_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_504, float_507)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_495, list_504)

# Processing the call keyword arguments (line 5)
kwargs_508 = {}
# Getting the type of 'np' (line 5)
np_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_493, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_494, *[list_495], **kwargs_508)

# Assigning a type to the variable 'A' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'A', array_call_result_509)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'A' (line 8)
A_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'A')
# Obtaining the member 'ndim' of a type (line 8)
ndim_511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), A_510, 'ndim')
# Assigning a type to the variable 'r' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r', ndim_511)

# Assigning a Call to a Name (line 10):

# Call to array(...): (line 10)
# Processing the call arguments (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 15), list_516, int_517)
# Adding element type (line 10)
int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 15), list_516, int_518)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_515, list_516)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 27), list_519, int_520)
# Adding element type (line 10)
int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 27), list_519, int_521)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_515, list_519)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), list_514, list_515)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 11)
list_522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'list' (line 11)
list_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
int_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 15), list_523, int_524)
# Adding element type (line 11)
int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 15), list_523, int_525)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 14), list_522, list_523)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'list' (line 11)
list_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 27), list_526, int_527)
# Adding element type (line 11)
int_528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 27), list_526, int_528)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 14), list_522, list_526)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), list_514, list_522)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 12)
list_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
int_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 15), list_530, int_531)
# Adding element type (line 12)
int_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 15), list_530, int_532)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_529, list_530)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 27), list_533, int_534)
# Adding element type (line 12)
int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 27), list_533, int_535)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_529, list_533)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), list_514, list_529)

# Processing the call keyword arguments (line 10)
kwargs_536 = {}
# Getting the type of 'np' (line 10)
np_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'np', False)
# Obtaining the member 'array' of a type (line 10)
array_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), np_512, 'array')
# Calling array(args, kwargs) (line 10)
array_call_result_537 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), array_513, *[list_514], **kwargs_536)

# Assigning a type to the variable 'B' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'B', array_call_result_537)

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'B' (line 14)
B_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'B')
# Obtaining the member 'ndim' of a type (line 14)
ndim_539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 6), B_538, 'ndim')
# Assigning a type to the variable 'r2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r2', ndim_539)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

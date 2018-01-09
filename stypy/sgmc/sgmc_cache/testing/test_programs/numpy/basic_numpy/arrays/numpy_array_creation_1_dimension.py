
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
6: V = np.array([3.4, 6.9, 99.8, 12.8])
7: 
8: r = F.dtype
9: r2 = V.dtype
10: r3 = np.ndim(F)
11: r4 = np.ndim(V)
12: 
13: # l = globals().copy()
14: # for v in l:
15: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_453 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_453) is not StypyTypeError):

    if (import_453 != 'pyd_module'):
        __import__(import_453)
        sys_modules_454 = sys.modules[import_453]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_454.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_453)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_458)
# Adding element type (line 5)
int_459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_459)
# Adding element type (line 5)
int_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_460)
# Adding element type (line 5)
int_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_461)
# Adding element type (line 5)
int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_462)
# Adding element type (line 5)
int_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_463)
# Adding element type (line 5)
int_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_464)
# Adding element type (line 5)
int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_457, int_465)

# Processing the call keyword arguments (line 5)
kwargs_466 = {}
# Getting the type of 'np' (line 5)
np_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_455, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_456, *[list_457], **kwargs_466)

# Assigning a type to the variable 'F' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'F', array_call_result_467)

# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
float_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_470, float_471)
# Adding element type (line 6)
float_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_470, float_472)
# Adding element type (line 6)
float_473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_470, float_473)
# Adding element type (line 6)
float_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_470, float_474)

# Processing the call keyword arguments (line 6)
kwargs_475 = {}
# Getting the type of 'np' (line 6)
np_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_468, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_469, *[list_470], **kwargs_475)

# Assigning a type to the variable 'V' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'V', array_call_result_476)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'F' (line 8)
F_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'F')
# Obtaining the member 'dtype' of a type (line 8)
dtype_478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), F_477, 'dtype')
# Assigning a type to the variable 'r' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r', dtype_478)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'V' (line 9)
V_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'V')
# Obtaining the member 'dtype' of a type (line 9)
dtype_480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), V_479, 'dtype')
# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', dtype_480)

# Assigning a Call to a Name (line 10):

# Call to ndim(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'F' (line 10)
F_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'F', False)
# Processing the call keyword arguments (line 10)
kwargs_484 = {}
# Getting the type of 'np' (line 10)
np_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'ndim' of a type (line 10)
ndim_482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_481, 'ndim')
# Calling ndim(args, kwargs) (line 10)
ndim_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), ndim_482, *[F_483], **kwargs_484)

# Assigning a type to the variable 'r3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r3', ndim_call_result_485)

# Assigning a Call to a Name (line 11):

# Call to ndim(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'V' (line 11)
V_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'V', False)
# Processing the call keyword arguments (line 11)
kwargs_489 = {}
# Getting the type of 'np' (line 11)
np_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'ndim' of a type (line 11)
ndim_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_486, 'ndim')
# Calling ndim(args, kwargs) (line 11)
ndim_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), ndim_487, *[V_488], **kwargs_489)

# Assigning a type to the variable 'r4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r4', ndim_call_result_490)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

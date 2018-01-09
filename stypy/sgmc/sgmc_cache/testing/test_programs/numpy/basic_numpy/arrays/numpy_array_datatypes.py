
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: x = np.array([1, 2])  # Let numpy choose the datatype
6: r = x.dtype  # Prints "int64"
7: 
8: x = np.array([1.0, 2.0])  # Let numpy choose the datatype
9: r2 = x.dtype  # Prints "float64"
10: 
11: x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
12: r3 = x.dtype  # Prints "int64"
13: 
14: # l = globals().copy()
15: # for v in l:
16: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_540 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_540) is not StypyTypeError):

    if (import_540 != 'pyd_module'):
        __import__(import_540)
        sys_modules_541 = sys.modules[import_540]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_541.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_540)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_544, int_545)
# Adding element type (line 5)
int_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_544, int_546)

# Processing the call keyword arguments (line 5)
kwargs_547 = {}
# Getting the type of 'np' (line 5)
np_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_542, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_548 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_543, *[list_544], **kwargs_547)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_548)

# Assigning a Attribute to a Name (line 6):
# Getting the type of 'x' (line 6)
x_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'x')
# Obtaining the member 'dtype' of a type (line 6)
dtype_550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), x_549, 'dtype')
# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', dtype_550)

# Assigning a Call to a Name (line 8):

# Call to array(...): (line 8)
# Processing the call arguments (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
float_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_553, float_554)
# Adding element type (line 8)
float_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_553, float_555)

# Processing the call keyword arguments (line 8)
kwargs_556 = {}
# Getting the type of 'np' (line 8)
np_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'array' of a type (line 8)
array_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_551, 'array')
# Calling array(args, kwargs) (line 8)
array_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), array_552, *[list_553], **kwargs_556)

# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x', array_call_result_557)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'x' (line 9)
x_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'x')
# Obtaining the member 'dtype' of a type (line 9)
dtype_559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), x_558, 'dtype')
# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', dtype_559)

# Assigning a Call to a Name (line 11):

# Call to array(...): (line 11)
# Processing the call arguments (line 11)

# Obtaining an instance of the builtin type 'list' (line 11)
list_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_562, int_563)
# Adding element type (line 11)
int_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 13), list_562, int_564)

# Processing the call keyword arguments (line 11)
# Getting the type of 'np' (line 11)
np_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'np', False)
# Obtaining the member 'int64' of a type (line 11)
int64_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 27), np_565, 'int64')
keyword_567 = int64_566
kwargs_568 = {'dtype': keyword_567}
# Getting the type of 'np' (line 11)
np_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'array' of a type (line 11)
array_561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_560, 'array')
# Calling array(args, kwargs) (line 11)
array_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), array_561, *[list_562], **kwargs_568)

# Assigning a type to the variable 'x' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'x', array_call_result_569)

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'x' (line 12)
x_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'x')
# Obtaining the member 'dtype' of a type (line 12)
dtype_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), x_570, 'dtype')
# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', dtype_571)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

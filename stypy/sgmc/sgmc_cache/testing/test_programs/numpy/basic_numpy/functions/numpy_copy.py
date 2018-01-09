
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: x = np.array([[42, 22, 12], [44, 53, 66]], order='F')
6: y = x.copy()
7: x[0, 0] = 1001
8: 
9: # l = globals().copy()
10: # for v in l:
11: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_560) is not StypyTypeError):

    if (import_560 != 'pyd_module'):
        __import__(import_560)
        sys_modules_561 = sys.modules[import_560]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_561.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_560)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_565, int_566)
# Adding element type (line 5)
int_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_565, int_567)
# Adding element type (line 5)
int_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_565, int_568)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_564, list_565)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 28), list_569, int_570)
# Adding element type (line 5)
int_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 28), list_569, int_571)
# Adding element type (line 5)
int_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 28), list_569, int_572)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_564, list_569)

# Processing the call keyword arguments (line 5)
str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 49), 'str', 'F')
keyword_574 = str_573
kwargs_575 = {'order': keyword_574}
# Getting the type of 'np' (line 5)
np_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_562, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_576 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_563, *[list_564], **kwargs_575)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_576)

# Assigning a Call to a Name (line 6):

# Call to copy(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_579 = {}
# Getting the type of 'x' (line 6)
x_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'x', False)
# Obtaining the member 'copy' of a type (line 6)
copy_578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), x_577, 'copy')
# Calling copy(args, kwargs) (line 6)
copy_call_result_580 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), copy_578, *[], **kwargs_579)

# Assigning a type to the variable 'y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'y', copy_call_result_580)

# Assigning a Num to a Subscript (line 7):
int_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'int')
# Getting the type of 'x' (line 7)
x_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x')

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 2), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 2), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 2), tuple_583, int_584)
# Adding element type (line 7)
int_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 2), tuple_583, int_585)

# Storing an element on a container (line 7)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), x_582, (tuple_583, int_581))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

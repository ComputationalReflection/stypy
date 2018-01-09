
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: 
4: import numpy as np
5: 
6: x = np.array([[42, 22, 12], [44, 53, 66]], order='F')
7: y = x.copy()
8: x[0, 0] = 1001
9: 
10: r = (x.flags['C_CONTIGUOUS'])
11: r2 = (y.flags['C_CONTIGUOUS'])
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_364) is not StypyTypeError):

    if (import_364 != 'pyd_module'):
        __import__(import_364)
        sys_modules_365 = sys.modules[import_364]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_365.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_364)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_369, int_370)
# Adding element type (line 6)
int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_369, int_371)
# Adding element type (line 6)
int_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_369, int_372)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_368, list_369)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), list_373, int_374)
# Adding element type (line 6)
int_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), list_373, int_375)
# Adding element type (line 6)
int_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), list_373, int_376)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_368, list_373)

# Processing the call keyword arguments (line 6)
str_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 49), 'str', 'F')
keyword_378 = str_377
kwargs_379 = {'order': keyword_378}
# Getting the type of 'np' (line 6)
np_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_366, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_367, *[list_368], **kwargs_379)

# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', array_call_result_380)

# Assigning a Call to a Name (line 7):

# Call to copy(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_383 = {}
# Getting the type of 'x' (line 7)
x_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'x', False)
# Obtaining the member 'copy' of a type (line 7)
copy_382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), x_381, 'copy')
# Calling copy(args, kwargs) (line 7)
copy_call_result_384 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), copy_382, *[], **kwargs_383)

# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'y', copy_call_result_384)

# Assigning a Num to a Subscript (line 8):
int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'int')
# Getting the type of 'x' (line 8)
x_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x')

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 2), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)
int_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 2), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 2), tuple_387, int_388)
# Adding element type (line 8)
int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 2), tuple_387, int_389)

# Storing an element on a container (line 8)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 0), x_386, (tuple_387, int_385))

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
str_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'str', 'C_CONTIGUOUS')
# Getting the type of 'x' (line 10)
x_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'x')
# Obtaining the member 'flags' of a type (line 10)
flags_392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), x_391, 'flags')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), flags_392, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_394 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), getitem___393, str_390)

# Assigning a type to the variable 'r' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r', subscript_call_result_394)

# Assigning a Subscript to a Name (line 11):

# Obtaining the type of the subscript
str_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'str', 'C_CONTIGUOUS')
# Getting the type of 'y' (line 11)
y_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'y')
# Obtaining the member 'flags' of a type (line 11)
flags_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 6), y_396, 'flags')
# Obtaining the member '__getitem__' of a type (line 11)
getitem___398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 6), flags_397, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 11, 6), getitem___398, str_395)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', subscript_call_result_399)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

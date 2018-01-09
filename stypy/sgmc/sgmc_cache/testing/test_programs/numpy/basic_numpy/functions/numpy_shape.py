
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: x = np.array([[67, 63, 87],
6:               [77, 69, 59],
7:               [85, 87, 99],
8:               [79, 72, 71],
9:               [63, 89, 93],
10:               [68, 92, 78]])
11: r = (np.shape(x))
12: 
13: x.shape = (3, 6)
14: r2 = (x)
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
import_2359 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2359) is not StypyTypeError):

    if (import_2359 != 'pyd_module'):
        __import__(import_2359)
        sys_modules_2360 = sys.modules[import_2359]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2360.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2359)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_2365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_2364, int_2365)
# Adding element type (line 5)
int_2366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_2364, int_2366)
# Adding element type (line 5)
int_2367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_2364, int_2367)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2363, list_2364)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 6)
list_2368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_2369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_2368, int_2369)
# Adding element type (line 6)
int_2370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_2368, int_2370)
# Adding element type (line 6)
int_2371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_2368, int_2371)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2363, list_2368)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 7)
list_2372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_2373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_2372, int_2373)
# Adding element type (line 7)
int_2374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_2372, int_2374)
# Adding element type (line 7)
int_2375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_2372, int_2375)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2363, list_2372)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 8)
list_2376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_2377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 14), list_2376, int_2377)
# Adding element type (line 8)
int_2378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 14), list_2376, int_2378)
# Adding element type (line 8)
int_2379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 14), list_2376, int_2379)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2363, list_2376)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 9)
list_2380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_2381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_2380, int_2381)
# Adding element type (line 9)
int_2382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_2380, int_2382)
# Adding element type (line 9)
int_2383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), list_2380, int_2383)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2363, list_2380)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 10)
list_2384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
int_2385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_2384, int_2385)
# Adding element type (line 10)
int_2386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_2384, int_2386)
# Adding element type (line 10)
int_2387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_2384, int_2387)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_2363, list_2384)

# Processing the call keyword arguments (line 5)
kwargs_2388 = {}
# Getting the type of 'np' (line 5)
np_2361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_2362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2361, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_2389 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_2362, *[list_2363], **kwargs_2388)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_2389)

# Assigning a Call to a Name (line 11):

# Call to shape(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x' (line 11)
x_2392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'x', False)
# Processing the call keyword arguments (line 11)
kwargs_2393 = {}
# Getting the type of 'np' (line 11)
np_2390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'shape' of a type (line 11)
shape_2391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_2390, 'shape')
# Calling shape(args, kwargs) (line 11)
shape_call_result_2394 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), shape_2391, *[x_2392], **kwargs_2393)

# Assigning a type to the variable 'r' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r', shape_call_result_2394)

# Assigning a Tuple to a Attribute (line 13):

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_2395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
int_2396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 11), tuple_2395, int_2396)
# Adding element type (line 13)
int_2397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 11), tuple_2395, int_2397)

# Getting the type of 'x' (line 13)
x_2398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'x')
# Setting the type of the member 'shape' of a type (line 13)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), x_2398, 'shape', tuple_2395)

# Assigning a Name to a Name (line 14):
# Getting the type of 'x' (line 14)
x_2399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'x')
# Assigning a type to the variable 'r2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r2', x_2399)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

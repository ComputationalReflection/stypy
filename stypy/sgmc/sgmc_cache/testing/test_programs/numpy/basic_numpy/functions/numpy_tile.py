
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
6: 
7: # l = globals().copy()
8: # for v in l:
9: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2517 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2517) is not StypyTypeError):

    if (import_2517 != 'pyd_module'):
        __import__(import_2517)
        sys_modules_2518 = sys.modules[import_2517]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2518.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2517)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to tile(...): (line 5)
# Processing the call arguments (line 5)

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_2525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_2524, int_2525)
# Adding element type (line 5)
int_2526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_2524, int_2526)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 21), list_2523, list_2524)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_2528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), list_2527, int_2528)
# Adding element type (line 5)
int_2529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), list_2527, int_2529)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 21), list_2523, list_2527)

# Processing the call keyword arguments (line 5)
kwargs_2530 = {}
# Getting the type of 'np' (line 5)
np_2521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 12), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_2522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 12), np_2521, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_2531 = invoke(stypy.reporting.localization.Localization(__file__, 5, 12), array_2522, *[list_2523], **kwargs_2530)


# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_2532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_2533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 41), tuple_2532, int_2533)
# Adding element type (line 5)
int_2534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 41), tuple_2532, int_2534)

# Processing the call keyword arguments (line 5)
kwargs_2535 = {}
# Getting the type of 'np' (line 5)
np_2519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'tile' of a type (line 5)
tile_2520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2519, 'tile')
# Calling tile(args, kwargs) (line 5)
tile_call_result_2536 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), tile_2520, *[array_call_result_2531, tuple_2532], **kwargs_2535)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', tile_call_result_2536)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

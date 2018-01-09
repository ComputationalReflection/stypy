
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: a = np.zeros((2, 2))  # Create an array of all zeros
6: 
7: b = np.ones((1, 2))  # Create an array of all ones
8: 
9: c = np.full((2, 2), 7)  # Create a constant array
10: 
11: d = np.eye(2)  # Create a 2x2 identity matrix
12: 
13: e = np.random.random((2, 2))  # Create an array filled with random values
14: 
15: # l = globals().copy()
16: # for v in l:
17: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_400 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_400) is not StypyTypeError):

    if (import_400 != 'pyd_module'):
        __import__(import_400)
        sys_modules_401 = sys.modules[import_400]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_401.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_400)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to zeros(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), tuple_404, int_405)
# Adding element type (line 5)
int_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), tuple_404, int_406)

# Processing the call keyword arguments (line 5)
kwargs_407 = {}
# Getting the type of 'np' (line 5)
np_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'zeros' of a type (line 5)
zeros_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_402, 'zeros')
# Calling zeros(args, kwargs) (line 5)
zeros_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), zeros_403, *[tuple_404], **kwargs_407)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', zeros_call_result_408)

# Assigning a Call to a Name (line 7):

# Call to ones(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_411, int_412)
# Adding element type (line 7)
int_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_411, int_413)

# Processing the call keyword arguments (line 7)
kwargs_414 = {}
# Getting the type of 'np' (line 7)
np_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 7)
ones_410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_409, 'ones')
# Calling ones(args, kwargs) (line 7)
ones_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), ones_410, *[tuple_411], **kwargs_414)

# Assigning a type to the variable 'b' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'b', ones_call_result_415)

# Assigning a Call to a Name (line 9):

# Call to full(...): (line 9)
# Processing the call arguments (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), tuple_418, int_419)
# Adding element type (line 9)
int_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), tuple_418, int_420)

int_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
# Processing the call keyword arguments (line 9)
kwargs_422 = {}
# Getting the type of 'np' (line 9)
np_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'full' of a type (line 9)
full_417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_416, 'full')
# Calling full(args, kwargs) (line 9)
full_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), full_417, *[tuple_418, int_421], **kwargs_422)

# Assigning a type to the variable 'c' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'c', full_call_result_423)

# Assigning a Call to a Name (line 11):

# Call to eye(...): (line 11)
# Processing the call arguments (line 11)
int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'int')
# Processing the call keyword arguments (line 11)
kwargs_427 = {}
# Getting the type of 'np' (line 11)
np_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'eye' of a type (line 11)
eye_425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_424, 'eye')
# Calling eye(args, kwargs) (line 11)
eye_call_result_428 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), eye_425, *[int_426], **kwargs_427)

# Assigning a type to the variable 'd' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'd', eye_call_result_428)

# Assigning a Call to a Name (line 13):

# Call to random(...): (line 13)
# Processing the call arguments (line 13)

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 22), tuple_432, int_433)
# Adding element type (line 13)
int_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 22), tuple_432, int_434)

# Processing the call keyword arguments (line 13)
kwargs_435 = {}
# Getting the type of 'np' (line 13)
np_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'np', False)
# Obtaining the member 'random' of a type (line 13)
random_430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), np_429, 'random')
# Obtaining the member 'random' of a type (line 13)
random_431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), random_430, 'random')
# Calling random(args, kwargs) (line 13)
random_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), random_431, *[tuple_432], **kwargs_435)

# Assigning a type to the variable 'e' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'e', random_call_result_436)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

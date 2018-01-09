
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.randint(0, 2, (6, 3))
6: T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
7: _, idx = np.unique(T, return_index=True)
8: uZ = Z[idx]
9: 
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
import_353 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_353) is not StypyTypeError):

    if (import_353 != 'pyd_module'):
        __import__(import_353)
        sys_modules_354 = sys.modules[import_353]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_354.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_353)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 29), tuple_360, int_361)
# Adding element type (line 5)
int_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 29), tuple_360, int_362)

# Processing the call keyword arguments (line 5)
kwargs_363 = {}
# Getting the type of 'np' (line 5)
np_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_355, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_356, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_364 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), randint_357, *[int_358, int_359, tuple_360], **kwargs_363)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', randint_call_result_364)

# Assigning a Call to a Name (line 6):

# Assigning a Call to a Name (line 6):

# Call to view(...): (line 6)
# Processing the call arguments (line 6)

# Call to dtype(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'np' (line 6)
np_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 43), 'np', False)
# Obtaining the member 'void' of a type (line 6)
void_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 43), np_374, 'void')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 43), tuple_373, void_375)
# Adding element type (line 6)
# Getting the type of 'Z' (line 6)
Z_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 52), 'Z', False)
# Obtaining the member 'dtype' of a type (line 6)
dtype_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 52), Z_376, 'dtype')
# Obtaining the member 'itemsize' of a type (line 6)
itemsize_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 52), dtype_377, 'itemsize')

# Obtaining the type of the subscript
int_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 79), 'int')
# Getting the type of 'Z' (line 6)
Z_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 71), 'Z', False)
# Obtaining the member 'shape' of a type (line 6)
shape_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 71), Z_380, 'shape')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 71), shape_381, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 6, 71), getitem___382, int_379)

# Applying the binary operator '*' (line 6)
result_mul_384 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 52), '*', itemsize_378, subscript_call_result_383)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 43), tuple_373, result_mul_384)

# Processing the call keyword arguments (line 6)
kwargs_385 = {}
# Getting the type of 'np' (line 6)
np_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 33), 'np', False)
# Obtaining the member 'dtype' of a type (line 6)
dtype_372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 33), np_371, 'dtype')
# Calling dtype(args, kwargs) (line 6)
dtype_call_result_386 = invoke(stypy.reporting.localization.Localization(__file__, 6, 33), dtype_372, *[tuple_373], **kwargs_385)

# Processing the call keyword arguments (line 6)
kwargs_387 = {}

# Call to ascontiguousarray(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 25), 'Z', False)
# Processing the call keyword arguments (line 6)
kwargs_368 = {}
# Getting the type of 'np' (line 6)
np_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'ascontiguousarray' of a type (line 6)
ascontiguousarray_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_365, 'ascontiguousarray')
# Calling ascontiguousarray(args, kwargs) (line 6)
ascontiguousarray_call_result_369 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), ascontiguousarray_366, *[Z_367], **kwargs_368)

# Obtaining the member 'view' of a type (line 6)
view_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), ascontiguousarray_call_result_369, 'view')
# Calling view(args, kwargs) (line 6)
view_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), view_370, *[dtype_call_result_386], **kwargs_387)

# Assigning a type to the variable 'T' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'T', view_call_result_388)

# Assigning a Call to a Tuple (line 7):

# Assigning a Call to a Name:

# Call to unique(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'T' (line 7)
T_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'T', False)
# Processing the call keyword arguments (line 7)
# Getting the type of 'True' (line 7)
True_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 35), 'True', False)
keyword_393 = True_392
kwargs_394 = {'return_index': keyword_393}
# Getting the type of 'np' (line 7)
np_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'np', False)
# Obtaining the member 'unique' of a type (line 7)
unique_390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), np_389, 'unique')
# Calling unique(args, kwargs) (line 7)
unique_call_result_395 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), unique_390, *[T_391], **kwargs_394)

# Assigning a type to the variable 'call_assignment_350' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_350', unique_call_result_395)

# Assigning a Call to a Name (line 7):

# Call to __getitem__(...):
# Processing the call arguments
int_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Processing the call keyword arguments
kwargs_399 = {}
# Getting the type of 'call_assignment_350' (line 7)
call_assignment_350_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_350', False)
# Obtaining the member '__getitem__' of a type (line 7)
getitem___397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), call_assignment_350_396, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___397, *[int_398], **kwargs_399)

# Assigning a type to the variable 'call_assignment_351' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_351', getitem___call_result_400)

# Assigning a Name to a Name (line 7):
# Getting the type of 'call_assignment_351' (line 7)
call_assignment_351_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_351')
# Assigning a type to the variable '_' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '_', call_assignment_351_401)

# Assigning a Call to a Name (line 7):

# Call to __getitem__(...):
# Processing the call arguments
int_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Processing the call keyword arguments
kwargs_405 = {}
# Getting the type of 'call_assignment_350' (line 7)
call_assignment_350_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_350', False)
# Obtaining the member '__getitem__' of a type (line 7)
getitem___403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), call_assignment_350_402, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_406 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___403, *[int_404], **kwargs_405)

# Assigning a type to the variable 'call_assignment_352' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_352', getitem___call_result_406)

# Assigning a Name to a Name (line 7):
# Getting the type of 'call_assignment_352' (line 7)
call_assignment_352_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_assignment_352')
# Assigning a type to the variable 'idx' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 3), 'idx', call_assignment_352_407)

# Assigning a Subscript to a Name (line 8):

# Assigning a Subscript to a Name (line 8):

# Obtaining the type of the subscript
# Getting the type of 'idx' (line 8)
idx_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'idx')
# Getting the type of 'Z' (line 8)
Z_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'Z')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), Z_409, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), getitem___410, idx_408)

# Assigning a type to the variable 'uZ' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'uZ', subscript_call_result_411)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

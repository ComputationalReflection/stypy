
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: 
6: def cartesian(arrays):
7:     arrays = [np.asarray(a) for a in arrays]
8:     shape = (len(x) for x in arrays)
9: 
10:     ix = np.indices(shape, dtype=int)
11:     ix = ix.reshape(len(arrays), -1).T
12: 
13:     for n, arr in enumerate(arrays):
14:         ix[:, n] = arrays[n][ix[:, n]]
15: 
16:     return ix
17: 
18: 
19: r = (cartesian(([1, 2, 3], [4, 5], [6, 7])))
20: 
21: # l = globals().copy()
22: # for v in l:
23: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1339 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1339) is not StypyTypeError):

    if (import_1339 != 'pyd_module'):
        __import__(import_1339)
        sys_modules_1340 = sys.modules[import_1339]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1340.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1339)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


@norecursion
def cartesian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cartesian'
    module_type_store = module_type_store.open_function_context('cartesian', 6, 0, False)
    
    # Passed parameters checking function
    cartesian.stypy_localization = localization
    cartesian.stypy_type_of_self = None
    cartesian.stypy_type_store = module_type_store
    cartesian.stypy_function_name = 'cartesian'
    cartesian.stypy_param_names_list = ['arrays']
    cartesian.stypy_varargs_param_name = None
    cartesian.stypy_kwargs_param_name = None
    cartesian.stypy_call_defaults = defaults
    cartesian.stypy_call_varargs = varargs
    cartesian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cartesian', ['arrays'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cartesian', localization, ['arrays'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cartesian(...)' code ##################

    
    # Assigning a ListComp to a Name (line 7):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 7)
    arrays_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 37), 'arrays')
    comprehension_1347 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), arrays_1346)
    # Assigning a type to the variable 'a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'a', comprehension_1347)
    
    # Call to asarray(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'a' (line 7)
    a_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 25), 'a', False)
    # Processing the call keyword arguments (line 7)
    kwargs_1344 = {}
    # Getting the type of 'np' (line 7)
    np_1341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 7)
    asarray_1342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 14), np_1341, 'asarray')
    # Calling asarray(args, kwargs) (line 7)
    asarray_call_result_1345 = invoke(stypy.reporting.localization.Localization(__file__, 7, 14), asarray_1342, *[a_1343], **kwargs_1344)
    
    list_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_1348, asarray_call_result_1345)
    # Assigning a type to the variable 'arrays' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'arrays', list_1348)
    
    # Assigning a GeneratorExp to a Name (line 8):
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 8, 13, True)
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 8)
    arrays_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 29), 'arrays')
    comprehension_1354 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), arrays_1353)
    # Assigning a type to the variable 'x' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'x', comprehension_1354)
    
    # Call to len(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'x' (line 8)
    x_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'x', False)
    # Processing the call keyword arguments (line 8)
    kwargs_1351 = {}
    # Getting the type of 'len' (line 8)
    len_1349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'len', False)
    # Calling len(args, kwargs) (line 8)
    len_call_result_1352 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), len_1349, *[x_1350], **kwargs_1351)
    
    list_1355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_1355, len_call_result_1352)
    # Assigning a type to the variable 'shape' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'shape', list_1355)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to indices(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'shape' (line 10)
    shape_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'shape', False)
    # Processing the call keyword arguments (line 10)
    # Getting the type of 'int' (line 10)
    int_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 33), 'int', False)
    keyword_1360 = int_1359
    kwargs_1361 = {'dtype': keyword_1360}
    # Getting the type of 'np' (line 10)
    np_1356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'np', False)
    # Obtaining the member 'indices' of a type (line 10)
    indices_1357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 9), np_1356, 'indices')
    # Calling indices(args, kwargs) (line 10)
    indices_call_result_1362 = invoke(stypy.reporting.localization.Localization(__file__, 10, 9), indices_1357, *[shape_1358], **kwargs_1361)
    
    # Assigning a type to the variable 'ix' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'ix', indices_call_result_1362)
    
    # Assigning a Attribute to a Name (line 11):
    
    # Call to reshape(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to len(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'arrays' (line 11)
    arrays_1366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'arrays', False)
    # Processing the call keyword arguments (line 11)
    kwargs_1367 = {}
    # Getting the type of 'len' (line 11)
    len_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'len', False)
    # Calling len(args, kwargs) (line 11)
    len_call_result_1368 = invoke(stypy.reporting.localization.Localization(__file__, 11, 20), len_1365, *[arrays_1366], **kwargs_1367)
    
    int_1369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 33), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_1370 = {}
    # Getting the type of 'ix' (line 11)
    ix_1363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'ix', False)
    # Obtaining the member 'reshape' of a type (line 11)
    reshape_1364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 9), ix_1363, 'reshape')
    # Calling reshape(args, kwargs) (line 11)
    reshape_call_result_1371 = invoke(stypy.reporting.localization.Localization(__file__, 11, 9), reshape_1364, *[len_call_result_1368, int_1369], **kwargs_1370)
    
    # Obtaining the member 'T' of a type (line 11)
    T_1372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 9), reshape_call_result_1371, 'T')
    # Assigning a type to the variable 'ix' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ix', T_1372)
    
    
    # Call to enumerate(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'arrays' (line 13)
    arrays_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'arrays', False)
    # Processing the call keyword arguments (line 13)
    kwargs_1375 = {}
    # Getting the type of 'enumerate' (line 13)
    enumerate_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 13)
    enumerate_call_result_1376 = invoke(stypy.reporting.localization.Localization(__file__, 13, 18), enumerate_1373, *[arrays_1374], **kwargs_1375)
    
    # Testing the type of a for loop iterable (line 13)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 13, 4), enumerate_call_result_1376)
    # Getting the type of the for loop variable (line 13)
    for_loop_var_1377 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 13, 4), enumerate_call_result_1376)
    # Assigning a type to the variable 'n' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), for_loop_var_1377))
    # Assigning a type to the variable 'arr' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'arr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), for_loop_var_1377))
    # SSA begins for a for statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 14):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    slice_1378 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 14, 29), None, None, None)
    # Getting the type of 'n' (line 14)
    n_1379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 35), 'n')
    # Getting the type of 'ix' (line 14)
    ix_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'ix')
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___1381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 29), ix_1380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_1382 = invoke(stypy.reporting.localization.Localization(__file__, 14, 29), getitem___1381, (slice_1378, n_1379))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 14)
    n_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'n')
    # Getting the type of 'arrays' (line 14)
    arrays_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'arrays')
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___1385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 19), arrays_1384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_1386 = invoke(stypy.reporting.localization.Localization(__file__, 14, 19), getitem___1385, n_1383)
    
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___1387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 19), subscript_call_result_1386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_1388 = invoke(stypy.reporting.localization.Localization(__file__, 14, 19), getitem___1387, subscript_call_result_1382)
    
    # Getting the type of 'ix' (line 14)
    ix_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'ix')
    slice_1390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 14, 8), None, None, None)
    # Getting the type of 'n' (line 14)
    n_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'n')
    # Storing an element on a container (line 14)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 8), ix_1389, ((slice_1390, n_1391), subscript_call_result_1388))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ix' (line 16)
    ix_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'ix')
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', ix_1392)
    
    # ################# End of 'cartesian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cartesian' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1393)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cartesian'
    return stypy_return_type_1393

# Assigning a type to the variable 'cartesian' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'cartesian', cartesian)

# Assigning a Call to a Name (line 19):

# Call to cartesian(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_1395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_1396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_1397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 16), list_1396, int_1397)
# Adding element type (line 19)
int_1398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 16), list_1396, int_1398)
# Adding element type (line 19)
int_1399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 16), list_1396, int_1399)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 16), tuple_1395, list_1396)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_1400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 27), list_1400, int_1401)
# Adding element type (line 19)
int_1402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 27), list_1400, int_1402)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 16), tuple_1395, list_1400)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_1403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_1404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 35), list_1403, int_1404)
# Adding element type (line 19)
int_1405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 35), list_1403, int_1405)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 16), tuple_1395, list_1403)

# Processing the call keyword arguments (line 19)
kwargs_1406 = {}
# Getting the type of 'cartesian' (line 19)
cartesian_1394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'cartesian', False)
# Calling cartesian(args, kwargs) (line 19)
cartesian_call_result_1407 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), cartesian_1394, *[tuple_1395], **kwargs_1406)

# Assigning a type to the variable 'r' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r', cartesian_call_result_1407)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

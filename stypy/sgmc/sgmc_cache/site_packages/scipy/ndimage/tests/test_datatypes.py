
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Testing data types for ndimage calls
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: import sys
6: 
7: import numpy as np
8: from numpy.testing import assert_array_almost_equal, assert_
9: import pytest
10: 
11: from scipy import ndimage
12: 
13: 
14: def test_map_coordinates_dts():
15:     # check that ndimage accepts different data types for interpolation
16:     data = np.array([[4, 1, 3, 2],
17:                      [7, 6, 8, 5],
18:                      [3, 5, 3, 6]])
19:     shifted_data = np.array([[0, 0, 0, 0],
20:                              [0, 4, 1, 3],
21:                              [0, 7, 6, 8]])
22:     idx = np.indices(data.shape)
23:     dts = (np.uint8, np.uint16, np.uint32, np.uint64,
24:            np.int8, np.int16, np.int32, np.int64,
25:            np.intp, np.uintp, np.float32, np.float64)
26:     for order in range(0, 6):
27:         for data_dt in dts:
28:             these_data = data.astype(data_dt)
29:             for coord_dt in dts:
30:                 # affine mapping
31:                 mat = np.eye(2, dtype=coord_dt)
32:                 off = np.zeros((2,), dtype=coord_dt)
33:                 out = ndimage.affine_transform(these_data, mat, off)
34:                 assert_array_almost_equal(these_data, out)
35:                 # map coordinates
36:                 coords_m1 = idx.astype(coord_dt) - 1
37:                 coords_p10 = idx.astype(coord_dt) + 10
38:                 out = ndimage.map_coordinates(these_data, coords_m1, order=order)
39:                 assert_array_almost_equal(out, shifted_data)
40:                 # check constant fill works
41:                 out = ndimage.map_coordinates(these_data, coords_p10, order=order)
42:                 assert_array_almost_equal(out, np.zeros((3,4)))
43:             # check shift and zoom
44:             out = ndimage.shift(these_data, 1)
45:             assert_array_almost_equal(out, shifted_data)
46:             out = ndimage.zoom(these_data, 1)
47:             assert_array_almost_equal(these_data, out)
48: 
49: 
50: @pytest.mark.xfail(not sys.platform == 'darwin', reason="runs only on darwin")
51: def test_uint64_max():
52:     # Test interpolation respects uint64 max.  Reported to fail at least on
53:     # win32 (due to the 32 bit visual C compiler using signed int64 when
54:     # converting between uint64 to double) and Debian on s390x.
55:     big = 2**64-1
56:     arr = np.array([big, big, big], dtype=np.uint64)
57:     # Tests geometric transform (map_coordinates, affine_transform)
58:     inds = np.indices(arr.shape) - 0.1
59:     x = ndimage.map_coordinates(arr, inds)
60:     assert_(x[1] > (2**63))
61:     # Tests zoom / shift
62:     x = ndimage.shift(arr, 0.1)
63:     assert_(x[1] > (2**63))
64: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_127279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Testing data types for ndimage calls\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127280 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_127280) is not StypyTypeError):

    if (import_127280 != 'pyd_module'):
        __import__(import_127280)
        sys_modules_127281 = sys.modules[import_127280]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_127281.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_127280)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127282 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_127282) is not StypyTypeError):

    if (import_127282 != 'pyd_module'):
        __import__(import_127282)
        sys_modules_127283 = sys.modules[import_127282]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_127283.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_127283, sys_modules_127283.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_'], [assert_array_almost_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_127282)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import pytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127284 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_127284) is not StypyTypeError):

    if (import_127284 != 'pyd_module'):
        __import__(import_127284)
        sys_modules_127285 = sys.modules[import_127284]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_127285.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_127284)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy import ndimage' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127286 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy')

if (type(import_127286) is not StypyTypeError):

    if (import_127286 != 'pyd_module'):
        __import__(import_127286)
        sys_modules_127287 = sys.modules[import_127286]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', sys_modules_127287.module_type_store, module_type_store, ['ndimage'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_127287, sys_modules_127287.module_type_store, module_type_store)
    else:
        from scipy import ndimage

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', None, module_type_store, ['ndimage'], [ndimage])

else:
    # Assigning a type to the variable 'scipy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', import_127286)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')


@norecursion
def test_map_coordinates_dts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_map_coordinates_dts'
    module_type_store = module_type_store.open_function_context('test_map_coordinates_dts', 14, 0, False)
    
    # Passed parameters checking function
    test_map_coordinates_dts.stypy_localization = localization
    test_map_coordinates_dts.stypy_type_of_self = None
    test_map_coordinates_dts.stypy_type_store = module_type_store
    test_map_coordinates_dts.stypy_function_name = 'test_map_coordinates_dts'
    test_map_coordinates_dts.stypy_param_names_list = []
    test_map_coordinates_dts.stypy_varargs_param_name = None
    test_map_coordinates_dts.stypy_kwargs_param_name = None
    test_map_coordinates_dts.stypy_call_defaults = defaults
    test_map_coordinates_dts.stypy_call_varargs = varargs
    test_map_coordinates_dts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_map_coordinates_dts', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_map_coordinates_dts', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_map_coordinates_dts(...)' code ##################

    
    # Assigning a Call to a Name (line 16):
    
    # Call to array(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_127290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_127291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_127292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), list_127291, int_127292)
    # Adding element type (line 16)
    int_127293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), list_127291, int_127293)
    # Adding element type (line 16)
    int_127294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), list_127291, int_127294)
    # Adding element type (line 16)
    int_127295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), list_127291, int_127295)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_127290, list_127291)
    # Adding element type (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_127296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_127297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_127296, int_127297)
    # Adding element type (line 17)
    int_127298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_127296, int_127298)
    # Adding element type (line 17)
    int_127299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_127296, int_127299)
    # Adding element type (line 17)
    int_127300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 21), list_127296, int_127300)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_127290, list_127296)
    # Adding element type (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_127301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_127302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), list_127301, int_127302)
    # Adding element type (line 18)
    int_127303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), list_127301, int_127303)
    # Adding element type (line 18)
    int_127304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), list_127301, int_127304)
    # Adding element type (line 18)
    int_127305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), list_127301, int_127305)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), list_127290, list_127301)
    
    # Processing the call keyword arguments (line 16)
    kwargs_127306 = {}
    # Getting the type of 'np' (line 16)
    np_127288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 16)
    array_127289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), np_127288, 'array')
    # Calling array(args, kwargs) (line 16)
    array_call_result_127307 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), array_127289, *[list_127290], **kwargs_127306)
    
    # Assigning a type to the variable 'data' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'data', array_call_result_127307)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to array(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_127310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_127311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    int_127312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 29), list_127311, int_127312)
    # Adding element type (line 19)
    int_127313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 29), list_127311, int_127313)
    # Adding element type (line 19)
    int_127314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 29), list_127311, int_127314)
    # Adding element type (line 19)
    int_127315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 29), list_127311, int_127315)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 28), list_127310, list_127311)
    # Adding element type (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_127316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    int_127317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 29), list_127316, int_127317)
    # Adding element type (line 20)
    int_127318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 29), list_127316, int_127318)
    # Adding element type (line 20)
    int_127319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 29), list_127316, int_127319)
    # Adding element type (line 20)
    int_127320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 29), list_127316, int_127320)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 28), list_127310, list_127316)
    # Adding element type (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_127321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_127322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 29), list_127321, int_127322)
    # Adding element type (line 21)
    int_127323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 29), list_127321, int_127323)
    # Adding element type (line 21)
    int_127324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 29), list_127321, int_127324)
    # Adding element type (line 21)
    int_127325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 29), list_127321, int_127325)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 28), list_127310, list_127321)
    
    # Processing the call keyword arguments (line 19)
    kwargs_127326 = {}
    # Getting the type of 'np' (line 19)
    np_127308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'np', False)
    # Obtaining the member 'array' of a type (line 19)
    array_127309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), np_127308, 'array')
    # Calling array(args, kwargs) (line 19)
    array_call_result_127327 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), array_127309, *[list_127310], **kwargs_127326)
    
    # Assigning a type to the variable 'shifted_data' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'shifted_data', array_call_result_127327)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to indices(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'data' (line 22)
    data_127330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'data', False)
    # Obtaining the member 'shape' of a type (line 22)
    shape_127331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 21), data_127330, 'shape')
    # Processing the call keyword arguments (line 22)
    kwargs_127332 = {}
    # Getting the type of 'np' (line 22)
    np_127328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'np', False)
    # Obtaining the member 'indices' of a type (line 22)
    indices_127329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 10), np_127328, 'indices')
    # Calling indices(args, kwargs) (line 22)
    indices_call_result_127333 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), indices_127329, *[shape_127331], **kwargs_127332)
    
    # Assigning a type to the variable 'idx' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'idx', indices_call_result_127333)
    
    # Assigning a Tuple to a Name (line 23):
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_127334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 23)
    np_127335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'np')
    # Obtaining the member 'uint8' of a type (line 23)
    uint8_127336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), np_127335, 'uint8')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, uint8_127336)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 23)
    np_127337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'np')
    # Obtaining the member 'uint16' of a type (line 23)
    uint16_127338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 21), np_127337, 'uint16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, uint16_127338)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 23)
    np_127339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'np')
    # Obtaining the member 'uint32' of a type (line 23)
    uint32_127340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 32), np_127339, 'uint32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, uint32_127340)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 23)
    np_127341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'np')
    # Obtaining the member 'uint64' of a type (line 23)
    uint64_127342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 43), np_127341, 'uint64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, uint64_127342)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 24)
    np_127343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'np')
    # Obtaining the member 'int8' of a type (line 24)
    int8_127344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 11), np_127343, 'int8')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, int8_127344)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 24)
    np_127345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'np')
    # Obtaining the member 'int16' of a type (line 24)
    int16_127346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), np_127345, 'int16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, int16_127346)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 24)
    np_127347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'np')
    # Obtaining the member 'int32' of a type (line 24)
    int32_127348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), np_127347, 'int32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, int32_127348)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 24)
    np_127349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 40), 'np')
    # Obtaining the member 'int64' of a type (line 24)
    int64_127350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 40), np_127349, 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, int64_127350)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 25)
    np_127351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'np')
    # Obtaining the member 'intp' of a type (line 25)
    intp_127352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), np_127351, 'intp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, intp_127352)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 25)
    np_127353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'np')
    # Obtaining the member 'uintp' of a type (line 25)
    uintp_127354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), np_127353, 'uintp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, uintp_127354)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 25)
    np_127355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'np')
    # Obtaining the member 'float32' of a type (line 25)
    float32_127356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 30), np_127355, 'float32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, float32_127356)
    # Adding element type (line 23)
    # Getting the type of 'np' (line 25)
    np_127357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 42), 'np')
    # Obtaining the member 'float64' of a type (line 25)
    float64_127358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 42), np_127357, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_127334, float64_127358)
    
    # Assigning a type to the variable 'dts' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'dts', tuple_127334)
    
    
    # Call to range(...): (line 26)
    # Processing the call arguments (line 26)
    int_127360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'int')
    int_127361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_127362 = {}
    # Getting the type of 'range' (line 26)
    range_127359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'range', False)
    # Calling range(args, kwargs) (line 26)
    range_call_result_127363 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), range_127359, *[int_127360, int_127361], **kwargs_127362)
    
    # Testing the type of a for loop iterable (line 26)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 4), range_call_result_127363)
    # Getting the type of the for loop variable (line 26)
    for_loop_var_127364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 4), range_call_result_127363)
    # Assigning a type to the variable 'order' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'order', for_loop_var_127364)
    # SSA begins for a for statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'dts' (line 27)
    dts_127365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'dts')
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), dts_127365)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_127366 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), dts_127365)
    # Assigning a type to the variable 'data_dt' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'data_dt', for_loop_var_127366)
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 28):
    
    # Call to astype(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'data_dt' (line 28)
    data_dt_127369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 37), 'data_dt', False)
    # Processing the call keyword arguments (line 28)
    kwargs_127370 = {}
    # Getting the type of 'data' (line 28)
    data_127367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'data', False)
    # Obtaining the member 'astype' of a type (line 28)
    astype_127368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), data_127367, 'astype')
    # Calling astype(args, kwargs) (line 28)
    astype_call_result_127371 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), astype_127368, *[data_dt_127369], **kwargs_127370)
    
    # Assigning a type to the variable 'these_data' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'these_data', astype_call_result_127371)
    
    # Getting the type of 'dts' (line 29)
    dts_127372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'dts')
    # Testing the type of a for loop iterable (line 29)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 12), dts_127372)
    # Getting the type of the for loop variable (line 29)
    for_loop_var_127373 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 12), dts_127372)
    # Assigning a type to the variable 'coord_dt' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'coord_dt', for_loop_var_127373)
    # SSA begins for a for statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 31):
    
    # Call to eye(...): (line 31)
    # Processing the call arguments (line 31)
    int_127376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
    # Processing the call keyword arguments (line 31)
    # Getting the type of 'coord_dt' (line 31)
    coord_dt_127377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 38), 'coord_dt', False)
    keyword_127378 = coord_dt_127377
    kwargs_127379 = {'dtype': keyword_127378}
    # Getting the type of 'np' (line 31)
    np_127374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'np', False)
    # Obtaining the member 'eye' of a type (line 31)
    eye_127375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 22), np_127374, 'eye')
    # Calling eye(args, kwargs) (line 31)
    eye_call_result_127380 = invoke(stypy.reporting.localization.Localization(__file__, 31, 22), eye_127375, *[int_127376], **kwargs_127379)
    
    # Assigning a type to the variable 'mat' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'mat', eye_call_result_127380)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to zeros(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Obtaining an instance of the builtin type 'tuple' (line 32)
    tuple_127383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 32)
    # Adding element type (line 32)
    int_127384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 32), tuple_127383, int_127384)
    
    # Processing the call keyword arguments (line 32)
    # Getting the type of 'coord_dt' (line 32)
    coord_dt_127385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 43), 'coord_dt', False)
    keyword_127386 = coord_dt_127385
    kwargs_127387 = {'dtype': keyword_127386}
    # Getting the type of 'np' (line 32)
    np_127381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'np', False)
    # Obtaining the member 'zeros' of a type (line 32)
    zeros_127382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 22), np_127381, 'zeros')
    # Calling zeros(args, kwargs) (line 32)
    zeros_call_result_127388 = invoke(stypy.reporting.localization.Localization(__file__, 32, 22), zeros_127382, *[tuple_127383], **kwargs_127387)
    
    # Assigning a type to the variable 'off' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'off', zeros_call_result_127388)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to affine_transform(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'these_data' (line 33)
    these_data_127391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 47), 'these_data', False)
    # Getting the type of 'mat' (line 33)
    mat_127392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 59), 'mat', False)
    # Getting the type of 'off' (line 33)
    off_127393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 64), 'off', False)
    # Processing the call keyword arguments (line 33)
    kwargs_127394 = {}
    # Getting the type of 'ndimage' (line 33)
    ndimage_127389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'ndimage', False)
    # Obtaining the member 'affine_transform' of a type (line 33)
    affine_transform_127390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), ndimage_127389, 'affine_transform')
    # Calling affine_transform(args, kwargs) (line 33)
    affine_transform_call_result_127395 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), affine_transform_127390, *[these_data_127391, mat_127392, off_127393], **kwargs_127394)
    
    # Assigning a type to the variable 'out' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'out', affine_transform_call_result_127395)
    
    # Call to assert_array_almost_equal(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'these_data' (line 34)
    these_data_127397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'these_data', False)
    # Getting the type of 'out' (line 34)
    out_127398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 54), 'out', False)
    # Processing the call keyword arguments (line 34)
    kwargs_127399 = {}
    # Getting the type of 'assert_array_almost_equal' (line 34)
    assert_array_almost_equal_127396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 34)
    assert_array_almost_equal_call_result_127400 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), assert_array_almost_equal_127396, *[these_data_127397, out_127398], **kwargs_127399)
    
    
    # Assigning a BinOp to a Name (line 36):
    
    # Call to astype(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'coord_dt' (line 36)
    coord_dt_127403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'coord_dt', False)
    # Processing the call keyword arguments (line 36)
    kwargs_127404 = {}
    # Getting the type of 'idx' (line 36)
    idx_127401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'idx', False)
    # Obtaining the member 'astype' of a type (line 36)
    astype_127402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 28), idx_127401, 'astype')
    # Calling astype(args, kwargs) (line 36)
    astype_call_result_127405 = invoke(stypy.reporting.localization.Localization(__file__, 36, 28), astype_127402, *[coord_dt_127403], **kwargs_127404)
    
    int_127406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 51), 'int')
    # Applying the binary operator '-' (line 36)
    result_sub_127407 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 28), '-', astype_call_result_127405, int_127406)
    
    # Assigning a type to the variable 'coords_m1' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'coords_m1', result_sub_127407)
    
    # Assigning a BinOp to a Name (line 37):
    
    # Call to astype(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'coord_dt' (line 37)
    coord_dt_127410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 40), 'coord_dt', False)
    # Processing the call keyword arguments (line 37)
    kwargs_127411 = {}
    # Getting the type of 'idx' (line 37)
    idx_127408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'idx', False)
    # Obtaining the member 'astype' of a type (line 37)
    astype_127409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 29), idx_127408, 'astype')
    # Calling astype(args, kwargs) (line 37)
    astype_call_result_127412 = invoke(stypy.reporting.localization.Localization(__file__, 37, 29), astype_127409, *[coord_dt_127410], **kwargs_127411)
    
    int_127413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 52), 'int')
    # Applying the binary operator '+' (line 37)
    result_add_127414 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 29), '+', astype_call_result_127412, int_127413)
    
    # Assigning a type to the variable 'coords_p10' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'coords_p10', result_add_127414)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to map_coordinates(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'these_data' (line 38)
    these_data_127417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 46), 'these_data', False)
    # Getting the type of 'coords_m1' (line 38)
    coords_m1_127418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 58), 'coords_m1', False)
    # Processing the call keyword arguments (line 38)
    # Getting the type of 'order' (line 38)
    order_127419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 75), 'order', False)
    keyword_127420 = order_127419
    kwargs_127421 = {'order': keyword_127420}
    # Getting the type of 'ndimage' (line 38)
    ndimage_127415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'ndimage', False)
    # Obtaining the member 'map_coordinates' of a type (line 38)
    map_coordinates_127416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), ndimage_127415, 'map_coordinates')
    # Calling map_coordinates(args, kwargs) (line 38)
    map_coordinates_call_result_127422 = invoke(stypy.reporting.localization.Localization(__file__, 38, 22), map_coordinates_127416, *[these_data_127417, coords_m1_127418], **kwargs_127421)
    
    # Assigning a type to the variable 'out' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'out', map_coordinates_call_result_127422)
    
    # Call to assert_array_almost_equal(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'out' (line 39)
    out_127424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 42), 'out', False)
    # Getting the type of 'shifted_data' (line 39)
    shifted_data_127425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 47), 'shifted_data', False)
    # Processing the call keyword arguments (line 39)
    kwargs_127426 = {}
    # Getting the type of 'assert_array_almost_equal' (line 39)
    assert_array_almost_equal_127423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 39)
    assert_array_almost_equal_call_result_127427 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), assert_array_almost_equal_127423, *[out_127424, shifted_data_127425], **kwargs_127426)
    
    
    # Assigning a Call to a Name (line 41):
    
    # Call to map_coordinates(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'these_data' (line 41)
    these_data_127430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 46), 'these_data', False)
    # Getting the type of 'coords_p10' (line 41)
    coords_p10_127431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 58), 'coords_p10', False)
    # Processing the call keyword arguments (line 41)
    # Getting the type of 'order' (line 41)
    order_127432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 76), 'order', False)
    keyword_127433 = order_127432
    kwargs_127434 = {'order': keyword_127433}
    # Getting the type of 'ndimage' (line 41)
    ndimage_127428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'ndimage', False)
    # Obtaining the member 'map_coordinates' of a type (line 41)
    map_coordinates_127429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 22), ndimage_127428, 'map_coordinates')
    # Calling map_coordinates(args, kwargs) (line 41)
    map_coordinates_call_result_127435 = invoke(stypy.reporting.localization.Localization(__file__, 41, 22), map_coordinates_127429, *[these_data_127430, coords_p10_127431], **kwargs_127434)
    
    # Assigning a type to the variable 'out' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'out', map_coordinates_call_result_127435)
    
    # Call to assert_array_almost_equal(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'out' (line 42)
    out_127437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'out', False)
    
    # Call to zeros(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_127440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    int_127441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 57), tuple_127440, int_127441)
    # Adding element type (line 42)
    int_127442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 57), tuple_127440, int_127442)
    
    # Processing the call keyword arguments (line 42)
    kwargs_127443 = {}
    # Getting the type of 'np' (line 42)
    np_127438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 47), 'np', False)
    # Obtaining the member 'zeros' of a type (line 42)
    zeros_127439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 47), np_127438, 'zeros')
    # Calling zeros(args, kwargs) (line 42)
    zeros_call_result_127444 = invoke(stypy.reporting.localization.Localization(__file__, 42, 47), zeros_127439, *[tuple_127440], **kwargs_127443)
    
    # Processing the call keyword arguments (line 42)
    kwargs_127445 = {}
    # Getting the type of 'assert_array_almost_equal' (line 42)
    assert_array_almost_equal_127436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 42)
    assert_array_almost_equal_call_result_127446 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), assert_array_almost_equal_127436, *[out_127437, zeros_call_result_127444], **kwargs_127445)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 44):
    
    # Call to shift(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'these_data' (line 44)
    these_data_127449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 32), 'these_data', False)
    int_127450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 44), 'int')
    # Processing the call keyword arguments (line 44)
    kwargs_127451 = {}
    # Getting the type of 'ndimage' (line 44)
    ndimage_127447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'ndimage', False)
    # Obtaining the member 'shift' of a type (line 44)
    shift_127448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 18), ndimage_127447, 'shift')
    # Calling shift(args, kwargs) (line 44)
    shift_call_result_127452 = invoke(stypy.reporting.localization.Localization(__file__, 44, 18), shift_127448, *[these_data_127449, int_127450], **kwargs_127451)
    
    # Assigning a type to the variable 'out' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'out', shift_call_result_127452)
    
    # Call to assert_array_almost_equal(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'out' (line 45)
    out_127454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'out', False)
    # Getting the type of 'shifted_data' (line 45)
    shifted_data_127455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 43), 'shifted_data', False)
    # Processing the call keyword arguments (line 45)
    kwargs_127456 = {}
    # Getting the type of 'assert_array_almost_equal' (line 45)
    assert_array_almost_equal_127453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 45)
    assert_array_almost_equal_call_result_127457 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), assert_array_almost_equal_127453, *[out_127454, shifted_data_127455], **kwargs_127456)
    
    
    # Assigning a Call to a Name (line 46):
    
    # Call to zoom(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'these_data' (line 46)
    these_data_127460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'these_data', False)
    int_127461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'int')
    # Processing the call keyword arguments (line 46)
    kwargs_127462 = {}
    # Getting the type of 'ndimage' (line 46)
    ndimage_127458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'ndimage', False)
    # Obtaining the member 'zoom' of a type (line 46)
    zoom_127459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), ndimage_127458, 'zoom')
    # Calling zoom(args, kwargs) (line 46)
    zoom_call_result_127463 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), zoom_127459, *[these_data_127460, int_127461], **kwargs_127462)
    
    # Assigning a type to the variable 'out' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'out', zoom_call_result_127463)
    
    # Call to assert_array_almost_equal(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'these_data' (line 47)
    these_data_127465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'these_data', False)
    # Getting the type of 'out' (line 47)
    out_127466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 50), 'out', False)
    # Processing the call keyword arguments (line 47)
    kwargs_127467 = {}
    # Getting the type of 'assert_array_almost_equal' (line 47)
    assert_array_almost_equal_127464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 47)
    assert_array_almost_equal_call_result_127468 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), assert_array_almost_equal_127464, *[these_data_127465, out_127466], **kwargs_127467)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_map_coordinates_dts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_map_coordinates_dts' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_127469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127469)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_map_coordinates_dts'
    return stypy_return_type_127469

# Assigning a type to the variable 'test_map_coordinates_dts' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'test_map_coordinates_dts', test_map_coordinates_dts)

@norecursion
def test_uint64_max(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_uint64_max'
    module_type_store = module_type_store.open_function_context('test_uint64_max', 50, 0, False)
    
    # Passed parameters checking function
    test_uint64_max.stypy_localization = localization
    test_uint64_max.stypy_type_of_self = None
    test_uint64_max.stypy_type_store = module_type_store
    test_uint64_max.stypy_function_name = 'test_uint64_max'
    test_uint64_max.stypy_param_names_list = []
    test_uint64_max.stypy_varargs_param_name = None
    test_uint64_max.stypy_kwargs_param_name = None
    test_uint64_max.stypy_call_defaults = defaults
    test_uint64_max.stypy_call_varargs = varargs
    test_uint64_max.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_uint64_max', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_uint64_max', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_uint64_max(...)' code ##################

    
    # Assigning a BinOp to a Name (line 55):
    int_127470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 10), 'int')
    int_127471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
    # Applying the binary operator '**' (line 55)
    result_pow_127472 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 10), '**', int_127470, int_127471)
    
    int_127473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'int')
    # Applying the binary operator '-' (line 55)
    result_sub_127474 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 10), '-', result_pow_127472, int_127473)
    
    # Assigning a type to the variable 'big' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'big', result_sub_127474)
    
    # Assigning a Call to a Name (line 56):
    
    # Call to array(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_127477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    # Getting the type of 'big' (line 56)
    big_127478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'big', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 19), list_127477, big_127478)
    # Adding element type (line 56)
    # Getting the type of 'big' (line 56)
    big_127479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'big', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 19), list_127477, big_127479)
    # Adding element type (line 56)
    # Getting the type of 'big' (line 56)
    big_127480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'big', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 19), list_127477, big_127480)
    
    # Processing the call keyword arguments (line 56)
    # Getting the type of 'np' (line 56)
    np_127481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'np', False)
    # Obtaining the member 'uint64' of a type (line 56)
    uint64_127482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 42), np_127481, 'uint64')
    keyword_127483 = uint64_127482
    kwargs_127484 = {'dtype': keyword_127483}
    # Getting the type of 'np' (line 56)
    np_127475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 56)
    array_127476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 10), np_127475, 'array')
    # Calling array(args, kwargs) (line 56)
    array_call_result_127485 = invoke(stypy.reporting.localization.Localization(__file__, 56, 10), array_127476, *[list_127477], **kwargs_127484)
    
    # Assigning a type to the variable 'arr' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'arr', array_call_result_127485)
    
    # Assigning a BinOp to a Name (line 58):
    
    # Call to indices(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'arr' (line 58)
    arr_127488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'arr', False)
    # Obtaining the member 'shape' of a type (line 58)
    shape_127489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 22), arr_127488, 'shape')
    # Processing the call keyword arguments (line 58)
    kwargs_127490 = {}
    # Getting the type of 'np' (line 58)
    np_127486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'np', False)
    # Obtaining the member 'indices' of a type (line 58)
    indices_127487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), np_127486, 'indices')
    # Calling indices(args, kwargs) (line 58)
    indices_call_result_127491 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), indices_127487, *[shape_127489], **kwargs_127490)
    
    float_127492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'float')
    # Applying the binary operator '-' (line 58)
    result_sub_127493 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), '-', indices_call_result_127491, float_127492)
    
    # Assigning a type to the variable 'inds' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'inds', result_sub_127493)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to map_coordinates(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'arr' (line 59)
    arr_127496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'arr', False)
    # Getting the type of 'inds' (line 59)
    inds_127497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'inds', False)
    # Processing the call keyword arguments (line 59)
    kwargs_127498 = {}
    # Getting the type of 'ndimage' (line 59)
    ndimage_127494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'ndimage', False)
    # Obtaining the member 'map_coordinates' of a type (line 59)
    map_coordinates_127495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), ndimage_127494, 'map_coordinates')
    # Calling map_coordinates(args, kwargs) (line 59)
    map_coordinates_call_result_127499 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), map_coordinates_127495, *[arr_127496, inds_127497], **kwargs_127498)
    
    # Assigning a type to the variable 'x' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'x', map_coordinates_call_result_127499)
    
    # Call to assert_(...): (line 60)
    # Processing the call arguments (line 60)
    
    
    # Obtaining the type of the subscript
    int_127501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 14), 'int')
    # Getting the type of 'x' (line 60)
    x_127502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___127503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), x_127502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_127504 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), getitem___127503, int_127501)
    
    int_127505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
    int_127506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'int')
    # Applying the binary operator '**' (line 60)
    result_pow_127507 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '**', int_127505, int_127506)
    
    # Applying the binary operator '>' (line 60)
    result_gt_127508 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), '>', subscript_call_result_127504, result_pow_127507)
    
    # Processing the call keyword arguments (line 60)
    kwargs_127509 = {}
    # Getting the type of 'assert_' (line 60)
    assert__127500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 60)
    assert__call_result_127510 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), assert__127500, *[result_gt_127508], **kwargs_127509)
    
    
    # Assigning a Call to a Name (line 62):
    
    # Call to shift(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'arr' (line 62)
    arr_127513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'arr', False)
    float_127514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'float')
    # Processing the call keyword arguments (line 62)
    kwargs_127515 = {}
    # Getting the type of 'ndimage' (line 62)
    ndimage_127511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'ndimage', False)
    # Obtaining the member 'shift' of a type (line 62)
    shift_127512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), ndimage_127511, 'shift')
    # Calling shift(args, kwargs) (line 62)
    shift_call_result_127516 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), shift_127512, *[arr_127513, float_127514], **kwargs_127515)
    
    # Assigning a type to the variable 'x' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'x', shift_call_result_127516)
    
    # Call to assert_(...): (line 63)
    # Processing the call arguments (line 63)
    
    
    # Obtaining the type of the subscript
    int_127518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'int')
    # Getting the type of 'x' (line 63)
    x_127519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___127520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), x_127519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_127521 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), getitem___127520, int_127518)
    
    int_127522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'int')
    int_127523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'int')
    # Applying the binary operator '**' (line 63)
    result_pow_127524 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 20), '**', int_127522, int_127523)
    
    # Applying the binary operator '>' (line 63)
    result_gt_127525 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 12), '>', subscript_call_result_127521, result_pow_127524)
    
    # Processing the call keyword arguments (line 63)
    kwargs_127526 = {}
    # Getting the type of 'assert_' (line 63)
    assert__127517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 63)
    assert__call_result_127527 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), assert__127517, *[result_gt_127525], **kwargs_127526)
    
    
    # ################# End of 'test_uint64_max(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_uint64_max' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_127528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127528)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_uint64_max'
    return stypy_return_type_127528

# Assigning a type to the variable 'test_uint64_max' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'test_uint64_max', test_uint64_max)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

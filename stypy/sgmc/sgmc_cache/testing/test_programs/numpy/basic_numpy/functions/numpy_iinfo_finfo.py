
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: for dtype in [np.int8, np.int32, np.int64]:
6:     rf1 = (np.iinfo(dtype).min)
7:     rf2 = (np.iinfo(dtype).max)
8: for dtype in [np.float32, np.float64]:
9:     rf3 = (np.finfo(dtype).min)
10:     rf4 = (np.finfo(dtype).max)
11:     rf5 = (np.finfo(dtype).eps)
12: 
13: r = np.finfo(np.float32).eps
14: 
15: r2 = np.finfo(np.float64).eps
16: 
17: r3 = np.float32(1e-8) + np.float32(1) == 1
18: 
19: r4 = np.float64(1e-8) + np.float64(1) == 1
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
import_1253 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1253) is not StypyTypeError):

    if (import_1253 != 'pyd_module'):
        __import__(import_1253)
        sys_modules_1254 = sys.modules[import_1253]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1254.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1253)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')



# Obtaining an instance of the builtin type 'list' (line 5)
list_1255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
# Getting the type of 'np' (line 5)
np_1256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 14), 'np')
# Obtaining the member 'int8' of a type (line 5)
int8_1257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 14), np_1256, 'int8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1255, int8_1257)
# Adding element type (line 5)
# Getting the type of 'np' (line 5)
np_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 23), 'np')
# Obtaining the member 'int32' of a type (line 5)
int32_1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 23), np_1258, 'int32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1255, int32_1259)
# Adding element type (line 5)
# Getting the type of 'np' (line 5)
np_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 33), 'np')
# Obtaining the member 'int64' of a type (line 5)
int64_1261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 33), np_1260, 'int64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1255, int64_1261)

# Testing the type of a for loop iterable (line 5)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 5, 0), list_1255)
# Getting the type of the for loop variable (line 5)
for_loop_var_1262 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 5, 0), list_1255)
# Assigning a type to the variable 'dtype' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'dtype', for_loop_var_1262)
# SSA begins for a for statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Attribute to a Name (line 6):

# Call to iinfo(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'dtype' (line 6)
dtype_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 20), 'dtype', False)
# Processing the call keyword arguments (line 6)
kwargs_1266 = {}
# Getting the type of 'np' (line 6)
np_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 11), 'np', False)
# Obtaining the member 'iinfo' of a type (line 6)
iinfo_1264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 11), np_1263, 'iinfo')
# Calling iinfo(args, kwargs) (line 6)
iinfo_call_result_1267 = invoke(stypy.reporting.localization.Localization(__file__, 6, 11), iinfo_1264, *[dtype_1265], **kwargs_1266)

# Obtaining the member 'min' of a type (line 6)
min_1268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 11), iinfo_call_result_1267, 'min')
# Assigning a type to the variable 'rf1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'rf1', min_1268)

# Assigning a Attribute to a Name (line 7):

# Call to iinfo(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'dtype' (line 7)
dtype_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'dtype', False)
# Processing the call keyword arguments (line 7)
kwargs_1272 = {}
# Getting the type of 'np' (line 7)
np_1269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'np', False)
# Obtaining the member 'iinfo' of a type (line 7)
iinfo_1270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 11), np_1269, 'iinfo')
# Calling iinfo(args, kwargs) (line 7)
iinfo_call_result_1273 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), iinfo_1270, *[dtype_1271], **kwargs_1272)

# Obtaining the member 'max' of a type (line 7)
max_1274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 11), iinfo_call_result_1273, 'max')
# Assigning a type to the variable 'rf2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'rf2', max_1274)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()



# Obtaining an instance of the builtin type 'list' (line 8)
list_1275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
# Getting the type of 'np' (line 8)
np_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'np')
# Obtaining the member 'float32' of a type (line 8)
float32_1277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), np_1276, 'float32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_1275, float32_1277)
# Adding element type (line 8)
# Getting the type of 'np' (line 8)
np_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 26), 'np')
# Obtaining the member 'float64' of a type (line 8)
float64_1279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 26), np_1278, 'float64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_1275, float64_1279)

# Testing the type of a for loop iterable (line 8)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 0), list_1275)
# Getting the type of the for loop variable (line 8)
for_loop_var_1280 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 0), list_1275)
# Assigning a type to the variable 'dtype' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'dtype', for_loop_var_1280)
# SSA begins for a for statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Attribute to a Name (line 9):

# Call to finfo(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'dtype' (line 9)
dtype_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'dtype', False)
# Processing the call keyword arguments (line 9)
kwargs_1284 = {}
# Getting the type of 'np' (line 9)
np_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'np', False)
# Obtaining the member 'finfo' of a type (line 9)
finfo_1282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), np_1281, 'finfo')
# Calling finfo(args, kwargs) (line 9)
finfo_call_result_1285 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), finfo_1282, *[dtype_1283], **kwargs_1284)

# Obtaining the member 'min' of a type (line 9)
min_1286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), finfo_call_result_1285, 'min')
# Assigning a type to the variable 'rf3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'rf3', min_1286)

# Assigning a Attribute to a Name (line 10):

# Call to finfo(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'dtype' (line 10)
dtype_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'dtype', False)
# Processing the call keyword arguments (line 10)
kwargs_1290 = {}
# Getting the type of 'np' (line 10)
np_1287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'np', False)
# Obtaining the member 'finfo' of a type (line 10)
finfo_1288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), np_1287, 'finfo')
# Calling finfo(args, kwargs) (line 10)
finfo_call_result_1291 = invoke(stypy.reporting.localization.Localization(__file__, 10, 11), finfo_1288, *[dtype_1289], **kwargs_1290)

# Obtaining the member 'max' of a type (line 10)
max_1292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), finfo_call_result_1291, 'max')
# Assigning a type to the variable 'rf4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'rf4', max_1292)

# Assigning a Attribute to a Name (line 11):

# Call to finfo(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'dtype' (line 11)
dtype_1295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'dtype', False)
# Processing the call keyword arguments (line 11)
kwargs_1296 = {}
# Getting the type of 'np' (line 11)
np_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'np', False)
# Obtaining the member 'finfo' of a type (line 11)
finfo_1294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), np_1293, 'finfo')
# Calling finfo(args, kwargs) (line 11)
finfo_call_result_1297 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), finfo_1294, *[dtype_1295], **kwargs_1296)

# Obtaining the member 'eps' of a type (line 11)
eps_1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), finfo_call_result_1297, 'eps')
# Assigning a type to the variable 'rf5' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'rf5', eps_1298)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Name (line 13):

# Call to finfo(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'np' (line 13)
np_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'np', False)
# Obtaining the member 'float32' of a type (line 13)
float32_1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), np_1301, 'float32')
# Processing the call keyword arguments (line 13)
kwargs_1303 = {}
# Getting the type of 'np' (line 13)
np_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'np', False)
# Obtaining the member 'finfo' of a type (line 13)
finfo_1300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), np_1299, 'finfo')
# Calling finfo(args, kwargs) (line 13)
finfo_call_result_1304 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), finfo_1300, *[float32_1302], **kwargs_1303)

# Obtaining the member 'eps' of a type (line 13)
eps_1305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), finfo_call_result_1304, 'eps')
# Assigning a type to the variable 'r' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r', eps_1305)

# Assigning a Attribute to a Name (line 15):

# Call to finfo(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'np' (line 15)
np_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'np', False)
# Obtaining the member 'float64' of a type (line 15)
float64_1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 14), np_1308, 'float64')
# Processing the call keyword arguments (line 15)
kwargs_1310 = {}
# Getting the type of 'np' (line 15)
np_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'finfo' of a type (line 15)
finfo_1307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_1306, 'finfo')
# Calling finfo(args, kwargs) (line 15)
finfo_call_result_1311 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), finfo_1307, *[float64_1309], **kwargs_1310)

# Obtaining the member 'eps' of a type (line 15)
eps_1312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), finfo_call_result_1311, 'eps')
# Assigning a type to the variable 'r2' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r2', eps_1312)

# Assigning a Compare to a Name (line 17):


# Call to float32(...): (line 17)
# Processing the call arguments (line 17)
float_1315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'float')
# Processing the call keyword arguments (line 17)
kwargs_1316 = {}
# Getting the type of 'np' (line 17)
np_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'float32' of a type (line 17)
float32_1314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_1313, 'float32')
# Calling float32(args, kwargs) (line 17)
float32_call_result_1317 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), float32_1314, *[float_1315], **kwargs_1316)


# Call to float32(...): (line 17)
# Processing the call arguments (line 17)
int_1320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
# Processing the call keyword arguments (line 17)
kwargs_1321 = {}
# Getting the type of 'np' (line 17)
np_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'np', False)
# Obtaining the member 'float32' of a type (line 17)
float32_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), np_1318, 'float32')
# Calling float32(args, kwargs) (line 17)
float32_call_result_1322 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), float32_1319, *[int_1320], **kwargs_1321)

# Applying the binary operator '+' (line 17)
result_add_1323 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 5), '+', float32_call_result_1317, float32_call_result_1322)

int_1324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'int')
# Applying the binary operator '==' (line 17)
result_eq_1325 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 5), '==', result_add_1323, int_1324)

# Assigning a type to the variable 'r3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r3', result_eq_1325)

# Assigning a Compare to a Name (line 19):


# Call to float64(...): (line 19)
# Processing the call arguments (line 19)
float_1328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'float')
# Processing the call keyword arguments (line 19)
kwargs_1329 = {}
# Getting the type of 'np' (line 19)
np_1326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'float64' of a type (line 19)
float64_1327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_1326, 'float64')
# Calling float64(args, kwargs) (line 19)
float64_call_result_1330 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), float64_1327, *[float_1328], **kwargs_1329)


# Call to float64(...): (line 19)
# Processing the call arguments (line 19)
int_1333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'int')
# Processing the call keyword arguments (line 19)
kwargs_1334 = {}
# Getting the type of 'np' (line 19)
np_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'np', False)
# Obtaining the member 'float64' of a type (line 19)
float64_1332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 24), np_1331, 'float64')
# Calling float64(args, kwargs) (line 19)
float64_call_result_1335 = invoke(stypy.reporting.localization.Localization(__file__, 19, 24), float64_1332, *[int_1333], **kwargs_1334)

# Applying the binary operator '+' (line 19)
result_add_1336 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 5), '+', float64_call_result_1330, float64_call_result_1335)

int_1337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'int')
# Applying the binary operator '==' (line 19)
result_eq_1338 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 5), '==', result_add_1336, int_1337)

# Assigning a type to the variable 'r4' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r4', result_eq_1338)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

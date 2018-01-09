
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: # 50 values between 1 and 10:
6: r = (np.linspace(1, 10))
7: # 7 values between 1 and 10:
8: r2 = (np.linspace(1, 10, 7))
9: # excluding the endpoint:
10: r3 = (np.linspace(1, 10, 7, endpoint=False))
11: 
12: samples, spacing = np.linspace(1, 10, retstep=True)
13: r4 = (spacing)
14: samples2, spacing = np.linspace(1, 10, 20, endpoint=True, retstep=True)
15: r5 = (spacing)
16: samples3, spacing = np.linspace(1, 10, 20, endpoint=False, retstep=True)
17: r6 = (spacing)
18: 
19: # l = globals().copy()
20: # for v in l:
21: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1417 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1417) is not StypyTypeError):

    if (import_1417 != 'pyd_module'):
        __import__(import_1417)
        sys_modules_1418 = sys.modules[import_1417]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1418.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1417)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 6):

# Assigning a Call to a Name (line 6):

# Call to linspace(...): (line 6)
# Processing the call arguments (line 6)
int_1421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 17), 'int')
int_1422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
# Processing the call keyword arguments (line 6)
kwargs_1423 = {}
# Getting the type of 'np' (line 6)
np_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'np', False)
# Obtaining the member 'linspace' of a type (line 6)
linspace_1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), np_1419, 'linspace')
# Calling linspace(args, kwargs) (line 6)
linspace_call_result_1424 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), linspace_1420, *[int_1421, int_1422], **kwargs_1423)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', linspace_call_result_1424)

# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to linspace(...): (line 8)
# Processing the call arguments (line 8)
int_1427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 18), 'int')
int_1428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 21), 'int')
int_1429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 25), 'int')
# Processing the call keyword arguments (line 8)
kwargs_1430 = {}
# Getting the type of 'np' (line 8)
np_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'np', False)
# Obtaining the member 'linspace' of a type (line 8)
linspace_1426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 6), np_1425, 'linspace')
# Calling linspace(args, kwargs) (line 8)
linspace_call_result_1431 = invoke(stypy.reporting.localization.Localization(__file__, 8, 6), linspace_1426, *[int_1427, int_1428, int_1429], **kwargs_1430)

# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', linspace_call_result_1431)

# Assigning a Call to a Name (line 10):

# Assigning a Call to a Name (line 10):

# Call to linspace(...): (line 10)
# Processing the call arguments (line 10)
int_1434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'int')
int_1435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'int')
int_1436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'int')
# Processing the call keyword arguments (line 10)
# Getting the type of 'False' (line 10)
False_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 37), 'False', False)
keyword_1438 = False_1437
kwargs_1439 = {'endpoint': keyword_1438}
# Getting the type of 'np' (line 10)
np_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'np', False)
# Obtaining the member 'linspace' of a type (line 10)
linspace_1433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 6), np_1432, 'linspace')
# Calling linspace(args, kwargs) (line 10)
linspace_call_result_1440 = invoke(stypy.reporting.localization.Localization(__file__, 10, 6), linspace_1433, *[int_1434, int_1435, int_1436], **kwargs_1439)

# Assigning a type to the variable 'r3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r3', linspace_call_result_1440)

# Assigning a Call to a Tuple (line 12):

# Assigning a Call to a Name:

# Call to linspace(...): (line 12)
# Processing the call arguments (line 12)
int_1443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
int_1444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'int')
# Processing the call keyword arguments (line 12)
# Getting the type of 'True' (line 12)
True_1445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 46), 'True', False)
keyword_1446 = True_1445
kwargs_1447 = {'retstep': keyword_1446}
# Getting the type of 'np' (line 12)
np_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'np', False)
# Obtaining the member 'linspace' of a type (line 12)
linspace_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), np_1441, 'linspace')
# Calling linspace(args, kwargs) (line 12)
linspace_call_result_1448 = invoke(stypy.reporting.localization.Localization(__file__, 12, 19), linspace_1442, *[int_1443, int_1444], **kwargs_1447)

# Assigning a type to the variable 'call_assignment_1408' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1408', linspace_call_result_1448)

# Assigning a Call to a Name (line 12):

# Call to __getitem__(...):
# Processing the call arguments
int_1451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Processing the call keyword arguments
kwargs_1452 = {}
# Getting the type of 'call_assignment_1408' (line 12)
call_assignment_1408_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1408', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___1450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), call_assignment_1408_1449, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1453 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1450, *[int_1451], **kwargs_1452)

# Assigning a type to the variable 'call_assignment_1409' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1409', getitem___call_result_1453)

# Assigning a Name to a Name (line 12):
# Getting the type of 'call_assignment_1409' (line 12)
call_assignment_1409_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1409')
# Assigning a type to the variable 'samples' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'samples', call_assignment_1409_1454)

# Assigning a Call to a Name (line 12):

# Call to __getitem__(...):
# Processing the call arguments
int_1457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Processing the call keyword arguments
kwargs_1458 = {}
# Getting the type of 'call_assignment_1408' (line 12)
call_assignment_1408_1455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1408', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___1456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), call_assignment_1408_1455, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1459 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1456, *[int_1457], **kwargs_1458)

# Assigning a type to the variable 'call_assignment_1410' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1410', getitem___call_result_1459)

# Assigning a Name to a Name (line 12):
# Getting the type of 'call_assignment_1410' (line 12)
call_assignment_1410_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'call_assignment_1410')
# Assigning a type to the variable 'spacing' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'spacing', call_assignment_1410_1460)

# Assigning a Name to a Name (line 13):

# Assigning a Name to a Name (line 13):
# Getting the type of 'spacing' (line 13)
spacing_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 6), 'spacing')
# Assigning a type to the variable 'r4' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r4', spacing_1461)

# Assigning a Call to a Tuple (line 14):

# Assigning a Call to a Name:

# Call to linspace(...): (line 14)
# Processing the call arguments (line 14)
int_1464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
int_1465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'int')
int_1466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'int')
# Processing the call keyword arguments (line 14)
# Getting the type of 'True' (line 14)
True_1467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 52), 'True', False)
keyword_1468 = True_1467
# Getting the type of 'True' (line 14)
True_1469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 66), 'True', False)
keyword_1470 = True_1469
kwargs_1471 = {'retstep': keyword_1470, 'endpoint': keyword_1468}
# Getting the type of 'np' (line 14)
np_1462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'np', False)
# Obtaining the member 'linspace' of a type (line 14)
linspace_1463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 20), np_1462, 'linspace')
# Calling linspace(args, kwargs) (line 14)
linspace_call_result_1472 = invoke(stypy.reporting.localization.Localization(__file__, 14, 20), linspace_1463, *[int_1464, int_1465, int_1466], **kwargs_1471)

# Assigning a type to the variable 'call_assignment_1411' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1411', linspace_call_result_1472)

# Assigning a Call to a Name (line 14):

# Call to __getitem__(...):
# Processing the call arguments
int_1475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 0), 'int')
# Processing the call keyword arguments
kwargs_1476 = {}
# Getting the type of 'call_assignment_1411' (line 14)
call_assignment_1411_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1411', False)
# Obtaining the member '__getitem__' of a type (line 14)
getitem___1474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 0), call_assignment_1411_1473, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1477 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1474, *[int_1475], **kwargs_1476)

# Assigning a type to the variable 'call_assignment_1412' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1412', getitem___call_result_1477)

# Assigning a Name to a Name (line 14):
# Getting the type of 'call_assignment_1412' (line 14)
call_assignment_1412_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1412')
# Assigning a type to the variable 'samples2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'samples2', call_assignment_1412_1478)

# Assigning a Call to a Name (line 14):

# Call to __getitem__(...):
# Processing the call arguments
int_1481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 0), 'int')
# Processing the call keyword arguments
kwargs_1482 = {}
# Getting the type of 'call_assignment_1411' (line 14)
call_assignment_1411_1479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1411', False)
# Obtaining the member '__getitem__' of a type (line 14)
getitem___1480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 0), call_assignment_1411_1479, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1483 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1480, *[int_1481], **kwargs_1482)

# Assigning a type to the variable 'call_assignment_1413' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1413', getitem___call_result_1483)

# Assigning a Name to a Name (line 14):
# Getting the type of 'call_assignment_1413' (line 14)
call_assignment_1413_1484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'call_assignment_1413')
# Assigning a type to the variable 'spacing' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'spacing', call_assignment_1413_1484)

# Assigning a Name to a Name (line 15):

# Assigning a Name to a Name (line 15):
# Getting the type of 'spacing' (line 15)
spacing_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 6), 'spacing')
# Assigning a type to the variable 'r5' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r5', spacing_1485)

# Assigning a Call to a Tuple (line 16):

# Assigning a Call to a Name:

# Call to linspace(...): (line 16)
# Processing the call arguments (line 16)
int_1488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'int')
int_1489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
int_1490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 39), 'int')
# Processing the call keyword arguments (line 16)
# Getting the type of 'False' (line 16)
False_1491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 52), 'False', False)
keyword_1492 = False_1491
# Getting the type of 'True' (line 16)
True_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 67), 'True', False)
keyword_1494 = True_1493
kwargs_1495 = {'retstep': keyword_1494, 'endpoint': keyword_1492}
# Getting the type of 'np' (line 16)
np_1486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'np', False)
# Obtaining the member 'linspace' of a type (line 16)
linspace_1487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), np_1486, 'linspace')
# Calling linspace(args, kwargs) (line 16)
linspace_call_result_1496 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), linspace_1487, *[int_1488, int_1489, int_1490], **kwargs_1495)

# Assigning a type to the variable 'call_assignment_1414' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1414', linspace_call_result_1496)

# Assigning a Call to a Name (line 16):

# Call to __getitem__(...):
# Processing the call arguments
int_1499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 0), 'int')
# Processing the call keyword arguments
kwargs_1500 = {}
# Getting the type of 'call_assignment_1414' (line 16)
call_assignment_1414_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1414', False)
# Obtaining the member '__getitem__' of a type (line 16)
getitem___1498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 0), call_assignment_1414_1497, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1501 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1498, *[int_1499], **kwargs_1500)

# Assigning a type to the variable 'call_assignment_1415' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1415', getitem___call_result_1501)

# Assigning a Name to a Name (line 16):
# Getting the type of 'call_assignment_1415' (line 16)
call_assignment_1415_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1415')
# Assigning a type to the variable 'samples3' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'samples3', call_assignment_1415_1502)

# Assigning a Call to a Name (line 16):

# Call to __getitem__(...):
# Processing the call arguments
int_1505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 0), 'int')
# Processing the call keyword arguments
kwargs_1506 = {}
# Getting the type of 'call_assignment_1414' (line 16)
call_assignment_1414_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1414', False)
# Obtaining the member '__getitem__' of a type (line 16)
getitem___1504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 0), call_assignment_1414_1503, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1507 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1504, *[int_1505], **kwargs_1506)

# Assigning a type to the variable 'call_assignment_1416' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1416', getitem___call_result_1507)

# Assigning a Name to a Name (line 16):
# Getting the type of 'call_assignment_1416' (line 16)
call_assignment_1416_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'call_assignment_1416')
# Assigning a type to the variable 'spacing' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'spacing', call_assignment_1416_1508)

# Assigning a Name to a Name (line 17):

# Assigning a Name to a Name (line 17):
# Getting the type of 'spacing' (line 17)
spacing_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 6), 'spacing')
# Assigning a type to the variable 'r6' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r6', spacing_1509)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

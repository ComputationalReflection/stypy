
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.randint(0, 2, 100)
6: r1 = np.logical_not(Z, out=Z)
7: 
8: Z = np.random.uniform(-1.0, 1.0, 100)
9: r2 = np.negative(Z, out=Z)
10: 
11: # l = globals().copy()
12: # for v in l:
13: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1510 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1510) is not StypyTypeError):

    if (import_1510 != 'pyd_module'):
        __import__(import_1510)
        sys_modules_1511 = sys.modules[import_1510]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1511.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1510)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_1515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_1516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
int_1517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1518 = {}
# Getting the type of 'np' (line 5)
np_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_1513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1512, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_1513, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_1519 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), randint_1514, *[int_1515, int_1516, int_1517], **kwargs_1518)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', randint_call_result_1519)

# Assigning a Call to a Name (line 6):

# Call to logical_not(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 20), 'Z', False)
# Processing the call keyword arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_1523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 27), 'Z', False)
keyword_1524 = Z_1523
kwargs_1525 = {'out': keyword_1524}
# Getting the type of 'np' (line 6)
np_1520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'np', False)
# Obtaining the member 'logical_not' of a type (line 6)
logical_not_1521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), np_1520, 'logical_not')
# Calling logical_not(args, kwargs) (line 6)
logical_not_call_result_1526 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), logical_not_1521, *[Z_1522], **kwargs_1525)

# Assigning a type to the variable 'r1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r1', logical_not_call_result_1526)

# Assigning a Call to a Name (line 8):

# Call to uniform(...): (line 8)
# Processing the call arguments (line 8)
float_1530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 22), 'float')
float_1531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 28), 'float')
int_1532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'int')
# Processing the call keyword arguments (line 8)
kwargs_1533 = {}
# Getting the type of 'np' (line 8)
np_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'random' of a type (line 8)
random_1528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_1527, 'random')
# Obtaining the member 'uniform' of a type (line 8)
uniform_1529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), random_1528, 'uniform')
# Calling uniform(args, kwargs) (line 8)
uniform_call_result_1534 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), uniform_1529, *[float_1530, float_1531, int_1532], **kwargs_1533)

# Assigning a type to the variable 'Z' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Z', uniform_call_result_1534)

# Assigning a Call to a Name (line 9):

# Call to negative(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'Z' (line 9)
Z_1537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'Z', False)
# Processing the call keyword arguments (line 9)
# Getting the type of 'Z' (line 9)
Z_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'Z', False)
keyword_1539 = Z_1538
kwargs_1540 = {'out': keyword_1539}
# Getting the type of 'np' (line 9)
np_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'negative' of a type (line 9)
negative_1536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_1535, 'negative')
# Calling negative(args, kwargs) (line 9)
negative_call_result_1541 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), negative_1536, *[Z_1537], **kwargs_1540)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', negative_call_result_1541)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: X = np.arange(8)
6: Y = X + 0.5
7: C = 1.0 / np.subtract.outer(X, Y)
8: r = (np.linalg.det(C))
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
import_2454 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2454) is not StypyTypeError):

    if (import_2454 != 'pyd_module'):
        __import__(import_2454)
        sys_modules_2455 = sys.modules[import_2454]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2455.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2454)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_2458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
kwargs_2459 = {}
# Getting the type of 'np' (line 5)
np_2456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_2457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2456, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_2460 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_2457, *[int_2458], **kwargs_2459)

# Assigning a type to the variable 'X' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'X', arange_call_result_2460)

# Assigning a BinOp to a Name (line 6):
# Getting the type of 'X' (line 6)
X_2461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'X')
float_2462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'float')
# Applying the binary operator '+' (line 6)
result_add_2463 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), '+', X_2461, float_2462)

# Assigning a type to the variable 'Y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Y', result_add_2463)

# Assigning a BinOp to a Name (line 7):
float_2464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'float')

# Call to outer(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'X' (line 7)
X_2468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 28), 'X', False)
# Getting the type of 'Y' (line 7)
Y_2469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 31), 'Y', False)
# Processing the call keyword arguments (line 7)
kwargs_2470 = {}
# Getting the type of 'np' (line 7)
np_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'np', False)
# Obtaining the member 'subtract' of a type (line 7)
subtract_2466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 10), np_2465, 'subtract')
# Obtaining the member 'outer' of a type (line 7)
outer_2467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 10), subtract_2466, 'outer')
# Calling outer(args, kwargs) (line 7)
outer_call_result_2471 = invoke(stypy.reporting.localization.Localization(__file__, 7, 10), outer_2467, *[X_2468, Y_2469], **kwargs_2470)

# Applying the binary operator 'div' (line 7)
result_div_2472 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 4), 'div', float_2464, outer_call_result_2471)

# Assigning a type to the variable 'C' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'C', result_div_2472)

# Assigning a Call to a Name (line 8):

# Call to det(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'C' (line 8)
C_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'C', False)
# Processing the call keyword arguments (line 8)
kwargs_2477 = {}
# Getting the type of 'np' (line 8)
np_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'linalg' of a type (line 8)
linalg_2474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_2473, 'linalg')
# Obtaining the member 'det' of a type (line 8)
det_2475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), linalg_2474, 'det')
# Calling det(args, kwargs) (line 8)
det_call_result_2478 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), det_2475, *[C_2476], **kwargs_2477)

# Assigning a type to the variable 'r' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r', det_call_result_2478)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

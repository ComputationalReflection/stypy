
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: 
4: import numpy as np
5: 
6: x = np.array(42)
7: 
8: r = type(x)
9: r2 = np.ndim(x)
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_437 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_437) is not StypyTypeError):

    if (import_437 != 'pyd_module'):
        __import__(import_437)
        sys_modules_438 = sys.modules[import_437]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_438.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_437)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)
int_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'int')
# Processing the call keyword arguments (line 6)
kwargs_442 = {}
# Getting the type of 'np' (line 6)
np_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_439, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_440, *[int_441], **kwargs_442)

# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', array_call_result_443)

# Assigning a Call to a Name (line 8):

# Call to type(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'x' (line 8)
x_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'x', False)
# Processing the call keyword arguments (line 8)
kwargs_446 = {}
# Getting the type of 'type' (line 8)
type_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'type', False)
# Calling type(args, kwargs) (line 8)
type_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), type_444, *[x_445], **kwargs_446)

# Assigning a type to the variable 'r' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r', type_call_result_447)

# Assigning a Call to a Name (line 9):

# Call to ndim(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'x', False)
# Processing the call keyword arguments (line 9)
kwargs_451 = {}
# Getting the type of 'np' (line 9)
np_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'ndim' of a type (line 9)
ndim_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_448, 'ndim')
# Calling ndim(args, kwargs) (line 9)
ndim_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), ndim_449, *[x_450], **kwargs_451)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', ndim_call_result_452)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

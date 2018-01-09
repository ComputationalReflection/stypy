
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.arange(10, dtype=np.int32)
6: Z2 = Z.astype(np.float32, copy=False)
7: 
8: # l = globals().copy()
9: # for v in l:
10: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_412 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_412) is not StypyTypeError):

    if (import_412 != 'pyd_module'):
        __import__(import_412)
        sys_modules_413 = sys.modules[import_412]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_413.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_412)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
# Getting the type of 'np' (line 5)
np_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 24), 'np', False)
# Obtaining the member 'int32' of a type (line 5)
int32_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 24), np_417, 'int32')
keyword_419 = int32_418
kwargs_420 = {'dtype': keyword_419}
# Getting the type of 'np' (line 5)
np_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_414, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_415, *[int_416], **kwargs_420)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', arange_call_result_421)

# Assigning a Call to a Name (line 6):

# Call to astype(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'np' (line 6)
np_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'np', False)
# Obtaining the member 'float32' of a type (line 6)
float32_425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 14), np_424, 'float32')
# Processing the call keyword arguments (line 6)
# Getting the type of 'False' (line 6)
False_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 31), 'False', False)
keyword_427 = False_426
kwargs_428 = {'copy': keyword_427}
# Getting the type of 'Z' (line 6)
Z_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'Z', False)
# Obtaining the member 'astype' of a type (line 6)
astype_423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), Z_422, 'astype')
# Calling astype(args, kwargs) (line 6)
astype_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), astype_423, *[float32_425], **kwargs_428)

# Assigning a type to the variable 'Z2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Z2', astype_call_result_429)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

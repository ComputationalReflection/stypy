
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http: //www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: r = np.identity(4)
6: 
7: r2 = np.identity(4, dtype=int)  # equivalent to np.identity(3, int)
8: 
9: # l = globals().copy()
10: # for v in l:
11: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1239 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1239) is not StypyTypeError):

    if (import_1239 != 'pyd_module'):
        __import__(import_1239)
        sys_modules_1240 = sys.modules[import_1239]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1240.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1239)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to identity(...): (line 5)
# Processing the call arguments (line 5)
int_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1244 = {}
# Getting the type of 'np' (line 5)
np_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'identity' of a type (line 5)
identity_1242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1241, 'identity')
# Calling identity(args, kwargs) (line 5)
identity_call_result_1245 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), identity_1242, *[int_1243], **kwargs_1244)

# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', identity_call_result_1245)

# Assigning a Call to a Name (line 7):

# Call to identity(...): (line 7)
# Processing the call arguments (line 7)
int_1248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'int')
# Processing the call keyword arguments (line 7)
# Getting the type of 'int' (line 7)
int_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'int', False)
keyword_1250 = int_1249
kwargs_1251 = {'dtype': keyword_1250}
# Getting the type of 'np' (line 7)
np_1246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'np', False)
# Obtaining the member 'identity' of a type (line 7)
identity_1247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), np_1246, 'identity')
# Calling identity(args, kwargs) (line 7)
identity_call_result_1252 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), identity_1247, *[int_1248], **kwargs_1251)

# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r2', identity_call_result_1252)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: # Print the numpy version and the configuration
6: 
7: r1 = np.__version__
8: r2 = np.show_config()
9: #
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_1172 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1172) is not StypyTypeError):

    if (import_1172 != 'pyd_module'):
        __import__(import_1172)
        sys_modules_1173 = sys.modules[import_1172]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1173.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1172)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Attribute to a Name (line 7):
# Getting the type of 'np' (line 7)
np_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'np')
# Obtaining the member '__version__' of a type (line 7)
version___1175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), np_1174, '__version__')
# Assigning a type to the variable 'r1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r1', version___1175)

# Assigning a Call to a Name (line 8):

# Call to show_config(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_1178 = {}
# Getting the type of 'np' (line 8)
np_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'show_config' of a type (line 8)
show_config_1177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_1176, 'show_config')
# Calling show_config(args, kwargs) (line 8)
show_config_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), show_config_1177, *[], **kwargs_1178)

# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', show_config_call_result_1179)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

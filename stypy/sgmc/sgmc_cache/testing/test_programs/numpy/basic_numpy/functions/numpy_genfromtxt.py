
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.genfromtxt("missing.dat", delimiter=",")
6: 
7: # l = globals().copy()
8: # for v in l:
9: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1183) is not StypyTypeError):

    if (import_1183 != 'pyd_module'):
        __import__(import_1183)
        sys_modules_1184 = sys.modules[import_1183]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1184.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1183)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to genfromtxt(...): (line 5)
# Processing the call arguments (line 5)
str_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'str', 'missing.dat')
# Processing the call keyword arguments (line 5)
str_1188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 43), 'str', ',')
keyword_1189 = str_1188
kwargs_1190 = {'delimiter': keyword_1189}
# Getting the type of 'np' (line 5)
np_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'genfromtxt' of a type (line 5)
genfromtxt_1186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1185, 'genfromtxt')
# Calling genfromtxt(args, kwargs) (line 5)
genfromtxt_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), genfromtxt_1186, *[str_1187], **kwargs_1190)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', genfromtxt_call_result_1191)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

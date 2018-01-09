
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.uniform(0, 10, 10)
6: 
7: r = (Z - Z % 1)
8: r2 = (np.floor(Z))
9: r3 = (np.ceil(Z) - 1)
10: r4 = (Z.astype(int))
11: r5 = (np.trunc(Z))
12: 
13: # l = globals().copy()
14: # for v in l:
15: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1125 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1125) is not StypyTypeError):

    if (import_1125 != 'pyd_module'):
        __import__(import_1125)
        sys_modules_1126 = sys.modules[import_1125]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1126.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1125)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to uniform(...): (line 5)
# Processing the call arguments (line 5)
int_1130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_1131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
int_1132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1133 = {}
# Getting the type of 'np' (line 5)
np_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_1128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1127, 'random')
# Obtaining the member 'uniform' of a type (line 5)
uniform_1129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_1128, 'uniform')
# Calling uniform(args, kwargs) (line 5)
uniform_call_result_1134 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), uniform_1129, *[int_1130, int_1131, int_1132], **kwargs_1133)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', uniform_call_result_1134)

# Assigning a BinOp to a Name (line 7):
# Getting the type of 'Z' (line 7)
Z_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'Z')
# Getting the type of 'Z' (line 7)
Z_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'Z')
int_1137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
# Applying the binary operator '%' (line 7)
result_mod_1138 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 9), '%', Z_1136, int_1137)

# Applying the binary operator '-' (line 7)
result_sub_1139 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 5), '-', Z_1135, result_mod_1138)

# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r', result_sub_1139)

# Assigning a Call to a Name (line 8):

# Call to floor(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'Z' (line 8)
Z_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'Z', False)
# Processing the call keyword arguments (line 8)
kwargs_1143 = {}
# Getting the type of 'np' (line 8)
np_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'np', False)
# Obtaining the member 'floor' of a type (line 8)
floor_1141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 6), np_1140, 'floor')
# Calling floor(args, kwargs) (line 8)
floor_call_result_1144 = invoke(stypy.reporting.localization.Localization(__file__, 8, 6), floor_1141, *[Z_1142], **kwargs_1143)

# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', floor_call_result_1144)

# Assigning a BinOp to a Name (line 9):

# Call to ceil(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'Z' (line 9)
Z_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'Z', False)
# Processing the call keyword arguments (line 9)
kwargs_1148 = {}
# Getting the type of 'np' (line 9)
np_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 6), 'np', False)
# Obtaining the member 'ceil' of a type (line 9)
ceil_1146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 6), np_1145, 'ceil')
# Calling ceil(args, kwargs) (line 9)
ceil_call_result_1149 = invoke(stypy.reporting.localization.Localization(__file__, 9, 6), ceil_1146, *[Z_1147], **kwargs_1148)

int_1150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
# Applying the binary operator '-' (line 9)
result_sub_1151 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 6), '-', ceil_call_result_1149, int_1150)

# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', result_sub_1151)

# Assigning a Call to a Name (line 10):

# Call to astype(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'int' (line 10)
int_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'int', False)
# Processing the call keyword arguments (line 10)
kwargs_1155 = {}
# Getting the type of 'Z' (line 10)
Z_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'Z', False)
# Obtaining the member 'astype' of a type (line 10)
astype_1153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 6), Z_1152, 'astype')
# Calling astype(args, kwargs) (line 10)
astype_call_result_1156 = invoke(stypy.reporting.localization.Localization(__file__, 10, 6), astype_1153, *[int_1154], **kwargs_1155)

# Assigning a type to the variable 'r4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r4', astype_call_result_1156)

# Assigning a Call to a Name (line 11):

# Call to trunc(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'Z' (line 11)
Z_1159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'Z', False)
# Processing the call keyword arguments (line 11)
kwargs_1160 = {}
# Getting the type of 'np' (line 11)
np_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'np', False)
# Obtaining the member 'trunc' of a type (line 11)
trunc_1158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 6), np_1157, 'trunc')
# Calling trunc(args, kwargs) (line 11)
trunc_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 11, 6), trunc_1158, *[Z_1159], **kwargs_1160)

# Assigning a type to the variable 'r5' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r5', trunc_call_result_1161)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

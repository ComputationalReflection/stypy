
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.uniform(-10, +10, 10)
6: r = (np.trunc(Z + np.copysign(0.5, Z)))
7: 
8: # l = globals().copy()
9: # for v in l:
10: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2198 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2198) is not StypyTypeError):

    if (import_2198 != 'pyd_module'):
        __import__(import_2198)
        sys_modules_2199 = sys.modules[import_2198]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2199.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2198)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to uniform(...): (line 5)
# Processing the call arguments (line 5)
int_2203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')

int_2204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
# Applying the 'uadd' unary operator (line 5)
result___pos___2205 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 27), 'uadd', int_2204)

int_2206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
# Processing the call keyword arguments (line 5)
kwargs_2207 = {}
# Getting the type of 'np' (line 5)
np_2200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_2201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2200, 'random')
# Obtaining the member 'uniform' of a type (line 5)
uniform_2202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_2201, 'uniform')
# Calling uniform(args, kwargs) (line 5)
uniform_call_result_2208 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), uniform_2202, *[int_2203, result___pos___2205, int_2206], **kwargs_2207)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', uniform_call_result_2208)

# Assigning a Call to a Name (line 6):

# Call to trunc(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_2211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'Z', False)

# Call to copysign(...): (line 6)
# Processing the call arguments (line 6)
float_2214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 30), 'float')
# Getting the type of 'Z' (line 6)
Z_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'Z', False)
# Processing the call keyword arguments (line 6)
kwargs_2216 = {}
# Getting the type of 'np' (line 6)
np_2212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 18), 'np', False)
# Obtaining the member 'copysign' of a type (line 6)
copysign_2213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 18), np_2212, 'copysign')
# Calling copysign(args, kwargs) (line 6)
copysign_call_result_2217 = invoke(stypy.reporting.localization.Localization(__file__, 6, 18), copysign_2213, *[float_2214, Z_2215], **kwargs_2216)

# Applying the binary operator '+' (line 6)
result_add_2218 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 14), '+', Z_2211, copysign_call_result_2217)

# Processing the call keyword arguments (line 6)
kwargs_2219 = {}
# Getting the type of 'np' (line 6)
np_2209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'np', False)
# Obtaining the member 'trunc' of a type (line 6)
trunc_2210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), np_2209, 'trunc')
# Calling trunc(args, kwargs) (line 6)
trunc_call_result_2220 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), trunc_2210, *[result_add_2218], **kwargs_2219)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', trunc_call_result_2220)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: A = np.random.randint(0, 2, 5)
6: B = np.random.randint(0, 2, 5)
7: equal = np.allclose(A, B)
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
import_2174 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2174) is not StypyTypeError):

    if (import_2174 != 'pyd_module'):
        __import__(import_2174)
        sys_modules_2175 = sys.modules[import_2174]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2175.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2174)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_2179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_2180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
int_2181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
# Processing the call keyword arguments (line 5)
kwargs_2182 = {}
# Getting the type of 'np' (line 5)
np_2176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_2177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2176, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_2178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_2177, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_2183 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), randint_2178, *[int_2179, int_2180, int_2181], **kwargs_2182)

# Assigning a type to the variable 'A' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'A', randint_call_result_2183)

# Assigning a Call to a Name (line 6):

# Call to randint(...): (line 6)
# Processing the call arguments (line 6)
int_2187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')
int_2188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'int')
int_2189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 28), 'int')
# Processing the call keyword arguments (line 6)
kwargs_2190 = {}
# Getting the type of 'np' (line 6)
np_2184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'random' of a type (line 6)
random_2185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_2184, 'random')
# Obtaining the member 'randint' of a type (line 6)
randint_2186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), random_2185, 'randint')
# Calling randint(args, kwargs) (line 6)
randint_call_result_2191 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), randint_2186, *[int_2187, int_2188, int_2189], **kwargs_2190)

# Assigning a type to the variable 'B' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'B', randint_call_result_2191)

# Assigning a Call to a Name (line 7):

# Call to allclose(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'A' (line 7)
A_2194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'A', False)
# Getting the type of 'B' (line 7)
B_2195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 23), 'B', False)
# Processing the call keyword arguments (line 7)
kwargs_2196 = {}
# Getting the type of 'np' (line 7)
np_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'np', False)
# Obtaining the member 'allclose' of a type (line 7)
allclose_2193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), np_2192, 'allclose')
# Calling allclose(args, kwargs) (line 7)
allclose_call_result_2197 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), allclose_2193, *[A_2194, B_2195], **kwargs_2196)

# Assigning a type to the variable 'equal' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'equal', allclose_call_result_2197)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

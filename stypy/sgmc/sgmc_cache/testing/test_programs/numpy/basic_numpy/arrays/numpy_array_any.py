
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.randint(0, 3, (3, 10))
6: r = ((~Z.any(axis=0)).any())
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 29), tuple_8, int_9)
# Adding element type (line 5)
int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 29), tuple_8, int_10)

# Processing the call keyword arguments (line 5)
kwargs_11 = {}
# Getting the type of 'np' (line 5)
np_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_3, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_4, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), randint_5, *[int_6, int_7, tuple_8], **kwargs_11)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', randint_call_result_12)

# Assigning a Call to a Name (line 6):

# Call to any(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_21 = {}


# Call to any(...): (line 6)
# Processing the call keyword arguments (line 6)
int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
keyword_16 = int_15
kwargs_17 = {'axis': keyword_16}
# Getting the type of 'Z' (line 6)
Z_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'Z', False)
# Obtaining the member 'any' of a type (line 6)
any_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 7), Z_13, 'any')
# Calling any(args, kwargs) (line 6)
any_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 6, 7), any_14, *[], **kwargs_17)

# Applying the '~' unary operator (line 6)
result_inv_19 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 6), '~', any_call_result_18)

# Obtaining the member 'any' of a type (line 6)
any_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 6), result_inv_19, 'any')
# Calling any(args, kwargs) (line 6)
any_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 6, 6), any_20, *[], **kwargs_21)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', any_call_result_22)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

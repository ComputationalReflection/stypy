
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.random(10)
6: Z[Z.argmax()] = 0
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
import_23 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_23) is not StypyTypeError):

    if (import_23 != 'pyd_module'):
        __import__(import_23)
        sys_modules_24 = sys.modules[import_23]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_24.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_23)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to random(...): (line 5)
# Processing the call arguments (line 5)
int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'int')
# Processing the call keyword arguments (line 5)
kwargs_29 = {}
# Getting the type of 'np' (line 5)
np_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_25, 'random')
# Obtaining the member 'random' of a type (line 5)
random_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_26, 'random')
# Calling random(args, kwargs) (line 5)
random_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), random_27, *[int_28], **kwargs_29)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', random_call_result_30)

# Assigning a Num to a Subscript (line 6):
int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'int')
# Getting the type of 'Z' (line 6)
Z_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Z')

# Call to argmax(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_35 = {}
# Getting the type of 'Z' (line 6)
Z_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 2), 'Z', False)
# Obtaining the member 'argmax' of a type (line 6)
argmax_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 2), Z_33, 'argmax')
# Calling argmax(args, kwargs) (line 6)
argmax_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 6, 2), argmax_34, *[], **kwargs_35)

# Storing an element on a container (line 6)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 0), Z_32, (argmax_call_result_36, int_31))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

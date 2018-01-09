
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: np.set_printoptions(threshold=np.nan)
6: Z = np.zeros((25, 25))
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
import_2343 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2343) is not StypyTypeError):

    if (import_2343 != 'pyd_module'):
        __import__(import_2343)
        sys_modules_2344 = sys.modules[import_2343]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2344.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2343)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Call to set_printoptions(...): (line 5)
# Processing the call keyword arguments (line 5)
# Getting the type of 'np' (line 5)
np_2347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 30), 'np', False)
# Obtaining the member 'nan' of a type (line 5)
nan_2348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 30), np_2347, 'nan')
keyword_2349 = nan_2348
kwargs_2350 = {'threshold': keyword_2349}
# Getting the type of 'np' (line 5)
np_2345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', False)
# Obtaining the member 'set_printoptions' of a type (line 5)
set_printoptions_2346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), np_2345, 'set_printoptions')
# Calling set_printoptions(args, kwargs) (line 5)
set_printoptions_call_result_2351 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), set_printoptions_2346, *[], **kwargs_2350)


# Assigning a Call to a Name (line 6):

# Call to zeros(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_2354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
int_2355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), tuple_2354, int_2355)
# Adding element type (line 6)
int_2356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), tuple_2354, int_2356)

# Processing the call keyword arguments (line 6)
kwargs_2357 = {}
# Getting the type of 'np' (line 6)
np_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'zeros' of a type (line 6)
zeros_2353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_2352, 'zeros')
# Calling zeros(args, kwargs) (line 6)
zeros_call_result_2358 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), zeros_2353, *[tuple_2354], **kwargs_2357)

# Assigning a type to the variable 'Z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Z', zeros_call_result_2358)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

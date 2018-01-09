
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.zeros(10)
6: Z.flags.writeable = False
7: # Type error
8: Z[0] = 1
9: 
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_296 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_296) is not StypyTypeError):

    if (import_296 != 'pyd_module'):
        __import__(import_296)
        sys_modules_297 = sys.modules[import_296]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_297.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_296)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to zeros(...): (line 5)
# Processing the call arguments (line 5)
int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Processing the call keyword arguments (line 5)
kwargs_301 = {}
# Getting the type of 'np' (line 5)
np_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'zeros' of a type (line 5)
zeros_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_298, 'zeros')
# Calling zeros(args, kwargs) (line 5)
zeros_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), zeros_299, *[int_300], **kwargs_301)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', zeros_call_result_302)

# Assigning a Name to a Attribute (line 6):
# Getting the type of 'False' (line 6)
False_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 20), 'False')
# Getting the type of 'Z' (line 6)
Z_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Z')
# Obtaining the member 'flags' of a type (line 6)
flags_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 0), Z_304, 'flags')
# Setting the type of the member 'writeable' of a type (line 6)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 0), flags_305, 'writeable', False_303)

# Assigning a Num to a Subscript (line 8):
int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 7), 'int')
# Getting the type of 'Z' (line 8)
Z_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Z')
int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 2), 'int')
# Storing an element on a container (line 8)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 0), Z_307, (int_308, int_306))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

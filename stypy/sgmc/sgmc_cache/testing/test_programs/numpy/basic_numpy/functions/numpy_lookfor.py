
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: 
3: import numpy as np
4: 
5: r = np.lookfor('create array')
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
import_1542 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1542) is not StypyTypeError):

    if (import_1542 != 'pyd_module'):
        __import__(import_1542)
        sys_modules_1543 = sys.modules[import_1542]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1543.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1542)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to lookfor(...): (line 5)
# Processing the call arguments (line 5)
str_1546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', 'create array')
# Processing the call keyword arguments (line 5)
kwargs_1547 = {}
# Getting the type of 'np' (line 5)
np_1544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'lookfor' of a type (line 5)
lookfor_1545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1544, 'lookfor')
# Calling lookfor(args, kwargs) (line 5)
lookfor_call_result_1548 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), lookfor_1545, *[str_1546], **kwargs_1547)

# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', lookfor_call_result_1548)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

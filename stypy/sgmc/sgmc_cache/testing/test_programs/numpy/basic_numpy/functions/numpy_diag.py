
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.diag(1 + np.arange(4), k=-1)
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
import_659 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_659) is not StypyTypeError):

    if (import_659 != 'pyd_module'):
        __import__(import_659)
        sys_modules_660 = sys.modules[import_659]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_660.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_659)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to diag(...): (line 5)
# Processing the call arguments (line 5)
int_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'int')

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
# Processing the call keyword arguments (line 5)
kwargs_667 = {}
# Getting the type of 'np' (line 5)
np_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 16), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 16), np_664, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 5, 16), arange_665, *[int_666], **kwargs_667)

# Applying the binary operator '+' (line 5)
result_add_669 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 12), '+', int_663, arange_call_result_668)

# Processing the call keyword arguments (line 5)
int_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
keyword_671 = int_670
kwargs_672 = {'k': keyword_671}
# Getting the type of 'np' (line 5)
np_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'diag' of a type (line 5)
diag_662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_661, 'diag')
# Calling diag(args, kwargs) (line 5)
diag_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), diag_662, *[result_add_669], **kwargs_672)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', diag_call_result_673)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

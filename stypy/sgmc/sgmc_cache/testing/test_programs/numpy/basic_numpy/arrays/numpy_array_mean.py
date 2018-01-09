
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: X = np.random.rand(5, 10)
6: 
7: # Recent versions of numpy
8: Y = X - X.mean(axis=1, keepdims=True)
9: 
10: # Older versions of numpy
11: Y2 = X - X.mean(axis=1).reshape(-1, 1)
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_821 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_821) is not StypyTypeError):

    if (import_821 != 'pyd_module'):
        __import__(import_821)
        sys_modules_822 = sys.modules[import_821]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_822.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_821)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to rand(...): (line 5)
# Processing the call arguments (line 5)
int_826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
int_827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
# Processing the call keyword arguments (line 5)
kwargs_828 = {}
# Getting the type of 'np' (line 5)
np_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_823, 'random')
# Obtaining the member 'rand' of a type (line 5)
rand_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_824, 'rand')
# Calling rand(args, kwargs) (line 5)
rand_call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), rand_825, *[int_826, int_827], **kwargs_828)

# Assigning a type to the variable 'X' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'X', rand_call_result_829)

# Assigning a BinOp to a Name (line 8):
# Getting the type of 'X' (line 8)
X_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'X')

# Call to mean(...): (line 8)
# Processing the call keyword arguments (line 8)
int_833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
keyword_834 = int_833
# Getting the type of 'True' (line 8)
True_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 32), 'True', False)
keyword_836 = True_835
kwargs_837 = {'keepdims': keyword_836, 'axis': keyword_834}
# Getting the type of 'X' (line 8)
X_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'X', False)
# Obtaining the member 'mean' of a type (line 8)
mean_832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), X_831, 'mean')
# Calling mean(args, kwargs) (line 8)
mean_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), mean_832, *[], **kwargs_837)

# Applying the binary operator '-' (line 8)
result_sub_839 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), '-', X_830, mean_call_result_838)

# Assigning a type to the variable 'Y' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Y', result_sub_839)

# Assigning a BinOp to a Name (line 11):
# Getting the type of 'X' (line 11)
X_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'X')

# Call to reshape(...): (line 11)
# Processing the call arguments (line 11)
int_848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 32), 'int')
int_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 36), 'int')
# Processing the call keyword arguments (line 11)
kwargs_850 = {}

# Call to mean(...): (line 11)
# Processing the call keyword arguments (line 11)
int_843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
keyword_844 = int_843
kwargs_845 = {'axis': keyword_844}
# Getting the type of 'X' (line 11)
X_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'X', False)
# Obtaining the member 'mean' of a type (line 11)
mean_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 9), X_841, 'mean')
# Calling mean(args, kwargs) (line 11)
mean_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 11, 9), mean_842, *[], **kwargs_845)

# Obtaining the member 'reshape' of a type (line 11)
reshape_847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 9), mean_call_result_846, 'reshape')
# Calling reshape(args, kwargs) (line 11)
reshape_call_result_851 = invoke(stypy.reporting.localization.Localization(__file__, 11, 9), reshape_847, *[int_848, int_849], **kwargs_850)

# Applying the binary operator '-' (line 11)
result_sub_852 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 5), '-', X_840, reshape_call_result_851)

# Assigning a type to the variable 'Y2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Y2', result_sub_852)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

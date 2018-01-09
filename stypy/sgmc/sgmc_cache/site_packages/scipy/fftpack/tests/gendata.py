
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy.io import loadmat
5: 
6: m = loadmat('test.mat', squeeze_me=True, struct_as_record=True,
7:         mat_dtype=True)
8: np.savez('test.npz', **m)
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_18705 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_18705) is not StypyTypeError):

    if (import_18705 != 'pyd_module'):
        __import__(import_18705)
        sys_modules_18706 = sys.modules[import_18705]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_18706.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_18705)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.io import loadmat' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_18707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.io')

if (type(import_18707) is not StypyTypeError):

    if (import_18707 != 'pyd_module'):
        __import__(import_18707)
        sys_modules_18708 = sys.modules[import_18707]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.io', sys_modules_18708.module_type_store, module_type_store, ['loadmat'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_18708, sys_modules_18708.module_type_store, module_type_store)
    else:
        from scipy.io import loadmat

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.io', None, module_type_store, ['loadmat'], [loadmat])

else:
    # Assigning a type to the variable 'scipy.io' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.io', import_18707)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')


# Assigning a Call to a Name (line 6):

# Call to loadmat(...): (line 6)
# Processing the call arguments (line 6)
str_18710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'test.mat')
# Processing the call keyword arguments (line 6)
# Getting the type of 'True' (line 6)
True_18711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'True', False)
keyword_18712 = True_18711
# Getting the type of 'True' (line 6)
True_18713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 58), 'True', False)
keyword_18714 = True_18713
# Getting the type of 'True' (line 7)
True_18715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'True', False)
keyword_18716 = True_18715
kwargs_18717 = {'mat_dtype': keyword_18716, 'squeeze_me': keyword_18712, 'struct_as_record': keyword_18714}
# Getting the type of 'loadmat' (line 6)
loadmat_18709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'loadmat', False)
# Calling loadmat(args, kwargs) (line 6)
loadmat_call_result_18718 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), loadmat_18709, *[str_18710], **kwargs_18717)

# Assigning a type to the variable 'm' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'm', loadmat_call_result_18718)

# Call to savez(...): (line 8)
# Processing the call arguments (line 8)
str_18721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 9), 'str', 'test.npz')
# Processing the call keyword arguments (line 8)
# Getting the type of 'm' (line 8)
m_18722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'm', False)
kwargs_18723 = {'m_18722': m_18722}
# Getting the type of 'np' (line 8)
np_18719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', False)
# Obtaining the member 'savez' of a type (line 8)
savez_18720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 0), np_18719, 'savez')
# Calling savez(args, kwargs) (line 8)
savez_call_result_18724 = invoke(stypy.reporting.localization.Localization(__file__, 8, 0), savez_18720, *[str_18721], **kwargs_18723)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

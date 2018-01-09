
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Utilities for dealing with MATLAB(R) files
3: 
4: Notes
5: -----
6: MATLAB(R) is a registered trademark of The MathWorks, Inc., 3 Apple Hill
7: Drive, Natick, MA 01760-2098, USA.
8: 
9: '''
10: from __future__ import division, print_function, absolute_import
11: 
12: # Matlab file read and write utilities
13: from .mio import loadmat, savemat, whosmat
14: from . import byteordercodes
15: 
16: __all__ = ['loadmat', 'savemat', 'whosmat', 'byteordercodes']
17: 
18: from scipy._lib._testutils import PytestTester
19: test = PytestTester(__name__)
20: del PytestTester
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_138001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\nUtilities for dealing with MATLAB(R) files\n\nNotes\n-----\nMATLAB(R) is a registered trademark of The MathWorks, Inc., 3 Apple Hill\nDrive, Natick, MA 01760-2098, USA.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.io.matlab.mio import loadmat, savemat, whosmat' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_138002 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio')

if (type(import_138002) is not StypyTypeError):

    if (import_138002 != 'pyd_module'):
        __import__(import_138002)
        sys_modules_138003 = sys.modules[import_138002]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio', sys_modules_138003.module_type_store, module_type_store, ['loadmat', 'savemat', 'whosmat'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_138003, sys_modules_138003.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio import loadmat, savemat, whosmat

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio', None, module_type_store, ['loadmat', 'savemat', 'whosmat'], [loadmat, savemat, whosmat])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio', import_138002)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.io.matlab import byteordercodes' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_138004 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab')

if (type(import_138004) is not StypyTypeError):

    if (import_138004 != 'pyd_module'):
        __import__(import_138004)
        sys_modules_138005 = sys.modules[import_138004]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab', sys_modules_138005.module_type_store, module_type_store, ['byteordercodes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_138005, sys_modules_138005.module_type_store, module_type_store)
    else:
        from scipy.io.matlab import byteordercodes

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab', None, module_type_store, ['byteordercodes'], [byteordercodes])

else:
    # Assigning a type to the variable 'scipy.io.matlab' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab', import_138004)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')


# Assigning a List to a Name (line 16):
__all__ = ['loadmat', 'savemat', 'whosmat', 'byteordercodes']
module_type_store.set_exportable_members(['loadmat', 'savemat', 'whosmat', 'byteordercodes'])

# Obtaining an instance of the builtin type 'list' (line 16)
list_138006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_138007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'loadmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_138006, str_138007)
# Adding element type (line 16)
str_138008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'str', 'savemat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_138006, str_138008)
# Adding element type (line 16)
str_138009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', 'whosmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_138006, str_138009)
# Adding element type (line 16)
str_138010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'str', 'byteordercodes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_138006, str_138010)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_138006)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_138011 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._testutils')

if (type(import_138011) is not StypyTypeError):

    if (import_138011 != 'pyd_module'):
        __import__(import_138011)
        sys_modules_138012 = sys.modules[import_138011]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._testutils', sys_modules_138012.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_138012, sys_modules_138012.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._testutils', import_138011)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')


# Assigning a Call to a Name (line 19):

# Call to PytestTester(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of '__name__' (line 19)
name___138014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), '__name__', False)
# Processing the call keyword arguments (line 19)
kwargs_138015 = {}
# Getting the type of 'PytestTester' (line 19)
PytestTester_138013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 19)
PytestTester_call_result_138016 = invoke(stypy.reporting.localization.Localization(__file__, 19, 7), PytestTester_138013, *[name___138014], **kwargs_138015)

# Assigning a type to the variable 'test' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'test', PytestTester_call_result_138016)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 20, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

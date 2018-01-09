
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Sub-package containing the matrix class and related functions.
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: from .defmatrix import *
7: 
8: __all__ = defmatrix.__all__
9: 
10: from numpy.testing.nosetester import _numpy_tester
11: test = _numpy_tester().test
12: bench = _numpy_tester().bench
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_161858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Sub-package containing the matrix class and related functions.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.matrixlib.defmatrix import ' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
import_161859 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.matrixlib.defmatrix')

if (type(import_161859) is not StypyTypeError):

    if (import_161859 != 'pyd_module'):
        __import__(import_161859)
        sys_modules_161860 = sys.modules[import_161859]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.matrixlib.defmatrix', sys_modules_161860.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_161860, sys_modules_161860.module_type_store, module_type_store)
    else:
        from numpy.matrixlib.defmatrix import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.matrixlib.defmatrix', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.matrixlib.defmatrix' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.matrixlib.defmatrix', import_161859)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')


# Assigning a Attribute to a Name (line 8):
# Getting the type of 'defmatrix' (line 8)
defmatrix_161861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'defmatrix')
# Obtaining the member '__all__' of a type (line 8)
all___161862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 10), defmatrix_161861, '__all__')
# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', all___161862)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
import_161863 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.nosetester')

if (type(import_161863) is not StypyTypeError):

    if (import_161863 != 'pyd_module'):
        __import__(import_161863)
        sys_modules_161864 = sys.modules[import_161863]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.nosetester', sys_modules_161864.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_161864, sys_modules_161864.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.nosetester', import_161863)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')


# Assigning a Attribute to a Name (line 11):

# Call to _numpy_tester(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_161866 = {}
# Getting the type of '_numpy_tester' (line 11)
_numpy_tester_161865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 11)
_numpy_tester_call_result_161867 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), _numpy_tester_161865, *[], **kwargs_161866)

# Obtaining the member 'test' of a type (line 11)
test_161868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), _numpy_tester_call_result_161867, 'test')
# Assigning a type to the variable 'test' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'test', test_161868)

# Assigning a Attribute to a Name (line 12):

# Call to _numpy_tester(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_161870 = {}
# Getting the type of '_numpy_tester' (line 12)
_numpy_tester_161869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 12)
_numpy_tester_call_result_161871 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), _numpy_tester_161869, *[], **kwargs_161870)

# Obtaining the member 'bench' of a type (line 12)
bench_161872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), _numpy_tester_call_result_161871, 'bench')
# Assigning a type to the variable 'bench' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'bench', bench_161872)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

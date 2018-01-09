
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Sparse Eigenvalue Solvers
3: -------------------------
4: 
5: The submodules of sparse.linalg.eigen:
6:     1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method
7: 
8: '''
9: from __future__ import division, print_function, absolute_import
10: 
11: from .arpack import *
12: from .lobpcg import *
13: 
14: __all__ = [s for s in dir() if not s.startswith('_')]
15: 
16: from scipy._lib._testutils import PytestTester
17: test = PytestTester(__name__)
18: del PytestTester
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_395971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nSparse Eigenvalue Solvers\n-------------------------\n\nThe submodules of sparse.linalg.eigen:\n    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg.eigen.arpack import ' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
import_395972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.eigen.arpack')

if (type(import_395972) is not StypyTypeError):

    if (import_395972 != 'pyd_module'):
        __import__(import_395972)
        sys_modules_395973 = sys.modules[import_395972]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.eigen.arpack', sys_modules_395973.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_395973, sys_modules_395973.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.eigen.arpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.eigen.arpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.eigen.arpack' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.eigen.arpack', import_395972)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse.linalg.eigen.lobpcg import ' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
import_395974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg')

if (type(import_395974) is not StypyTypeError):

    if (import_395974 != 'pyd_module'):
        __import__(import_395974)
        sys_modules_395975 = sys.modules[import_395974]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg', sys_modules_395975.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_395975, sys_modules_395975.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.eigen.lobpcg import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.eigen.lobpcg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg', import_395974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')


# Assigning a ListComp to a Name (line 14):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_395984 = {}
# Getting the type of 'dir' (line 14)
dir_395983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'dir', False)
# Calling dir(args, kwargs) (line 14)
dir_call_result_395985 = invoke(stypy.reporting.localization.Localization(__file__, 14, 22), dir_395983, *[], **kwargs_395984)

comprehension_395986 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 11), dir_call_result_395985)
# Assigning a type to the variable 's' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 's', comprehension_395986)


# Call to startswith(...): (line 14)
# Processing the call arguments (line 14)
str_395979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'str', '_')
# Processing the call keyword arguments (line 14)
kwargs_395980 = {}
# Getting the type of 's' (line 14)
s_395977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 14)
startswith_395978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 35), s_395977, 'startswith')
# Calling startswith(args, kwargs) (line 14)
startswith_call_result_395981 = invoke(stypy.reporting.localization.Localization(__file__, 14, 35), startswith_395978, *[str_395979], **kwargs_395980)

# Applying the 'not' unary operator (line 14)
result_not__395982 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 31), 'not', startswith_call_result_395981)

# Getting the type of 's' (line 14)
s_395976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 's')
list_395987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 11), list_395987, s_395976)
# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_395987)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')
import_395988 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils')

if (type(import_395988) is not StypyTypeError):

    if (import_395988 != 'pyd_module'):
        __import__(import_395988)
        sys_modules_395989 = sys.modules[import_395988]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils', sys_modules_395989.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_395989, sys_modules_395989.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils', import_395988)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/')


# Assigning a Call to a Name (line 17):

# Call to PytestTester(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of '__name__' (line 17)
name___395991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), '__name__', False)
# Processing the call keyword arguments (line 17)
kwargs_395992 = {}
# Getting the type of 'PytestTester' (line 17)
PytestTester_395990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 17)
PytestTester_call_result_395993 = invoke(stypy.reporting.localization.Localization(__file__, 17, 7), PytestTester_395990, *[name___395991], **kwargs_395992)

# Assigning a type to the variable 'test' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test', PytestTester_call_result_395993)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 18, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

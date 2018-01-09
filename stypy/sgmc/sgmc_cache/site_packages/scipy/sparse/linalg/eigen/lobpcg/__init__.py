
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)
3: 
4: LOBPCG is a preconditioned eigensolver for large symmetric positive definite
5: (SPD) generalized eigenproblems.
6: 
7: Call the function lobpcg - see help for lobpcg.lobpcg.
8: 
9: '''
10: from __future__ import division, print_function, absolute_import
11: 
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

str_406909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\nLocally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)\n\nLOBPCG is a preconditioned eigensolver for large symmetric positive definite\n(SPD) generalized eigenproblems.\n\nCall the function lobpcg - see help for lobpcg.lobpcg.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse.linalg.eigen.lobpcg.lobpcg import ' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_406910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg.lobpcg')

if (type(import_406910) is not StypyTypeError):

    if (import_406910 != 'pyd_module'):
        __import__(import_406910)
        sys_modules_406911 = sys.modules[import_406910]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg.lobpcg', sys_modules_406911.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_406911, sys_modules_406911.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.eigen.lobpcg.lobpcg import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg.lobpcg', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.eigen.lobpcg.lobpcg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg.lobpcg', import_406910)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')


# Assigning a ListComp to a Name (line 14):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_406920 = {}
# Getting the type of 'dir' (line 14)
dir_406919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'dir', False)
# Calling dir(args, kwargs) (line 14)
dir_call_result_406921 = invoke(stypy.reporting.localization.Localization(__file__, 14, 22), dir_406919, *[], **kwargs_406920)

comprehension_406922 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 11), dir_call_result_406921)
# Assigning a type to the variable 's' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 's', comprehension_406922)


# Call to startswith(...): (line 14)
# Processing the call arguments (line 14)
str_406915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'str', '_')
# Processing the call keyword arguments (line 14)
kwargs_406916 = {}
# Getting the type of 's' (line 14)
s_406913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 14)
startswith_406914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 35), s_406913, 'startswith')
# Calling startswith(args, kwargs) (line 14)
startswith_call_result_406917 = invoke(stypy.reporting.localization.Localization(__file__, 14, 35), startswith_406914, *[str_406915], **kwargs_406916)

# Applying the 'not' unary operator (line 14)
result_not__406918 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 31), 'not', startswith_call_result_406917)

# Getting the type of 's' (line 14)
s_406912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 's')
list_406923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 11), list_406923, s_406912)
# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_406923)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')
import_406924 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils')

if (type(import_406924) is not StypyTypeError):

    if (import_406924 != 'pyd_module'):
        __import__(import_406924)
        sys_modules_406925 = sys.modules[import_406924]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils', sys_modules_406925.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_406925, sys_modules_406925.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib._testutils', import_406924)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/')


# Assigning a Call to a Name (line 17):

# Call to PytestTester(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of '__name__' (line 17)
name___406927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), '__name__', False)
# Processing the call keyword arguments (line 17)
kwargs_406928 = {}
# Getting the type of 'PytestTester' (line 17)
PytestTester_406926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 17)
PytestTester_call_result_406929 = invoke(stypy.reporting.localization.Localization(__file__, 17, 7), PytestTester_406926, *[name___406927], **kwargs_406928)

# Assigning a type to the variable 'test' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test', PytestTester_call_result_406929)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 18, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

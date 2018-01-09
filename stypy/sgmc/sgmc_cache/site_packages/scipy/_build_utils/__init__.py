
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy as np
2: from ._fortran import *
3: from scipy._lib._version import NumpyVersion
4: 
5: 
6: # Don't use deprecated Numpy C API.  Define this to a fixed version instead of
7: # NPY_API_VERSION in order not to break compilation for released Scipy versions
8: # when Numpy introduces a new deprecation.  Use in setup.py::
9: #
10: #   config.add_extension('_name', sources=['source_fname'], **numpy_nodepr_api)
11: #
12: if NumpyVersion(np.__version__) >= '1.10.0.dev':
13:     numpy_nodepr_api = dict(define_macros=[("NPY_NO_DEPRECATED_API",
14:                                             "NPY_1_9_API_VERSION")])
15: else:
16:     numpy_nodepr_api = dict()
17: 
18: 
19: from scipy._lib._testutils import PytestTester
20: test = PytestTester(__name__)
21: del PytestTester
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_build_utils/')
import_706329 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy')

if (type(import_706329) is not StypyTypeError):

    if (import_706329 != 'pyd_module'):
        __import__(import_706329)
        sys_modules_706330 = sys.modules[import_706329]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', sys_modules_706330.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', import_706329)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_build_utils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from scipy._build_utils._fortran import ' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_build_utils/')
import_706331 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy._build_utils._fortran')

if (type(import_706331) is not StypyTypeError):

    if (import_706331 != 'pyd_module'):
        __import__(import_706331)
        sys_modules_706332 = sys.modules[import_706331]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy._build_utils._fortran', sys_modules_706332.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_706332, sys_modules_706332.module_type_store, module_type_store)
    else:
        from scipy._build_utils._fortran import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy._build_utils._fortran', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy._build_utils._fortran' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy._build_utils._fortran', import_706331)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_build_utils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_build_utils/')
import_706333 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._version')

if (type(import_706333) is not StypyTypeError):

    if (import_706333 != 'pyd_module'):
        __import__(import_706333)
        sys_modules_706334 = sys.modules[import_706333]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._version', sys_modules_706334.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_706334, sys_modules_706334.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._version', import_706333)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_build_utils/')




# Call to NumpyVersion(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'np' (line 12)
np_706336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 12)
version___706337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), np_706336, '__version__')
# Processing the call keyword arguments (line 12)
kwargs_706338 = {}
# Getting the type of 'NumpyVersion' (line 12)
NumpyVersion_706335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 3), 'NumpyVersion', False)
# Calling NumpyVersion(args, kwargs) (line 12)
NumpyVersion_call_result_706339 = invoke(stypy.reporting.localization.Localization(__file__, 12, 3), NumpyVersion_706335, *[version___706337], **kwargs_706338)

str_706340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'str', '1.10.0.dev')
# Applying the binary operator '>=' (line 12)
result_ge_706341 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 3), '>=', NumpyVersion_call_result_706339, str_706340)

# Testing the type of an if condition (line 12)
if_condition_706342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 0), result_ge_706341)
# Assigning a type to the variable 'if_condition_706342' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'if_condition_706342', if_condition_706342)
# SSA begins for if statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 13):

# Call to dict(...): (line 13)
# Processing the call keyword arguments (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_706344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 42), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_706345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 44), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
str_706346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 44), 'str', 'NPY_NO_DEPRECATED_API')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 44), tuple_706345, str_706346)
# Adding element type (line 13)
str_706347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 44), 'str', 'NPY_1_9_API_VERSION')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 44), tuple_706345, str_706347)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 42), list_706344, tuple_706345)

keyword_706348 = list_706344
kwargs_706349 = {'define_macros': keyword_706348}
# Getting the type of 'dict' (line 13)
dict_706343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'dict', False)
# Calling dict(args, kwargs) (line 13)
dict_call_result_706350 = invoke(stypy.reporting.localization.Localization(__file__, 13, 23), dict_706343, *[], **kwargs_706349)

# Assigning a type to the variable 'numpy_nodepr_api' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy_nodepr_api', dict_call_result_706350)
# SSA branch for the else part of an if statement (line 12)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 16):

# Call to dict(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_706352 = {}
# Getting the type of 'dict' (line 16)
dict_706351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'dict', False)
# Calling dict(args, kwargs) (line 16)
dict_call_result_706353 = invoke(stypy.reporting.localization.Localization(__file__, 16, 23), dict_706351, *[], **kwargs_706352)

# Assigning a type to the variable 'numpy_nodepr_api' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy_nodepr_api', dict_call_result_706353)
# SSA join for if statement (line 12)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_build_utils/')
import_706354 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._testutils')

if (type(import_706354) is not StypyTypeError):

    if (import_706354 != 'pyd_module'):
        __import__(import_706354)
        sys_modules_706355 = sys.modules[import_706354]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._testutils', sys_modules_706355.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_706355, sys_modules_706355.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._testutils', import_706354)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_build_utils/')


# Assigning a Call to a Name (line 20):

# Call to PytestTester(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of '__name__' (line 20)
name___706357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), '__name__', False)
# Processing the call keyword arguments (line 20)
kwargs_706358 = {}
# Getting the type of 'PytestTester' (line 20)
PytestTester_706356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 20)
PytestTester_call_result_706359 = invoke(stypy.reporting.localization.Localization(__file__, 20, 7), PytestTester_706356, *[name___706357], **kwargs_706358)

# Assigning a type to the variable 'test' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'test', PytestTester_call_result_706359)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 21, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

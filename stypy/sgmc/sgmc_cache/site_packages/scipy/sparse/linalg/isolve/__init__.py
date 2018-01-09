
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: "Iterative Solvers for Sparse Linear Systems"
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: #from info import __doc__
6: from .iterative import *
7: from .minres import minres
8: from .lgmres import lgmres
9: from .lsqr import lsqr
10: from .lsmr import lsmr
11: from ._gcrotmk import gcrotmk
12: 
13: __all__ = [s for s in dir() if not s.startswith('_')]
14: 
15: from scipy._lib._testutils import PytestTester
16: test = PytestTester(__name__)
17: del PytestTester
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_416350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Iterative Solvers for Sparse Linear Systems')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse.linalg.isolve.iterative import ' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416351 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.iterative')

if (type(import_416351) is not StypyTypeError):

    if (import_416351 != 'pyd_module'):
        __import__(import_416351)
        sys_modules_416352 = sys.modules[import_416351]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.iterative', sys_modules_416352.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_416352, sys_modules_416352.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.iterative import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.iterative', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.iterative' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.linalg.isolve.iterative', import_416351)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse.linalg.isolve.minres import minres' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416353 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.minres')

if (type(import_416353) is not StypyTypeError):

    if (import_416353 != 'pyd_module'):
        __import__(import_416353)
        sys_modules_416354 = sys.modules[import_416353]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.minres', sys_modules_416354.module_type_store, module_type_store, ['minres'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_416354, sys_modules_416354.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.minres import minres

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.minres', None, module_type_store, ['minres'], [minres])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.minres' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg.isolve.minres', import_416353)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.sparse.linalg.isolve.lgmres import lgmres' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416355 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg.isolve.lgmres')

if (type(import_416355) is not StypyTypeError):

    if (import_416355 != 'pyd_module'):
        __import__(import_416355)
        sys_modules_416356 = sys.modules[import_416355]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg.isolve.lgmres', sys_modules_416356.module_type_store, module_type_store, ['lgmres'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_416356, sys_modules_416356.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.lgmres import lgmres

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg.isolve.lgmres', None, module_type_store, ['lgmres'], [lgmres])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.lgmres' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg.isolve.lgmres', import_416355)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.sparse.linalg.isolve.lsqr import lsqr' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416357 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.isolve.lsqr')

if (type(import_416357) is not StypyTypeError):

    if (import_416357 != 'pyd_module'):
        __import__(import_416357)
        sys_modules_416358 = sys.modules[import_416357]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.isolve.lsqr', sys_modules_416358.module_type_store, module_type_store, ['lsqr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_416358, sys_modules_416358.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.lsqr import lsqr

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.isolve.lsqr', None, module_type_store, ['lsqr'], [lsqr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.lsqr' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg.isolve.lsqr', import_416357)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg.isolve.lsmr import lsmr' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416359 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.lsmr')

if (type(import_416359) is not StypyTypeError):

    if (import_416359 != 'pyd_module'):
        __import__(import_416359)
        sys_modules_416360 = sys.modules[import_416359]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.lsmr', sys_modules_416360.module_type_store, module_type_store, ['lsmr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_416360, sys_modules_416360.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.lsmr import lsmr

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.lsmr', None, module_type_store, ['lsmr'], [lsmr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.lsmr' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.lsmr', import_416359)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg.isolve._gcrotmk import gcrotmk' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416361 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve._gcrotmk')

if (type(import_416361) is not StypyTypeError):

    if (import_416361 != 'pyd_module'):
        __import__(import_416361)
        sys_modules_416362 = sys.modules[import_416361]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve._gcrotmk', sys_modules_416362.module_type_store, module_type_store, ['gcrotmk'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_416362, sys_modules_416362.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve._gcrotmk import gcrotmk

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve._gcrotmk', None, module_type_store, ['gcrotmk'], [gcrotmk])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve._gcrotmk' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg.isolve._gcrotmk', import_416361)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a ListComp to a Name (line 13):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_416371 = {}
# Getting the type of 'dir' (line 13)
dir_416370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'dir', False)
# Calling dir(args, kwargs) (line 13)
dir_call_result_416372 = invoke(stypy.reporting.localization.Localization(__file__, 13, 22), dir_416370, *[], **kwargs_416371)

comprehension_416373 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 11), dir_call_result_416372)
# Assigning a type to the variable 's' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 's', comprehension_416373)


# Call to startswith(...): (line 13)
# Processing the call arguments (line 13)
str_416366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 48), 'str', '_')
# Processing the call keyword arguments (line 13)
kwargs_416367 = {}
# Getting the type of 's' (line 13)
s_416364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 13)
startswith_416365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 35), s_416364, 'startswith')
# Calling startswith(args, kwargs) (line 13)
startswith_call_result_416368 = invoke(stypy.reporting.localization.Localization(__file__, 13, 35), startswith_416365, *[str_416366], **kwargs_416367)

# Applying the 'not' unary operator (line 13)
result_not__416369 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 31), 'not', startswith_call_result_416368)

# Getting the type of 's' (line 13)
s_416363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 's')
list_416374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 11), list_416374, s_416363)
# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_416374)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_416375 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib._testutils')

if (type(import_416375) is not StypyTypeError):

    if (import_416375 != 'pyd_module'):
        __import__(import_416375)
        sys_modules_416376 = sys.modules[import_416375]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib._testutils', sys_modules_416376.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_416376, sys_modules_416376.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib._testutils', import_416375)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a Call to a Name (line 16):

# Call to PytestTester(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of '__name__' (line 16)
name___416378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), '__name__', False)
# Processing the call keyword arguments (line 16)
kwargs_416379 = {}
# Getting the type of 'PytestTester' (line 16)
PytestTester_416377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 16)
PytestTester_call_result_416380 = invoke(stypy.reporting.localization.Localization(__file__, 16, 7), PytestTester_416377, *[name___416378], **kwargs_416379)

# Assigning a type to the variable 'test' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'test', PytestTester_call_result_416380)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 17, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

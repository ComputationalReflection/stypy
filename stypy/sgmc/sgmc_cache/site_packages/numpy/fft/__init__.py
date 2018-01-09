
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: # To get sub-modules
4: from .info import __doc__
5: 
6: from .fftpack import *
7: from .helper import *
8: 
9: from numpy.testing.nosetester import _numpy_tester
10: test = _numpy_tester().test
11: bench = _numpy_tester().bench
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.fft.info import __doc__' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_101117 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.fft.info')

if (type(import_101117) is not StypyTypeError):

    if (import_101117 != 'pyd_module'):
        __import__(import_101117)
        sys_modules_101118 = sys.modules[import_101117]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.fft.info', sys_modules_101118.module_type_store, module_type_store, ['__doc__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_101118, sys_modules_101118.module_type_store, module_type_store)
    else:
        from numpy.fft.info import __doc__

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.fft.info', None, module_type_store, ['__doc__'], [__doc__])

else:
    # Assigning a type to the variable 'numpy.fft.info' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.fft.info', import_101117)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.fft.fftpack import ' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_101119 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.fft.fftpack')

if (type(import_101119) is not StypyTypeError):

    if (import_101119 != 'pyd_module'):
        __import__(import_101119)
        sys_modules_101120 = sys.modules[import_101119]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.fft.fftpack', sys_modules_101120.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_101120, sys_modules_101120.module_type_store, module_type_store)
    else:
        from numpy.fft.fftpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.fft.fftpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.fft.fftpack' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.fft.fftpack', import_101119)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.fft.helper import ' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_101121 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft.helper')

if (type(import_101121) is not StypyTypeError):

    if (import_101121 != 'pyd_module'):
        __import__(import_101121)
        sys_modules_101122 = sys.modules[import_101121]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft.helper', sys_modules_101122.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_101122, sys_modules_101122.module_type_store, module_type_store)
    else:
        from numpy.fft.helper import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft.helper', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.fft.helper' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft.helper', import_101121)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_101123 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing.nosetester')

if (type(import_101123) is not StypyTypeError):

    if (import_101123 != 'pyd_module'):
        __import__(import_101123)
        sys_modules_101124 = sys.modules[import_101123]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing.nosetester', sys_modules_101124.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_101124, sys_modules_101124.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing.nosetester', import_101123)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')


# Assigning a Attribute to a Name (line 10):

# Call to _numpy_tester(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_101126 = {}
# Getting the type of '_numpy_tester' (line 10)
_numpy_tester_101125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 10)
_numpy_tester_call_result_101127 = invoke(stypy.reporting.localization.Localization(__file__, 10, 7), _numpy_tester_101125, *[], **kwargs_101126)

# Obtaining the member 'test' of a type (line 10)
test_101128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 7), _numpy_tester_call_result_101127, 'test')
# Assigning a type to the variable 'test' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test', test_101128)

# Assigning a Attribute to a Name (line 11):

# Call to _numpy_tester(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_101130 = {}
# Getting the type of '_numpy_tester' (line 11)
_numpy_tester_101129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 11)
_numpy_tester_call_result_101131 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), _numpy_tester_101129, *[], **kwargs_101130)

# Obtaining the member 'bench' of a type (line 11)
bench_101132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), _numpy_tester_call_result_101131, 'bench')
# Assigning a type to the variable 'bench' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'bench', bench_101132)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

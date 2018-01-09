
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: 
5: from .__version__ import version as __version__
6: # Must import local ccompiler ASAP in order to get
7: # customized CCompiler.spawn effective.
8: from . import ccompiler
9: from . import unixccompiler
10: 
11: from .info import __doc__
12: from .npy_pkg_config import *
13: 
14: # If numpy is installed, add distutils.test()
15: try:
16:     from . import __config__
17:     # Normally numpy is installed if the above import works, but an interrupted
18:     # in-place build could also have left a __config__.py.  In that case the
19:     # next import may still fail, so keep it inside the try block.
20:     from numpy.testing.nosetester import _numpy_tester
21:     test = _numpy_tester().test
22: except ImportError:
23:     pass
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.distutils.__version__ import __version__' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52252 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.__version__')

if (type(import_52252) is not StypyTypeError):

    if (import_52252 != 'pyd_module'):
        __import__(import_52252)
        sys_modules_52253 = sys.modules[import_52252]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.__version__', sys_modules_52253.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_52253, sys_modules_52253.module_type_store, module_type_store)
    else:
        from numpy.distutils.__version__ import version as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.__version__', None, module_type_store, ['version'], [__version__])

else:
    # Assigning a type to the variable 'numpy.distutils.__version__' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.__version__', import_52252)

# Adding an alias
module_type_store.add_alias('__version__', 'version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.distutils import ccompiler' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52254 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils')

if (type(import_52254) is not StypyTypeError):

    if (import_52254 != 'pyd_module'):
        __import__(import_52254)
        sys_modules_52255 = sys.modules[import_52254]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils', sys_modules_52255.module_type_store, module_type_store, ['ccompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_52255, sys_modules_52255.module_type_store, module_type_store)
    else:
        from numpy.distutils import ccompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils', None, module_type_store, ['ccompiler'], [ccompiler])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils', import_52254)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.distutils import unixccompiler' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52256 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils')

if (type(import_52256) is not StypyTypeError):

    if (import_52256 != 'pyd_module'):
        __import__(import_52256)
        sys_modules_52257 = sys.modules[import_52256]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils', sys_modules_52257.module_type_store, module_type_store, ['unixccompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_52257, sys_modules_52257.module_type_store, module_type_store)
    else:
        from numpy.distutils import unixccompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils', None, module_type_store, ['unixccompiler'], [unixccompiler])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils', import_52256)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.distutils.info import __doc__' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52258 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.info')

if (type(import_52258) is not StypyTypeError):

    if (import_52258 != 'pyd_module'):
        __import__(import_52258)
        sys_modules_52259 = sys.modules[import_52258]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.info', sys_modules_52259.module_type_store, module_type_store, ['__doc__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_52259, sys_modules_52259.module_type_store, module_type_store)
    else:
        from numpy.distutils.info import __doc__

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.info', None, module_type_store, ['__doc__'], [__doc__])

else:
    # Assigning a type to the variable 'numpy.distutils.info' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.info', import_52258)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.distutils.npy_pkg_config import ' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52260 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.npy_pkg_config')

if (type(import_52260) is not StypyTypeError):

    if (import_52260 != 'pyd_module'):
        __import__(import_52260)
        sys_modules_52261 = sys.modules[import_52260]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.npy_pkg_config', sys_modules_52261.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_52261, sys_modules_52261.module_type_store, module_type_store)
    else:
        from numpy.distutils.npy_pkg_config import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.npy_pkg_config', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.distutils.npy_pkg_config' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.npy_pkg_config', import_52260)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')



# SSA begins for try-except statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 4))

# 'from numpy.distutils import __config__' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52262 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils')

if (type(import_52262) is not StypyTypeError):

    if (import_52262 != 'pyd_module'):
        __import__(import_52262)
        sys_modules_52263 = sys.modules[import_52262]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils', sys_modules_52263.module_type_store, module_type_store, ['__config__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 4), __file__, sys_modules_52263, sys_modules_52263.module_type_store, module_type_store)
    else:
        from numpy.distutils import __config__

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils', None, module_type_store, ['__config__'], [__config__])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils', import_52262)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 4))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_52264 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'numpy.testing.nosetester')

if (type(import_52264) is not StypyTypeError):

    if (import_52264 != 'pyd_module'):
        __import__(import_52264)
        sys_modules_52265 = sys.modules[import_52264]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'numpy.testing.nosetester', sys_modules_52265.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 4), __file__, sys_modules_52265, sys_modules_52265.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'numpy.testing.nosetester', import_52264)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


# Assigning a Attribute to a Name (line 21):

# Call to _numpy_tester(...): (line 21)
# Processing the call keyword arguments (line 21)
kwargs_52267 = {}
# Getting the type of '_numpy_tester' (line 21)
_numpy_tester_52266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 21)
_numpy_tester_call_result_52268 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), _numpy_tester_52266, *[], **kwargs_52267)

# Obtaining the member 'test' of a type (line 21)
test_52269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), _numpy_tester_call_result_52268, 'test')
# Assigning a type to the variable 'test' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'test', test_52269)
# SSA branch for the except part of a try statement (line 15)
# SSA branch for the except 'ImportError' branch of a try statement (line 15)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 15)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

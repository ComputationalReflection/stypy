
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Compatibility module.
3: 
4: This module contains duplicated code from Python itself or 3rd party
5: extensions, which may be included for the following reasons:
6: 
7:   * compatibility
8:   * we may only need a small subset of the copied library/module
9: 
10: '''
11: from __future__ import division, absolute_import, print_function
12: 
13: from . import _inspect
14: from . import py3k
15: from ._inspect import getargspec, formatargspec
16: from .py3k import *
17: 
18: __all__ = []
19: __all__.extend(_inspect.__all__)
20: __all__.extend(py3k.__all__)
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_26313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', '\nCompatibility module.\n\nThis module contains duplicated code from Python itself or 3rd party\nextensions, which may be included for the following reasons:\n\n  * compatibility\n  * we may only need a small subset of the copied library/module\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.compat import _inspect' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/compat/')
import_26314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat')

if (type(import_26314) is not StypyTypeError):

    if (import_26314 != 'pyd_module'):
        __import__(import_26314)
        sys_modules_26315 = sys.modules[import_26314]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat', sys_modules_26315.module_type_store, module_type_store, ['_inspect'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_26315, sys_modules_26315.module_type_store, module_type_store)
    else:
        from numpy.compat import _inspect

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat', None, module_type_store, ['_inspect'], [_inspect])

else:
    # Assigning a type to the variable 'numpy.compat' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat', import_26314)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/compat/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.compat import py3k' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/compat/')
import_26316 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.compat')

if (type(import_26316) is not StypyTypeError):

    if (import_26316 != 'pyd_module'):
        __import__(import_26316)
        sys_modules_26317 = sys.modules[import_26316]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.compat', sys_modules_26317.module_type_store, module_type_store, ['py3k'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_26317, sys_modules_26317.module_type_store, module_type_store)
    else:
        from numpy.compat import py3k

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.compat', None, module_type_store, ['py3k'], [py3k])

else:
    # Assigning a type to the variable 'numpy.compat' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.compat', import_26316)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/compat/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.compat._inspect import getargspec, formatargspec' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/compat/')
import_26318 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat._inspect')

if (type(import_26318) is not StypyTypeError):

    if (import_26318 != 'pyd_module'):
        __import__(import_26318)
        sys_modules_26319 = sys.modules[import_26318]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat._inspect', sys_modules_26319.module_type_store, module_type_store, ['getargspec', 'formatargspec'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_26319, sys_modules_26319.module_type_store, module_type_store)
    else:
        from numpy.compat._inspect import getargspec, formatargspec

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat._inspect', None, module_type_store, ['getargspec', 'formatargspec'], [getargspec, formatargspec])

else:
    # Assigning a type to the variable 'numpy.compat._inspect' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat._inspect', import_26318)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/compat/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from numpy.compat.py3k import ' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/compat/')
import_26320 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat.py3k')

if (type(import_26320) is not StypyTypeError):

    if (import_26320 != 'pyd_module'):
        __import__(import_26320)
        sys_modules_26321 = sys.modules[import_26320]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat.py3k', sys_modules_26321.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_26321, sys_modules_26321.module_type_store, module_type_store)
    else:
        from numpy.compat.py3k import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat.py3k', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.compat.py3k' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat.py3k', import_26320)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/compat/')


# Assigning a List to a Name (line 18):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 18)
list_26322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)

# Assigning a type to the variable '__all__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__all__', list_26322)

# Call to extend(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of '_inspect' (line 19)
_inspect_26325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), '_inspect', False)
# Obtaining the member '__all__' of a type (line 19)
all___26326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), _inspect_26325, '__all__')
# Processing the call keyword arguments (line 19)
kwargs_26327 = {}
# Getting the type of '__all__' (line 19)
all___26323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__all__', False)
# Obtaining the member 'extend' of a type (line 19)
extend_26324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), all___26323, 'extend')
# Calling extend(args, kwargs) (line 19)
extend_call_result_26328 = invoke(stypy.reporting.localization.Localization(__file__, 19, 0), extend_26324, *[all___26326], **kwargs_26327)


# Call to extend(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'py3k' (line 20)
py3k_26331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'py3k', False)
# Obtaining the member '__all__' of a type (line 20)
all___26332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 15), py3k_26331, '__all__')
# Processing the call keyword arguments (line 20)
kwargs_26333 = {}
# Getting the type of '__all__' (line 20)
all___26329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__all__', False)
# Obtaining the member 'extend' of a type (line 20)
extend_26330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), all___26329, 'extend')
# Calling extend(args, kwargs) (line 20)
extend_call_result_26334 = invoke(stypy.reporting.localization.Localization(__file__, 20, 0), extend_26330, *[all___26332], **kwargs_26333)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

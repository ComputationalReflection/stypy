
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unstructured triangular grid functions.
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: 
9: from .triangulation import *
10: from .tricontour import *
11: from .tritools import *
12: from .trifinder import *
13: from .triinterpolate import *
14: from .trirefine import *
15: from .tripcolor import *
16: from .triplot import *
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_302345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nUnstructured triangular grid functions.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302346 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_302346) is not StypyTypeError):

    if (import_302346 != 'pyd_module'):
        __import__(import_302346)
        sys_modules_302347 = sys.modules[import_302346]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_302347.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_302346)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.tri.triangulation import ' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302348 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri.triangulation')

if (type(import_302348) is not StypyTypeError):

    if (import_302348 != 'pyd_module'):
        __import__(import_302348)
        sys_modules_302349 = sys.modules[import_302348]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri.triangulation', sys_modules_302349.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_302349, sys_modules_302349.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triangulation import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri.triangulation', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.triangulation' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri.triangulation', import_302348)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.tri.tricontour import ' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302350 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.tricontour')

if (type(import_302350) is not StypyTypeError):

    if (import_302350 != 'pyd_module'):
        __import__(import_302350)
        sys_modules_302351 = sys.modules[import_302350]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.tricontour', sys_modules_302351.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_302351, sys_modules_302351.module_type_store, module_type_store)
    else:
        from matplotlib.tri.tricontour import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.tricontour', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.tricontour' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.tricontour', import_302350)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib.tri.tritools import ' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302352 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.tritools')

if (type(import_302352) is not StypyTypeError):

    if (import_302352 != 'pyd_module'):
        __import__(import_302352)
        sys_modules_302353 = sys.modules[import_302352]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.tritools', sys_modules_302353.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_302353, sys_modules_302353.module_type_store, module_type_store)
    else:
        from matplotlib.tri.tritools import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.tritools', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.tritools' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.tritools', import_302352)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.tri.trifinder import ' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302354 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.tri.trifinder')

if (type(import_302354) is not StypyTypeError):

    if (import_302354 != 'pyd_module'):
        __import__(import_302354)
        sys_modules_302355 = sys.modules[import_302354]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.tri.trifinder', sys_modules_302355.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_302355, sys_modules_302355.module_type_store, module_type_store)
    else:
        from matplotlib.tri.trifinder import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.tri.trifinder', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.trifinder' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.tri.trifinder', import_302354)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.tri.triinterpolate import ' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302356 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.tri.triinterpolate')

if (type(import_302356) is not StypyTypeError):

    if (import_302356 != 'pyd_module'):
        __import__(import_302356)
        sys_modules_302357 = sys.modules[import_302356]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.tri.triinterpolate', sys_modules_302357.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_302357, sys_modules_302357.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triinterpolate import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.tri.triinterpolate', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.triinterpolate' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.tri.triinterpolate', import_302356)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib.tri.trirefine import ' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302358 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.tri.trirefine')

if (type(import_302358) is not StypyTypeError):

    if (import_302358 != 'pyd_module'):
        __import__(import_302358)
        sys_modules_302359 = sys.modules[import_302358]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.tri.trirefine', sys_modules_302359.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_302359, sys_modules_302359.module_type_store, module_type_store)
    else:
        from matplotlib.tri.trirefine import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.tri.trirefine', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.trirefine' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.tri.trirefine', import_302358)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.tri.tripcolor import ' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302360 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.tri.tripcolor')

if (type(import_302360) is not StypyTypeError):

    if (import_302360 != 'pyd_module'):
        __import__(import_302360)
        sys_modules_302361 = sys.modules[import_302360]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.tri.tripcolor', sys_modules_302361.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_302361, sys_modules_302361.module_type_store, module_type_store)
    else:
        from matplotlib.tri.tripcolor import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.tri.tripcolor', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.tripcolor' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.tri.tripcolor', import_302360)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.tri.triplot import ' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_302362 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.tri.triplot')

if (type(import_302362) is not StypyTypeError):

    if (import_302362 != 'pyd_module'):
        __import__(import_302362)
        sys_modules_302363 = sys.modules[import_302362]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.tri.triplot', sys_modules_302363.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_302363, sys_modules_302363.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triplot import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.tri.triplot', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.tri.triplot' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.tri.triplot', import_302362)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

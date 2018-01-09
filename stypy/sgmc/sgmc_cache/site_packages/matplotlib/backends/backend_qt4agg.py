
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Render to qt from agg
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: 
9: from .backend_agg import FigureCanvasAgg
10: from .backend_qt4 import (
11:     QtCore, _BackendQT4, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT)
12: from .backend_qt5agg import FigureCanvasQTAggBase
13: 
14: 
15: class FigureCanvasQTAgg(FigureCanvasQTAggBase, FigureCanvasQT):
16:     '''
17:     The canvas the figure renders into.  Calls the draw and print fig
18:     methods, creates the renderers, etc...
19: 
20:     Attributes
21:     ----------
22:     figure : `matplotlib.figure.Figure`
23:         A high-level Figure instance
24: 
25:     '''
26: 
27: 
28: @_BackendQT4.export
29: class _BackendQT4Agg(_BackendQT4):
30:     FigureCanvas = FigureCanvasQTAgg
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_249681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nRender to qt from agg\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_249682) is not StypyTypeError):

    if (import_249682 != 'pyd_module'):
        __import__(import_249682)
        sys_modules_249683 = sys.modules[import_249682]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_249683.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_249682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.backends.backend_agg import FigureCanvasAgg' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg')

if (type(import_249684) is not StypyTypeError):

    if (import_249684 != 'pyd_module'):
        __import__(import_249684)
        sys_modules_249685 = sys.modules[import_249684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg', sys_modules_249685.module_type_store, module_type_store, ['FigureCanvasAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_249685, sys_modules_249685.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg', None, module_type_store, ['FigureCanvasAgg'], [FigureCanvasAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg', import_249684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.backends.backend_qt4 import QtCore, _BackendQT4, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_qt4')

if (type(import_249686) is not StypyTypeError):

    if (import_249686 != 'pyd_module'):
        __import__(import_249686)
        sys_modules_249687 = sys.modules[import_249686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_qt4', sys_modules_249687.module_type_store, module_type_store, ['QtCore', '_BackendQT4', 'FigureCanvasQT', 'FigureManagerQT', 'NavigationToolbar2QT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_249687, sys_modules_249687.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_qt4 import QtCore, _BackendQT4, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_qt4', None, module_type_store, ['QtCore', '_BackendQT4', 'FigureCanvasQT', 'FigureManagerQT', 'NavigationToolbar2QT'], [QtCore, _BackendQT4, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_qt4' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_qt4', import_249686)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.backends.backend_qt5agg import FigureCanvasQTAggBase' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends.backend_qt5agg')

if (type(import_249688) is not StypyTypeError):

    if (import_249688 != 'pyd_module'):
        __import__(import_249688)
        sys_modules_249689 = sys.modules[import_249688]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends.backend_qt5agg', sys_modules_249689.module_type_store, module_type_store, ['FigureCanvasQTAggBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_249689, sys_modules_249689.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAggBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends.backend_qt5agg', None, module_type_store, ['FigureCanvasQTAggBase'], [FigureCanvasQTAggBase])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_qt5agg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends.backend_qt5agg', import_249688)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'FigureCanvasQTAgg' class
# Getting the type of 'FigureCanvasQTAggBase' (line 15)
FigureCanvasQTAggBase_249690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'FigureCanvasQTAggBase')
# Getting the type of 'FigureCanvasQT' (line 15)
FigureCanvasQT_249691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 47), 'FigureCanvasQT')

class FigureCanvasQTAgg(FigureCanvasQTAggBase_249690, FigureCanvasQT_249691, ):
    unicode_249692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'unicode', u'\n    The canvas the figure renders into.  Calls the draw and print fig\n    methods, creates the renderers, etc...\n\n    Attributes\n    ----------\n    figure : `matplotlib.figure.Figure`\n        A high-level Figure instance\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAgg.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'FigureCanvasQTAgg' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'FigureCanvasQTAgg', FigureCanvasQTAgg)
# Declaration of the '_BackendQT4Agg' class
# Getting the type of '_BackendQT4' (line 29)
_BackendQT4_249693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), '_BackendQT4')

class _BackendQT4Agg(_BackendQT4_249693, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 0, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendQT4Agg.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_BackendQT4Agg' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '_BackendQT4Agg', _BackendQT4Agg)

# Assigning a Name to a Name (line 30):
# Getting the type of 'FigureCanvasQTAgg' (line 30)
FigureCanvasQTAgg_249694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'FigureCanvasQTAgg')
# Getting the type of '_BackendQT4Agg'
_BackendQT4Agg_249695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendQT4Agg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendQT4Agg_249695, 'FigureCanvas', FigureCanvasQTAgg_249694)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

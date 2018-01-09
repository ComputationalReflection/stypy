
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: from six import unichr
6: import os
7: import re
8: import signal
9: import sys
10: 
11: from matplotlib._pylab_helpers import Gcf
12: from matplotlib.backend_bases import (
13:     FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase,
14:     cursors)
15: from matplotlib.figure import Figure
16: from matplotlib.widgets import SubplotTool
17: 
18: from .qt_compat import QtCore, QtWidgets, _getSaveFileName, __version__
19: 
20: from .backend_qt5 import (
21:     backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS,
22:     cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureManagerQT,
23:     NavigationToolbar2QT, SubplotToolQt, error_msg_qt, exception_handler)
24: from .backend_qt5 import FigureCanvasQT as FigureCanvasQT5
25: 
26: DEBUG = False
27: 
28: 
29: class FigureCanvasQT(FigureCanvasQT5):
30: 
31:     def wheelEvent(self, event):
32:         x = event.x()
33:         # flipy so y=0 is bottom of canvas
34:         y = self.figure.bbox.height - event.y()
35:         # from QWheelEvent::delta doc
36:         steps = event.delta()/120
37:         if (event.orientation() == QtCore.Qt.Vertical):
38:             FigureCanvasBase.scroll_event(self, x, y, steps)
39:             if DEBUG:
40:                 print('scroll event: delta = %i, '
41:                       'steps = %i ' % (event.delta(), steps))
42: 
43: 
44: @_BackendQT5.export
45: class _BackendQT4(_BackendQT5):
46:     FigureCanvas = FigureCanvasQT
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249608 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_249608) is not StypyTypeError):

    if (import_249608 != 'pyd_module'):
        __import__(import_249608)
        sys_modules_249609 = sys.modules[import_249608]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_249609.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_249608)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from six import unichr' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249610 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six')

if (type(import_249610) is not StypyTypeError):

    if (import_249610 != 'pyd_module'):
        __import__(import_249610)
        sys_modules_249611 = sys.modules[import_249610]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six', sys_modules_249611.module_type_store, module_type_store, ['unichr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_249611, sys_modules_249611.module_type_store, module_type_store)
    else:
        from six import unichr

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six', None, module_type_store, ['unichr'], [unichr])

else:
    # Assigning a type to the variable 'six' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'six', import_249610)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import re' statement (line 7)
import re

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import signal' statement (line 8)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'signal', signal, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249612 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib._pylab_helpers')

if (type(import_249612) is not StypyTypeError):

    if (import_249612 != 'pyd_module'):
        __import__(import_249612)
        sys_modules_249613 = sys.modules[import_249612]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib._pylab_helpers', sys_modules_249613.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_249613, sys_modules_249613.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib._pylab_helpers', import_249612)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.backend_bases import FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249614 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases')

if (type(import_249614) is not StypyTypeError):

    if (import_249614 != 'pyd_module'):
        __import__(import_249614)
        sys_modules_249615 = sys.modules[import_249614]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases', sys_modules_249615.module_type_store, module_type_store, ['FigureCanvasBase', 'FigureManagerBase', 'NavigationToolbar2', 'TimerBase', 'cursors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_249615, sys_modules_249615.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases', None, module_type_store, ['FigureCanvasBase', 'FigureManagerBase', 'NavigationToolbar2', 'TimerBase', 'cursors'], [FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases', import_249614)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.figure import Figure' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249616 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.figure')

if (type(import_249616) is not StypyTypeError):

    if (import_249616 != 'pyd_module'):
        __import__(import_249616)
        sys_modules_249617 = sys.modules[import_249616]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.figure', sys_modules_249617.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_249617, sys_modules_249617.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.figure', import_249616)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.widgets import SubplotTool' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249618 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets')

if (type(import_249618) is not StypyTypeError):

    if (import_249618 != 'pyd_module'):
        __import__(import_249618)
        sys_modules_249619 = sys.modules[import_249618]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets', sys_modules_249619.module_type_store, module_type_store, ['SubplotTool'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_249619, sys_modules_249619.module_type_store, module_type_store)
    else:
        from matplotlib.widgets import SubplotTool

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets', None, module_type_store, ['SubplotTool'], [SubplotTool])

else:
    # Assigning a type to the variable 'matplotlib.widgets' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets', import_249618)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from matplotlib.backends.qt_compat import QtCore, QtWidgets, _getSaveFileName, __version__' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249620 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.backends.qt_compat')

if (type(import_249620) is not StypyTypeError):

    if (import_249620 != 'pyd_module'):
        __import__(import_249620)
        sys_modules_249621 = sys.modules[import_249620]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.backends.qt_compat', sys_modules_249621.module_type_store, module_type_store, ['QtCore', 'QtWidgets', '_getSaveFileName', '__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_249621, sys_modules_249621.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_compat import QtCore, QtWidgets, _getSaveFileName, __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.backends.qt_compat', None, module_type_store, ['QtCore', 'QtWidgets', '_getSaveFileName', '__version__'], [QtCore, QtWidgets, _getSaveFileName, __version__])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_compat' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.backends.qt_compat', import_249620)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.backends.backend_qt5 import backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS, cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureManagerQT, NavigationToolbar2QT, SubplotToolQt, error_msg_qt, exception_handler' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249622 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.backends.backend_qt5')

if (type(import_249622) is not StypyTypeError):

    if (import_249622 != 'pyd_module'):
        __import__(import_249622)
        sys_modules_249623 = sys.modules[import_249622]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.backends.backend_qt5', sys_modules_249623.module_type_store, module_type_store, ['backend_version', 'SPECIAL_KEYS', 'SUPER', 'ALT', 'CTRL', 'SHIFT', 'MODIFIER_KEYS', 'cursord', '_create_qApp', '_BackendQT5', 'TimerQT', 'MainWindow', 'FigureManagerQT', 'NavigationToolbar2QT', 'SubplotToolQt', 'error_msg_qt', 'exception_handler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_249623, sys_modules_249623.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_qt5 import backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS, cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureManagerQT, NavigationToolbar2QT, SubplotToolQt, error_msg_qt, exception_handler

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.backends.backend_qt5', None, module_type_store, ['backend_version', 'SPECIAL_KEYS', 'SUPER', 'ALT', 'CTRL', 'SHIFT', 'MODIFIER_KEYS', 'cursord', '_create_qApp', '_BackendQT5', 'TimerQT', 'MainWindow', 'FigureManagerQT', 'NavigationToolbar2QT', 'SubplotToolQt', 'error_msg_qt', 'exception_handler'], [backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS, cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureManagerQT, NavigationToolbar2QT, SubplotToolQt, error_msg_qt, exception_handler])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_qt5' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.backends.backend_qt5', import_249622)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib.backends.backend_qt5 import FigureCanvasQT5' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249624 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.backends.backend_qt5')

if (type(import_249624) is not StypyTypeError):

    if (import_249624 != 'pyd_module'):
        __import__(import_249624)
        sys_modules_249625 = sys.modules[import_249624]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.backends.backend_qt5', sys_modules_249625.module_type_store, module_type_store, ['FigureCanvasQT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_249625, sys_modules_249625.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_qt5 import FigureCanvasQT as FigureCanvasQT5

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.backends.backend_qt5', None, module_type_store, ['FigureCanvasQT'], [FigureCanvasQT5])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_qt5' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.backends.backend_qt5', import_249624)

# Adding an alias
module_type_store.add_alias('FigureCanvasQT5', 'FigureCanvasQT')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Name to a Name (line 26):
# Getting the type of 'False' (line 26)
False_249626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'False')
# Assigning a type to the variable 'DEBUG' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'DEBUG', False_249626)
# Declaration of the 'FigureCanvasQT' class
# Getting the type of 'FigureCanvasQT5' (line 29)
FigureCanvasQT5_249627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'FigureCanvasQT5')

class FigureCanvasQT(FigureCanvasQT5_249627, ):

    @norecursion
    def wheelEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wheelEvent'
        module_type_store = module_type_store.open_function_context('wheelEvent', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.wheelEvent')
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.wheelEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wheelEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wheelEvent(...)' code ##################

        
        # Assigning a Call to a Name (line 32):
        
        # Call to x(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_249630 = {}
        # Getting the type of 'event' (line 32)
        event_249628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'event', False)
        # Obtaining the member 'x' of a type (line 32)
        x_249629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), event_249628, 'x')
        # Calling x(args, kwargs) (line 32)
        x_call_result_249631 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), x_249629, *[], **kwargs_249630)
        
        # Assigning a type to the variable 'x' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x', x_call_result_249631)
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'self' (line 34)
        self_249632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
        # Obtaining the member 'figure' of a type (line 34)
        figure_249633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_249632, 'figure')
        # Obtaining the member 'bbox' of a type (line 34)
        bbox_249634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), figure_249633, 'bbox')
        # Obtaining the member 'height' of a type (line 34)
        height_249635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), bbox_249634, 'height')
        
        # Call to y(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_249638 = {}
        # Getting the type of 'event' (line 34)
        event_249636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'event', False)
        # Obtaining the member 'y' of a type (line 34)
        y_249637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), event_249636, 'y')
        # Calling y(args, kwargs) (line 34)
        y_call_result_249639 = invoke(stypy.reporting.localization.Localization(__file__, 34, 38), y_249637, *[], **kwargs_249638)
        
        # Applying the binary operator '-' (line 34)
        result_sub_249640 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '-', height_249635, y_call_result_249639)
        
        # Assigning a type to the variable 'y' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'y', result_sub_249640)
        
        # Assigning a BinOp to a Name (line 36):
        
        # Call to delta(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_249643 = {}
        # Getting the type of 'event' (line 36)
        event_249641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'event', False)
        # Obtaining the member 'delta' of a type (line 36)
        delta_249642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), event_249641, 'delta')
        # Calling delta(args, kwargs) (line 36)
        delta_call_result_249644 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), delta_249642, *[], **kwargs_249643)
        
        int_249645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
        # Applying the binary operator 'div' (line 36)
        result_div_249646 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 16), 'div', delta_call_result_249644, int_249645)
        
        # Assigning a type to the variable 'steps' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'steps', result_div_249646)
        
        
        
        # Call to orientation(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_249649 = {}
        # Getting the type of 'event' (line 37)
        event_249647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'event', False)
        # Obtaining the member 'orientation' of a type (line 37)
        orientation_249648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), event_249647, 'orientation')
        # Calling orientation(args, kwargs) (line 37)
        orientation_call_result_249650 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), orientation_249648, *[], **kwargs_249649)
        
        # Getting the type of 'QtCore' (line 37)
        QtCore_249651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'QtCore')
        # Obtaining the member 'Qt' of a type (line 37)
        Qt_249652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 35), QtCore_249651, 'Qt')
        # Obtaining the member 'Vertical' of a type (line 37)
        Vertical_249653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 35), Qt_249652, 'Vertical')
        # Applying the binary operator '==' (line 37)
        result_eq_249654 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '==', orientation_call_result_249650, Vertical_249653)
        
        # Testing the type of an if condition (line 37)
        if_condition_249655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 8), result_eq_249654)
        # Assigning a type to the variable 'if_condition_249655' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'if_condition_249655', if_condition_249655)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to scroll_event(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_249658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'self', False)
        # Getting the type of 'x' (line 38)
        x_249659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 48), 'x', False)
        # Getting the type of 'y' (line 38)
        y_249660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 51), 'y', False)
        # Getting the type of 'steps' (line 38)
        steps_249661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 54), 'steps', False)
        # Processing the call keyword arguments (line 38)
        kwargs_249662 = {}
        # Getting the type of 'FigureCanvasBase' (line 38)
        FigureCanvasBase_249656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'FigureCanvasBase', False)
        # Obtaining the member 'scroll_event' of a type (line 38)
        scroll_event_249657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), FigureCanvasBase_249656, 'scroll_event')
        # Calling scroll_event(args, kwargs) (line 38)
        scroll_event_call_result_249663 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), scroll_event_249657, *[self_249658, x_249659, y_249660, steps_249661], **kwargs_249662)
        
        
        # Getting the type of 'DEBUG' (line 39)
        DEBUG_249664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'DEBUG')
        # Testing the type of an if condition (line 39)
        if_condition_249665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 12), DEBUG_249664)
        # Assigning a type to the variable 'if_condition_249665' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'if_condition_249665', if_condition_249665)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 40)
        # Processing the call arguments (line 40)
        unicode_249667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'unicode', u'scroll event: delta = %i, steps = %i ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_249668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        
        # Call to delta(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_249671 = {}
        # Getting the type of 'event' (line 41)
        event_249669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'event', False)
        # Obtaining the member 'delta' of a type (line 41)
        delta_249670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 39), event_249669, 'delta')
        # Calling delta(args, kwargs) (line 41)
        delta_call_result_249672 = invoke(stypy.reporting.localization.Localization(__file__, 41, 39), delta_249670, *[], **kwargs_249671)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), tuple_249668, delta_call_result_249672)
        # Adding element type (line 41)
        # Getting the type of 'steps' (line 41)
        steps_249673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 54), 'steps', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), tuple_249668, steps_249673)
        
        # Applying the binary operator '%' (line 40)
        result_mod_249674 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 22), '%', unicode_249667, tuple_249668)
        
        # Processing the call keyword arguments (line 40)
        kwargs_249675 = {}
        # Getting the type of 'print' (line 40)
        print_249666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'print', False)
        # Calling print(args, kwargs) (line 40)
        print_call_result_249676 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), print_249666, *[result_mod_249674], **kwargs_249675)
        
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'wheelEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wheelEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_249677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249677)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wheelEvent'
        return stypy_return_type_249677


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasQT' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'FigureCanvasQT', FigureCanvasQT)
# Declaration of the '_BackendQT4' class
# Getting the type of '_BackendQT5' (line 45)
_BackendQT5_249678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), '_BackendQT5')

class _BackendQT4(_BackendQT5_249678, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 0, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendQT4.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendQT4' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), '_BackendQT4', _BackendQT4)

# Assigning a Name to a Name (line 46):
# Getting the type of 'FigureCanvasQT' (line 46)
FigureCanvasQT_249679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'FigureCanvasQT')
# Getting the type of '_BackendQT4'
_BackendQT4_249680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendQT4')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendQT4_249680, 'FigureCanvas', FigureCanvasQT_249679)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

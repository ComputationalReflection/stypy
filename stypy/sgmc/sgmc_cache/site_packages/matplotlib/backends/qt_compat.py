
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' A Qt API selector that can be used to switch between PyQt and PySide.
2: '''
3: from __future__ import (absolute_import, division, print_function,
4:                         unicode_literals)
5: 
6: import six
7: 
8: import os
9: import sys
10: from matplotlib import rcParams, verbose
11: 
12: # Available APIs.
13: QT_API_PYQT = 'PyQt4'       # API is not set here; Python 2.x default is V 1
14: QT_API_PYQTv2 = 'PyQt4v2'   # forced to Version 2 API
15: QT_API_PYSIDE = 'PySide'    # only supports Version 2 API
16: QT_API_PYQT5 = 'PyQt5'      # use PyQt5 API; Version 2 with module shim
17: QT_API_PYSIDE2 = 'PySide2'  # Version 2 API with module shim
18: 
19: ETS = dict(pyqt=(QT_API_PYQTv2, 4), pyside=(QT_API_PYSIDE, 4),
20:            pyqt5=(QT_API_PYQT5, 5), pyside2=(QT_API_PYSIDE2, 5))
21: # ETS is a dict of env variable to (QT_API, QT_MAJOR_VERSION)
22: # If the ETS QT_API environment variable is set, use it, but only
23: # if the varible if of the same major QT version.  Note that
24: # ETS requires the version 2 of PyQt4, which is not the platform
25: # default for Python 2.x.
26: 
27: QT_API_ENV = os.environ.get('QT_API')
28: 
29: if rcParams['backend'] == 'Qt5Agg':
30:     QT_RC_MAJOR_VERSION = 5
31: elif rcParams['backend'] == 'Qt4Agg':
32:     QT_RC_MAJOR_VERSION = 4
33: else:
34:     # A different backend was specified, but we still got here because a Qt
35:     # related file was imported. This is allowed, so lets try and guess
36:     # what we should be using.
37:     if "PyQt4" in sys.modules or "PySide" in sys.modules:
38:         # PyQt4 or PySide is actually used.
39:         QT_RC_MAJOR_VERSION = 4
40:     else:
41:         # This is a fallback: PyQt5
42:         QT_RC_MAJOR_VERSION = 5
43: 
44: QT_API = None
45: 
46: # check if any binding is already imported, if so silently ignore the
47: # rcparams/ENV settings and use what ever is already imported.
48: if 'PySide' in sys.modules:
49:     # user has imported PySide before importing mpl
50:     QT_API = QT_API_PYSIDE
51: 
52: if 'PySide2' in sys.modules:
53:     # user has imported PySide before importing mpl
54:     QT_API = QT_API_PYSIDE2
55: 
56: if 'PyQt4' in sys.modules:
57:     # user has imported PyQt4 before importing mpl
58:     # this case also handles the PyQt4v2 case as once sip is imported
59:     # the API versions can not be changed so do not try
60:     QT_API = QT_API_PYQT
61: 
62: if 'PyQt5' in sys.modules:
63:     # the user has imported PyQt5 before importing mpl
64:     QT_API = QT_API_PYQT5
65: 
66: if (QT_API_ENV is not None) and QT_API is None:
67:     try:
68:         QT_ENV_MAJOR_VERSION = ETS[QT_API_ENV][1]
69:     except KeyError:
70:         raise RuntimeError(
71:             ('Unrecognized environment variable %r, valid values are:'
72:              ' %r, %r, %r or %r'
73:              % (QT_API_ENV, 'pyqt', 'pyside', 'pyqt5', 'pyside2')))
74:     if QT_ENV_MAJOR_VERSION == QT_RC_MAJOR_VERSION:
75:         # Only if backend and env qt major version are
76:         # compatible use the env variable.
77:         QT_API = ETS[QT_API_ENV][0]
78: 
79: _fallback_to_qt4 = False
80: if QT_API is None:
81:     # No ETS environment or incompatible so use rcParams.
82:     if rcParams['backend'] == 'Qt5Agg':
83:         QT_API = rcParams['backend.qt5']
84:     elif rcParams['backend'] == 'Qt4Agg':
85:         QT_API = rcParams['backend.qt4']
86:     else:
87:         # A non-Qt backend was specified, no version of the Qt
88:         # bindings is imported, but we still got here because a Qt
89:         # related file was imported. This is allowed, fall back to Qt5
90:         # using which ever binding the rparams ask for.
91:         _fallback_to_qt4 = True
92:         QT_API = rcParams['backend.qt5']
93: 
94: # We will define an appropriate wrapper for the differing versions
95: # of file dialog.
96: _getSaveFileName = None
97: 
98: # Flag to check if sip could be imported
99: _sip_imported = False
100: 
101: # Now perform the imports.
102: if QT_API in (QT_API_PYQT, QT_API_PYQTv2, QT_API_PYQT5):
103:     try:
104:         import sip
105:         _sip_imported = True
106:     except ImportError:
107:         # Try using PySide
108:         if QT_RC_MAJOR_VERSION == 5:
109:             QT_API = QT_API_PYSIDE2
110:         else:
111:             QT_API = QT_API_PYSIDE
112:         cond = ("Could not import sip; falling back on PySide\n"
113:                 "in place of PyQt4 or PyQt5.\n")
114:         verbose.report(cond, 'helpful')
115: 
116: if _sip_imported:
117:     if QT_API == QT_API_PYQTv2:
118:         if QT_API_ENV == 'pyqt':
119:             cond = ("Found 'QT_API=pyqt' environment variable. "
120:                     "Setting PyQt4 API accordingly.\n")
121:         else:
122:             cond = "PyQt API v2 specified."
123:         try:
124:             sip.setapi('QString', 2)
125:         except:
126:             res = 'QString API v2 specification failed. Defaulting to v1.'
127:             verbose.report(cond + res, 'helpful')
128:             # condition has now been reported, no need to repeat it:
129:             cond = ""
130:         try:
131:             sip.setapi('QVariant', 2)
132:         except:
133:             res = 'QVariant API v2 specification failed. Defaulting to v1.'
134:             verbose.report(cond + res, 'helpful')
135:     if QT_API == QT_API_PYQT5:
136:         try:
137:             from PyQt5 import QtCore, QtGui, QtWidgets
138:             _getSaveFileName = QtWidgets.QFileDialog.getSaveFileName
139:         except ImportError:
140:             if _fallback_to_qt4:
141:                 # fell through, tried PyQt5, failed fall back to PyQt4
142:                 QT_API = rcParams['backend.qt4']
143:                 QT_RC_MAJOR_VERSION = 4
144:             else:
145:                 raise
146: 
147:     # needs to be if so we can re-test the value of QT_API which may
148:     # have been changed in the above if block
149:     if QT_API in [QT_API_PYQT, QT_API_PYQTv2]:  # PyQt4 API
150:         from PyQt4 import QtCore, QtGui
151: 
152:         try:
153:             if sip.getapi("QString") > 1:
154:                 # Use new getSaveFileNameAndFilter()
155:                 _getSaveFileName = QtGui.QFileDialog.getSaveFileNameAndFilter
156:             else:
157: 
158:                 # Use old getSaveFileName()
159:                 def _getSaveFileName(*args, **kwargs):
160:                     return (QtGui.QFileDialog.getSaveFileName(*args, **kwargs),
161:                             None)
162: 
163:         except (AttributeError, KeyError):
164: 
165:             # call to getapi() can fail in older versions of sip
166:             def _getSaveFileName(*args, **kwargs):
167:                 return QtGui.QFileDialog.getSaveFileName(*args, **kwargs), None
168:     try:
169:         # Alias PyQt-specific functions for PySide compatibility.
170:         QtCore.Signal = QtCore.pyqtSignal
171:         try:
172:             QtCore.Slot = QtCore.pyqtSlot
173:         except AttributeError:
174:             # Not a perfect match but works in simple cases
175:             QtCore.Slot = QtCore.pyqtSignature
176: 
177:         QtCore.Property = QtCore.pyqtProperty
178:         __version__ = QtCore.PYQT_VERSION_STR
179:     except NameError:
180:         # QtCore did not get imported, fall back to pyside
181:         if QT_RC_MAJOR_VERSION == 5:
182:             QT_API = QT_API_PYSIDE2
183:         else:
184:             QT_API = QT_API_PYSIDE
185: 
186: 
187: if QT_API == QT_API_PYSIDE2:
188:     try:
189:         from PySide2 import QtCore, QtGui, QtWidgets, __version__
190:         _getSaveFileName = QtWidgets.QFileDialog.getSaveFileName
191:     except ImportError:
192:         # tried PySide2, failed, fall back to PySide
193:         QT_RC_MAJOR_VERSION = 4
194:         QT_API = QT_API_PYSIDE
195: 
196: if QT_API == QT_API_PYSIDE:  # try importing pyside
197:     try:
198:         from PySide import QtCore, QtGui, __version__, __version_info__
199:     except ImportError:
200:         raise ImportError(
201:             "Matplotlib qt-based backends require an external PyQt4, PyQt5,\n"
202:             "PySide or PySide2 package to be installed, but it was not found.")
203: 
204:     if __version_info__ < (1, 0, 3):
205:         raise ImportError(
206:             "Matplotlib backend_qt4 and backend_qt4agg require PySide >=1.0.3")
207: 
208:     _getSaveFileName = QtGui.QFileDialog.getSaveFileName
209: 
210: 
211: # Apply shim to Qt4 APIs to make them look like Qt5
212: if QT_API in (QT_API_PYQT, QT_API_PYQTv2, QT_API_PYSIDE):
213:     '''Import all used QtGui objects into QtWidgets
214: 
215:     Here I've opted to simple copy QtGui into QtWidgets as that
216:     achieves the same result as copying over the objects, and will
217:     continue to work if other objects are used.
218: 
219:     '''
220:     QtWidgets = QtGui
221: 
222: 
223: def is_pyqt5():
224:     return QT_API == QT_API_PYQT5
225: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_268989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'unicode', u' A Qt API selector that can be used to switch between PyQt and PySide.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import six' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268990 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six')

if (type(import_268990) is not StypyTypeError):

    if (import_268990 != 'pyd_module'):
        __import__(import_268990)
        sys_modules_268991 = sys.modules[import_268990]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', sys_modules_268991.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', import_268990)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib import rcParams, verbose' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268992 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib')

if (type(import_268992) is not StypyTypeError):

    if (import_268992 != 'pyd_module'):
        __import__(import_268992)
        sys_modules_268993 = sys.modules[import_268992]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', sys_modules_268993.module_type_store, module_type_store, ['rcParams', 'verbose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_268993, sys_modules_268993.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams, verbose

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', None, module_type_store, ['rcParams', 'verbose'], [rcParams, verbose])

else:
    # Assigning a type to the variable 'matplotlib' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', import_268992)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Str to a Name (line 13):
unicode_268994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'unicode', u'PyQt4')
# Assigning a type to the variable 'QT_API_PYQT' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'QT_API_PYQT', unicode_268994)

# Assigning a Str to a Name (line 14):
unicode_268995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'unicode', u'PyQt4v2')
# Assigning a type to the variable 'QT_API_PYQTv2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'QT_API_PYQTv2', unicode_268995)

# Assigning a Str to a Name (line 15):
unicode_268996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'unicode', u'PySide')
# Assigning a type to the variable 'QT_API_PYSIDE' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'QT_API_PYSIDE', unicode_268996)

# Assigning a Str to a Name (line 16):
unicode_268997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'unicode', u'PyQt5')
# Assigning a type to the variable 'QT_API_PYQT5' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'QT_API_PYQT5', unicode_268997)

# Assigning a Str to a Name (line 17):
unicode_268998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'unicode', u'PySide2')
# Assigning a type to the variable 'QT_API_PYSIDE2' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'QT_API_PYSIDE2', unicode_268998)

# Assigning a Call to a Name (line 19):

# Call to dict(...): (line 19)
# Processing the call keyword arguments (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_269000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
# Getting the type of 'QT_API_PYQTv2' (line 19)
QT_API_PYQTv2_269001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'QT_API_PYQTv2', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), tuple_269000, QT_API_PYQTv2_269001)
# Adding element type (line 19)
int_269002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), tuple_269000, int_269002)

keyword_269003 = tuple_269000

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_269004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 44), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
# Getting the type of 'QT_API_PYSIDE' (line 19)
QT_API_PYSIDE_269005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 44), 'QT_API_PYSIDE', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 44), tuple_269004, QT_API_PYSIDE_269005)
# Adding element type (line 19)
int_269006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 44), tuple_269004, int_269006)

keyword_269007 = tuple_269004

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_269008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
# Getting the type of 'QT_API_PYQT5' (line 20)
QT_API_PYQT5_269009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'QT_API_PYQT5', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), tuple_269008, QT_API_PYQT5_269009)
# Adding element type (line 20)
int_269010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), tuple_269008, int_269010)

keyword_269011 = tuple_269008

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_269012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 45), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
# Getting the type of 'QT_API_PYSIDE2' (line 20)
QT_API_PYSIDE2_269013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 45), 'QT_API_PYSIDE2', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 45), tuple_269012, QT_API_PYSIDE2_269013)
# Adding element type (line 20)
int_269014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 45), tuple_269012, int_269014)

keyword_269015 = tuple_269012
kwargs_269016 = {'pyqt5': keyword_269011, 'pyside': keyword_269007, 'pyqt': keyword_269003, 'pyside2': keyword_269015}
# Getting the type of 'dict' (line 19)
dict_268999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 6), 'dict', False)
# Calling dict(args, kwargs) (line 19)
dict_call_result_269017 = invoke(stypy.reporting.localization.Localization(__file__, 19, 6), dict_268999, *[], **kwargs_269016)

# Assigning a type to the variable 'ETS' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'ETS', dict_call_result_269017)

# Assigning a Call to a Name (line 27):

# Call to get(...): (line 27)
# Processing the call arguments (line 27)
unicode_269021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'unicode', u'QT_API')
# Processing the call keyword arguments (line 27)
kwargs_269022 = {}
# Getting the type of 'os' (line 27)
os_269018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'os', False)
# Obtaining the member 'environ' of a type (line 27)
environ_269019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 13), os_269018, 'environ')
# Obtaining the member 'get' of a type (line 27)
get_269020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 13), environ_269019, 'get')
# Calling get(args, kwargs) (line 27)
get_call_result_269023 = invoke(stypy.reporting.localization.Localization(__file__, 27, 13), get_269020, *[unicode_269021], **kwargs_269022)

# Assigning a type to the variable 'QT_API_ENV' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'QT_API_ENV', get_call_result_269023)



# Obtaining the type of the subscript
unicode_269024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'unicode', u'backend')
# Getting the type of 'rcParams' (line 29)
rcParams_269025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 3), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 29)
getitem___269026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 3), rcParams_269025, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 29)
subscript_call_result_269027 = invoke(stypy.reporting.localization.Localization(__file__, 29, 3), getitem___269026, unicode_269024)

unicode_269028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'unicode', u'Qt5Agg')
# Applying the binary operator '==' (line 29)
result_eq_269029 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 3), '==', subscript_call_result_269027, unicode_269028)

# Testing the type of an if condition (line 29)
if_condition_269030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 0), result_eq_269029)
# Assigning a type to the variable 'if_condition_269030' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'if_condition_269030', if_condition_269030)
# SSA begins for if statement (line 29)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 30):
int_269031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'int')
# Assigning a type to the variable 'QT_RC_MAJOR_VERSION' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'QT_RC_MAJOR_VERSION', int_269031)
# SSA branch for the else part of an if statement (line 29)
module_type_store.open_ssa_branch('else')



# Obtaining the type of the subscript
unicode_269032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'unicode', u'backend')
# Getting the type of 'rcParams' (line 31)
rcParams_269033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 5), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 31)
getitem___269034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 5), rcParams_269033, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 31)
subscript_call_result_269035 = invoke(stypy.reporting.localization.Localization(__file__, 31, 5), getitem___269034, unicode_269032)

unicode_269036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'unicode', u'Qt4Agg')
# Applying the binary operator '==' (line 31)
result_eq_269037 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 5), '==', subscript_call_result_269035, unicode_269036)

# Testing the type of an if condition (line 31)
if_condition_269038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 5), result_eq_269037)
# Assigning a type to the variable 'if_condition_269038' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 5), 'if_condition_269038', if_condition_269038)
# SSA begins for if statement (line 31)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 32):
int_269039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'int')
# Assigning a type to the variable 'QT_RC_MAJOR_VERSION' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'QT_RC_MAJOR_VERSION', int_269039)
# SSA branch for the else part of an if statement (line 31)
module_type_store.open_ssa_branch('else')


# Evaluating a boolean operation

unicode_269040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 7), 'unicode', u'PyQt4')
# Getting the type of 'sys' (line 37)
sys_269041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'sys')
# Obtaining the member 'modules' of a type (line 37)
modules_269042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 18), sys_269041, 'modules')
# Applying the binary operator 'in' (line 37)
result_contains_269043 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), 'in', unicode_269040, modules_269042)


unicode_269044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'unicode', u'PySide')
# Getting the type of 'sys' (line 37)
sys_269045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 45), 'sys')
# Obtaining the member 'modules' of a type (line 37)
modules_269046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 45), sys_269045, 'modules')
# Applying the binary operator 'in' (line 37)
result_contains_269047 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 33), 'in', unicode_269044, modules_269046)

# Applying the binary operator 'or' (line 37)
result_or_keyword_269048 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), 'or', result_contains_269043, result_contains_269047)

# Testing the type of an if condition (line 37)
if_condition_269049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_or_keyword_269048)
# Assigning a type to the variable 'if_condition_269049' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_269049', if_condition_269049)
# SSA begins for if statement (line 37)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 39):
int_269050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
# Assigning a type to the variable 'QT_RC_MAJOR_VERSION' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'QT_RC_MAJOR_VERSION', int_269050)
# SSA branch for the else part of an if statement (line 37)
module_type_store.open_ssa_branch('else')

# Assigning a Num to a Name (line 42):
int_269051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'int')
# Assigning a type to the variable 'QT_RC_MAJOR_VERSION' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'QT_RC_MAJOR_VERSION', int_269051)
# SSA join for if statement (line 37)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 31)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 29)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 44):
# Getting the type of 'None' (line 44)
None_269052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'None')
# Assigning a type to the variable 'QT_API' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'QT_API', None_269052)


unicode_269053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 3), 'unicode', u'PySide')
# Getting the type of 'sys' (line 48)
sys_269054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'sys')
# Obtaining the member 'modules' of a type (line 48)
modules_269055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), sys_269054, 'modules')
# Applying the binary operator 'in' (line 48)
result_contains_269056 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 3), 'in', unicode_269053, modules_269055)

# Testing the type of an if condition (line 48)
if_condition_269057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 0), result_contains_269056)
# Assigning a type to the variable 'if_condition_269057' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'if_condition_269057', if_condition_269057)
# SSA begins for if statement (line 48)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 50):
# Getting the type of 'QT_API_PYSIDE' (line 50)
QT_API_PYSIDE_269058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'QT_API_PYSIDE')
# Assigning a type to the variable 'QT_API' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'QT_API', QT_API_PYSIDE_269058)
# SSA join for if statement (line 48)
module_type_store = module_type_store.join_ssa_context()



unicode_269059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 3), 'unicode', u'PySide2')
# Getting the type of 'sys' (line 52)
sys_269060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'sys')
# Obtaining the member 'modules' of a type (line 52)
modules_269061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), sys_269060, 'modules')
# Applying the binary operator 'in' (line 52)
result_contains_269062 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 3), 'in', unicode_269059, modules_269061)

# Testing the type of an if condition (line 52)
if_condition_269063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 0), result_contains_269062)
# Assigning a type to the variable 'if_condition_269063' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'if_condition_269063', if_condition_269063)
# SSA begins for if statement (line 52)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 54):
# Getting the type of 'QT_API_PYSIDE2' (line 54)
QT_API_PYSIDE2_269064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'QT_API_PYSIDE2')
# Assigning a type to the variable 'QT_API' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'QT_API', QT_API_PYSIDE2_269064)
# SSA join for if statement (line 52)
module_type_store = module_type_store.join_ssa_context()



unicode_269065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 3), 'unicode', u'PyQt4')
# Getting the type of 'sys' (line 56)
sys_269066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'sys')
# Obtaining the member 'modules' of a type (line 56)
modules_269067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 14), sys_269066, 'modules')
# Applying the binary operator 'in' (line 56)
result_contains_269068 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 3), 'in', unicode_269065, modules_269067)

# Testing the type of an if condition (line 56)
if_condition_269069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 0), result_contains_269068)
# Assigning a type to the variable 'if_condition_269069' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'if_condition_269069', if_condition_269069)
# SSA begins for if statement (line 56)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 60):
# Getting the type of 'QT_API_PYQT' (line 60)
QT_API_PYQT_269070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'QT_API_PYQT')
# Assigning a type to the variable 'QT_API' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'QT_API', QT_API_PYQT_269070)
# SSA join for if statement (line 56)
module_type_store = module_type_store.join_ssa_context()



unicode_269071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 3), 'unicode', u'PyQt5')
# Getting the type of 'sys' (line 62)
sys_269072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 14), 'sys')
# Obtaining the member 'modules' of a type (line 62)
modules_269073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 14), sys_269072, 'modules')
# Applying the binary operator 'in' (line 62)
result_contains_269074 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 3), 'in', unicode_269071, modules_269073)

# Testing the type of an if condition (line 62)
if_condition_269075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 0), result_contains_269074)
# Assigning a type to the variable 'if_condition_269075' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'if_condition_269075', if_condition_269075)
# SSA begins for if statement (line 62)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 64):
# Getting the type of 'QT_API_PYQT5' (line 64)
QT_API_PYQT5_269076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'QT_API_PYQT5')
# Assigning a type to the variable 'QT_API' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'QT_API', QT_API_PYQT5_269076)
# SSA join for if statement (line 62)
module_type_store = module_type_store.join_ssa_context()



# Evaluating a boolean operation

# Getting the type of 'QT_API_ENV' (line 66)
QT_API_ENV_269077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'QT_API_ENV')
# Getting the type of 'None' (line 66)
None_269078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'None')
# Applying the binary operator 'isnot' (line 66)
result_is_not_269079 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 4), 'isnot', QT_API_ENV_269077, None_269078)


# Getting the type of 'QT_API' (line 66)
QT_API_269080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'QT_API')
# Getting the type of 'None' (line 66)
None_269081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'None')
# Applying the binary operator 'is' (line 66)
result_is__269082 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 32), 'is', QT_API_269080, None_269081)

# Applying the binary operator 'and' (line 66)
result_and_keyword_269083 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 3), 'and', result_is_not_269079, result_is__269082)

# Testing the type of an if condition (line 66)
if_condition_269084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 0), result_and_keyword_269083)
# Assigning a type to the variable 'if_condition_269084' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'if_condition_269084', if_condition_269084)
# SSA begins for if statement (line 66)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 67)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Subscript to a Name (line 68):

# Obtaining the type of the subscript
int_269085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 47), 'int')

# Obtaining the type of the subscript
# Getting the type of 'QT_API_ENV' (line 68)
QT_API_ENV_269086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'QT_API_ENV')
# Getting the type of 'ETS' (line 68)
ETS_269087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'ETS')
# Obtaining the member '__getitem__' of a type (line 68)
getitem___269088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 31), ETS_269087, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 68)
subscript_call_result_269089 = invoke(stypy.reporting.localization.Localization(__file__, 68, 31), getitem___269088, QT_API_ENV_269086)

# Obtaining the member '__getitem__' of a type (line 68)
getitem___269090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 31), subscript_call_result_269089, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 68)
subscript_call_result_269091 = invoke(stypy.reporting.localization.Localization(__file__, 68, 31), getitem___269090, int_269085)

# Assigning a type to the variable 'QT_ENV_MAJOR_VERSION' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'QT_ENV_MAJOR_VERSION', subscript_call_result_269091)
# SSA branch for the except part of a try statement (line 67)
# SSA branch for the except 'KeyError' branch of a try statement (line 67)
module_type_store.open_ssa_branch('except')

# Call to RuntimeError(...): (line 70)
# Processing the call arguments (line 70)
unicode_269093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'unicode', u'Unrecognized environment variable %r, valid values are: %r, %r, %r or %r')

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_269094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
# Getting the type of 'QT_API_ENV' (line 73)
QT_API_ENV_269095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'QT_API_ENV', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), tuple_269094, QT_API_ENV_269095)
# Adding element type (line 73)
unicode_269096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'unicode', u'pyqt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), tuple_269094, unicode_269096)
# Adding element type (line 73)
unicode_269097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'unicode', u'pyside')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), tuple_269094, unicode_269097)
# Adding element type (line 73)
unicode_269098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 46), 'unicode', u'pyqt5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), tuple_269094, unicode_269098)
# Adding element type (line 73)
unicode_269099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 55), 'unicode', u'pyside2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), tuple_269094, unicode_269099)

# Applying the binary operator '%' (line 71)
result_mod_269100 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 13), '%', unicode_269093, tuple_269094)

# Processing the call keyword arguments (line 70)
kwargs_269101 = {}
# Getting the type of 'RuntimeError' (line 70)
RuntimeError_269092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'RuntimeError', False)
# Calling RuntimeError(args, kwargs) (line 70)
RuntimeError_call_result_269102 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), RuntimeError_269092, *[result_mod_269100], **kwargs_269101)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 70, 8), RuntimeError_call_result_269102, 'raise parameter', BaseException)
# SSA join for try-except statement (line 67)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of 'QT_ENV_MAJOR_VERSION' (line 74)
QT_ENV_MAJOR_VERSION_269103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'QT_ENV_MAJOR_VERSION')
# Getting the type of 'QT_RC_MAJOR_VERSION' (line 74)
QT_RC_MAJOR_VERSION_269104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'QT_RC_MAJOR_VERSION')
# Applying the binary operator '==' (line 74)
result_eq_269105 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), '==', QT_ENV_MAJOR_VERSION_269103, QT_RC_MAJOR_VERSION_269104)

# Testing the type of an if condition (line 74)
if_condition_269106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), result_eq_269105)
# Assigning a type to the variable 'if_condition_269106' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_269106', if_condition_269106)
# SSA begins for if statement (line 74)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Subscript to a Name (line 77):

# Obtaining the type of the subscript
int_269107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 33), 'int')

# Obtaining the type of the subscript
# Getting the type of 'QT_API_ENV' (line 77)
QT_API_ENV_269108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'QT_API_ENV')
# Getting the type of 'ETS' (line 77)
ETS_269109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'ETS')
# Obtaining the member '__getitem__' of a type (line 77)
getitem___269110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 17), ETS_269109, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 77)
subscript_call_result_269111 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), getitem___269110, QT_API_ENV_269108)

# Obtaining the member '__getitem__' of a type (line 77)
getitem___269112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 17), subscript_call_result_269111, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 77)
subscript_call_result_269113 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), getitem___269112, int_269107)

# Assigning a type to the variable 'QT_API' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'QT_API', subscript_call_result_269113)
# SSA join for if statement (line 74)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 66)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 79):
# Getting the type of 'False' (line 79)
False_269114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'False')
# Assigning a type to the variable '_fallback_to_qt4' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), '_fallback_to_qt4', False_269114)

# Type idiom detected: calculating its left and rigth part (line 80)
# Getting the type of 'QT_API' (line 80)
QT_API_269115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 3), 'QT_API')
# Getting the type of 'None' (line 80)
None_269116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'None')

(may_be_269117, more_types_in_union_269118) = may_be_none(QT_API_269115, None_269116)

if may_be_269117:

    if more_types_in_union_269118:
        # Runtime conditional SSA (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    
    
    # Obtaining the type of the subscript
    unicode_269119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'unicode', u'backend')
    # Getting the type of 'rcParams' (line 82)
    rcParams_269120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___269121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 7), rcParams_269120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_269122 = invoke(stypy.reporting.localization.Localization(__file__, 82, 7), getitem___269121, unicode_269119)
    
    unicode_269123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'unicode', u'Qt5Agg')
    # Applying the binary operator '==' (line 82)
    result_eq_269124 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 7), '==', subscript_call_result_269122, unicode_269123)
    
    # Testing the type of an if condition (line 82)
    if_condition_269125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_eq_269124)
    # Assigning a type to the variable 'if_condition_269125' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_269125', if_condition_269125)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 83):
    
    # Obtaining the type of the subscript
    unicode_269126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'unicode', u'backend.qt5')
    # Getting the type of 'rcParams' (line 83)
    rcParams_269127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 83)
    getitem___269128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), rcParams_269127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 83)
    subscript_call_result_269129 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), getitem___269128, unicode_269126)
    
    # Assigning a type to the variable 'QT_API' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'QT_API', subscript_call_result_269129)
    # SSA branch for the else part of an if statement (line 82)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    unicode_269130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 18), 'unicode', u'backend')
    # Getting the type of 'rcParams' (line 84)
    rcParams_269131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___269132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 9), rcParams_269131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_269133 = invoke(stypy.reporting.localization.Localization(__file__, 84, 9), getitem___269132, unicode_269130)
    
    unicode_269134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 32), 'unicode', u'Qt4Agg')
    # Applying the binary operator '==' (line 84)
    result_eq_269135 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 9), '==', subscript_call_result_269133, unicode_269134)
    
    # Testing the type of an if condition (line 84)
    if_condition_269136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 9), result_eq_269135)
    # Assigning a type to the variable 'if_condition_269136' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'if_condition_269136', if_condition_269136)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    unicode_269137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'unicode', u'backend.qt4')
    # Getting the type of 'rcParams' (line 85)
    rcParams_269138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___269139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 17), rcParams_269138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_269140 = invoke(stypy.reporting.localization.Localization(__file__, 85, 17), getitem___269139, unicode_269137)
    
    # Assigning a type to the variable 'QT_API' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'QT_API', subscript_call_result_269140)
    # SSA branch for the else part of an if statement (line 84)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 91):
    # Getting the type of 'True' (line 91)
    True_269141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'True')
    # Assigning a type to the variable '_fallback_to_qt4' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), '_fallback_to_qt4', True_269141)
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    unicode_269142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'unicode', u'backend.qt5')
    # Getting the type of 'rcParams' (line 92)
    rcParams_269143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___269144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), rcParams_269143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_269145 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), getitem___269144, unicode_269142)
    
    # Assigning a type to the variable 'QT_API' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'QT_API', subscript_call_result_269145)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    

    if more_types_in_union_269118:
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Name to a Name (line 96):
# Getting the type of 'None' (line 96)
None_269146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'None')
# Assigning a type to the variable '_getSaveFileName' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), '_getSaveFileName', None_269146)

# Assigning a Name to a Name (line 99):
# Getting the type of 'False' (line 99)
False_269147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'False')
# Assigning a type to the variable '_sip_imported' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), '_sip_imported', False_269147)


# Getting the type of 'QT_API' (line 102)
QT_API_269148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 3), 'QT_API')

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_269149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
# Getting the type of 'QT_API_PYQT' (line 102)
QT_API_PYQT_269150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'QT_API_PYQT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 14), tuple_269149, QT_API_PYQT_269150)
# Adding element type (line 102)
# Getting the type of 'QT_API_PYQTv2' (line 102)
QT_API_PYQTv2_269151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'QT_API_PYQTv2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 14), tuple_269149, QT_API_PYQTv2_269151)
# Adding element type (line 102)
# Getting the type of 'QT_API_PYQT5' (line 102)
QT_API_PYQT5_269152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'QT_API_PYQT5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 14), tuple_269149, QT_API_PYQT5_269152)

# Applying the binary operator 'in' (line 102)
result_contains_269153 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 3), 'in', QT_API_269148, tuple_269149)

# Testing the type of an if condition (line 102)
if_condition_269154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 0), result_contains_269153)
# Assigning a type to the variable 'if_condition_269154' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'if_condition_269154', if_condition_269154)
# SSA begins for if statement (line 102)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 103)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 104, 8))

# 'import sip' statement (line 104)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269155 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 104, 8), 'sip')

if (type(import_269155) is not StypyTypeError):

    if (import_269155 != 'pyd_module'):
        __import__(import_269155)
        sys_modules_269156 = sys.modules[import_269155]
        import_module(stypy.reporting.localization.Localization(__file__, 104, 8), 'sip', sys_modules_269156.module_type_store, module_type_store)
    else:
        import sip

        import_module(stypy.reporting.localization.Localization(__file__, 104, 8), 'sip', sip, module_type_store)

else:
    # Assigning a type to the variable 'sip' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'sip', import_269155)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Name to a Name (line 105):
# Getting the type of 'True' (line 105)
True_269157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'True')
# Assigning a type to the variable '_sip_imported' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), '_sip_imported', True_269157)
# SSA branch for the except part of a try statement (line 103)
# SSA branch for the except 'ImportError' branch of a try statement (line 103)
module_type_store.open_ssa_branch('except')


# Getting the type of 'QT_RC_MAJOR_VERSION' (line 108)
QT_RC_MAJOR_VERSION_269158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'QT_RC_MAJOR_VERSION')
int_269159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'int')
# Applying the binary operator '==' (line 108)
result_eq_269160 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), '==', QT_RC_MAJOR_VERSION_269158, int_269159)

# Testing the type of an if condition (line 108)
if_condition_269161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 8), result_eq_269160)
# Assigning a type to the variable 'if_condition_269161' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'if_condition_269161', if_condition_269161)
# SSA begins for if statement (line 108)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 109):
# Getting the type of 'QT_API_PYSIDE2' (line 109)
QT_API_PYSIDE2_269162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'QT_API_PYSIDE2')
# Assigning a type to the variable 'QT_API' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'QT_API', QT_API_PYSIDE2_269162)
# SSA branch for the else part of an if statement (line 108)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 111):
# Getting the type of 'QT_API_PYSIDE' (line 111)
QT_API_PYSIDE_269163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'QT_API_PYSIDE')
# Assigning a type to the variable 'QT_API' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'QT_API', QT_API_PYSIDE_269163)
# SSA join for if statement (line 108)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 112):
unicode_269164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'unicode', u'Could not import sip; falling back on PySide\nin place of PyQt4 or PyQt5.\n')
# Assigning a type to the variable 'cond' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'cond', unicode_269164)

# Call to report(...): (line 114)
# Processing the call arguments (line 114)
# Getting the type of 'cond' (line 114)
cond_269167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'cond', False)
unicode_269168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'unicode', u'helpful')
# Processing the call keyword arguments (line 114)
kwargs_269169 = {}
# Getting the type of 'verbose' (line 114)
verbose_269165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'verbose', False)
# Obtaining the member 'report' of a type (line 114)
report_269166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), verbose_269165, 'report')
# Calling report(args, kwargs) (line 114)
report_call_result_269170 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), report_269166, *[cond_269167, unicode_269168], **kwargs_269169)

# SSA join for try-except statement (line 103)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 102)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of '_sip_imported' (line 116)
_sip_imported_269171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 3), '_sip_imported')
# Testing the type of an if condition (line 116)
if_condition_269172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 0), _sip_imported_269171)
# Assigning a type to the variable 'if_condition_269172' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'if_condition_269172', if_condition_269172)
# SSA begins for if statement (line 116)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# Getting the type of 'QT_API' (line 117)
QT_API_269173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'QT_API')
# Getting the type of 'QT_API_PYQTv2' (line 117)
QT_API_PYQTv2_269174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'QT_API_PYQTv2')
# Applying the binary operator '==' (line 117)
result_eq_269175 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), '==', QT_API_269173, QT_API_PYQTv2_269174)

# Testing the type of an if condition (line 117)
if_condition_269176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_eq_269175)
# Assigning a type to the variable 'if_condition_269176' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_269176', if_condition_269176)
# SSA begins for if statement (line 117)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# Getting the type of 'QT_API_ENV' (line 118)
QT_API_ENV_269177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'QT_API_ENV')
unicode_269178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'unicode', u'pyqt')
# Applying the binary operator '==' (line 118)
result_eq_269179 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 11), '==', QT_API_ENV_269177, unicode_269178)

# Testing the type of an if condition (line 118)
if_condition_269180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), result_eq_269179)
# Assigning a type to the variable 'if_condition_269180' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_269180', if_condition_269180)
# SSA begins for if statement (line 118)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 119):
unicode_269181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'unicode', u"Found 'QT_API=pyqt' environment variable. Setting PyQt4 API accordingly.\n")
# Assigning a type to the variable 'cond' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'cond', unicode_269181)
# SSA branch for the else part of an if statement (line 118)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 122):
unicode_269182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 19), 'unicode', u'PyQt API v2 specified.')
# Assigning a type to the variable 'cond' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'cond', unicode_269182)
# SSA join for if statement (line 118)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 123)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to setapi(...): (line 124)
# Processing the call arguments (line 124)
unicode_269185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'unicode', u'QString')
int_269186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 34), 'int')
# Processing the call keyword arguments (line 124)
kwargs_269187 = {}
# Getting the type of 'sip' (line 124)
sip_269183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'sip', False)
# Obtaining the member 'setapi' of a type (line 124)
setapi_269184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), sip_269183, 'setapi')
# Calling setapi(args, kwargs) (line 124)
setapi_call_result_269188 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), setapi_269184, *[unicode_269185, int_269186], **kwargs_269187)

# SSA branch for the except part of a try statement (line 123)
# SSA branch for the except '<any exception>' branch of a try statement (line 123)
module_type_store.open_ssa_branch('except')

# Assigning a Str to a Name (line 126):
unicode_269189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 18), 'unicode', u'QString API v2 specification failed. Defaulting to v1.')
# Assigning a type to the variable 'res' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'res', unicode_269189)

# Call to report(...): (line 127)
# Processing the call arguments (line 127)
# Getting the type of 'cond' (line 127)
cond_269192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'cond', False)
# Getting the type of 'res' (line 127)
res_269193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'res', False)
# Applying the binary operator '+' (line 127)
result_add_269194 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 27), '+', cond_269192, res_269193)

unicode_269195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'unicode', u'helpful')
# Processing the call keyword arguments (line 127)
kwargs_269196 = {}
# Getting the type of 'verbose' (line 127)
verbose_269190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'verbose', False)
# Obtaining the member 'report' of a type (line 127)
report_269191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), verbose_269190, 'report')
# Calling report(args, kwargs) (line 127)
report_call_result_269197 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), report_269191, *[result_add_269194, unicode_269195], **kwargs_269196)


# Assigning a Str to a Name (line 129):
unicode_269198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'unicode', u'')
# Assigning a type to the variable 'cond' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'cond', unicode_269198)
# SSA join for try-except statement (line 123)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 130)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to setapi(...): (line 131)
# Processing the call arguments (line 131)
unicode_269201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'unicode', u'QVariant')
int_269202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 35), 'int')
# Processing the call keyword arguments (line 131)
kwargs_269203 = {}
# Getting the type of 'sip' (line 131)
sip_269199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'sip', False)
# Obtaining the member 'setapi' of a type (line 131)
setapi_269200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), sip_269199, 'setapi')
# Calling setapi(args, kwargs) (line 131)
setapi_call_result_269204 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), setapi_269200, *[unicode_269201, int_269202], **kwargs_269203)

# SSA branch for the except part of a try statement (line 130)
# SSA branch for the except '<any exception>' branch of a try statement (line 130)
module_type_store.open_ssa_branch('except')

# Assigning a Str to a Name (line 133):
unicode_269205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'unicode', u'QVariant API v2 specification failed. Defaulting to v1.')
# Assigning a type to the variable 'res' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'res', unicode_269205)

# Call to report(...): (line 134)
# Processing the call arguments (line 134)
# Getting the type of 'cond' (line 134)
cond_269208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'cond', False)
# Getting the type of 'res' (line 134)
res_269209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 34), 'res', False)
# Applying the binary operator '+' (line 134)
result_add_269210 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 27), '+', cond_269208, res_269209)

unicode_269211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 39), 'unicode', u'helpful')
# Processing the call keyword arguments (line 134)
kwargs_269212 = {}
# Getting the type of 'verbose' (line 134)
verbose_269206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'verbose', False)
# Obtaining the member 'report' of a type (line 134)
report_269207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), verbose_269206, 'report')
# Calling report(args, kwargs) (line 134)
report_call_result_269213 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), report_269207, *[result_add_269210, unicode_269211], **kwargs_269212)

# SSA join for try-except statement (line 130)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 117)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of 'QT_API' (line 135)
QT_API_269214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 7), 'QT_API')
# Getting the type of 'QT_API_PYQT5' (line 135)
QT_API_PYQT5_269215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'QT_API_PYQT5')
# Applying the binary operator '==' (line 135)
result_eq_269216 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 7), '==', QT_API_269214, QT_API_PYQT5_269215)

# Testing the type of an if condition (line 135)
if_condition_269217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 4), result_eq_269216)
# Assigning a type to the variable 'if_condition_269217' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'if_condition_269217', if_condition_269217)
# SSA begins for if statement (line 135)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 136)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 137, 12))

# 'from PyQt5 import QtCore, QtGui, QtWidgets' statement (line 137)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269218 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 137, 12), 'PyQt5')

if (type(import_269218) is not StypyTypeError):

    if (import_269218 != 'pyd_module'):
        __import__(import_269218)
        sys_modules_269219 = sys.modules[import_269218]
        import_from_module(stypy.reporting.localization.Localization(__file__, 137, 12), 'PyQt5', sys_modules_269219.module_type_store, module_type_store, ['QtCore', 'QtGui', 'QtWidgets'])
        nest_module(stypy.reporting.localization.Localization(__file__, 137, 12), __file__, sys_modules_269219, sys_modules_269219.module_type_store, module_type_store)
    else:
        from PyQt5 import QtCore, QtGui, QtWidgets

        import_from_module(stypy.reporting.localization.Localization(__file__, 137, 12), 'PyQt5', None, module_type_store, ['QtCore', 'QtGui', 'QtWidgets'], [QtCore, QtGui, QtWidgets])

else:
    # Assigning a type to the variable 'PyQt5' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'PyQt5', import_269218)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Attribute to a Name (line 138):
# Getting the type of 'QtWidgets' (line 138)
QtWidgets_269220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'QtWidgets')
# Obtaining the member 'QFileDialog' of a type (line 138)
QFileDialog_269221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 31), QtWidgets_269220, 'QFileDialog')
# Obtaining the member 'getSaveFileName' of a type (line 138)
getSaveFileName_269222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 31), QFileDialog_269221, 'getSaveFileName')
# Assigning a type to the variable '_getSaveFileName' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), '_getSaveFileName', getSaveFileName_269222)
# SSA branch for the except part of a try statement (line 136)
# SSA branch for the except 'ImportError' branch of a try statement (line 136)
module_type_store.open_ssa_branch('except')

# Getting the type of '_fallback_to_qt4' (line 140)
_fallback_to_qt4_269223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), '_fallback_to_qt4')
# Testing the type of an if condition (line 140)
if_condition_269224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 12), _fallback_to_qt4_269223)
# Assigning a type to the variable 'if_condition_269224' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'if_condition_269224', if_condition_269224)
# SSA begins for if statement (line 140)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Subscript to a Name (line 142):

# Obtaining the type of the subscript
unicode_269225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'unicode', u'backend.qt4')
# Getting the type of 'rcParams' (line 142)
rcParams_269226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 142)
getitem___269227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 25), rcParams_269226, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 142)
subscript_call_result_269228 = invoke(stypy.reporting.localization.Localization(__file__, 142, 25), getitem___269227, unicode_269225)

# Assigning a type to the variable 'QT_API' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'QT_API', subscript_call_result_269228)

# Assigning a Num to a Name (line 143):
int_269229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 38), 'int')
# Assigning a type to the variable 'QT_RC_MAJOR_VERSION' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'QT_RC_MAJOR_VERSION', int_269229)
# SSA branch for the else part of an if statement (line 140)
module_type_store.open_ssa_branch('else')
# SSA join for if statement (line 140)
module_type_store = module_type_store.join_ssa_context()

# SSA join for try-except statement (line 136)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 135)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of 'QT_API' (line 149)
QT_API_269230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'QT_API')

# Obtaining an instance of the builtin type 'list' (line 149)
list_269231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 149)
# Adding element type (line 149)
# Getting the type of 'QT_API_PYQT' (line 149)
QT_API_PYQT_269232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'QT_API_PYQT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 17), list_269231, QT_API_PYQT_269232)
# Adding element type (line 149)
# Getting the type of 'QT_API_PYQTv2' (line 149)
QT_API_PYQTv2_269233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'QT_API_PYQTv2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 17), list_269231, QT_API_PYQTv2_269233)

# Applying the binary operator 'in' (line 149)
result_contains_269234 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 7), 'in', QT_API_269230, list_269231)

# Testing the type of an if condition (line 149)
if_condition_269235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), result_contains_269234)
# Assigning a type to the variable 'if_condition_269235' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_269235', if_condition_269235)
# SSA begins for if statement (line 149)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 150, 8))

# 'from PyQt4 import QtCore, QtGui' statement (line 150)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269236 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 150, 8), 'PyQt4')

if (type(import_269236) is not StypyTypeError):

    if (import_269236 != 'pyd_module'):
        __import__(import_269236)
        sys_modules_269237 = sys.modules[import_269236]
        import_from_module(stypy.reporting.localization.Localization(__file__, 150, 8), 'PyQt4', sys_modules_269237.module_type_store, module_type_store, ['QtCore', 'QtGui'])
        nest_module(stypy.reporting.localization.Localization(__file__, 150, 8), __file__, sys_modules_269237, sys_modules_269237.module_type_store, module_type_store)
    else:
        from PyQt4 import QtCore, QtGui

        import_from_module(stypy.reporting.localization.Localization(__file__, 150, 8), 'PyQt4', None, module_type_store, ['QtCore', 'QtGui'], [QtCore, QtGui])

else:
    # Assigning a type to the variable 'PyQt4' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'PyQt4', import_269236)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# SSA begins for try-except statement (line 152)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')



# Call to getapi(...): (line 153)
# Processing the call arguments (line 153)
unicode_269240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'unicode', u'QString')
# Processing the call keyword arguments (line 153)
kwargs_269241 = {}
# Getting the type of 'sip' (line 153)
sip_269238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'sip', False)
# Obtaining the member 'getapi' of a type (line 153)
getapi_269239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), sip_269238, 'getapi')
# Calling getapi(args, kwargs) (line 153)
getapi_call_result_269242 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), getapi_269239, *[unicode_269240], **kwargs_269241)

int_269243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 39), 'int')
# Applying the binary operator '>' (line 153)
result_gt_269244 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '>', getapi_call_result_269242, int_269243)

# Testing the type of an if condition (line 153)
if_condition_269245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), result_gt_269244)
# Assigning a type to the variable 'if_condition_269245' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_269245', if_condition_269245)
# SSA begins for if statement (line 153)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Attribute to a Name (line 155):
# Getting the type of 'QtGui' (line 155)
QtGui_269246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'QtGui')
# Obtaining the member 'QFileDialog' of a type (line 155)
QFileDialog_269247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 35), QtGui_269246, 'QFileDialog')
# Obtaining the member 'getSaveFileNameAndFilter' of a type (line 155)
getSaveFileNameAndFilter_269248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 35), QFileDialog_269247, 'getSaveFileNameAndFilter')
# Assigning a type to the variable '_getSaveFileName' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), '_getSaveFileName', getSaveFileNameAndFilter_269248)
# SSA branch for the else part of an if statement (line 153)
module_type_store.open_ssa_branch('else')

@norecursion
def _getSaveFileName(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getSaveFileName'
    module_type_store = module_type_store.open_function_context('_getSaveFileName', 159, 16, False)
    
    # Passed parameters checking function
    _getSaveFileName.stypy_localization = localization
    _getSaveFileName.stypy_type_of_self = None
    _getSaveFileName.stypy_type_store = module_type_store
    _getSaveFileName.stypy_function_name = '_getSaveFileName'
    _getSaveFileName.stypy_param_names_list = []
    _getSaveFileName.stypy_varargs_param_name = 'args'
    _getSaveFileName.stypy_kwargs_param_name = 'kwargs'
    _getSaveFileName.stypy_call_defaults = defaults
    _getSaveFileName.stypy_call_varargs = varargs
    _getSaveFileName.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getSaveFileName', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getSaveFileName', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getSaveFileName(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 160)
    tuple_269249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 160)
    # Adding element type (line 160)
    
    # Call to getSaveFileName(...): (line 160)
    # Getting the type of 'args' (line 160)
    args_269253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 63), 'args', False)
    # Processing the call keyword arguments (line 160)
    # Getting the type of 'kwargs' (line 160)
    kwargs_269254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 71), 'kwargs', False)
    kwargs_269255 = {'kwargs_269254': kwargs_269254}
    # Getting the type of 'QtGui' (line 160)
    QtGui_269250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'QtGui', False)
    # Obtaining the member 'QFileDialog' of a type (line 160)
    QFileDialog_269251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), QtGui_269250, 'QFileDialog')
    # Obtaining the member 'getSaveFileName' of a type (line 160)
    getSaveFileName_269252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), QFileDialog_269251, 'getSaveFileName')
    # Calling getSaveFileName(args, kwargs) (line 160)
    getSaveFileName_call_result_269256 = invoke(stypy.reporting.localization.Localization(__file__, 160, 28), getSaveFileName_269252, *[args_269253], **kwargs_269255)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_269249, getSaveFileName_call_result_269256)
    # Adding element type (line 160)
    # Getting the type of 'None' (line 161)
    None_269257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_269249, None_269257)
    
    # Assigning a type to the variable 'stypy_return_type' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'stypy_return_type', tuple_269249)
    
    # ################# End of '_getSaveFileName(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getSaveFileName' in the type store
    # Getting the type of 'stypy_return_type' (line 159)
    stypy_return_type_269258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getSaveFileName'
    return stypy_return_type_269258

# Assigning a type to the variable '_getSaveFileName' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), '_getSaveFileName', _getSaveFileName)
# SSA join for if statement (line 153)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the except part of a try statement (line 152)
# SSA branch for the except 'Tuple' branch of a try statement (line 152)
module_type_store.open_ssa_branch('except')

@norecursion
def _getSaveFileName(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getSaveFileName'
    module_type_store = module_type_store.open_function_context('_getSaveFileName', 166, 12, False)
    
    # Passed parameters checking function
    _getSaveFileName.stypy_localization = localization
    _getSaveFileName.stypy_type_of_self = None
    _getSaveFileName.stypy_type_store = module_type_store
    _getSaveFileName.stypy_function_name = '_getSaveFileName'
    _getSaveFileName.stypy_param_names_list = []
    _getSaveFileName.stypy_varargs_param_name = 'args'
    _getSaveFileName.stypy_kwargs_param_name = 'kwargs'
    _getSaveFileName.stypy_call_defaults = defaults
    _getSaveFileName.stypy_call_varargs = varargs
    _getSaveFileName.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getSaveFileName', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getSaveFileName', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getSaveFileName(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_269259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    
    # Call to getSaveFileName(...): (line 167)
    # Getting the type of 'args' (line 167)
    args_269263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 58), 'args', False)
    # Processing the call keyword arguments (line 167)
    # Getting the type of 'kwargs' (line 167)
    kwargs_269264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 66), 'kwargs', False)
    kwargs_269265 = {'kwargs_269264': kwargs_269264}
    # Getting the type of 'QtGui' (line 167)
    QtGui_269260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'QtGui', False)
    # Obtaining the member 'QFileDialog' of a type (line 167)
    QFileDialog_269261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 23), QtGui_269260, 'QFileDialog')
    # Obtaining the member 'getSaveFileName' of a type (line 167)
    getSaveFileName_269262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 23), QFileDialog_269261, 'getSaveFileName')
    # Calling getSaveFileName(args, kwargs) (line 167)
    getSaveFileName_call_result_269266 = invoke(stypy.reporting.localization.Localization(__file__, 167, 23), getSaveFileName_269262, *[args_269263], **kwargs_269265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), tuple_269259, getSaveFileName_call_result_269266)
    # Adding element type (line 167)
    # Getting the type of 'None' (line 167)
    None_269267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 75), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), tuple_269259, None_269267)
    
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'stypy_return_type', tuple_269259)
    
    # ################# End of '_getSaveFileName(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getSaveFileName' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_269268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getSaveFileName'
    return stypy_return_type_269268

# Assigning a type to the variable '_getSaveFileName' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), '_getSaveFileName', _getSaveFileName)
# SSA join for try-except statement (line 152)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 149)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 168)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Attribute to a Attribute (line 170):
# Getting the type of 'QtCore' (line 170)
QtCore_269269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'QtCore')
# Obtaining the member 'pyqtSignal' of a type (line 170)
pyqtSignal_269270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 24), QtCore_269269, 'pyqtSignal')
# Getting the type of 'QtCore' (line 170)
QtCore_269271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'QtCore')
# Setting the type of the member 'Signal' of a type (line 170)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), QtCore_269271, 'Signal', pyqtSignal_269270)


# SSA begins for try-except statement (line 171)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Attribute to a Attribute (line 172):
# Getting the type of 'QtCore' (line 172)
QtCore_269272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 26), 'QtCore')
# Obtaining the member 'pyqtSlot' of a type (line 172)
pyqtSlot_269273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 26), QtCore_269272, 'pyqtSlot')
# Getting the type of 'QtCore' (line 172)
QtCore_269274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'QtCore')
# Setting the type of the member 'Slot' of a type (line 172)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), QtCore_269274, 'Slot', pyqtSlot_269273)
# SSA branch for the except part of a try statement (line 171)
# SSA branch for the except 'AttributeError' branch of a try statement (line 171)
module_type_store.open_ssa_branch('except')

# Assigning a Attribute to a Attribute (line 175):
# Getting the type of 'QtCore' (line 175)
QtCore_269275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'QtCore')
# Obtaining the member 'pyqtSignature' of a type (line 175)
pyqtSignature_269276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 26), QtCore_269275, 'pyqtSignature')
# Getting the type of 'QtCore' (line 175)
QtCore_269277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'QtCore')
# Setting the type of the member 'Slot' of a type (line 175)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), QtCore_269277, 'Slot', pyqtSignature_269276)
# SSA join for try-except statement (line 171)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Attribute (line 177):
# Getting the type of 'QtCore' (line 177)
QtCore_269278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'QtCore')
# Obtaining the member 'pyqtProperty' of a type (line 177)
pyqtProperty_269279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 26), QtCore_269278, 'pyqtProperty')
# Getting the type of 'QtCore' (line 177)
QtCore_269280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'QtCore')
# Setting the type of the member 'Property' of a type (line 177)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), QtCore_269280, 'Property', pyqtProperty_269279)

# Assigning a Attribute to a Name (line 178):
# Getting the type of 'QtCore' (line 178)
QtCore_269281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'QtCore')
# Obtaining the member 'PYQT_VERSION_STR' of a type (line 178)
PYQT_VERSION_STR_269282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), QtCore_269281, 'PYQT_VERSION_STR')
# Assigning a type to the variable '__version__' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), '__version__', PYQT_VERSION_STR_269282)
# SSA branch for the except part of a try statement (line 168)
# SSA branch for the except 'NameError' branch of a try statement (line 168)
module_type_store.open_ssa_branch('except')


# Getting the type of 'QT_RC_MAJOR_VERSION' (line 181)
QT_RC_MAJOR_VERSION_269283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'QT_RC_MAJOR_VERSION')
int_269284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 34), 'int')
# Applying the binary operator '==' (line 181)
result_eq_269285 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), '==', QT_RC_MAJOR_VERSION_269283, int_269284)

# Testing the type of an if condition (line 181)
if_condition_269286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_eq_269285)
# Assigning a type to the variable 'if_condition_269286' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_269286', if_condition_269286)
# SSA begins for if statement (line 181)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 182):
# Getting the type of 'QT_API_PYSIDE2' (line 182)
QT_API_PYSIDE2_269287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'QT_API_PYSIDE2')
# Assigning a type to the variable 'QT_API' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'QT_API', QT_API_PYSIDE2_269287)
# SSA branch for the else part of an if statement (line 181)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 184):
# Getting the type of 'QT_API_PYSIDE' (line 184)
QT_API_PYSIDE_269288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'QT_API_PYSIDE')
# Assigning a type to the variable 'QT_API' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'QT_API', QT_API_PYSIDE_269288)
# SSA join for if statement (line 181)
module_type_store = module_type_store.join_ssa_context()

# SSA join for try-except statement (line 168)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 116)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of 'QT_API' (line 187)
QT_API_269289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 3), 'QT_API')
# Getting the type of 'QT_API_PYSIDE2' (line 187)
QT_API_PYSIDE2_269290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'QT_API_PYSIDE2')
# Applying the binary operator '==' (line 187)
result_eq_269291 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 3), '==', QT_API_269289, QT_API_PYSIDE2_269290)

# Testing the type of an if condition (line 187)
if_condition_269292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 0), result_eq_269291)
# Assigning a type to the variable 'if_condition_269292' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'if_condition_269292', if_condition_269292)
# SSA begins for if statement (line 187)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 188)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 189, 8))

# 'from PySide2 import QtCore, QtGui, QtWidgets, __version__' statement (line 189)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269293 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 189, 8), 'PySide2')

if (type(import_269293) is not StypyTypeError):

    if (import_269293 != 'pyd_module'):
        __import__(import_269293)
        sys_modules_269294 = sys.modules[import_269293]
        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 8), 'PySide2', sys_modules_269294.module_type_store, module_type_store, ['QtCore', 'QtGui', 'QtWidgets', '__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 189, 8), __file__, sys_modules_269294, sys_modules_269294.module_type_store, module_type_store)
    else:
        from PySide2 import QtCore, QtGui, QtWidgets, __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 8), 'PySide2', None, module_type_store, ['QtCore', 'QtGui', 'QtWidgets', '__version__'], [QtCore, QtGui, QtWidgets, __version__])

else:
    # Assigning a type to the variable 'PySide2' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'PySide2', import_269293)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Attribute to a Name (line 190):
# Getting the type of 'QtWidgets' (line 190)
QtWidgets_269295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'QtWidgets')
# Obtaining the member 'QFileDialog' of a type (line 190)
QFileDialog_269296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), QtWidgets_269295, 'QFileDialog')
# Obtaining the member 'getSaveFileName' of a type (line 190)
getSaveFileName_269297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), QFileDialog_269296, 'getSaveFileName')
# Assigning a type to the variable '_getSaveFileName' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), '_getSaveFileName', getSaveFileName_269297)
# SSA branch for the except part of a try statement (line 188)
# SSA branch for the except 'ImportError' branch of a try statement (line 188)
module_type_store.open_ssa_branch('except')

# Assigning a Num to a Name (line 193):
int_269298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 30), 'int')
# Assigning a type to the variable 'QT_RC_MAJOR_VERSION' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'QT_RC_MAJOR_VERSION', int_269298)

# Assigning a Name to a Name (line 194):
# Getting the type of 'QT_API_PYSIDE' (line 194)
QT_API_PYSIDE_269299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'QT_API_PYSIDE')
# Assigning a type to the variable 'QT_API' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'QT_API', QT_API_PYSIDE_269299)
# SSA join for try-except statement (line 188)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 187)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of 'QT_API' (line 196)
QT_API_269300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 3), 'QT_API')
# Getting the type of 'QT_API_PYSIDE' (line 196)
QT_API_PYSIDE_269301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 13), 'QT_API_PYSIDE')
# Applying the binary operator '==' (line 196)
result_eq_269302 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 3), '==', QT_API_269300, QT_API_PYSIDE_269301)

# Testing the type of an if condition (line 196)
if_condition_269303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 0), result_eq_269302)
# Assigning a type to the variable 'if_condition_269303' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'if_condition_269303', if_condition_269303)
# SSA begins for if statement (line 196)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 197)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 198, 8))

# 'from PySide import QtCore, QtGui, __version__, __version_info__' statement (line 198)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269304 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 198, 8), 'PySide')

if (type(import_269304) is not StypyTypeError):

    if (import_269304 != 'pyd_module'):
        __import__(import_269304)
        sys_modules_269305 = sys.modules[import_269304]
        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 8), 'PySide', sys_modules_269305.module_type_store, module_type_store, ['QtCore', 'QtGui', '__version__', '__version_info__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 198, 8), __file__, sys_modules_269305, sys_modules_269305.module_type_store, module_type_store)
    else:
        from PySide import QtCore, QtGui, __version__, __version_info__

        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 8), 'PySide', None, module_type_store, ['QtCore', 'QtGui', '__version__', '__version_info__'], [QtCore, QtGui, __version__, __version_info__])

else:
    # Assigning a type to the variable 'PySide' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'PySide', import_269304)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 197)
# SSA branch for the except 'ImportError' branch of a try statement (line 197)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 200)
# Processing the call arguments (line 200)
unicode_269307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 12), 'unicode', u'Matplotlib qt-based backends require an external PyQt4, PyQt5,\nPySide or PySide2 package to be installed, but it was not found.')
# Processing the call keyword arguments (line 200)
kwargs_269308 = {}
# Getting the type of 'ImportError' (line 200)
ImportError_269306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 200)
ImportError_call_result_269309 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), ImportError_269306, *[unicode_269307], **kwargs_269308)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 200, 8), ImportError_call_result_269309, 'raise parameter', BaseException)
# SSA join for try-except statement (line 197)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of '__version_info__' (line 204)
version_info___269310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 7), '__version_info__')

# Obtaining an instance of the builtin type 'tuple' (line 204)
tuple_269311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 204)
# Adding element type (line 204)
int_269312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 27), tuple_269311, int_269312)
# Adding element type (line 204)
int_269313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 27), tuple_269311, int_269313)
# Adding element type (line 204)
int_269314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 27), tuple_269311, int_269314)

# Applying the binary operator '<' (line 204)
result_lt_269315 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 7), '<', version_info___269310, tuple_269311)

# Testing the type of an if condition (line 204)
if_condition_269316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 4), result_lt_269315)
# Assigning a type to the variable 'if_condition_269316' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'if_condition_269316', if_condition_269316)
# SSA begins for if statement (line 204)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to ImportError(...): (line 205)
# Processing the call arguments (line 205)
unicode_269318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 12), 'unicode', u'Matplotlib backend_qt4 and backend_qt4agg require PySide >=1.0.3')
# Processing the call keyword arguments (line 205)
kwargs_269319 = {}
# Getting the type of 'ImportError' (line 205)
ImportError_269317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 14), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 205)
ImportError_call_result_269320 = invoke(stypy.reporting.localization.Localization(__file__, 205, 14), ImportError_269317, *[unicode_269318], **kwargs_269319)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 205, 8), ImportError_call_result_269320, 'raise parameter', BaseException)
# SSA join for if statement (line 204)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Name (line 208):
# Getting the type of 'QtGui' (line 208)
QtGui_269321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'QtGui')
# Obtaining the member 'QFileDialog' of a type (line 208)
QFileDialog_269322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), QtGui_269321, 'QFileDialog')
# Obtaining the member 'getSaveFileName' of a type (line 208)
getSaveFileName_269323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), QFileDialog_269322, 'getSaveFileName')
# Assigning a type to the variable '_getSaveFileName' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), '_getSaveFileName', getSaveFileName_269323)
# SSA join for if statement (line 196)
module_type_store = module_type_store.join_ssa_context()



# Getting the type of 'QT_API' (line 212)
QT_API_269324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 3), 'QT_API')

# Obtaining an instance of the builtin type 'tuple' (line 212)
tuple_269325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 212)
# Adding element type (line 212)
# Getting the type of 'QT_API_PYQT' (line 212)
QT_API_PYQT_269326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'QT_API_PYQT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 14), tuple_269325, QT_API_PYQT_269326)
# Adding element type (line 212)
# Getting the type of 'QT_API_PYQTv2' (line 212)
QT_API_PYQTv2_269327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'QT_API_PYQTv2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 14), tuple_269325, QT_API_PYQTv2_269327)
# Adding element type (line 212)
# Getting the type of 'QT_API_PYSIDE' (line 212)
QT_API_PYSIDE_269328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 42), 'QT_API_PYSIDE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 14), tuple_269325, QT_API_PYSIDE_269328)

# Applying the binary operator 'in' (line 212)
result_contains_269329 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 3), 'in', QT_API_269324, tuple_269325)

# Testing the type of an if condition (line 212)
if_condition_269330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 0), result_contains_269329)
# Assigning a type to the variable 'if_condition_269330' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'if_condition_269330', if_condition_269330)
# SSA begins for if statement (line 212)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
unicode_269331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'unicode', u"Import all used QtGui objects into QtWidgets\n\n    Here I've opted to simple copy QtGui into QtWidgets as that\n    achieves the same result as copying over the objects, and will\n    continue to work if other objects are used.\n\n    ")

# Assigning a Name to a Name (line 220):
# Getting the type of 'QtGui' (line 220)
QtGui_269332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'QtGui')
# Assigning a type to the variable 'QtWidgets' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'QtWidgets', QtGui_269332)
# SSA join for if statement (line 212)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def is_pyqt5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_pyqt5'
    module_type_store = module_type_store.open_function_context('is_pyqt5', 223, 0, False)
    
    # Passed parameters checking function
    is_pyqt5.stypy_localization = localization
    is_pyqt5.stypy_type_of_self = None
    is_pyqt5.stypy_type_store = module_type_store
    is_pyqt5.stypy_function_name = 'is_pyqt5'
    is_pyqt5.stypy_param_names_list = []
    is_pyqt5.stypy_varargs_param_name = None
    is_pyqt5.stypy_kwargs_param_name = None
    is_pyqt5.stypy_call_defaults = defaults
    is_pyqt5.stypy_call_varargs = varargs
    is_pyqt5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_pyqt5', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_pyqt5', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_pyqt5(...)' code ##################

    
    # Getting the type of 'QT_API' (line 224)
    QT_API_269333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'QT_API')
    # Getting the type of 'QT_API_PYQT5' (line 224)
    QT_API_PYQT5_269334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'QT_API_PYQT5')
    # Applying the binary operator '==' (line 224)
    result_eq_269335 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), '==', QT_API_269333, QT_API_PYQT5_269334)
    
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type', result_eq_269335)
    
    # ################# End of 'is_pyqt5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_pyqt5' in the type store
    # Getting the type of 'stypy_return_type' (line 223)
    stypy_return_type_269336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269336)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_pyqt5'
    return stypy_return_type_269336

# Assigning a type to the variable 'is_pyqt5' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'is_pyqt5', is_pyqt5)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

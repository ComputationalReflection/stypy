
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: A wx API adapter to hide differences between wxPython classic and phoenix.
4: 
5: It is assumed that the user code is selecting what version it wants to use,
6: here we just ensure that it meets the minimum required by matplotlib.
7: 
8: For an example see embedding_in_wx2.py
9: '''
10: from __future__ import (absolute_import, division, print_function,
11:                         unicode_literals)
12: 
13: import six
14: from distutils.version import LooseVersion
15: 
16: missingwx = "Matplotlib backend_wx and backend_wxagg require wxPython >=2.8.12"
17: 
18: 
19: try:
20:     import wx
21:     backend_version = wx.VERSION_STRING
22:     is_phoenix = 'phoenix' in wx.PlatformInfo
23: except ImportError:
24:     raise ImportError(missingwx)
25: 
26: # Ensure we have the correct version imported
27: if LooseVersion(wx.VERSION_STRING) < LooseVersion("2.8.12"):
28:     print(" wxPython version %s was imported." % backend_version)
29:     raise ImportError(missingwx)
30: 
31: if is_phoenix:
32:     # define all the wxPython phoenix stuff
33: 
34:     # font styles, families and weight
35:     fontweights = {
36:         100: wx.FONTWEIGHT_LIGHT,
37:         200: wx.FONTWEIGHT_LIGHT,
38:         300: wx.FONTWEIGHT_LIGHT,
39:         400: wx.FONTWEIGHT_NORMAL,
40:         500: wx.FONTWEIGHT_NORMAL,
41:         600: wx.FONTWEIGHT_NORMAL,
42:         700: wx.FONTWEIGHT_BOLD,
43:         800: wx.FONTWEIGHT_BOLD,
44:         900: wx.FONTWEIGHT_BOLD,
45:         'ultralight': wx.FONTWEIGHT_LIGHT,
46:         'light': wx.FONTWEIGHT_LIGHT,
47:         'normal': wx.FONTWEIGHT_NORMAL,
48:         'medium': wx.FONTWEIGHT_NORMAL,
49:         'semibold': wx.FONTWEIGHT_NORMAL,
50:         'bold': wx.FONTWEIGHT_BOLD,
51:         'heavy': wx.FONTWEIGHT_BOLD,
52:         'ultrabold': wx.FONTWEIGHT_BOLD,
53:         'black': wx.FONTWEIGHT_BOLD
54:     }
55:     fontangles = {
56:         'italic': wx.FONTSTYLE_ITALIC,
57:         'normal': wx.FONTSTYLE_NORMAL,
58:         'oblique': wx.FONTSTYLE_SLANT}
59: 
60:     # wxPython allows for portable font styles, choosing them appropriately
61:     # for the target platform. Map some standard font names to the portable
62:     # styles
63:     # QUESTION: Is it be wise to agree standard fontnames across all backends?
64:     fontnames = {'Sans': wx.FONTFAMILY_SWISS,
65:                  'Roman': wx.FONTFAMILY_ROMAN,
66:                  'Script': wx.FONTFAMILY_SCRIPT,
67:                  'Decorative': wx.FONTFAMILY_DECORATIVE,
68:                  'Modern': wx.FONTFAMILY_MODERN,
69:                  'Courier': wx.FONTFAMILY_MODERN,
70:                  'courier': wx.FONTFAMILY_MODERN}
71: 
72:     dashd_wx = {'solid': wx.PENSTYLE_SOLID,
73:                 'dashed': wx.PENSTYLE_SHORT_DASH,
74:                 'dashdot': wx.PENSTYLE_DOT_DASH,
75:                 'dotted': wx.PENSTYLE_DOT}
76: 
77:     # functions changes
78:     BitmapFromBuffer = wx.Bitmap.FromBufferRGBA
79:     EmptyBitmap = wx.Bitmap
80:     EmptyImage = wx.Image
81:     Cursor = wx.Cursor
82:     EventLoop = wx.GUIEventLoop
83:     NamedColour = wx.Colour
84:     StockCursor = wx.Cursor
85: 
86: else:
87:     # define all the wxPython classic stuff
88: 
89:     # font styles, families and weight
90:     fontweights = {
91:         100: wx.LIGHT,
92:         200: wx.LIGHT,
93:         300: wx.LIGHT,
94:         400: wx.NORMAL,
95:         500: wx.NORMAL,
96:         600: wx.NORMAL,
97:         700: wx.BOLD,
98:         800: wx.BOLD,
99:         900: wx.BOLD,
100:         'ultralight': wx.LIGHT,
101:         'light': wx.LIGHT,
102:         'normal': wx.NORMAL,
103:         'medium': wx.NORMAL,
104:         'semibold': wx.NORMAL,
105:         'bold': wx.BOLD,
106:         'heavy': wx.BOLD,
107:         'ultrabold': wx.BOLD,
108:         'black': wx.BOLD
109:     }
110:     fontangles = {
111:         'italic': wx.ITALIC,
112:         'normal': wx.NORMAL,
113:         'oblique': wx.SLANT}
114: 
115:     # wxPython allows for portable font styles, choosing them appropriately
116:     # for the target platform. Map some standard font names to the portable
117:     # styles
118:     # QUESTION: Is it be wise to agree standard fontnames across all backends?
119:     fontnames = {'Sans': wx.SWISS,
120:                  'Roman': wx.ROMAN,
121:                  'Script': wx.SCRIPT,
122:                  'Decorative': wx.DECORATIVE,
123:                  'Modern': wx.MODERN,
124:                  'Courier': wx.MODERN,
125:                  'courier': wx.MODERN}
126: 
127:     dashd_wx = {'solid': wx.SOLID,
128:                 'dashed': wx.SHORT_DASH,
129:                 'dashdot': wx.DOT_DASH,
130:                 'dotted': wx.DOT}
131: 
132:     # functions changes
133:     BitmapFromBuffer = wx.BitmapFromBufferRGBA
134:     EmptyBitmap = wx.EmptyBitmap
135:     EmptyImage = wx.EmptyImage
136:     Cursor = wx.StockCursor
137:     EventLoop = wx.EventLoop
138:     NamedColour = wx.NamedColour
139:     StockCursor = wx.StockCursor
140: 
141: 
142: # wxPython Classic's DoAddTool has become AddTool in Phoenix. Otherwise
143: # they are the same, except for early betas and prerelease builds of
144: # Phoenix. This function provides a shim that does the RightThing based on
145: # which wxPython is in use.
146: def _AddTool(parent, wx_ids, text, bmp, tooltip_text):
147:     if text in ['Pan', 'Zoom']:
148:         kind = wx.ITEM_CHECK
149:     else:
150:         kind = wx.ITEM_NORMAL
151:     if is_phoenix:
152:         add_tool = parent.AddTool
153:     else:
154:         add_tool = parent.DoAddTool
155: 
156:     if not is_phoenix or LooseVersion(wx.VERSION_STRING) >= str("4.0.0b2"):
157:         # NOTE: when support for Phoenix prior to 4.0.0b2 is dropped then
158:         # all that is needed is this clause, and the if and else clause can
159:         # be removed.
160:         kwargs = dict(label=text,
161:                       bitmap=bmp,
162:                       bmpDisabled=wx.NullBitmap,
163:                       shortHelp=text,
164:                       longHelp=tooltip_text,
165:                       kind=kind)
166:     else:
167:         kwargs = dict(label=text,
168:                       bitmap=bmp,
169:                       bmpDisabled=wx.NullBitmap,
170:                       shortHelpString=text,
171:                       longHelpString=tooltip_text,
172:                       kind=kind)
173: 
174:     return add_tool(wx_ids[text], **kwargs)
175: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_269494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'unicode', u'\nA wx API adapter to hide differences between wxPython classic and phoenix.\n\nIt is assumed that the user code is selecting what version it wants to use,\nhere we just ensure that it meets the minimum required by matplotlib.\n\nFor an example see embedding_in_wx2.py\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import six' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269495 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six')

if (type(import_269495) is not StypyTypeError):

    if (import_269495 != 'pyd_module'):
        __import__(import_269495)
        sys_modules_269496 = sys.modules[import_269495]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six', sys_modules_269496.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'six', import_269495)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.version import LooseVersion' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269497 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version')

if (type(import_269497) is not StypyTypeError):

    if (import_269497 != 'pyd_module'):
        __import__(import_269497)
        sys_modules_269498 = sys.modules[import_269497]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version', sys_modules_269498.module_type_store, module_type_store, ['LooseVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_269498, sys_modules_269498.module_type_store, module_type_store)
    else:
        from distutils.version import LooseVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version', import_269497)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Str to a Name (line 16):
unicode_269499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'unicode', u'Matplotlib backend_wx and backend_wxagg require wxPython >=2.8.12')
# Assigning a type to the variable 'missingwx' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'missingwx', unicode_269499)


# SSA begins for try-except statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 4))

# 'import wx' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269500 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'wx')

if (type(import_269500) is not StypyTypeError):

    if (import_269500 != 'pyd_module'):
        __import__(import_269500)
        sys_modules_269501 = sys.modules[import_269500]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'wx', sys_modules_269501.module_type_store, module_type_store)
    else:
        import wx

        import_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'wx', wx, module_type_store)

else:
    # Assigning a type to the variable 'wx' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'wx', import_269500)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Attribute to a Name (line 21):
# Getting the type of 'wx' (line 21)
wx_269502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'wx')
# Obtaining the member 'VERSION_STRING' of a type (line 21)
VERSION_STRING_269503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 22), wx_269502, 'VERSION_STRING')
# Assigning a type to the variable 'backend_version' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'backend_version', VERSION_STRING_269503)

# Assigning a Compare to a Name (line 22):

unicode_269504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'unicode', u'phoenix')
# Getting the type of 'wx' (line 22)
wx_269505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'wx')
# Obtaining the member 'PlatformInfo' of a type (line 22)
PlatformInfo_269506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 30), wx_269505, 'PlatformInfo')
# Applying the binary operator 'in' (line 22)
result_contains_269507 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 17), 'in', unicode_269504, PlatformInfo_269506)

# Assigning a type to the variable 'is_phoenix' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'is_phoenix', result_contains_269507)
# SSA branch for the except part of a try statement (line 19)
# SSA branch for the except 'ImportError' branch of a try statement (line 19)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'missingwx' (line 24)
missingwx_269509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'missingwx', False)
# Processing the call keyword arguments (line 24)
kwargs_269510 = {}
# Getting the type of 'ImportError' (line 24)
ImportError_269508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 24)
ImportError_call_result_269511 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), ImportError_269508, *[missingwx_269509], **kwargs_269510)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 24, 4), ImportError_call_result_269511, 'raise parameter', BaseException)
# SSA join for try-except statement (line 19)
module_type_store = module_type_store.join_ssa_context()




# Call to LooseVersion(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'wx' (line 27)
wx_269513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'wx', False)
# Obtaining the member 'VERSION_STRING' of a type (line 27)
VERSION_STRING_269514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), wx_269513, 'VERSION_STRING')
# Processing the call keyword arguments (line 27)
kwargs_269515 = {}
# Getting the type of 'LooseVersion' (line 27)
LooseVersion_269512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 3), 'LooseVersion', False)
# Calling LooseVersion(args, kwargs) (line 27)
LooseVersion_call_result_269516 = invoke(stypy.reporting.localization.Localization(__file__, 27, 3), LooseVersion_269512, *[VERSION_STRING_269514], **kwargs_269515)


# Call to LooseVersion(...): (line 27)
# Processing the call arguments (line 27)
unicode_269518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 50), 'unicode', u'2.8.12')
# Processing the call keyword arguments (line 27)
kwargs_269519 = {}
# Getting the type of 'LooseVersion' (line 27)
LooseVersion_269517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'LooseVersion', False)
# Calling LooseVersion(args, kwargs) (line 27)
LooseVersion_call_result_269520 = invoke(stypy.reporting.localization.Localization(__file__, 27, 37), LooseVersion_269517, *[unicode_269518], **kwargs_269519)

# Applying the binary operator '<' (line 27)
result_lt_269521 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 3), '<', LooseVersion_call_result_269516, LooseVersion_call_result_269520)

# Testing the type of an if condition (line 27)
if_condition_269522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 0), result_lt_269521)
# Assigning a type to the variable 'if_condition_269522' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'if_condition_269522', if_condition_269522)
# SSA begins for if statement (line 27)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 28)
# Processing the call arguments (line 28)
unicode_269524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'unicode', u' wxPython version %s was imported.')
# Getting the type of 'backend_version' (line 28)
backend_version_269525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 49), 'backend_version', False)
# Applying the binary operator '%' (line 28)
result_mod_269526 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 10), '%', unicode_269524, backend_version_269525)

# Processing the call keyword arguments (line 28)
kwargs_269527 = {}
# Getting the type of 'print' (line 28)
print_269523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'print', False)
# Calling print(args, kwargs) (line 28)
print_call_result_269528 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), print_269523, *[result_mod_269526], **kwargs_269527)


# Call to ImportError(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'missingwx' (line 29)
missingwx_269530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'missingwx', False)
# Processing the call keyword arguments (line 29)
kwargs_269531 = {}
# Getting the type of 'ImportError' (line 29)
ImportError_269529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 29)
ImportError_call_result_269532 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), ImportError_269529, *[missingwx_269530], **kwargs_269531)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 29, 4), ImportError_call_result_269532, 'raise parameter', BaseException)
# SSA join for if statement (line 27)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'is_phoenix' (line 31)
is_phoenix_269533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 3), 'is_phoenix')
# Testing the type of an if condition (line 31)
if_condition_269534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 0), is_phoenix_269533)
# Assigning a type to the variable 'if_condition_269534' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'if_condition_269534', if_condition_269534)
# SSA begins for if statement (line 31)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Name (line 35):

# Obtaining an instance of the builtin type 'dict' (line 35)
dict_269535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 35)
# Adding element type (key, value) (line 35)
int_269536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'int')
# Getting the type of 'wx' (line 36)
wx_269537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_LIGHT' of a type (line 36)
FONTWEIGHT_LIGHT_269538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), wx_269537, 'FONTWEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269536, FONTWEIGHT_LIGHT_269538))
# Adding element type (key, value) (line 35)
int_269539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'int')
# Getting the type of 'wx' (line 37)
wx_269540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_LIGHT' of a type (line 37)
FONTWEIGHT_LIGHT_269541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), wx_269540, 'FONTWEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269539, FONTWEIGHT_LIGHT_269541))
# Adding element type (key, value) (line 35)
int_269542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 8), 'int')
# Getting the type of 'wx' (line 38)
wx_269543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_LIGHT' of a type (line 38)
FONTWEIGHT_LIGHT_269544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), wx_269543, 'FONTWEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269542, FONTWEIGHT_LIGHT_269544))
# Adding element type (key, value) (line 35)
int_269545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'int')
# Getting the type of 'wx' (line 39)
wx_269546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_NORMAL' of a type (line 39)
FONTWEIGHT_NORMAL_269547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 13), wx_269546, 'FONTWEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269545, FONTWEIGHT_NORMAL_269547))
# Adding element type (key, value) (line 35)
int_269548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
# Getting the type of 'wx' (line 40)
wx_269549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_NORMAL' of a type (line 40)
FONTWEIGHT_NORMAL_269550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), wx_269549, 'FONTWEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269548, FONTWEIGHT_NORMAL_269550))
# Adding element type (key, value) (line 35)
int_269551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'int')
# Getting the type of 'wx' (line 41)
wx_269552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_NORMAL' of a type (line 41)
FONTWEIGHT_NORMAL_269553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 13), wx_269552, 'FONTWEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269551, FONTWEIGHT_NORMAL_269553))
# Adding element type (key, value) (line 35)
int_269554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'int')
# Getting the type of 'wx' (line 42)
wx_269555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 42)
FONTWEIGHT_BOLD_269556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), wx_269555, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269554, FONTWEIGHT_BOLD_269556))
# Adding element type (key, value) (line 35)
int_269557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'int')
# Getting the type of 'wx' (line 43)
wx_269558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 43)
FONTWEIGHT_BOLD_269559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), wx_269558, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269557, FONTWEIGHT_BOLD_269559))
# Adding element type (key, value) (line 35)
int_269560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
# Getting the type of 'wx' (line 44)
wx_269561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 44)
FONTWEIGHT_BOLD_269562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), wx_269561, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (int_269560, FONTWEIGHT_BOLD_269562))
# Adding element type (key, value) (line 35)
unicode_269563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'unicode', u'ultralight')
# Getting the type of 'wx' (line 45)
wx_269564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'wx')
# Obtaining the member 'FONTWEIGHT_LIGHT' of a type (line 45)
FONTWEIGHT_LIGHT_269565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), wx_269564, 'FONTWEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269563, FONTWEIGHT_LIGHT_269565))
# Adding element type (key, value) (line 35)
unicode_269566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'unicode', u'light')
# Getting the type of 'wx' (line 46)
wx_269567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'wx')
# Obtaining the member 'FONTWEIGHT_LIGHT' of a type (line 46)
FONTWEIGHT_LIGHT_269568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 17), wx_269567, 'FONTWEIGHT_LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269566, FONTWEIGHT_LIGHT_269568))
# Adding element type (key, value) (line 35)
unicode_269569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'unicode', u'normal')
# Getting the type of 'wx' (line 47)
wx_269570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'wx')
# Obtaining the member 'FONTWEIGHT_NORMAL' of a type (line 47)
FONTWEIGHT_NORMAL_269571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 18), wx_269570, 'FONTWEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269569, FONTWEIGHT_NORMAL_269571))
# Adding element type (key, value) (line 35)
unicode_269572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'unicode', u'medium')
# Getting the type of 'wx' (line 48)
wx_269573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'wx')
# Obtaining the member 'FONTWEIGHT_NORMAL' of a type (line 48)
FONTWEIGHT_NORMAL_269574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 18), wx_269573, 'FONTWEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269572, FONTWEIGHT_NORMAL_269574))
# Adding element type (key, value) (line 35)
unicode_269575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'unicode', u'semibold')
# Getting the type of 'wx' (line 49)
wx_269576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'wx')
# Obtaining the member 'FONTWEIGHT_NORMAL' of a type (line 49)
FONTWEIGHT_NORMAL_269577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 20), wx_269576, 'FONTWEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269575, FONTWEIGHT_NORMAL_269577))
# Adding element type (key, value) (line 35)
unicode_269578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'unicode', u'bold')
# Getting the type of 'wx' (line 50)
wx_269579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 50)
FONTWEIGHT_BOLD_269580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), wx_269579, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269578, FONTWEIGHT_BOLD_269580))
# Adding element type (key, value) (line 35)
unicode_269581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'unicode', u'heavy')
# Getting the type of 'wx' (line 51)
wx_269582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 51)
FONTWEIGHT_BOLD_269583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), wx_269582, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269581, FONTWEIGHT_BOLD_269583))
# Adding element type (key, value) (line 35)
unicode_269584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'unicode', u'ultrabold')
# Getting the type of 'wx' (line 52)
wx_269585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 52)
FONTWEIGHT_BOLD_269586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 21), wx_269585, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269584, FONTWEIGHT_BOLD_269586))
# Adding element type (key, value) (line 35)
unicode_269587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'unicode', u'black')
# Getting the type of 'wx' (line 53)
wx_269588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'wx')
# Obtaining the member 'FONTWEIGHT_BOLD' of a type (line 53)
FONTWEIGHT_BOLD_269589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), wx_269588, 'FONTWEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 18), dict_269535, (unicode_269587, FONTWEIGHT_BOLD_269589))

# Assigning a type to the variable 'fontweights' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'fontweights', dict_269535)

# Assigning a Dict to a Name (line 55):

# Obtaining an instance of the builtin type 'dict' (line 55)
dict_269590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 55)
# Adding element type (key, value) (line 55)
unicode_269591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'unicode', u'italic')
# Getting the type of 'wx' (line 56)
wx_269592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'wx')
# Obtaining the member 'FONTSTYLE_ITALIC' of a type (line 56)
FONTSTYLE_ITALIC_269593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 18), wx_269592, 'FONTSTYLE_ITALIC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), dict_269590, (unicode_269591, FONTSTYLE_ITALIC_269593))
# Adding element type (key, value) (line 55)
unicode_269594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'unicode', u'normal')
# Getting the type of 'wx' (line 57)
wx_269595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'wx')
# Obtaining the member 'FONTSTYLE_NORMAL' of a type (line 57)
FONTSTYLE_NORMAL_269596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), wx_269595, 'FONTSTYLE_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), dict_269590, (unicode_269594, FONTSTYLE_NORMAL_269596))
# Adding element type (key, value) (line 55)
unicode_269597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'unicode', u'oblique')
# Getting the type of 'wx' (line 58)
wx_269598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'wx')
# Obtaining the member 'FONTSTYLE_SLANT' of a type (line 58)
FONTSTYLE_SLANT_269599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), wx_269598, 'FONTSTYLE_SLANT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), dict_269590, (unicode_269597, FONTSTYLE_SLANT_269599))

# Assigning a type to the variable 'fontangles' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'fontangles', dict_269590)

# Assigning a Dict to a Name (line 64):

# Obtaining an instance of the builtin type 'dict' (line 64)
dict_269600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 64)
# Adding element type (key, value) (line 64)
unicode_269601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 17), 'unicode', u'Sans')
# Getting the type of 'wx' (line 64)
wx_269602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'wx')
# Obtaining the member 'FONTFAMILY_SWISS' of a type (line 64)
FONTFAMILY_SWISS_269603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 25), wx_269602, 'FONTFAMILY_SWISS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269601, FONTFAMILY_SWISS_269603))
# Adding element type (key, value) (line 64)
unicode_269604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'unicode', u'Roman')
# Getting the type of 'wx' (line 65)
wx_269605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'wx')
# Obtaining the member 'FONTFAMILY_ROMAN' of a type (line 65)
FONTFAMILY_ROMAN_269606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), wx_269605, 'FONTFAMILY_ROMAN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269604, FONTFAMILY_ROMAN_269606))
# Adding element type (key, value) (line 64)
unicode_269607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 17), 'unicode', u'Script')
# Getting the type of 'wx' (line 66)
wx_269608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'wx')
# Obtaining the member 'FONTFAMILY_SCRIPT' of a type (line 66)
FONTFAMILY_SCRIPT_269609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 27), wx_269608, 'FONTFAMILY_SCRIPT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269607, FONTFAMILY_SCRIPT_269609))
# Adding element type (key, value) (line 64)
unicode_269610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'unicode', u'Decorative')
# Getting the type of 'wx' (line 67)
wx_269611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'wx')
# Obtaining the member 'FONTFAMILY_DECORATIVE' of a type (line 67)
FONTFAMILY_DECORATIVE_269612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 31), wx_269611, 'FONTFAMILY_DECORATIVE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269610, FONTFAMILY_DECORATIVE_269612))
# Adding element type (key, value) (line 64)
unicode_269613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'unicode', u'Modern')
# Getting the type of 'wx' (line 68)
wx_269614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'wx')
# Obtaining the member 'FONTFAMILY_MODERN' of a type (line 68)
FONTFAMILY_MODERN_269615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 27), wx_269614, 'FONTFAMILY_MODERN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269613, FONTFAMILY_MODERN_269615))
# Adding element type (key, value) (line 64)
unicode_269616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 17), 'unicode', u'Courier')
# Getting the type of 'wx' (line 69)
wx_269617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'wx')
# Obtaining the member 'FONTFAMILY_MODERN' of a type (line 69)
FONTFAMILY_MODERN_269618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 28), wx_269617, 'FONTFAMILY_MODERN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269616, FONTFAMILY_MODERN_269618))
# Adding element type (key, value) (line 64)
unicode_269619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 17), 'unicode', u'courier')
# Getting the type of 'wx' (line 70)
wx_269620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'wx')
# Obtaining the member 'FONTFAMILY_MODERN' of a type (line 70)
FONTFAMILY_MODERN_269621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 28), wx_269620, 'FONTFAMILY_MODERN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), dict_269600, (unicode_269619, FONTFAMILY_MODERN_269621))

# Assigning a type to the variable 'fontnames' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'fontnames', dict_269600)

# Assigning a Dict to a Name (line 72):

# Obtaining an instance of the builtin type 'dict' (line 72)
dict_269622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 72)
# Adding element type (key, value) (line 72)
unicode_269623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'unicode', u'solid')
# Getting the type of 'wx' (line 72)
wx_269624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'wx')
# Obtaining the member 'PENSTYLE_SOLID' of a type (line 72)
PENSTYLE_SOLID_269625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), wx_269624, 'PENSTYLE_SOLID')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 15), dict_269622, (unicode_269623, PENSTYLE_SOLID_269625))
# Adding element type (key, value) (line 72)
unicode_269626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'unicode', u'dashed')
# Getting the type of 'wx' (line 73)
wx_269627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'wx')
# Obtaining the member 'PENSTYLE_SHORT_DASH' of a type (line 73)
PENSTYLE_SHORT_DASH_269628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 26), wx_269627, 'PENSTYLE_SHORT_DASH')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 15), dict_269622, (unicode_269626, PENSTYLE_SHORT_DASH_269628))
# Adding element type (key, value) (line 72)
unicode_269629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'unicode', u'dashdot')
# Getting the type of 'wx' (line 74)
wx_269630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'wx')
# Obtaining the member 'PENSTYLE_DOT_DASH' of a type (line 74)
PENSTYLE_DOT_DASH_269631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 27), wx_269630, 'PENSTYLE_DOT_DASH')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 15), dict_269622, (unicode_269629, PENSTYLE_DOT_DASH_269631))
# Adding element type (key, value) (line 72)
unicode_269632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'unicode', u'dotted')
# Getting the type of 'wx' (line 75)
wx_269633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'wx')
# Obtaining the member 'PENSTYLE_DOT' of a type (line 75)
PENSTYLE_DOT_269634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), wx_269633, 'PENSTYLE_DOT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 15), dict_269622, (unicode_269632, PENSTYLE_DOT_269634))

# Assigning a type to the variable 'dashd_wx' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'dashd_wx', dict_269622)

# Assigning a Attribute to a Name (line 78):
# Getting the type of 'wx' (line 78)
wx_269635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'wx')
# Obtaining the member 'Bitmap' of a type (line 78)
Bitmap_269636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), wx_269635, 'Bitmap')
# Obtaining the member 'FromBufferRGBA' of a type (line 78)
FromBufferRGBA_269637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), Bitmap_269636, 'FromBufferRGBA')
# Assigning a type to the variable 'BitmapFromBuffer' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'BitmapFromBuffer', FromBufferRGBA_269637)

# Assigning a Attribute to a Name (line 79):
# Getting the type of 'wx' (line 79)
wx_269638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'wx')
# Obtaining the member 'Bitmap' of a type (line 79)
Bitmap_269639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 18), wx_269638, 'Bitmap')
# Assigning a type to the variable 'EmptyBitmap' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'EmptyBitmap', Bitmap_269639)

# Assigning a Attribute to a Name (line 80):
# Getting the type of 'wx' (line 80)
wx_269640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'wx')
# Obtaining the member 'Image' of a type (line 80)
Image_269641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), wx_269640, 'Image')
# Assigning a type to the variable 'EmptyImage' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'EmptyImage', Image_269641)

# Assigning a Attribute to a Name (line 81):
# Getting the type of 'wx' (line 81)
wx_269642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'wx')
# Obtaining the member 'Cursor' of a type (line 81)
Cursor_269643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 13), wx_269642, 'Cursor')
# Assigning a type to the variable 'Cursor' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'Cursor', Cursor_269643)

# Assigning a Attribute to a Name (line 82):
# Getting the type of 'wx' (line 82)
wx_269644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'wx')
# Obtaining the member 'GUIEventLoop' of a type (line 82)
GUIEventLoop_269645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), wx_269644, 'GUIEventLoop')
# Assigning a type to the variable 'EventLoop' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'EventLoop', GUIEventLoop_269645)

# Assigning a Attribute to a Name (line 83):
# Getting the type of 'wx' (line 83)
wx_269646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'wx')
# Obtaining the member 'Colour' of a type (line 83)
Colour_269647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 18), wx_269646, 'Colour')
# Assigning a type to the variable 'NamedColour' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'NamedColour', Colour_269647)

# Assigning a Attribute to a Name (line 84):
# Getting the type of 'wx' (line 84)
wx_269648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'wx')
# Obtaining the member 'Cursor' of a type (line 84)
Cursor_269649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 18), wx_269648, 'Cursor')
# Assigning a type to the variable 'StockCursor' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'StockCursor', Cursor_269649)
# SSA branch for the else part of an if statement (line 31)
module_type_store.open_ssa_branch('else')

# Assigning a Dict to a Name (line 90):

# Obtaining an instance of the builtin type 'dict' (line 90)
dict_269650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 90)
# Adding element type (key, value) (line 90)
int_269651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
# Getting the type of 'wx' (line 91)
wx_269652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'wx')
# Obtaining the member 'LIGHT' of a type (line 91)
LIGHT_269653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), wx_269652, 'LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269651, LIGHT_269653))
# Adding element type (key, value) (line 90)
int_269654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
# Getting the type of 'wx' (line 92)
wx_269655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'wx')
# Obtaining the member 'LIGHT' of a type (line 92)
LIGHT_269656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), wx_269655, 'LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269654, LIGHT_269656))
# Adding element type (key, value) (line 90)
int_269657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
# Getting the type of 'wx' (line 93)
wx_269658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'wx')
# Obtaining the member 'LIGHT' of a type (line 93)
LIGHT_269659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 13), wx_269658, 'LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269657, LIGHT_269659))
# Adding element type (key, value) (line 90)
int_269660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
# Getting the type of 'wx' (line 94)
wx_269661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'wx')
# Obtaining the member 'NORMAL' of a type (line 94)
NORMAL_269662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 13), wx_269661, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269660, NORMAL_269662))
# Adding element type (key, value) (line 90)
int_269663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
# Getting the type of 'wx' (line 95)
wx_269664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'wx')
# Obtaining the member 'NORMAL' of a type (line 95)
NORMAL_269665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 13), wx_269664, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269663, NORMAL_269665))
# Adding element type (key, value) (line 90)
int_269666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
# Getting the type of 'wx' (line 96)
wx_269667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'wx')
# Obtaining the member 'NORMAL' of a type (line 96)
NORMAL_269668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), wx_269667, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269666, NORMAL_269668))
# Adding element type (key, value) (line 90)
int_269669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
# Getting the type of 'wx' (line 97)
wx_269670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'wx')
# Obtaining the member 'BOLD' of a type (line 97)
BOLD_269671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), wx_269670, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269669, BOLD_269671))
# Adding element type (key, value) (line 90)
int_269672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'int')
# Getting the type of 'wx' (line 98)
wx_269673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'wx')
# Obtaining the member 'BOLD' of a type (line 98)
BOLD_269674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), wx_269673, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269672, BOLD_269674))
# Adding element type (key, value) (line 90)
int_269675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'int')
# Getting the type of 'wx' (line 99)
wx_269676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'wx')
# Obtaining the member 'BOLD' of a type (line 99)
BOLD_269677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), wx_269676, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (int_269675, BOLD_269677))
# Adding element type (key, value) (line 90)
unicode_269678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'unicode', u'ultralight')
# Getting the type of 'wx' (line 100)
wx_269679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'wx')
# Obtaining the member 'LIGHT' of a type (line 100)
LIGHT_269680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 22), wx_269679, 'LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269678, LIGHT_269680))
# Adding element type (key, value) (line 90)
unicode_269681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'unicode', u'light')
# Getting the type of 'wx' (line 101)
wx_269682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'wx')
# Obtaining the member 'LIGHT' of a type (line 101)
LIGHT_269683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), wx_269682, 'LIGHT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269681, LIGHT_269683))
# Adding element type (key, value) (line 90)
unicode_269684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'unicode', u'normal')
# Getting the type of 'wx' (line 102)
wx_269685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), 'wx')
# Obtaining the member 'NORMAL' of a type (line 102)
NORMAL_269686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 18), wx_269685, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269684, NORMAL_269686))
# Adding element type (key, value) (line 90)
unicode_269687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'unicode', u'medium')
# Getting the type of 'wx' (line 103)
wx_269688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'wx')
# Obtaining the member 'NORMAL' of a type (line 103)
NORMAL_269689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 18), wx_269688, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269687, NORMAL_269689))
# Adding element type (key, value) (line 90)
unicode_269690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'unicode', u'semibold')
# Getting the type of 'wx' (line 104)
wx_269691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'wx')
# Obtaining the member 'NORMAL' of a type (line 104)
NORMAL_269692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), wx_269691, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269690, NORMAL_269692))
# Adding element type (key, value) (line 90)
unicode_269693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'unicode', u'bold')
# Getting the type of 'wx' (line 105)
wx_269694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'wx')
# Obtaining the member 'BOLD' of a type (line 105)
BOLD_269695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), wx_269694, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269693, BOLD_269695))
# Adding element type (key, value) (line 90)
unicode_269696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'unicode', u'heavy')
# Getting the type of 'wx' (line 106)
wx_269697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'wx')
# Obtaining the member 'BOLD' of a type (line 106)
BOLD_269698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 17), wx_269697, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269696, BOLD_269698))
# Adding element type (key, value) (line 90)
unicode_269699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'unicode', u'ultrabold')
# Getting the type of 'wx' (line 107)
wx_269700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'wx')
# Obtaining the member 'BOLD' of a type (line 107)
BOLD_269701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 21), wx_269700, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269699, BOLD_269701))
# Adding element type (key, value) (line 90)
unicode_269702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'unicode', u'black')
# Getting the type of 'wx' (line 108)
wx_269703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'wx')
# Obtaining the member 'BOLD' of a type (line 108)
BOLD_269704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), wx_269703, 'BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), dict_269650, (unicode_269702, BOLD_269704))

# Assigning a type to the variable 'fontweights' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'fontweights', dict_269650)

# Assigning a Dict to a Name (line 110):

# Obtaining an instance of the builtin type 'dict' (line 110)
dict_269705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 110)
# Adding element type (key, value) (line 110)
unicode_269706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 8), 'unicode', u'italic')
# Getting the type of 'wx' (line 111)
wx_269707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'wx')
# Obtaining the member 'ITALIC' of a type (line 111)
ITALIC_269708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 18), wx_269707, 'ITALIC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), dict_269705, (unicode_269706, ITALIC_269708))
# Adding element type (key, value) (line 110)
unicode_269709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'unicode', u'normal')
# Getting the type of 'wx' (line 112)
wx_269710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'wx')
# Obtaining the member 'NORMAL' of a type (line 112)
NORMAL_269711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), wx_269710, 'NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), dict_269705, (unicode_269709, NORMAL_269711))
# Adding element type (key, value) (line 110)
unicode_269712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'unicode', u'oblique')
# Getting the type of 'wx' (line 113)
wx_269713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'wx')
# Obtaining the member 'SLANT' of a type (line 113)
SLANT_269714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), wx_269713, 'SLANT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), dict_269705, (unicode_269712, SLANT_269714))

# Assigning a type to the variable 'fontangles' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'fontangles', dict_269705)

# Assigning a Dict to a Name (line 119):

# Obtaining an instance of the builtin type 'dict' (line 119)
dict_269715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 119)
# Adding element type (key, value) (line 119)
unicode_269716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 17), 'unicode', u'Sans')
# Getting the type of 'wx' (line 119)
wx_269717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'wx')
# Obtaining the member 'SWISS' of a type (line 119)
SWISS_269718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), wx_269717, 'SWISS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269716, SWISS_269718))
# Adding element type (key, value) (line 119)
unicode_269719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 17), 'unicode', u'Roman')
# Getting the type of 'wx' (line 120)
wx_269720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'wx')
# Obtaining the member 'ROMAN' of a type (line 120)
ROMAN_269721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), wx_269720, 'ROMAN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269719, ROMAN_269721))
# Adding element type (key, value) (line 119)
unicode_269722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'unicode', u'Script')
# Getting the type of 'wx' (line 121)
wx_269723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'wx')
# Obtaining the member 'SCRIPT' of a type (line 121)
SCRIPT_269724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 27), wx_269723, 'SCRIPT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269722, SCRIPT_269724))
# Adding element type (key, value) (line 119)
unicode_269725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 17), 'unicode', u'Decorative')
# Getting the type of 'wx' (line 122)
wx_269726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'wx')
# Obtaining the member 'DECORATIVE' of a type (line 122)
DECORATIVE_269727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 31), wx_269726, 'DECORATIVE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269725, DECORATIVE_269727))
# Adding element type (key, value) (line 119)
unicode_269728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'unicode', u'Modern')
# Getting the type of 'wx' (line 123)
wx_269729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 27), 'wx')
# Obtaining the member 'MODERN' of a type (line 123)
MODERN_269730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 27), wx_269729, 'MODERN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269728, MODERN_269730))
# Adding element type (key, value) (line 119)
unicode_269731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 17), 'unicode', u'Courier')
# Getting the type of 'wx' (line 124)
wx_269732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'wx')
# Obtaining the member 'MODERN' of a type (line 124)
MODERN_269733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 28), wx_269732, 'MODERN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269731, MODERN_269733))
# Adding element type (key, value) (line 119)
unicode_269734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'unicode', u'courier')
# Getting the type of 'wx' (line 125)
wx_269735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'wx')
# Obtaining the member 'MODERN' of a type (line 125)
MODERN_269736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 28), wx_269735, 'MODERN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), dict_269715, (unicode_269734, MODERN_269736))

# Assigning a type to the variable 'fontnames' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'fontnames', dict_269715)

# Assigning a Dict to a Name (line 127):

# Obtaining an instance of the builtin type 'dict' (line 127)
dict_269737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 127)
# Adding element type (key, value) (line 127)
unicode_269738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 16), 'unicode', u'solid')
# Getting the type of 'wx' (line 127)
wx_269739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'wx')
# Obtaining the member 'SOLID' of a type (line 127)
SOLID_269740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), wx_269739, 'SOLID')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), dict_269737, (unicode_269738, SOLID_269740))
# Adding element type (key, value) (line 127)
unicode_269741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 16), 'unicode', u'dashed')
# Getting the type of 'wx' (line 128)
wx_269742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'wx')
# Obtaining the member 'SHORT_DASH' of a type (line 128)
SHORT_DASH_269743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 26), wx_269742, 'SHORT_DASH')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), dict_269737, (unicode_269741, SHORT_DASH_269743))
# Adding element type (key, value) (line 127)
unicode_269744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'unicode', u'dashdot')
# Getting the type of 'wx' (line 129)
wx_269745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'wx')
# Obtaining the member 'DOT_DASH' of a type (line 129)
DOT_DASH_269746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 27), wx_269745, 'DOT_DASH')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), dict_269737, (unicode_269744, DOT_DASH_269746))
# Adding element type (key, value) (line 127)
unicode_269747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'unicode', u'dotted')
# Getting the type of 'wx' (line 130)
wx_269748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'wx')
# Obtaining the member 'DOT' of a type (line 130)
DOT_269749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 26), wx_269748, 'DOT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), dict_269737, (unicode_269747, DOT_269749))

# Assigning a type to the variable 'dashd_wx' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'dashd_wx', dict_269737)

# Assigning a Attribute to a Name (line 133):
# Getting the type of 'wx' (line 133)
wx_269750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'wx')
# Obtaining the member 'BitmapFromBufferRGBA' of a type (line 133)
BitmapFromBufferRGBA_269751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), wx_269750, 'BitmapFromBufferRGBA')
# Assigning a type to the variable 'BitmapFromBuffer' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'BitmapFromBuffer', BitmapFromBufferRGBA_269751)

# Assigning a Attribute to a Name (line 134):
# Getting the type of 'wx' (line 134)
wx_269752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'wx')
# Obtaining the member 'EmptyBitmap' of a type (line 134)
EmptyBitmap_269753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 18), wx_269752, 'EmptyBitmap')
# Assigning a type to the variable 'EmptyBitmap' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'EmptyBitmap', EmptyBitmap_269753)

# Assigning a Attribute to a Name (line 135):
# Getting the type of 'wx' (line 135)
wx_269754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'wx')
# Obtaining the member 'EmptyImage' of a type (line 135)
EmptyImage_269755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), wx_269754, 'EmptyImage')
# Assigning a type to the variable 'EmptyImage' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'EmptyImage', EmptyImage_269755)

# Assigning a Attribute to a Name (line 136):
# Getting the type of 'wx' (line 136)
wx_269756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'wx')
# Obtaining the member 'StockCursor' of a type (line 136)
StockCursor_269757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 13), wx_269756, 'StockCursor')
# Assigning a type to the variable 'Cursor' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'Cursor', StockCursor_269757)

# Assigning a Attribute to a Name (line 137):
# Getting the type of 'wx' (line 137)
wx_269758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'wx')
# Obtaining the member 'EventLoop' of a type (line 137)
EventLoop_269759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), wx_269758, 'EventLoop')
# Assigning a type to the variable 'EventLoop' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'EventLoop', EventLoop_269759)

# Assigning a Attribute to a Name (line 138):
# Getting the type of 'wx' (line 138)
wx_269760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'wx')
# Obtaining the member 'NamedColour' of a type (line 138)
NamedColour_269761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 18), wx_269760, 'NamedColour')
# Assigning a type to the variable 'NamedColour' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'NamedColour', NamedColour_269761)

# Assigning a Attribute to a Name (line 139):
# Getting the type of 'wx' (line 139)
wx_269762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'wx')
# Obtaining the member 'StockCursor' of a type (line 139)
StockCursor_269763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 18), wx_269762, 'StockCursor')
# Assigning a type to the variable 'StockCursor' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'StockCursor', StockCursor_269763)
# SSA join for if statement (line 31)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _AddTool(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_AddTool'
    module_type_store = module_type_store.open_function_context('_AddTool', 146, 0, False)
    
    # Passed parameters checking function
    _AddTool.stypy_localization = localization
    _AddTool.stypy_type_of_self = None
    _AddTool.stypy_type_store = module_type_store
    _AddTool.stypy_function_name = '_AddTool'
    _AddTool.stypy_param_names_list = ['parent', 'wx_ids', 'text', 'bmp', 'tooltip_text']
    _AddTool.stypy_varargs_param_name = None
    _AddTool.stypy_kwargs_param_name = None
    _AddTool.stypy_call_defaults = defaults
    _AddTool.stypy_call_varargs = varargs
    _AddTool.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_AddTool', ['parent', 'wx_ids', 'text', 'bmp', 'tooltip_text'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_AddTool', localization, ['parent', 'wx_ids', 'text', 'bmp', 'tooltip_text'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_AddTool(...)' code ##################

    
    
    # Getting the type of 'text' (line 147)
    text_269764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'text')
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_269765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    # Adding element type (line 147)
    unicode_269766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 16), 'unicode', u'Pan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), list_269765, unicode_269766)
    # Adding element type (line 147)
    unicode_269767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'unicode', u'Zoom')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), list_269765, unicode_269767)
    
    # Applying the binary operator 'in' (line 147)
    result_contains_269768 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 7), 'in', text_269764, list_269765)
    
    # Testing the type of an if condition (line 147)
    if_condition_269769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), result_contains_269768)
    # Assigning a type to the variable 'if_condition_269769' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_269769', if_condition_269769)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 148):
    # Getting the type of 'wx' (line 148)
    wx_269770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'wx')
    # Obtaining the member 'ITEM_CHECK' of a type (line 148)
    ITEM_CHECK_269771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), wx_269770, 'ITEM_CHECK')
    # Assigning a type to the variable 'kind' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'kind', ITEM_CHECK_269771)
    # SSA branch for the else part of an if statement (line 147)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 150):
    # Getting the type of 'wx' (line 150)
    wx_269772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'wx')
    # Obtaining the member 'ITEM_NORMAL' of a type (line 150)
    ITEM_NORMAL_269773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), wx_269772, 'ITEM_NORMAL')
    # Assigning a type to the variable 'kind' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'kind', ITEM_NORMAL_269773)
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'is_phoenix' (line 151)
    is_phoenix_269774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'is_phoenix')
    # Testing the type of an if condition (line 151)
    if_condition_269775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), is_phoenix_269774)
    # Assigning a type to the variable 'if_condition_269775' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_269775', if_condition_269775)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 152):
    # Getting the type of 'parent' (line 152)
    parent_269776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'parent')
    # Obtaining the member 'AddTool' of a type (line 152)
    AddTool_269777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 19), parent_269776, 'AddTool')
    # Assigning a type to the variable 'add_tool' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'add_tool', AddTool_269777)
    # SSA branch for the else part of an if statement (line 151)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 154):
    # Getting the type of 'parent' (line 154)
    parent_269778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'parent')
    # Obtaining the member 'DoAddTool' of a type (line 154)
    DoAddTool_269779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 19), parent_269778, 'DoAddTool')
    # Assigning a type to the variable 'add_tool' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'add_tool', DoAddTool_269779)
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'is_phoenix' (line 156)
    is_phoenix_269780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'is_phoenix')
    # Applying the 'not' unary operator (line 156)
    result_not__269781 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 7), 'not', is_phoenix_269780)
    
    
    
    # Call to LooseVersion(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'wx' (line 156)
    wx_269783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 38), 'wx', False)
    # Obtaining the member 'VERSION_STRING' of a type (line 156)
    VERSION_STRING_269784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 38), wx_269783, 'VERSION_STRING')
    # Processing the call keyword arguments (line 156)
    kwargs_269785 = {}
    # Getting the type of 'LooseVersion' (line 156)
    LooseVersion_269782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 156)
    LooseVersion_call_result_269786 = invoke(stypy.reporting.localization.Localization(__file__, 156, 25), LooseVersion_269782, *[VERSION_STRING_269784], **kwargs_269785)
    
    
    # Call to str(...): (line 156)
    # Processing the call arguments (line 156)
    unicode_269788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 64), 'unicode', u'4.0.0b2')
    # Processing the call keyword arguments (line 156)
    kwargs_269789 = {}
    # Getting the type of 'str' (line 156)
    str_269787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'str', False)
    # Calling str(args, kwargs) (line 156)
    str_call_result_269790 = invoke(stypy.reporting.localization.Localization(__file__, 156, 60), str_269787, *[unicode_269788], **kwargs_269789)
    
    # Applying the binary operator '>=' (line 156)
    result_ge_269791 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 25), '>=', LooseVersion_call_result_269786, str_call_result_269790)
    
    # Applying the binary operator 'or' (line 156)
    result_or_keyword_269792 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 7), 'or', result_not__269781, result_ge_269791)
    
    # Testing the type of an if condition (line 156)
    if_condition_269793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 4), result_or_keyword_269792)
    # Assigning a type to the variable 'if_condition_269793' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'if_condition_269793', if_condition_269793)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 160):
    
    # Call to dict(...): (line 160)
    # Processing the call keyword arguments (line 160)
    # Getting the type of 'text' (line 160)
    text_269795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'text', False)
    keyword_269796 = text_269795
    # Getting the type of 'bmp' (line 161)
    bmp_269797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'bmp', False)
    keyword_269798 = bmp_269797
    # Getting the type of 'wx' (line 162)
    wx_269799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'wx', False)
    # Obtaining the member 'NullBitmap' of a type (line 162)
    NullBitmap_269800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 34), wx_269799, 'NullBitmap')
    keyword_269801 = NullBitmap_269800
    # Getting the type of 'text' (line 163)
    text_269802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'text', False)
    keyword_269803 = text_269802
    # Getting the type of 'tooltip_text' (line 164)
    tooltip_text_269804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'tooltip_text', False)
    keyword_269805 = tooltip_text_269804
    # Getting the type of 'kind' (line 165)
    kind_269806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'kind', False)
    keyword_269807 = kind_269806
    kwargs_269808 = {'kind': keyword_269807, 'bmpDisabled': keyword_269801, 'bitmap': keyword_269798, 'longHelp': keyword_269805, 'label': keyword_269796, 'shortHelp': keyword_269803}
    # Getting the type of 'dict' (line 160)
    dict_269794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'dict', False)
    # Calling dict(args, kwargs) (line 160)
    dict_call_result_269809 = invoke(stypy.reporting.localization.Localization(__file__, 160, 17), dict_269794, *[], **kwargs_269808)
    
    # Assigning a type to the variable 'kwargs' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'kwargs', dict_call_result_269809)
    # SSA branch for the else part of an if statement (line 156)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 167):
    
    # Call to dict(...): (line 167)
    # Processing the call keyword arguments (line 167)
    # Getting the type of 'text' (line 167)
    text_269811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'text', False)
    keyword_269812 = text_269811
    # Getting the type of 'bmp' (line 168)
    bmp_269813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'bmp', False)
    keyword_269814 = bmp_269813
    # Getting the type of 'wx' (line 169)
    wx_269815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'wx', False)
    # Obtaining the member 'NullBitmap' of a type (line 169)
    NullBitmap_269816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), wx_269815, 'NullBitmap')
    keyword_269817 = NullBitmap_269816
    # Getting the type of 'text' (line 170)
    text_269818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 38), 'text', False)
    keyword_269819 = text_269818
    # Getting the type of 'tooltip_text' (line 171)
    tooltip_text_269820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'tooltip_text', False)
    keyword_269821 = tooltip_text_269820
    # Getting the type of 'kind' (line 172)
    kind_269822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'kind', False)
    keyword_269823 = kind_269822
    kwargs_269824 = {'kind': keyword_269823, 'shortHelpString': keyword_269819, 'bmpDisabled': keyword_269817, 'bitmap': keyword_269814, 'label': keyword_269812, 'longHelpString': keyword_269821}
    # Getting the type of 'dict' (line 167)
    dict_269810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'dict', False)
    # Calling dict(args, kwargs) (line 167)
    dict_call_result_269825 = invoke(stypy.reporting.localization.Localization(__file__, 167, 17), dict_269810, *[], **kwargs_269824)
    
    # Assigning a type to the variable 'kwargs' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'kwargs', dict_call_result_269825)
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add_tool(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'text' (line 174)
    text_269827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'text', False)
    # Getting the type of 'wx_ids' (line 174)
    wx_ids_269828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'wx_ids', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___269829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 20), wx_ids_269828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_269830 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), getitem___269829, text_269827)
    
    # Processing the call keyword arguments (line 174)
    # Getting the type of 'kwargs' (line 174)
    kwargs_269831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 36), 'kwargs', False)
    kwargs_269832 = {'kwargs_269831': kwargs_269831}
    # Getting the type of 'add_tool' (line 174)
    add_tool_269826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'add_tool', False)
    # Calling add_tool(args, kwargs) (line 174)
    add_tool_call_result_269833 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), add_tool_269826, *[subscript_call_result_269830], **kwargs_269832)
    
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', add_tool_call_result_269833)
    
    # ################# End of '_AddTool(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_AddTool' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_269834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_AddTool'
    return stypy_return_type_269834

# Assigning a type to the variable '_AddTool' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), '_AddTool', _AddTool)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

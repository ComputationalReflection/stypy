
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: #
3: # Copyright © 2009 Pierre Raybaut
4: # Licensed under the terms of the MIT License
5: # see the mpl licenses directory for a copy of the license
6: 
7: 
8: '''Module that provides a GUI-based editor for matplotlib's figure options'''
9: 
10: from __future__ import (absolute_import, division, print_function,
11:                         unicode_literals)
12: 
13: import six
14: 
15: import os.path as osp
16: import re
17: 
18: import matplotlib
19: from matplotlib import cm, markers, colors as mcolors
20: import matplotlib.backends.qt_editor.formlayout as formlayout
21: from matplotlib.backends.qt_compat import QtGui
22: 
23: 
24: def get_icon(name):
25:     basedir = osp.join(matplotlib.rcParams['datapath'], 'images')
26:     return QtGui.QIcon(osp.join(basedir, name))
27: 
28: 
29: LINESTYLES = {'-': 'Solid',
30:               '--': 'Dashed',
31:               '-.': 'DashDot',
32:               ':': 'Dotted',
33:               'None': 'None',
34:               }
35: 
36: DRAWSTYLES = {
37:     'default': 'Default',
38:     'steps-pre': 'Steps (Pre)', 'steps': 'Steps (Pre)',
39:     'steps-mid': 'Steps (Mid)',
40:     'steps-post': 'Steps (Post)'}
41: 
42: MARKERS = markers.MarkerStyle.markers
43: 
44: 
45: def figure_edit(axes, parent=None):
46:     '''Edit matplotlib figure options'''
47:     sep = (None, None)  # separator
48: 
49:     # Get / General
50:     # Cast to builtin floats as they have nicer reprs.
51:     xmin, xmax = map(float, axes.get_xlim())
52:     ymin, ymax = map(float, axes.get_ylim())
53:     general = [('Title', axes.get_title()),
54:                sep,
55:                (None, "<b>X-Axis</b>"),
56:                ('Left', xmin), ('Right', xmax),
57:                ('Label', axes.get_xlabel()),
58:                ('Scale', [axes.get_xscale(), 'linear', 'log', 'logit']),
59:                sep,
60:                (None, "<b>Y-Axis</b>"),
61:                ('Bottom', ymin), ('Top', ymax),
62:                ('Label', axes.get_ylabel()),
63:                ('Scale', [axes.get_yscale(), 'linear', 'log', 'logit']),
64:                sep,
65:                ('(Re-)Generate automatic legend', False),
66:                ]
67: 
68:     # Save the unit data
69:     xconverter = axes.xaxis.converter
70:     yconverter = axes.yaxis.converter
71:     xunits = axes.xaxis.get_units()
72:     yunits = axes.yaxis.get_units()
73: 
74:     # Sorting for default labels (_lineXXX, _imageXXX).
75:     def cmp_key(label):
76:         match = re.match(r"(_line|_image)(\d+)", label)
77:         if match:
78:             return match.group(1), int(match.group(2))
79:         else:
80:             return label, 0
81: 
82:     # Get / Curves
83:     linedict = {}
84:     for line in axes.get_lines():
85:         label = line.get_label()
86:         if label == '_nolegend_':
87:             continue
88:         linedict[label] = line
89:     curves = []
90: 
91:     def prepare_data(d, init):
92:         '''Prepare entry for FormLayout.
93: 
94:         `d` is a mapping of shorthands to style names (a single style may
95:         have multiple shorthands, in particular the shorthands `None`,
96:         `"None"`, `"none"` and `""` are synonyms); `init` is one shorthand
97:         of the initial style.
98: 
99:         This function returns an list suitable for initializing a
100:         FormLayout combobox, namely `[initial_name, (shorthand,
101:         style_name), (shorthand, style_name), ...]`.
102:         '''
103:         # Drop duplicate shorthands from dict (by overwriting them during
104:         # the dict comprehension).
105:         name2short = {name: short for short, name in d.items()}
106:         # Convert back to {shorthand: name}.
107:         short2name = {short: name for name, short in name2short.items()}
108:         # Find the kept shorthand for the style specified by init.
109:         canonical_init = name2short[d[init]]
110:         # Sort by representation and prepend the initial value.
111:         return ([canonical_init] +
112:                 sorted(short2name.items(),
113:                        key=lambda short_and_name: short_and_name[1]))
114: 
115:     curvelabels = sorted(linedict, key=cmp_key)
116:     for label in curvelabels:
117:         line = linedict[label]
118:         color = mcolors.to_hex(
119:             mcolors.to_rgba(line.get_color(), line.get_alpha()),
120:             keep_alpha=True)
121:         ec = mcolors.to_hex(
122:             mcolors.to_rgba(line.get_markeredgecolor(), line.get_alpha()),
123:             keep_alpha=True)
124:         fc = mcolors.to_hex(
125:             mcolors.to_rgba(line.get_markerfacecolor(), line.get_alpha()),
126:             keep_alpha=True)
127:         curvedata = [
128:             ('Label', label),
129:             sep,
130:             (None, '<b>Line</b>'),
131:             ('Line style', prepare_data(LINESTYLES, line.get_linestyle())),
132:             ('Draw style', prepare_data(DRAWSTYLES, line.get_drawstyle())),
133:             ('Width', line.get_linewidth()),
134:             ('Color (RGBA)', color),
135:             sep,
136:             (None, '<b>Marker</b>'),
137:             ('Style', prepare_data(MARKERS, line.get_marker())),
138:             ('Size', line.get_markersize()),
139:             ('Face color (RGBA)', fc),
140:             ('Edge color (RGBA)', ec)]
141:         curves.append([curvedata, label, ""])
142:     # Is there a curve displayed?
143:     has_curve = bool(curves)
144: 
145:     # Get / Images
146:     imagedict = {}
147:     for image in axes.get_images():
148:         label = image.get_label()
149:         if label == '_nolegend_':
150:             continue
151:         imagedict[label] = image
152:     imagelabels = sorted(imagedict, key=cmp_key)
153:     images = []
154:     cmaps = [(cmap, name) for name, cmap in sorted(cm.cmap_d.items())]
155:     for label in imagelabels:
156:         image = imagedict[label]
157:         cmap = image.get_cmap()
158:         if cmap not in cm.cmap_d.values():
159:             cmaps = [(cmap, cmap.name)] + cmaps
160:         low, high = image.get_clim()
161:         imagedata = [
162:             ('Label', label),
163:             ('Colormap', [cmap.name] + cmaps),
164:             ('Min. value', low),
165:             ('Max. value', high),
166:             ('Interpolation',
167:              [image.get_interpolation()]
168:              + [(name, name) for name in sorted(image.iterpnames)])]
169:         images.append([imagedata, label, ""])
170:     # Is there an image displayed?
171:     has_image = bool(images)
172: 
173:     datalist = [(general, "Axes", "")]
174:     if curves:
175:         datalist.append((curves, "Curves", ""))
176:     if images:
177:         datalist.append((images, "Images", ""))
178: 
179:     def apply_callback(data):
180:         '''This function will be called to apply changes'''
181:         orig_xlim = axes.get_xlim()
182:         orig_ylim = axes.get_ylim()
183: 
184:         general = data.pop(0)
185:         curves = data.pop(0) if has_curve else []
186:         images = data.pop(0) if has_image else []
187:         if data:
188:             raise ValueError("Unexpected field")
189: 
190:         # Set / General
191:         (title, xmin, xmax, xlabel, xscale, ymin, ymax, ylabel, yscale,
192:          generate_legend) = general
193: 
194:         if axes.get_xscale() != xscale:
195:             axes.set_xscale(xscale)
196:         if axes.get_yscale() != yscale:
197:             axes.set_yscale(yscale)
198: 
199:         axes.set_title(title)
200:         axes.set_xlim(xmin, xmax)
201:         axes.set_xlabel(xlabel)
202:         axes.set_ylim(ymin, ymax)
203:         axes.set_ylabel(ylabel)
204: 
205:         # Restore the unit data
206:         axes.xaxis.converter = xconverter
207:         axes.yaxis.converter = yconverter
208:         axes.xaxis.set_units(xunits)
209:         axes.yaxis.set_units(yunits)
210:         axes.xaxis._update_axisinfo()
211:         axes.yaxis._update_axisinfo()
212: 
213:         # Set / Curves
214:         for index, curve in enumerate(curves):
215:             line = linedict[curvelabels[index]]
216:             (label, linestyle, drawstyle, linewidth, color, marker, markersize,
217:              markerfacecolor, markeredgecolor) = curve
218:             line.set_label(label)
219:             line.set_linestyle(linestyle)
220:             line.set_drawstyle(drawstyle)
221:             line.set_linewidth(linewidth)
222:             rgba = mcolors.to_rgba(color)
223:             line.set_alpha(None)
224:             line.set_color(rgba)
225:             if marker is not 'none':
226:                 line.set_marker(marker)
227:                 line.set_markersize(markersize)
228:                 line.set_markerfacecolor(markerfacecolor)
229:                 line.set_markeredgecolor(markeredgecolor)
230: 
231:         # Set / Images
232:         for index, image_settings in enumerate(images):
233:             image = imagedict[imagelabels[index]]
234:             label, cmap, low, high, interpolation = image_settings
235:             image.set_label(label)
236:             image.set_cmap(cm.get_cmap(cmap))
237:             image.set_clim(*sorted([low, high]))
238:             image.set_interpolation(interpolation)
239: 
240:         # re-generate legend, if checkbox is checked
241:         if generate_legend:
242:             draggable = None
243:             ncol = 1
244:             if axes.legend_ is not None:
245:                 old_legend = axes.get_legend()
246:                 draggable = old_legend._draggable is not None
247:                 ncol = old_legend._ncol
248:             new_legend = axes.legend(ncol=ncol)
249:             if new_legend:
250:                 new_legend.draggable(draggable)
251: 
252:         # Redraw
253:         figure = axes.get_figure()
254:         figure.canvas.draw()
255:         if not (axes.get_xlim() == orig_xlim and axes.get_ylim() == orig_ylim):
256:             figure.canvas.toolbar.push_current()
257: 
258:     data = formlayout.fedit(datalist, title="Figure options", parent=parent,
259:                             icon=get_icon('qt4_editor_options.svg'),
260:                             apply=apply_callback)
261:     if data is not None:
262:         apply_callback(data)
263: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_270005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u"Module that provides a GUI-based editor for matplotlib's figure options")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import six' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_270006 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six')

if (type(import_270006) is not StypyTypeError):

    if (import_270006 != 'pyd_module'):
        __import__(import_270006)
        sys_modules_270007 = sys.modules[import_270006]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six', sys_modules_270007.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'six', import_270006)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import os.path' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_270008 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'os.path')

if (type(import_270008) is not StypyTypeError):

    if (import_270008 != 'pyd_module'):
        __import__(import_270008)
        sys_modules_270009 = sys.modules[import_270008]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'osp', sys_modules_270009.module_type_store, module_type_store)
    else:
        import os.path as osp

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'osp', os.path, module_type_store)

else:
    # Assigning a type to the variable 'os.path' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'os.path', import_270008)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import re' statement (line 16)
import re

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_270010 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib')

if (type(import_270010) is not StypyTypeError):

    if (import_270010 != 'pyd_module'):
        __import__(import_270010)
        sys_modules_270011 = sys.modules[import_270010]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', sys_modules_270011.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', import_270010)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib import cm, markers, mcolors' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_270012 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib')

if (type(import_270012) is not StypyTypeError):

    if (import_270012 != 'pyd_module'):
        __import__(import_270012)
        sys_modules_270013 = sys.modules[import_270012]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', sys_modules_270013.module_type_store, module_type_store, ['cm', 'markers', 'colors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_270013, sys_modules_270013.module_type_store, module_type_store)
    else:
        from matplotlib import cm, markers, colors as mcolors

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', None, module_type_store, ['cm', 'markers', 'colors'], [cm, markers, mcolors])

else:
    # Assigning a type to the variable 'matplotlib' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', import_270012)

# Adding an alias
module_type_store.add_alias('mcolors', 'colors')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import matplotlib.backends.qt_editor.formlayout' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_270014 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.backends.qt_editor.formlayout')

if (type(import_270014) is not StypyTypeError):

    if (import_270014 != 'pyd_module'):
        __import__(import_270014)
        sys_modules_270015 = sys.modules[import_270014]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'formlayout', sys_modules_270015.module_type_store, module_type_store)
    else:
        import matplotlib.backends.qt_editor.formlayout as formlayout

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'formlayout', matplotlib.backends.qt_editor.formlayout, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_editor.formlayout' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.backends.qt_editor.formlayout', import_270014)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from matplotlib.backends.qt_compat import QtGui' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_270016 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.qt_compat')

if (type(import_270016) is not StypyTypeError):

    if (import_270016 != 'pyd_module'):
        __import__(import_270016)
        sys_modules_270017 = sys.modules[import_270016]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.qt_compat', sys_modules_270017.module_type_store, module_type_store, ['QtGui'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_270017, sys_modules_270017.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_compat import QtGui

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.qt_compat', None, module_type_store, ['QtGui'], [QtGui])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_compat' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.qt_compat', import_270016)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')


@norecursion
def get_icon(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_icon'
    module_type_store = module_type_store.open_function_context('get_icon', 24, 0, False)
    
    # Passed parameters checking function
    get_icon.stypy_localization = localization
    get_icon.stypy_type_of_self = None
    get_icon.stypy_type_store = module_type_store
    get_icon.stypy_function_name = 'get_icon'
    get_icon.stypy_param_names_list = ['name']
    get_icon.stypy_varargs_param_name = None
    get_icon.stypy_kwargs_param_name = None
    get_icon.stypy_call_defaults = defaults
    get_icon.stypy_call_varargs = varargs
    get_icon.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_icon', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_icon', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_icon(...)' code ##################

    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to join(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining the type of the subscript
    unicode_270020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 43), 'unicode', u'datapath')
    # Getting the type of 'matplotlib' (line 25)
    matplotlib_270021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'matplotlib', False)
    # Obtaining the member 'rcParams' of a type (line 25)
    rcParams_270022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 23), matplotlib_270021, 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___270023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 23), rcParams_270022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_270024 = invoke(stypy.reporting.localization.Localization(__file__, 25, 23), getitem___270023, unicode_270020)
    
    unicode_270025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 56), 'unicode', u'images')
    # Processing the call keyword arguments (line 25)
    kwargs_270026 = {}
    # Getting the type of 'osp' (line 25)
    osp_270018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'osp', False)
    # Obtaining the member 'join' of a type (line 25)
    join_270019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 14), osp_270018, 'join')
    # Calling join(args, kwargs) (line 25)
    join_call_result_270027 = invoke(stypy.reporting.localization.Localization(__file__, 25, 14), join_270019, *[subscript_call_result_270024, unicode_270025], **kwargs_270026)
    
    # Assigning a type to the variable 'basedir' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'basedir', join_call_result_270027)
    
    # Call to QIcon(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to join(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'basedir' (line 26)
    basedir_270032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'basedir', False)
    # Getting the type of 'name' (line 26)
    name_270033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 41), 'name', False)
    # Processing the call keyword arguments (line 26)
    kwargs_270034 = {}
    # Getting the type of 'osp' (line 26)
    osp_270030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'osp', False)
    # Obtaining the member 'join' of a type (line 26)
    join_270031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), osp_270030, 'join')
    # Calling join(args, kwargs) (line 26)
    join_call_result_270035 = invoke(stypy.reporting.localization.Localization(__file__, 26, 23), join_270031, *[basedir_270032, name_270033], **kwargs_270034)
    
    # Processing the call keyword arguments (line 26)
    kwargs_270036 = {}
    # Getting the type of 'QtGui' (line 26)
    QtGui_270028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'QtGui', False)
    # Obtaining the member 'QIcon' of a type (line 26)
    QIcon_270029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), QtGui_270028, 'QIcon')
    # Calling QIcon(args, kwargs) (line 26)
    QIcon_call_result_270037 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), QIcon_270029, *[join_call_result_270035], **kwargs_270036)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', QIcon_call_result_270037)
    
    # ################# End of 'get_icon(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_icon' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_270038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_270038)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_icon'
    return stypy_return_type_270038

# Assigning a type to the variable 'get_icon' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'get_icon', get_icon)

# Assigning a Dict to a Name (line 29):

# Assigning a Dict to a Name (line 29):

# Obtaining an instance of the builtin type 'dict' (line 29)
dict_270039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 29)
# Adding element type (key, value) (line 29)
unicode_270040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'unicode', u'-')
unicode_270041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'unicode', u'Solid')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_270039, (unicode_270040, unicode_270041))
# Adding element type (key, value) (line 29)
unicode_270042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'unicode', u'--')
unicode_270043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'unicode', u'Dashed')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_270039, (unicode_270042, unicode_270043))
# Adding element type (key, value) (line 29)
unicode_270044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'unicode', u'-.')
unicode_270045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'unicode', u'DashDot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_270039, (unicode_270044, unicode_270045))
# Adding element type (key, value) (line 29)
unicode_270046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'unicode', u':')
unicode_270047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'unicode', u'Dotted')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_270039, (unicode_270046, unicode_270047))
# Adding element type (key, value) (line 29)
unicode_270048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 14), 'unicode', u'None')
unicode_270049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'unicode', u'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_270039, (unicode_270048, unicode_270049))

# Assigning a type to the variable 'LINESTYLES' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'LINESTYLES', dict_270039)

# Assigning a Dict to a Name (line 36):

# Assigning a Dict to a Name (line 36):

# Obtaining an instance of the builtin type 'dict' (line 36)
dict_270050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 36)
# Adding element type (key, value) (line 36)
unicode_270051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'unicode', u'default')
unicode_270052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'unicode', u'Default')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_270050, (unicode_270051, unicode_270052))
# Adding element type (key, value) (line 36)
unicode_270053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'unicode', u'steps-pre')
unicode_270054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'unicode', u'Steps (Pre)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_270050, (unicode_270053, unicode_270054))
# Adding element type (key, value) (line 36)
unicode_270055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'unicode', u'steps')
unicode_270056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'unicode', u'Steps (Pre)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_270050, (unicode_270055, unicode_270056))
# Adding element type (key, value) (line 36)
unicode_270057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'unicode', u'steps-mid')
unicode_270058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'unicode', u'Steps (Mid)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_270050, (unicode_270057, unicode_270058))
# Adding element type (key, value) (line 36)
unicode_270059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'unicode', u'steps-post')
unicode_270060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'unicode', u'Steps (Post)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_270050, (unicode_270059, unicode_270060))

# Assigning a type to the variable 'DRAWSTYLES' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'DRAWSTYLES', dict_270050)

# Assigning a Attribute to a Name (line 42):

# Assigning a Attribute to a Name (line 42):
# Getting the type of 'markers' (line 42)
markers_270061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'markers')
# Obtaining the member 'MarkerStyle' of a type (line 42)
MarkerStyle_270062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), markers_270061, 'MarkerStyle')
# Obtaining the member 'markers' of a type (line 42)
markers_270063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), MarkerStyle_270062, 'markers')
# Assigning a type to the variable 'MARKERS' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'MARKERS', markers_270063)

@norecursion
def figure_edit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 45)
    None_270064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'None')
    defaults = [None_270064]
    # Create a new context for function 'figure_edit'
    module_type_store = module_type_store.open_function_context('figure_edit', 45, 0, False)
    
    # Passed parameters checking function
    figure_edit.stypy_localization = localization
    figure_edit.stypy_type_of_self = None
    figure_edit.stypy_type_store = module_type_store
    figure_edit.stypy_function_name = 'figure_edit'
    figure_edit.stypy_param_names_list = ['axes', 'parent']
    figure_edit.stypy_varargs_param_name = None
    figure_edit.stypy_kwargs_param_name = None
    figure_edit.stypy_call_defaults = defaults
    figure_edit.stypy_call_varargs = varargs
    figure_edit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'figure_edit', ['axes', 'parent'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'figure_edit', localization, ['axes', 'parent'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'figure_edit(...)' code ##################

    unicode_270065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'unicode', u'Edit matplotlib figure options')
    
    # Assigning a Tuple to a Name (line 47):
    
    # Assigning a Tuple to a Name (line 47):
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_270066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    # Getting the type of 'None' (line 47)
    None_270067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 11), tuple_270066, None_270067)
    # Adding element type (line 47)
    # Getting the type of 'None' (line 47)
    None_270068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 11), tuple_270066, None_270068)
    
    # Assigning a type to the variable 'sep' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'sep', tuple_270066)
    
    # Assigning a Call to a Tuple (line 51):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'float' (line 51)
    float_270070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'float', False)
    
    # Call to get_xlim(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_270073 = {}
    # Getting the type of 'axes' (line 51)
    axes_270071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'axes', False)
    # Obtaining the member 'get_xlim' of a type (line 51)
    get_xlim_270072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), axes_270071, 'get_xlim')
    # Calling get_xlim(args, kwargs) (line 51)
    get_xlim_call_result_270074 = invoke(stypy.reporting.localization.Localization(__file__, 51, 28), get_xlim_270072, *[], **kwargs_270073)
    
    # Processing the call keyword arguments (line 51)
    kwargs_270075 = {}
    # Getting the type of 'map' (line 51)
    map_270069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'map', False)
    # Calling map(args, kwargs) (line 51)
    map_call_result_270076 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), map_270069, *[float_270070, get_xlim_call_result_270074], **kwargs_270075)
    
    # Assigning a type to the variable 'call_assignment_269972' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269972', map_call_result_270076)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_270079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'int')
    # Processing the call keyword arguments
    kwargs_270080 = {}
    # Getting the type of 'call_assignment_269972' (line 51)
    call_assignment_269972_270077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269972', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___270078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), call_assignment_269972_270077, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_270081 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___270078, *[int_270079], **kwargs_270080)
    
    # Assigning a type to the variable 'call_assignment_269973' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269973', getitem___call_result_270081)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'call_assignment_269973' (line 51)
    call_assignment_269973_270082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269973')
    # Assigning a type to the variable 'xmin' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'xmin', call_assignment_269973_270082)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_270085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'int')
    # Processing the call keyword arguments
    kwargs_270086 = {}
    # Getting the type of 'call_assignment_269972' (line 51)
    call_assignment_269972_270083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269972', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___270084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), call_assignment_269972_270083, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_270087 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___270084, *[int_270085], **kwargs_270086)
    
    # Assigning a type to the variable 'call_assignment_269974' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269974', getitem___call_result_270087)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'call_assignment_269974' (line 51)
    call_assignment_269974_270088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'call_assignment_269974')
    # Assigning a type to the variable 'xmax' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'xmax', call_assignment_269974_270088)
    
    # Assigning a Call to a Tuple (line 52):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'float' (line 52)
    float_270090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'float', False)
    
    # Call to get_ylim(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_270093 = {}
    # Getting the type of 'axes' (line 52)
    axes_270091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'axes', False)
    # Obtaining the member 'get_ylim' of a type (line 52)
    get_ylim_270092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), axes_270091, 'get_ylim')
    # Calling get_ylim(args, kwargs) (line 52)
    get_ylim_call_result_270094 = invoke(stypy.reporting.localization.Localization(__file__, 52, 28), get_ylim_270092, *[], **kwargs_270093)
    
    # Processing the call keyword arguments (line 52)
    kwargs_270095 = {}
    # Getting the type of 'map' (line 52)
    map_270089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'map', False)
    # Calling map(args, kwargs) (line 52)
    map_call_result_270096 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), map_270089, *[float_270090, get_ylim_call_result_270094], **kwargs_270095)
    
    # Assigning a type to the variable 'call_assignment_269975' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269975', map_call_result_270096)
    
    # Assigning a Call to a Name (line 52):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_270099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Processing the call keyword arguments
    kwargs_270100 = {}
    # Getting the type of 'call_assignment_269975' (line 52)
    call_assignment_269975_270097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269975', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___270098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), call_assignment_269975_270097, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_270101 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___270098, *[int_270099], **kwargs_270100)
    
    # Assigning a type to the variable 'call_assignment_269976' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269976', getitem___call_result_270101)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'call_assignment_269976' (line 52)
    call_assignment_269976_270102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269976')
    # Assigning a type to the variable 'ymin' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'ymin', call_assignment_269976_270102)
    
    # Assigning a Call to a Name (line 52):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_270105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Processing the call keyword arguments
    kwargs_270106 = {}
    # Getting the type of 'call_assignment_269975' (line 52)
    call_assignment_269975_270103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269975', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___270104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), call_assignment_269975_270103, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_270107 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___270104, *[int_270105], **kwargs_270106)
    
    # Assigning a type to the variable 'call_assignment_269977' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269977', getitem___call_result_270107)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'call_assignment_269977' (line 52)
    call_assignment_269977_270108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_269977')
    # Assigning a type to the variable 'ymax' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'ymax', call_assignment_269977_270108)
    
    # Assigning a List to a Name (line 53):
    
    # Assigning a List to a Name (line 53):
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_270109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_270110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    unicode_270111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'unicode', u'Title')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), tuple_270110, unicode_270111)
    # Adding element type (line 53)
    
    # Call to get_title(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_270114 = {}
    # Getting the type of 'axes' (line 53)
    axes_270112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'axes', False)
    # Obtaining the member 'get_title' of a type (line 53)
    get_title_270113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 25), axes_270112, 'get_title')
    # Calling get_title(args, kwargs) (line 53)
    get_title_call_result_270115 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), get_title_270113, *[], **kwargs_270114)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), tuple_270110, get_title_call_result_270115)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270110)
    # Adding element type (line 53)
    # Getting the type of 'sep' (line 54)
    sep_270116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, sep_270116)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_270117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    # Getting the type of 'None' (line 55)
    None_270118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 16), tuple_270117, None_270118)
    # Adding element type (line 55)
    unicode_270119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'unicode', u'<b>X-Axis</b>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 16), tuple_270117, unicode_270119)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270117)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_270120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    unicode_270121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'unicode', u'Left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 16), tuple_270120, unicode_270121)
    # Adding element type (line 56)
    # Getting the type of 'xmin' (line 56)
    xmin_270122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'xmin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 16), tuple_270120, xmin_270122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270120)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_270123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    unicode_270124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'unicode', u'Right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 32), tuple_270123, unicode_270124)
    # Adding element type (line 56)
    # Getting the type of 'xmax' (line 56)
    xmax_270125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'xmax')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 32), tuple_270123, xmax_270125)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270123)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_270126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    unicode_270127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'unicode', u'Label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), tuple_270126, unicode_270127)
    # Adding element type (line 57)
    
    # Call to get_xlabel(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_270130 = {}
    # Getting the type of 'axes' (line 57)
    axes_270128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'axes', False)
    # Obtaining the member 'get_xlabel' of a type (line 57)
    get_xlabel_270129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), axes_270128, 'get_xlabel')
    # Calling get_xlabel(args, kwargs) (line 57)
    get_xlabel_call_result_270131 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), get_xlabel_270129, *[], **kwargs_270130)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), tuple_270126, get_xlabel_call_result_270131)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270126)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_270132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    unicode_270133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'unicode', u'Scale')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), tuple_270132, unicode_270133)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_270134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    
    # Call to get_xscale(...): (line 58)
    # Processing the call keyword arguments (line 58)
    kwargs_270137 = {}
    # Getting the type of 'axes' (line 58)
    axes_270135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'axes', False)
    # Obtaining the member 'get_xscale' of a type (line 58)
    get_xscale_270136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 26), axes_270135, 'get_xscale')
    # Calling get_xscale(args, kwargs) (line 58)
    get_xscale_call_result_270138 = invoke(stypy.reporting.localization.Localization(__file__, 58, 26), get_xscale_270136, *[], **kwargs_270137)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 25), list_270134, get_xscale_call_result_270138)
    # Adding element type (line 58)
    unicode_270139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'unicode', u'linear')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 25), list_270134, unicode_270139)
    # Adding element type (line 58)
    unicode_270140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 55), 'unicode', u'log')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 25), list_270134, unicode_270140)
    # Adding element type (line 58)
    unicode_270141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 62), 'unicode', u'logit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 25), list_270134, unicode_270141)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), tuple_270132, list_270134)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270132)
    # Adding element type (line 53)
    # Getting the type of 'sep' (line 59)
    sep_270142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, sep_270142)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_270143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'None' (line 60)
    None_270144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 16), tuple_270143, None_270144)
    # Adding element type (line 60)
    unicode_270145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'unicode', u'<b>Y-Axis</b>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 16), tuple_270143, unicode_270145)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270143)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_270146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    unicode_270147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'unicode', u'Bottom')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), tuple_270146, unicode_270147)
    # Adding element type (line 61)
    # Getting the type of 'ymin' (line 61)
    ymin_270148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'ymin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), tuple_270146, ymin_270148)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270146)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_270149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    unicode_270150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 34), 'unicode', u'Top')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 34), tuple_270149, unicode_270150)
    # Adding element type (line 61)
    # Getting the type of 'ymax' (line 61)
    ymax_270151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'ymax')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 34), tuple_270149, ymax_270151)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270149)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_270152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    unicode_270153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'unicode', u'Label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), tuple_270152, unicode_270153)
    # Adding element type (line 62)
    
    # Call to get_ylabel(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_270156 = {}
    # Getting the type of 'axes' (line 62)
    axes_270154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'axes', False)
    # Obtaining the member 'get_ylabel' of a type (line 62)
    get_ylabel_270155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), axes_270154, 'get_ylabel')
    # Calling get_ylabel(args, kwargs) (line 62)
    get_ylabel_call_result_270157 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), get_ylabel_270155, *[], **kwargs_270156)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), tuple_270152, get_ylabel_call_result_270157)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270152)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_270158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    unicode_270159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'unicode', u'Scale')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 16), tuple_270158, unicode_270159)
    # Adding element type (line 63)
    
    # Obtaining an instance of the builtin type 'list' (line 63)
    list_270160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 63)
    # Adding element type (line 63)
    
    # Call to get_yscale(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_270163 = {}
    # Getting the type of 'axes' (line 63)
    axes_270161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'axes', False)
    # Obtaining the member 'get_yscale' of a type (line 63)
    get_yscale_270162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 26), axes_270161, 'get_yscale')
    # Calling get_yscale(args, kwargs) (line 63)
    get_yscale_call_result_270164 = invoke(stypy.reporting.localization.Localization(__file__, 63, 26), get_yscale_270162, *[], **kwargs_270163)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 25), list_270160, get_yscale_call_result_270164)
    # Adding element type (line 63)
    unicode_270165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 45), 'unicode', u'linear')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 25), list_270160, unicode_270165)
    # Adding element type (line 63)
    unicode_270166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'unicode', u'log')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 25), list_270160, unicode_270166)
    # Adding element type (line 63)
    unicode_270167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 62), 'unicode', u'logit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 25), list_270160, unicode_270167)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 16), tuple_270158, list_270160)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270158)
    # Adding element type (line 53)
    # Getting the type of 'sep' (line 64)
    sep_270168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, sep_270168)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_270169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    unicode_270170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 16), 'unicode', u'(Re-)Generate automatic legend')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 16), tuple_270169, unicode_270170)
    # Adding element type (line 65)
    # Getting the type of 'False' (line 65)
    False_270171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 50), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 16), tuple_270169, False_270171)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 14), list_270109, tuple_270169)
    
    # Assigning a type to the variable 'general' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'general', list_270109)
    
    # Assigning a Attribute to a Name (line 69):
    
    # Assigning a Attribute to a Name (line 69):
    # Getting the type of 'axes' (line 69)
    axes_270172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'axes')
    # Obtaining the member 'xaxis' of a type (line 69)
    xaxis_270173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), axes_270172, 'xaxis')
    # Obtaining the member 'converter' of a type (line 69)
    converter_270174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), xaxis_270173, 'converter')
    # Assigning a type to the variable 'xconverter' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'xconverter', converter_270174)
    
    # Assigning a Attribute to a Name (line 70):
    
    # Assigning a Attribute to a Name (line 70):
    # Getting the type of 'axes' (line 70)
    axes_270175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'axes')
    # Obtaining the member 'yaxis' of a type (line 70)
    yaxis_270176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), axes_270175, 'yaxis')
    # Obtaining the member 'converter' of a type (line 70)
    converter_270177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), yaxis_270176, 'converter')
    # Assigning a type to the variable 'yconverter' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'yconverter', converter_270177)
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to get_units(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_270181 = {}
    # Getting the type of 'axes' (line 71)
    axes_270178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'axes', False)
    # Obtaining the member 'xaxis' of a type (line 71)
    xaxis_270179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 13), axes_270178, 'xaxis')
    # Obtaining the member 'get_units' of a type (line 71)
    get_units_270180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 13), xaxis_270179, 'get_units')
    # Calling get_units(args, kwargs) (line 71)
    get_units_call_result_270182 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), get_units_270180, *[], **kwargs_270181)
    
    # Assigning a type to the variable 'xunits' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'xunits', get_units_call_result_270182)
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to get_units(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_270186 = {}
    # Getting the type of 'axes' (line 72)
    axes_270183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'axes', False)
    # Obtaining the member 'yaxis' of a type (line 72)
    yaxis_270184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), axes_270183, 'yaxis')
    # Obtaining the member 'get_units' of a type (line 72)
    get_units_270185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), yaxis_270184, 'get_units')
    # Calling get_units(args, kwargs) (line 72)
    get_units_call_result_270187 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), get_units_270185, *[], **kwargs_270186)
    
    # Assigning a type to the variable 'yunits' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'yunits', get_units_call_result_270187)

    @norecursion
    def cmp_key(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cmp_key'
        module_type_store = module_type_store.open_function_context('cmp_key', 75, 4, False)
        
        # Passed parameters checking function
        cmp_key.stypy_localization = localization
        cmp_key.stypy_type_of_self = None
        cmp_key.stypy_type_store = module_type_store
        cmp_key.stypy_function_name = 'cmp_key'
        cmp_key.stypy_param_names_list = ['label']
        cmp_key.stypy_varargs_param_name = None
        cmp_key.stypy_kwargs_param_name = None
        cmp_key.stypy_call_defaults = defaults
        cmp_key.stypy_call_varargs = varargs
        cmp_key.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'cmp_key', ['label'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cmp_key', localization, ['label'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cmp_key(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to match(...): (line 76)
        # Processing the call arguments (line 76)
        unicode_270190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'unicode', u'(_line|_image)(\\d+)')
        # Getting the type of 'label' (line 76)
        label_270191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'label', False)
        # Processing the call keyword arguments (line 76)
        kwargs_270192 = {}
        # Getting the type of 're' (line 76)
        re_270188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 're', False)
        # Obtaining the member 'match' of a type (line 76)
        match_270189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), re_270188, 'match')
        # Calling match(args, kwargs) (line 76)
        match_call_result_270193 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), match_270189, *[unicode_270190, label_270191], **kwargs_270192)
        
        # Assigning a type to the variable 'match' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'match', match_call_result_270193)
        
        # Getting the type of 'match' (line 77)
        match_270194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'match')
        # Testing the type of an if condition (line 77)
        if_condition_270195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), match_270194)
        # Assigning a type to the variable 'if_condition_270195' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_270195', if_condition_270195)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_270196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        
        # Call to group(...): (line 78)
        # Processing the call arguments (line 78)
        int_270199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 31), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_270200 = {}
        # Getting the type of 'match' (line 78)
        match_270197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'match', False)
        # Obtaining the member 'group' of a type (line 78)
        group_270198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), match_270197, 'group')
        # Calling group(args, kwargs) (line 78)
        group_call_result_270201 = invoke(stypy.reporting.localization.Localization(__file__, 78, 19), group_270198, *[int_270199], **kwargs_270200)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), tuple_270196, group_call_result_270201)
        # Adding element type (line 78)
        
        # Call to int(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to group(...): (line 78)
        # Processing the call arguments (line 78)
        int_270205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 51), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_270206 = {}
        # Getting the type of 'match' (line 78)
        match_270203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 39), 'match', False)
        # Obtaining the member 'group' of a type (line 78)
        group_270204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 39), match_270203, 'group')
        # Calling group(args, kwargs) (line 78)
        group_call_result_270207 = invoke(stypy.reporting.localization.Localization(__file__, 78, 39), group_270204, *[int_270205], **kwargs_270206)
        
        # Processing the call keyword arguments (line 78)
        kwargs_270208 = {}
        # Getting the type of 'int' (line 78)
        int_270202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'int', False)
        # Calling int(args, kwargs) (line 78)
        int_call_result_270209 = invoke(stypy.reporting.localization.Localization(__file__, 78, 35), int_270202, *[group_call_result_270207], **kwargs_270208)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), tuple_270196, int_call_result_270209)
        
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', tuple_270196)
        # SSA branch for the else part of an if statement (line 77)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_270210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        # Getting the type of 'label' (line 80)
        label_270211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'label')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 19), tuple_270210, label_270211)
        # Adding element type (line 80)
        int_270212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 19), tuple_270210, int_270212)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_return_type', tuple_270210)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'cmp_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cmp_key' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_270213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_270213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cmp_key'
        return stypy_return_type_270213

    # Assigning a type to the variable 'cmp_key' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'cmp_key', cmp_key)
    
    # Assigning a Dict to a Name (line 83):
    
    # Assigning a Dict to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'dict' (line 83)
    dict_270214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 83)
    
    # Assigning a type to the variable 'linedict' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'linedict', dict_270214)
    
    
    # Call to get_lines(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_270217 = {}
    # Getting the type of 'axes' (line 84)
    axes_270215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'axes', False)
    # Obtaining the member 'get_lines' of a type (line 84)
    get_lines_270216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), axes_270215, 'get_lines')
    # Calling get_lines(args, kwargs) (line 84)
    get_lines_call_result_270218 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), get_lines_270216, *[], **kwargs_270217)
    
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), get_lines_call_result_270218)
    # Getting the type of the for loop variable (line 84)
    for_loop_var_270219 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), get_lines_call_result_270218)
    # Assigning a type to the variable 'line' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'line', for_loop_var_270219)
    # SSA begins for a for statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to get_label(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_270222 = {}
    # Getting the type of 'line' (line 85)
    line_270220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'line', False)
    # Obtaining the member 'get_label' of a type (line 85)
    get_label_270221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), line_270220, 'get_label')
    # Calling get_label(args, kwargs) (line 85)
    get_label_call_result_270223 = invoke(stypy.reporting.localization.Localization(__file__, 85, 16), get_label_270221, *[], **kwargs_270222)
    
    # Assigning a type to the variable 'label' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'label', get_label_call_result_270223)
    
    
    # Getting the type of 'label' (line 86)
    label_270224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'label')
    unicode_270225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'unicode', u'_nolegend_')
    # Applying the binary operator '==' (line 86)
    result_eq_270226 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), '==', label_270224, unicode_270225)
    
    # Testing the type of an if condition (line 86)
    if_condition_270227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_270226)
    # Assigning a type to the variable 'if_condition_270227' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_270227', if_condition_270227)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 88):
    
    # Assigning a Name to a Subscript (line 88):
    # Getting the type of 'line' (line 88)
    line_270228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'line')
    # Getting the type of 'linedict' (line 88)
    linedict_270229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'linedict')
    # Getting the type of 'label' (line 88)
    label_270230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'label')
    # Storing an element on a container (line 88)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), linedict_270229, (label_270230, line_270228))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 89):
    
    # Assigning a List to a Name (line 89):
    
    # Obtaining an instance of the builtin type 'list' (line 89)
    list_270231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 89)
    
    # Assigning a type to the variable 'curves' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'curves', list_270231)

    @norecursion
    def prepare_data(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prepare_data'
        module_type_store = module_type_store.open_function_context('prepare_data', 91, 4, False)
        
        # Passed parameters checking function
        prepare_data.stypy_localization = localization
        prepare_data.stypy_type_of_self = None
        prepare_data.stypy_type_store = module_type_store
        prepare_data.stypy_function_name = 'prepare_data'
        prepare_data.stypy_param_names_list = ['d', 'init']
        prepare_data.stypy_varargs_param_name = None
        prepare_data.stypy_kwargs_param_name = None
        prepare_data.stypy_call_defaults = defaults
        prepare_data.stypy_call_varargs = varargs
        prepare_data.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'prepare_data', ['d', 'init'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prepare_data', localization, ['d', 'init'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prepare_data(...)' code ##################

        unicode_270232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'unicode', u'Prepare entry for FormLayout.\n\n        `d` is a mapping of shorthands to style names (a single style may\n        have multiple shorthands, in particular the shorthands `None`,\n        `"None"`, `"none"` and `""` are synonyms); `init` is one shorthand\n        of the initial style.\n\n        This function returns an list suitable for initializing a\n        FormLayout combobox, namely `[initial_name, (shorthand,\n        style_name), (shorthand, style_name), ...]`.\n        ')
        
        # Assigning a DictComp to a Name (line 105):
        
        # Assigning a DictComp to a Name (line 105):
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 105, 22, True)
        # Calculating comprehension expression
        
        # Call to items(...): (line 105)
        # Processing the call keyword arguments (line 105)
        kwargs_270237 = {}
        # Getting the type of 'd' (line 105)
        d_270235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 53), 'd', False)
        # Obtaining the member 'items' of a type (line 105)
        items_270236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 53), d_270235, 'items')
        # Calling items(args, kwargs) (line 105)
        items_call_result_270238 = invoke(stypy.reporting.localization.Localization(__file__, 105, 53), items_270236, *[], **kwargs_270237)
        
        comprehension_270239 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), items_call_result_270238)
        # Assigning a type to the variable 'short' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'short', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), comprehension_270239))
        # Assigning a type to the variable 'name' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), comprehension_270239))
        # Getting the type of 'name' (line 105)
        name_270233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'name')
        # Getting the type of 'short' (line 105)
        short_270234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'short')
        dict_270240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 22), dict_270240, (name_270233, short_270234))
        # Assigning a type to the variable 'name2short' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'name2short', dict_270240)
        
        # Assigning a DictComp to a Name (line 107):
        
        # Assigning a DictComp to a Name (line 107):
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 107, 22, True)
        # Calculating comprehension expression
        
        # Call to items(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_270245 = {}
        # Getting the type of 'name2short' (line 107)
        name2short_270243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 53), 'name2short', False)
        # Obtaining the member 'items' of a type (line 107)
        items_270244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 53), name2short_270243, 'items')
        # Calling items(args, kwargs) (line 107)
        items_call_result_270246 = invoke(stypy.reporting.localization.Localization(__file__, 107, 53), items_270244, *[], **kwargs_270245)
        
        comprehension_270247 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), items_call_result_270246)
        # Assigning a type to the variable 'name' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), comprehension_270247))
        # Assigning a type to the variable 'short' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'short', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), comprehension_270247))
        # Getting the type of 'short' (line 107)
        short_270241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'short')
        # Getting the type of 'name' (line 107)
        name_270242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'name')
        dict_270248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 22), dict_270248, (short_270241, name_270242))
        # Assigning a type to the variable 'short2name' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'short2name', dict_270248)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'init' (line 109)
        init_270249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 38), 'init')
        # Getting the type of 'd' (line 109)
        d_270250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 36), 'd')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___270251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 36), d_270250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_270252 = invoke(stypy.reporting.localization.Localization(__file__, 109, 36), getitem___270251, init_270249)
        
        # Getting the type of 'name2short' (line 109)
        name2short_270253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'name2short')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___270254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), name2short_270253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_270255 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), getitem___270254, subscript_call_result_270252)
        
        # Assigning a type to the variable 'canonical_init' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'canonical_init', subscript_call_result_270255)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_270256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'canonical_init' (line 111)
        canonical_init_270257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'canonical_init')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 16), list_270256, canonical_init_270257)
        
        
        # Call to sorted(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to items(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_270261 = {}
        # Getting the type of 'short2name' (line 112)
        short2name_270259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'short2name', False)
        # Obtaining the member 'items' of a type (line 112)
        items_270260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 23), short2name_270259, 'items')
        # Calling items(args, kwargs) (line 112)
        items_call_result_270262 = invoke(stypy.reporting.localization.Localization(__file__, 112, 23), items_270260, *[], **kwargs_270261)
        
        # Processing the call keyword arguments (line 112)

        @norecursion
        def _stypy_temp_lambda_114(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_114'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_114', 113, 27, True)
            # Passed parameters checking function
            _stypy_temp_lambda_114.stypy_localization = localization
            _stypy_temp_lambda_114.stypy_type_of_self = None
            _stypy_temp_lambda_114.stypy_type_store = module_type_store
            _stypy_temp_lambda_114.stypy_function_name = '_stypy_temp_lambda_114'
            _stypy_temp_lambda_114.stypy_param_names_list = ['short_and_name']
            _stypy_temp_lambda_114.stypy_varargs_param_name = None
            _stypy_temp_lambda_114.stypy_kwargs_param_name = None
            _stypy_temp_lambda_114.stypy_call_defaults = defaults
            _stypy_temp_lambda_114.stypy_call_varargs = varargs
            _stypy_temp_lambda_114.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_114', ['short_and_name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_114', ['short_and_name'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_270263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 65), 'int')
            # Getting the type of 'short_and_name' (line 113)
            short_and_name_270264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'short_and_name', False)
            # Obtaining the member '__getitem__' of a type (line 113)
            getitem___270265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 50), short_and_name_270264, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 113)
            subscript_call_result_270266 = invoke(stypy.reporting.localization.Localization(__file__, 113, 50), getitem___270265, int_270263)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'stypy_return_type', subscript_call_result_270266)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_114' in the type store
            # Getting the type of 'stypy_return_type' (line 113)
            stypy_return_type_270267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_270267)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_114'
            return stypy_return_type_270267

        # Assigning a type to the variable '_stypy_temp_lambda_114' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), '_stypy_temp_lambda_114', _stypy_temp_lambda_114)
        # Getting the type of '_stypy_temp_lambda_114' (line 113)
        _stypy_temp_lambda_114_270268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), '_stypy_temp_lambda_114')
        keyword_270269 = _stypy_temp_lambda_114_270268
        kwargs_270270 = {'key': keyword_270269}
        # Getting the type of 'sorted' (line 112)
        sorted_270258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'sorted', False)
        # Calling sorted(args, kwargs) (line 112)
        sorted_call_result_270271 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), sorted_270258, *[items_call_result_270262], **kwargs_270270)
        
        # Applying the binary operator '+' (line 111)
        result_add_270272 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '+', list_270256, sorted_call_result_270271)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', result_add_270272)
        
        # ################# End of 'prepare_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prepare_data' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_270273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_270273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prepare_data'
        return stypy_return_type_270273

    # Assigning a type to the variable 'prepare_data' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'prepare_data', prepare_data)
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to sorted(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'linedict' (line 115)
    linedict_270275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'linedict', False)
    # Processing the call keyword arguments (line 115)
    # Getting the type of 'cmp_key' (line 115)
    cmp_key_270276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 39), 'cmp_key', False)
    keyword_270277 = cmp_key_270276
    kwargs_270278 = {'key': keyword_270277}
    # Getting the type of 'sorted' (line 115)
    sorted_270274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'sorted', False)
    # Calling sorted(args, kwargs) (line 115)
    sorted_call_result_270279 = invoke(stypy.reporting.localization.Localization(__file__, 115, 18), sorted_270274, *[linedict_270275], **kwargs_270278)
    
    # Assigning a type to the variable 'curvelabels' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'curvelabels', sorted_call_result_270279)
    
    # Getting the type of 'curvelabels' (line 116)
    curvelabels_270280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'curvelabels')
    # Testing the type of a for loop iterable (line 116)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 4), curvelabels_270280)
    # Getting the type of the for loop variable (line 116)
    for_loop_var_270281 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 4), curvelabels_270280)
    # Assigning a type to the variable 'label' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'label', for_loop_var_270281)
    # SSA begins for a for statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 117):
    
    # Assigning a Subscript to a Name (line 117):
    
    # Obtaining the type of the subscript
    # Getting the type of 'label' (line 117)
    label_270282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'label')
    # Getting the type of 'linedict' (line 117)
    linedict_270283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'linedict')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___270284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), linedict_270283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_270285 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), getitem___270284, label_270282)
    
    # Assigning a type to the variable 'line' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'line', subscript_call_result_270285)
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to to_hex(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to to_rgba(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Call to get_color(...): (line 119)
    # Processing the call keyword arguments (line 119)
    kwargs_270292 = {}
    # Getting the type of 'line' (line 119)
    line_270290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'line', False)
    # Obtaining the member 'get_color' of a type (line 119)
    get_color_270291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 28), line_270290, 'get_color')
    # Calling get_color(args, kwargs) (line 119)
    get_color_call_result_270293 = invoke(stypy.reporting.localization.Localization(__file__, 119, 28), get_color_270291, *[], **kwargs_270292)
    
    
    # Call to get_alpha(...): (line 119)
    # Processing the call keyword arguments (line 119)
    kwargs_270296 = {}
    # Getting the type of 'line' (line 119)
    line_270294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 46), 'line', False)
    # Obtaining the member 'get_alpha' of a type (line 119)
    get_alpha_270295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 46), line_270294, 'get_alpha')
    # Calling get_alpha(args, kwargs) (line 119)
    get_alpha_call_result_270297 = invoke(stypy.reporting.localization.Localization(__file__, 119, 46), get_alpha_270295, *[], **kwargs_270296)
    
    # Processing the call keyword arguments (line 119)
    kwargs_270298 = {}
    # Getting the type of 'mcolors' (line 119)
    mcolors_270288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 119)
    to_rgba_270289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), mcolors_270288, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 119)
    to_rgba_call_result_270299 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), to_rgba_270289, *[get_color_call_result_270293, get_alpha_call_result_270297], **kwargs_270298)
    
    # Processing the call keyword arguments (line 118)
    # Getting the type of 'True' (line 120)
    True_270300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'True', False)
    keyword_270301 = True_270300
    kwargs_270302 = {'keep_alpha': keyword_270301}
    # Getting the type of 'mcolors' (line 118)
    mcolors_270286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'mcolors', False)
    # Obtaining the member 'to_hex' of a type (line 118)
    to_hex_270287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), mcolors_270286, 'to_hex')
    # Calling to_hex(args, kwargs) (line 118)
    to_hex_call_result_270303 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), to_hex_270287, *[to_rgba_call_result_270299], **kwargs_270302)
    
    # Assigning a type to the variable 'color' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'color', to_hex_call_result_270303)
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to to_hex(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Call to to_rgba(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to get_markeredgecolor(...): (line 122)
    # Processing the call keyword arguments (line 122)
    kwargs_270310 = {}
    # Getting the type of 'line' (line 122)
    line_270308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'line', False)
    # Obtaining the member 'get_markeredgecolor' of a type (line 122)
    get_markeredgecolor_270309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 28), line_270308, 'get_markeredgecolor')
    # Calling get_markeredgecolor(args, kwargs) (line 122)
    get_markeredgecolor_call_result_270311 = invoke(stypy.reporting.localization.Localization(__file__, 122, 28), get_markeredgecolor_270309, *[], **kwargs_270310)
    
    
    # Call to get_alpha(...): (line 122)
    # Processing the call keyword arguments (line 122)
    kwargs_270314 = {}
    # Getting the type of 'line' (line 122)
    line_270312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'line', False)
    # Obtaining the member 'get_alpha' of a type (line 122)
    get_alpha_270313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 56), line_270312, 'get_alpha')
    # Calling get_alpha(args, kwargs) (line 122)
    get_alpha_call_result_270315 = invoke(stypy.reporting.localization.Localization(__file__, 122, 56), get_alpha_270313, *[], **kwargs_270314)
    
    # Processing the call keyword arguments (line 122)
    kwargs_270316 = {}
    # Getting the type of 'mcolors' (line 122)
    mcolors_270306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 122)
    to_rgba_270307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), mcolors_270306, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 122)
    to_rgba_call_result_270317 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), to_rgba_270307, *[get_markeredgecolor_call_result_270311, get_alpha_call_result_270315], **kwargs_270316)
    
    # Processing the call keyword arguments (line 121)
    # Getting the type of 'True' (line 123)
    True_270318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'True', False)
    keyword_270319 = True_270318
    kwargs_270320 = {'keep_alpha': keyword_270319}
    # Getting the type of 'mcolors' (line 121)
    mcolors_270304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'mcolors', False)
    # Obtaining the member 'to_hex' of a type (line 121)
    to_hex_270305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 13), mcolors_270304, 'to_hex')
    # Calling to_hex(args, kwargs) (line 121)
    to_hex_call_result_270321 = invoke(stypy.reporting.localization.Localization(__file__, 121, 13), to_hex_270305, *[to_rgba_call_result_270317], **kwargs_270320)
    
    # Assigning a type to the variable 'ec' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'ec', to_hex_call_result_270321)
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to to_hex(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to to_rgba(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Call to get_markerfacecolor(...): (line 125)
    # Processing the call keyword arguments (line 125)
    kwargs_270328 = {}
    # Getting the type of 'line' (line 125)
    line_270326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'line', False)
    # Obtaining the member 'get_markerfacecolor' of a type (line 125)
    get_markerfacecolor_270327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 28), line_270326, 'get_markerfacecolor')
    # Calling get_markerfacecolor(args, kwargs) (line 125)
    get_markerfacecolor_call_result_270329 = invoke(stypy.reporting.localization.Localization(__file__, 125, 28), get_markerfacecolor_270327, *[], **kwargs_270328)
    
    
    # Call to get_alpha(...): (line 125)
    # Processing the call keyword arguments (line 125)
    kwargs_270332 = {}
    # Getting the type of 'line' (line 125)
    line_270330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 56), 'line', False)
    # Obtaining the member 'get_alpha' of a type (line 125)
    get_alpha_270331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 56), line_270330, 'get_alpha')
    # Calling get_alpha(args, kwargs) (line 125)
    get_alpha_call_result_270333 = invoke(stypy.reporting.localization.Localization(__file__, 125, 56), get_alpha_270331, *[], **kwargs_270332)
    
    # Processing the call keyword arguments (line 125)
    kwargs_270334 = {}
    # Getting the type of 'mcolors' (line 125)
    mcolors_270324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 125)
    to_rgba_270325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), mcolors_270324, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 125)
    to_rgba_call_result_270335 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), to_rgba_270325, *[get_markerfacecolor_call_result_270329, get_alpha_call_result_270333], **kwargs_270334)
    
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'True' (line 126)
    True_270336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'True', False)
    keyword_270337 = True_270336
    kwargs_270338 = {'keep_alpha': keyword_270337}
    # Getting the type of 'mcolors' (line 124)
    mcolors_270322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'mcolors', False)
    # Obtaining the member 'to_hex' of a type (line 124)
    to_hex_270323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), mcolors_270322, 'to_hex')
    # Calling to_hex(args, kwargs) (line 124)
    to_hex_call_result_270339 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), to_hex_270323, *[to_rgba_call_result_270335], **kwargs_270338)
    
    # Assigning a type to the variable 'fc' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'fc', to_hex_call_result_270339)
    
    # Assigning a List to a Name (line 127):
    
    # Assigning a List to a Name (line 127):
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_270340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_270341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    # Adding element type (line 128)
    unicode_270342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 13), 'unicode', u'Label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 13), tuple_270341, unicode_270342)
    # Adding element type (line 128)
    # Getting the type of 'label' (line 128)
    label_270343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 13), tuple_270341, label_270343)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270341)
    # Adding element type (line 127)
    # Getting the type of 'sep' (line 129)
    sep_270344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, sep_270344)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 130)
    tuple_270345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 130)
    # Adding element type (line 130)
    # Getting the type of 'None' (line 130)
    None_270346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 13), tuple_270345, None_270346)
    # Adding element type (line 130)
    unicode_270347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'unicode', u'<b>Line</b>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 13), tuple_270345, unicode_270347)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270345)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 131)
    tuple_270348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 131)
    # Adding element type (line 131)
    unicode_270349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 13), 'unicode', u'Line style')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 13), tuple_270348, unicode_270349)
    # Adding element type (line 131)
    
    # Call to prepare_data(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'LINESTYLES' (line 131)
    LINESTYLES_270351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'LINESTYLES', False)
    
    # Call to get_linestyle(...): (line 131)
    # Processing the call keyword arguments (line 131)
    kwargs_270354 = {}
    # Getting the type of 'line' (line 131)
    line_270352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 52), 'line', False)
    # Obtaining the member 'get_linestyle' of a type (line 131)
    get_linestyle_270353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 52), line_270352, 'get_linestyle')
    # Calling get_linestyle(args, kwargs) (line 131)
    get_linestyle_call_result_270355 = invoke(stypy.reporting.localization.Localization(__file__, 131, 52), get_linestyle_270353, *[], **kwargs_270354)
    
    # Processing the call keyword arguments (line 131)
    kwargs_270356 = {}
    # Getting the type of 'prepare_data' (line 131)
    prepare_data_270350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'prepare_data', False)
    # Calling prepare_data(args, kwargs) (line 131)
    prepare_data_call_result_270357 = invoke(stypy.reporting.localization.Localization(__file__, 131, 27), prepare_data_270350, *[LINESTYLES_270351, get_linestyle_call_result_270355], **kwargs_270356)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 13), tuple_270348, prepare_data_call_result_270357)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270348)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_270358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    unicode_270359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'unicode', u'Draw style')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 13), tuple_270358, unicode_270359)
    # Adding element type (line 132)
    
    # Call to prepare_data(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'DRAWSTYLES' (line 132)
    DRAWSTYLES_270361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 40), 'DRAWSTYLES', False)
    
    # Call to get_drawstyle(...): (line 132)
    # Processing the call keyword arguments (line 132)
    kwargs_270364 = {}
    # Getting the type of 'line' (line 132)
    line_270362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 52), 'line', False)
    # Obtaining the member 'get_drawstyle' of a type (line 132)
    get_drawstyle_270363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 52), line_270362, 'get_drawstyle')
    # Calling get_drawstyle(args, kwargs) (line 132)
    get_drawstyle_call_result_270365 = invoke(stypy.reporting.localization.Localization(__file__, 132, 52), get_drawstyle_270363, *[], **kwargs_270364)
    
    # Processing the call keyword arguments (line 132)
    kwargs_270366 = {}
    # Getting the type of 'prepare_data' (line 132)
    prepare_data_270360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'prepare_data', False)
    # Calling prepare_data(args, kwargs) (line 132)
    prepare_data_call_result_270367 = invoke(stypy.reporting.localization.Localization(__file__, 132, 27), prepare_data_270360, *[DRAWSTYLES_270361, get_drawstyle_call_result_270365], **kwargs_270366)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 13), tuple_270358, prepare_data_call_result_270367)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270358)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_270368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    unicode_270369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 13), 'unicode', u'Width')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 13), tuple_270368, unicode_270369)
    # Adding element type (line 133)
    
    # Call to get_linewidth(...): (line 133)
    # Processing the call keyword arguments (line 133)
    kwargs_270372 = {}
    # Getting the type of 'line' (line 133)
    line_270370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'line', False)
    # Obtaining the member 'get_linewidth' of a type (line 133)
    get_linewidth_270371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), line_270370, 'get_linewidth')
    # Calling get_linewidth(args, kwargs) (line 133)
    get_linewidth_call_result_270373 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), get_linewidth_270371, *[], **kwargs_270372)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 13), tuple_270368, get_linewidth_call_result_270373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270368)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 134)
    tuple_270374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 134)
    # Adding element type (line 134)
    unicode_270375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 13), 'unicode', u'Color (RGBA)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), tuple_270374, unicode_270375)
    # Adding element type (line 134)
    # Getting the type of 'color' (line 134)
    color_270376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'color')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), tuple_270374, color_270376)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270374)
    # Adding element type (line 127)
    # Getting the type of 'sep' (line 135)
    sep_270377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, sep_270377)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 136)
    tuple_270378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 136)
    # Adding element type (line 136)
    # Getting the type of 'None' (line 136)
    None_270379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 13), tuple_270378, None_270379)
    # Adding element type (line 136)
    unicode_270380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'unicode', u'<b>Marker</b>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 13), tuple_270378, unicode_270380)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270378)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_270381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    unicode_270382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 13), 'unicode', u'Style')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 13), tuple_270381, unicode_270382)
    # Adding element type (line 137)
    
    # Call to prepare_data(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'MARKERS' (line 137)
    MARKERS_270384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 35), 'MARKERS', False)
    
    # Call to get_marker(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_270387 = {}
    # Getting the type of 'line' (line 137)
    line_270385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), 'line', False)
    # Obtaining the member 'get_marker' of a type (line 137)
    get_marker_270386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 44), line_270385, 'get_marker')
    # Calling get_marker(args, kwargs) (line 137)
    get_marker_call_result_270388 = invoke(stypy.reporting.localization.Localization(__file__, 137, 44), get_marker_270386, *[], **kwargs_270387)
    
    # Processing the call keyword arguments (line 137)
    kwargs_270389 = {}
    # Getting the type of 'prepare_data' (line 137)
    prepare_data_270383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'prepare_data', False)
    # Calling prepare_data(args, kwargs) (line 137)
    prepare_data_call_result_270390 = invoke(stypy.reporting.localization.Localization(__file__, 137, 22), prepare_data_270383, *[MARKERS_270384, get_marker_call_result_270388], **kwargs_270389)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 13), tuple_270381, prepare_data_call_result_270390)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270381)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 138)
    tuple_270391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 138)
    # Adding element type (line 138)
    unicode_270392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 13), 'unicode', u'Size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 13), tuple_270391, unicode_270392)
    # Adding element type (line 138)
    
    # Call to get_markersize(...): (line 138)
    # Processing the call keyword arguments (line 138)
    kwargs_270395 = {}
    # Getting the type of 'line' (line 138)
    line_270393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'line', False)
    # Obtaining the member 'get_markersize' of a type (line 138)
    get_markersize_270394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), line_270393, 'get_markersize')
    # Calling get_markersize(args, kwargs) (line 138)
    get_markersize_call_result_270396 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), get_markersize_270394, *[], **kwargs_270395)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 13), tuple_270391, get_markersize_call_result_270396)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270391)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 139)
    tuple_270397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 139)
    # Adding element type (line 139)
    unicode_270398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 13), 'unicode', u'Face color (RGBA)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 13), tuple_270397, unicode_270398)
    # Adding element type (line 139)
    # Getting the type of 'fc' (line 139)
    fc_270399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'fc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 13), tuple_270397, fc_270399)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270397)
    # Adding element type (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 140)
    tuple_270400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 140)
    # Adding element type (line 140)
    unicode_270401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 13), 'unicode', u'Edge color (RGBA)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 13), tuple_270400, unicode_270401)
    # Adding element type (line 140)
    # Getting the type of 'ec' (line 140)
    ec_270402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 34), 'ec')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 13), tuple_270400, ec_270402)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), list_270340, tuple_270400)
    
    # Assigning a type to the variable 'curvedata' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'curvedata', list_270340)
    
    # Call to append(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_270405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    # Getting the type of 'curvedata' (line 141)
    curvedata_270406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'curvedata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_270405, curvedata_270406)
    # Adding element type (line 141)
    # Getting the type of 'label' (line 141)
    label_270407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'label', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_270405, label_270407)
    # Adding element type (line 141)
    unicode_270408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 41), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_270405, unicode_270408)
    
    # Processing the call keyword arguments (line 141)
    kwargs_270409 = {}
    # Getting the type of 'curves' (line 141)
    curves_270403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'curves', False)
    # Obtaining the member 'append' of a type (line 141)
    append_270404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), curves_270403, 'append')
    # Calling append(args, kwargs) (line 141)
    append_call_result_270410 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), append_270404, *[list_270405], **kwargs_270409)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to bool(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'curves' (line 143)
    curves_270412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'curves', False)
    # Processing the call keyword arguments (line 143)
    kwargs_270413 = {}
    # Getting the type of 'bool' (line 143)
    bool_270411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'bool', False)
    # Calling bool(args, kwargs) (line 143)
    bool_call_result_270414 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), bool_270411, *[curves_270412], **kwargs_270413)
    
    # Assigning a type to the variable 'has_curve' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'has_curve', bool_call_result_270414)
    
    # Assigning a Dict to a Name (line 146):
    
    # Assigning a Dict to a Name (line 146):
    
    # Obtaining an instance of the builtin type 'dict' (line 146)
    dict_270415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 146)
    
    # Assigning a type to the variable 'imagedict' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'imagedict', dict_270415)
    
    
    # Call to get_images(...): (line 147)
    # Processing the call keyword arguments (line 147)
    kwargs_270418 = {}
    # Getting the type of 'axes' (line 147)
    axes_270416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'axes', False)
    # Obtaining the member 'get_images' of a type (line 147)
    get_images_270417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 17), axes_270416, 'get_images')
    # Calling get_images(args, kwargs) (line 147)
    get_images_call_result_270419 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), get_images_270417, *[], **kwargs_270418)
    
    # Testing the type of a for loop iterable (line 147)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 4), get_images_call_result_270419)
    # Getting the type of the for loop variable (line 147)
    for_loop_var_270420 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 4), get_images_call_result_270419)
    # Assigning a type to the variable 'image' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'image', for_loop_var_270420)
    # SSA begins for a for statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to get_label(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_270423 = {}
    # Getting the type of 'image' (line 148)
    image_270421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'image', False)
    # Obtaining the member 'get_label' of a type (line 148)
    get_label_270422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), image_270421, 'get_label')
    # Calling get_label(args, kwargs) (line 148)
    get_label_call_result_270424 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), get_label_270422, *[], **kwargs_270423)
    
    # Assigning a type to the variable 'label' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'label', get_label_call_result_270424)
    
    
    # Getting the type of 'label' (line 149)
    label_270425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'label')
    unicode_270426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'unicode', u'_nolegend_')
    # Applying the binary operator '==' (line 149)
    result_eq_270427 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '==', label_270425, unicode_270426)
    
    # Testing the type of an if condition (line 149)
    if_condition_270428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_eq_270427)
    # Assigning a type to the variable 'if_condition_270428' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_270428', if_condition_270428)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 151):
    
    # Assigning a Name to a Subscript (line 151):
    # Getting the type of 'image' (line 151)
    image_270429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'image')
    # Getting the type of 'imagedict' (line 151)
    imagedict_270430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'imagedict')
    # Getting the type of 'label' (line 151)
    label_270431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'label')
    # Storing an element on a container (line 151)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 8), imagedict_270430, (label_270431, image_270429))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to sorted(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'imagedict' (line 152)
    imagedict_270433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'imagedict', False)
    # Processing the call keyword arguments (line 152)
    # Getting the type of 'cmp_key' (line 152)
    cmp_key_270434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 40), 'cmp_key', False)
    keyword_270435 = cmp_key_270434
    kwargs_270436 = {'key': keyword_270435}
    # Getting the type of 'sorted' (line 152)
    sorted_270432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'sorted', False)
    # Calling sorted(args, kwargs) (line 152)
    sorted_call_result_270437 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), sorted_270432, *[imagedict_270433], **kwargs_270436)
    
    # Assigning a type to the variable 'imagelabels' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'imagelabels', sorted_call_result_270437)
    
    # Assigning a List to a Name (line 153):
    
    # Assigning a List to a Name (line 153):
    
    # Obtaining an instance of the builtin type 'list' (line 153)
    list_270438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 153)
    
    # Assigning a type to the variable 'images' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'images', list_270438)
    
    # Assigning a ListComp to a Name (line 154):
    
    # Assigning a ListComp to a Name (line 154):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to sorted(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Call to items(...): (line 154)
    # Processing the call keyword arguments (line 154)
    kwargs_270446 = {}
    # Getting the type of 'cm' (line 154)
    cm_270443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'cm', False)
    # Obtaining the member 'cmap_d' of a type (line 154)
    cmap_d_270444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 51), cm_270443, 'cmap_d')
    # Obtaining the member 'items' of a type (line 154)
    items_270445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 51), cmap_d_270444, 'items')
    # Calling items(args, kwargs) (line 154)
    items_call_result_270447 = invoke(stypy.reporting.localization.Localization(__file__, 154, 51), items_270445, *[], **kwargs_270446)
    
    # Processing the call keyword arguments (line 154)
    kwargs_270448 = {}
    # Getting the type of 'sorted' (line 154)
    sorted_270442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 44), 'sorted', False)
    # Calling sorted(args, kwargs) (line 154)
    sorted_call_result_270449 = invoke(stypy.reporting.localization.Localization(__file__, 154, 44), sorted_270442, *[items_call_result_270447], **kwargs_270448)
    
    comprehension_270450 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), sorted_call_result_270449)
    # Assigning a type to the variable 'name' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), comprehension_270450))
    # Assigning a type to the variable 'cmap' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'cmap', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), comprehension_270450))
    
    # Obtaining an instance of the builtin type 'tuple' (line 154)
    tuple_270439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 154)
    # Adding element type (line 154)
    # Getting the type of 'cmap' (line 154)
    cmap_270440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'cmap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 14), tuple_270439, cmap_270440)
    # Adding element type (line 154)
    # Getting the type of 'name' (line 154)
    name_270441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 14), tuple_270439, name_270441)
    
    list_270451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), list_270451, tuple_270439)
    # Assigning a type to the variable 'cmaps' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'cmaps', list_270451)
    
    # Getting the type of 'imagelabels' (line 155)
    imagelabels_270452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'imagelabels')
    # Testing the type of a for loop iterable (line 155)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 4), imagelabels_270452)
    # Getting the type of the for loop variable (line 155)
    for_loop_var_270453 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 4), imagelabels_270452)
    # Assigning a type to the variable 'label' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'label', for_loop_var_270453)
    # SSA begins for a for statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 156):
    
    # Assigning a Subscript to a Name (line 156):
    
    # Obtaining the type of the subscript
    # Getting the type of 'label' (line 156)
    label_270454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'label')
    # Getting the type of 'imagedict' (line 156)
    imagedict_270455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'imagedict')
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___270456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), imagedict_270455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_270457 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), getitem___270456, label_270454)
    
    # Assigning a type to the variable 'image' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'image', subscript_call_result_270457)
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to get_cmap(...): (line 157)
    # Processing the call keyword arguments (line 157)
    kwargs_270460 = {}
    # Getting the type of 'image' (line 157)
    image_270458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'image', False)
    # Obtaining the member 'get_cmap' of a type (line 157)
    get_cmap_270459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 15), image_270458, 'get_cmap')
    # Calling get_cmap(args, kwargs) (line 157)
    get_cmap_call_result_270461 = invoke(stypy.reporting.localization.Localization(__file__, 157, 15), get_cmap_270459, *[], **kwargs_270460)
    
    # Assigning a type to the variable 'cmap' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'cmap', get_cmap_call_result_270461)
    
    
    # Getting the type of 'cmap' (line 158)
    cmap_270462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'cmap')
    
    # Call to values(...): (line 158)
    # Processing the call keyword arguments (line 158)
    kwargs_270466 = {}
    # Getting the type of 'cm' (line 158)
    cm_270463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'cm', False)
    # Obtaining the member 'cmap_d' of a type (line 158)
    cmap_d_270464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 23), cm_270463, 'cmap_d')
    # Obtaining the member 'values' of a type (line 158)
    values_270465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 23), cmap_d_270464, 'values')
    # Calling values(args, kwargs) (line 158)
    values_call_result_270467 = invoke(stypy.reporting.localization.Localization(__file__, 158, 23), values_270465, *[], **kwargs_270466)
    
    # Applying the binary operator 'notin' (line 158)
    result_contains_270468 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), 'notin', cmap_270462, values_call_result_270467)
    
    # Testing the type of an if condition (line 158)
    if_condition_270469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_contains_270468)
    # Assigning a type to the variable 'if_condition_270469' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_270469', if_condition_270469)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 159):
    
    # Assigning a BinOp to a Name (line 159):
    
    # Obtaining an instance of the builtin type 'list' (line 159)
    list_270470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 159)
    # Adding element type (line 159)
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_270471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    # Getting the type of 'cmap' (line 159)
    cmap_270472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'cmap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), tuple_270471, cmap_270472)
    # Adding element type (line 159)
    # Getting the type of 'cmap' (line 159)
    cmap_270473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'cmap')
    # Obtaining the member 'name' of a type (line 159)
    name_270474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), cmap_270473, 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), tuple_270471, name_270474)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 20), list_270470, tuple_270471)
    
    # Getting the type of 'cmaps' (line 159)
    cmaps_270475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 42), 'cmaps')
    # Applying the binary operator '+' (line 159)
    result_add_270476 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 20), '+', list_270470, cmaps_270475)
    
    # Assigning a type to the variable 'cmaps' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'cmaps', result_add_270476)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 160):
    
    # Assigning a Call to a Name:
    
    # Call to get_clim(...): (line 160)
    # Processing the call keyword arguments (line 160)
    kwargs_270479 = {}
    # Getting the type of 'image' (line 160)
    image_270477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'image', False)
    # Obtaining the member 'get_clim' of a type (line 160)
    get_clim_270478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), image_270477, 'get_clim')
    # Calling get_clim(args, kwargs) (line 160)
    get_clim_call_result_270480 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), get_clim_270478, *[], **kwargs_270479)
    
    # Assigning a type to the variable 'call_assignment_269978' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269978', get_clim_call_result_270480)
    
    # Assigning a Call to a Name (line 160):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_270483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
    # Processing the call keyword arguments
    kwargs_270484 = {}
    # Getting the type of 'call_assignment_269978' (line 160)
    call_assignment_269978_270481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269978', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___270482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), call_assignment_269978_270481, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_270485 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___270482, *[int_270483], **kwargs_270484)
    
    # Assigning a type to the variable 'call_assignment_269979' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269979', getitem___call_result_270485)
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'call_assignment_269979' (line 160)
    call_assignment_269979_270486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269979')
    # Assigning a type to the variable 'low' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'low', call_assignment_269979_270486)
    
    # Assigning a Call to a Name (line 160):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_270489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
    # Processing the call keyword arguments
    kwargs_270490 = {}
    # Getting the type of 'call_assignment_269978' (line 160)
    call_assignment_269978_270487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269978', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___270488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), call_assignment_269978_270487, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_270491 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___270488, *[int_270489], **kwargs_270490)
    
    # Assigning a type to the variable 'call_assignment_269980' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269980', getitem___call_result_270491)
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'call_assignment_269980' (line 160)
    call_assignment_269980_270492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'call_assignment_269980')
    # Assigning a type to the variable 'high' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'high', call_assignment_269980_270492)
    
    # Assigning a List to a Name (line 161):
    
    # Assigning a List to a Name (line 161):
    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_270493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_270494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    unicode_270495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 13), 'unicode', u'Label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 13), tuple_270494, unicode_270495)
    # Adding element type (line 162)
    # Getting the type of 'label' (line 162)
    label_270496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 13), tuple_270494, label_270496)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), list_270493, tuple_270494)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_270497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    unicode_270498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 13), 'unicode', u'Colormap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 13), tuple_270497, unicode_270498)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_270499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    # Getting the type of 'cmap' (line 163)
    cmap_270500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'cmap')
    # Obtaining the member 'name' of a type (line 163)
    name_270501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 26), cmap_270500, 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 25), list_270499, name_270501)
    
    # Getting the type of 'cmaps' (line 163)
    cmaps_270502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 39), 'cmaps')
    # Applying the binary operator '+' (line 163)
    result_add_270503 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 25), '+', list_270499, cmaps_270502)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 13), tuple_270497, result_add_270503)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), list_270493, tuple_270497)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_270504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    unicode_270505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 13), 'unicode', u'Min. value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), tuple_270504, unicode_270505)
    # Adding element type (line 164)
    # Getting the type of 'low' (line 164)
    low_270506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'low')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), tuple_270504, low_270506)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), list_270493, tuple_270504)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'tuple' (line 165)
    tuple_270507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 165)
    # Adding element type (line 165)
    unicode_270508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 13), 'unicode', u'Max. value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), tuple_270507, unicode_270508)
    # Adding element type (line 165)
    # Getting the type of 'high' (line 165)
    high_270509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'high')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), tuple_270507, high_270509)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), list_270493, tuple_270507)
    # Adding element type (line 161)
    
    # Obtaining an instance of the builtin type 'tuple' (line 166)
    tuple_270510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 166)
    # Adding element type (line 166)
    unicode_270511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'unicode', u'Interpolation')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), tuple_270510, unicode_270511)
    # Adding element type (line 166)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_270512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    
    # Call to get_interpolation(...): (line 167)
    # Processing the call keyword arguments (line 167)
    kwargs_270515 = {}
    # Getting the type of 'image' (line 167)
    image_270513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'image', False)
    # Obtaining the member 'get_interpolation' of a type (line 167)
    get_interpolation_270514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 14), image_270513, 'get_interpolation')
    # Calling get_interpolation(args, kwargs) (line 167)
    get_interpolation_call_result_270516 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), get_interpolation_270514, *[], **kwargs_270515)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), list_270512, get_interpolation_call_result_270516)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to sorted(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'image' (line 168)
    image_270521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 48), 'image', False)
    # Obtaining the member 'iterpnames' of a type (line 168)
    iterpnames_270522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 48), image_270521, 'iterpnames')
    # Processing the call keyword arguments (line 168)
    kwargs_270523 = {}
    # Getting the type of 'sorted' (line 168)
    sorted_270520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 41), 'sorted', False)
    # Calling sorted(args, kwargs) (line 168)
    sorted_call_result_270524 = invoke(stypy.reporting.localization.Localization(__file__, 168, 41), sorted_270520, *[iterpnames_270522], **kwargs_270523)
    
    comprehension_270525 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 16), sorted_call_result_270524)
    # Assigning a type to the variable 'name' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'name', comprehension_270525)
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_270517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    # Getting the type of 'name' (line 168)
    name_270518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 17), tuple_270517, name_270518)
    # Adding element type (line 168)
    # Getting the type of 'name' (line 168)
    name_270519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 17), tuple_270517, name_270519)
    
    list_270526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 16), list_270526, tuple_270517)
    # Applying the binary operator '+' (line 167)
    result_add_270527 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 13), '+', list_270512, list_270526)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), tuple_270510, result_add_270527)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), list_270493, tuple_270510)
    
    # Assigning a type to the variable 'imagedata' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'imagedata', list_270493)
    
    # Call to append(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining an instance of the builtin type 'list' (line 169)
    list_270530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 169)
    # Adding element type (line 169)
    # Getting the type of 'imagedata' (line 169)
    imagedata_270531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'imagedata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), list_270530, imagedata_270531)
    # Adding element type (line 169)
    # Getting the type of 'label' (line 169)
    label_270532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'label', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), list_270530, label_270532)
    # Adding element type (line 169)
    unicode_270533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 41), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), list_270530, unicode_270533)
    
    # Processing the call keyword arguments (line 169)
    kwargs_270534 = {}
    # Getting the type of 'images' (line 169)
    images_270528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'images', False)
    # Obtaining the member 'append' of a type (line 169)
    append_270529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), images_270528, 'append')
    # Calling append(args, kwargs) (line 169)
    append_call_result_270535 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), append_270529, *[list_270530], **kwargs_270534)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to bool(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'images' (line 171)
    images_270537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'images', False)
    # Processing the call keyword arguments (line 171)
    kwargs_270538 = {}
    # Getting the type of 'bool' (line 171)
    bool_270536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'bool', False)
    # Calling bool(args, kwargs) (line 171)
    bool_call_result_270539 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), bool_270536, *[images_270537], **kwargs_270538)
    
    # Assigning a type to the variable 'has_image' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'has_image', bool_call_result_270539)
    
    # Assigning a List to a Name (line 173):
    
    # Assigning a List to a Name (line 173):
    
    # Obtaining an instance of the builtin type 'list' (line 173)
    list_270540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 173)
    # Adding element type (line 173)
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_270541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    # Adding element type (line 173)
    # Getting the type of 'general' (line 173)
    general_270542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'general')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 17), tuple_270541, general_270542)
    # Adding element type (line 173)
    unicode_270543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 26), 'unicode', u'Axes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 17), tuple_270541, unicode_270543)
    # Adding element type (line 173)
    unicode_270544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 34), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 17), tuple_270541, unicode_270544)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 15), list_270540, tuple_270541)
    
    # Assigning a type to the variable 'datalist' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'datalist', list_270540)
    
    # Getting the type of 'curves' (line 174)
    curves_270545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'curves')
    # Testing the type of an if condition (line 174)
    if_condition_270546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), curves_270545)
    # Assigning a type to the variable 'if_condition_270546' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_270546', if_condition_270546)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Obtaining an instance of the builtin type 'tuple' (line 175)
    tuple_270549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 175)
    # Adding element type (line 175)
    # Getting the type of 'curves' (line 175)
    curves_270550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'curves', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 25), tuple_270549, curves_270550)
    # Adding element type (line 175)
    unicode_270551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 33), 'unicode', u'Curves')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 25), tuple_270549, unicode_270551)
    # Adding element type (line 175)
    unicode_270552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 43), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 25), tuple_270549, unicode_270552)
    
    # Processing the call keyword arguments (line 175)
    kwargs_270553 = {}
    # Getting the type of 'datalist' (line 175)
    datalist_270547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'datalist', False)
    # Obtaining the member 'append' of a type (line 175)
    append_270548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), datalist_270547, 'append')
    # Calling append(args, kwargs) (line 175)
    append_call_result_270554 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), append_270548, *[tuple_270549], **kwargs_270553)
    
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'images' (line 176)
    images_270555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'images')
    # Testing the type of an if condition (line 176)
    if_condition_270556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), images_270555)
    # Assigning a type to the variable 'if_condition_270556' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_270556', if_condition_270556)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 177)
    # Processing the call arguments (line 177)
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_270559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    # Getting the type of 'images' (line 177)
    images_270560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'images', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), tuple_270559, images_270560)
    # Adding element type (line 177)
    unicode_270561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 33), 'unicode', u'Images')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), tuple_270559, unicode_270561)
    # Adding element type (line 177)
    unicode_270562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 43), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), tuple_270559, unicode_270562)
    
    # Processing the call keyword arguments (line 177)
    kwargs_270563 = {}
    # Getting the type of 'datalist' (line 177)
    datalist_270557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'datalist', False)
    # Obtaining the member 'append' of a type (line 177)
    append_270558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), datalist_270557, 'append')
    # Calling append(args, kwargs) (line 177)
    append_call_result_270564 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), append_270558, *[tuple_270559], **kwargs_270563)
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def apply_callback(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'apply_callback'
        module_type_store = module_type_store.open_function_context('apply_callback', 179, 4, False)
        
        # Passed parameters checking function
        apply_callback.stypy_localization = localization
        apply_callback.stypy_type_of_self = None
        apply_callback.stypy_type_store = module_type_store
        apply_callback.stypy_function_name = 'apply_callback'
        apply_callback.stypy_param_names_list = ['data']
        apply_callback.stypy_varargs_param_name = None
        apply_callback.stypy_kwargs_param_name = None
        apply_callback.stypy_call_defaults = defaults
        apply_callback.stypy_call_varargs = varargs
        apply_callback.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'apply_callback', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'apply_callback', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'apply_callback(...)' code ##################

        unicode_270565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'unicode', u'This function will be called to apply changes')
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to get_xlim(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_270568 = {}
        # Getting the type of 'axes' (line 181)
        axes_270566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'axes', False)
        # Obtaining the member 'get_xlim' of a type (line 181)
        get_xlim_270567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 20), axes_270566, 'get_xlim')
        # Calling get_xlim(args, kwargs) (line 181)
        get_xlim_call_result_270569 = invoke(stypy.reporting.localization.Localization(__file__, 181, 20), get_xlim_270567, *[], **kwargs_270568)
        
        # Assigning a type to the variable 'orig_xlim' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'orig_xlim', get_xlim_call_result_270569)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to get_ylim(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_270572 = {}
        # Getting the type of 'axes' (line 182)
        axes_270570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'axes', False)
        # Obtaining the member 'get_ylim' of a type (line 182)
        get_ylim_270571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), axes_270570, 'get_ylim')
        # Calling get_ylim(args, kwargs) (line 182)
        get_ylim_call_result_270573 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), get_ylim_270571, *[], **kwargs_270572)
        
        # Assigning a type to the variable 'orig_ylim' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'orig_ylim', get_ylim_call_result_270573)
        
        # Assigning a Call to a Name (line 184):
        
        # Assigning a Call to a Name (line 184):
        
        # Call to pop(...): (line 184)
        # Processing the call arguments (line 184)
        int_270576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 27), 'int')
        # Processing the call keyword arguments (line 184)
        kwargs_270577 = {}
        # Getting the type of 'data' (line 184)
        data_270574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'data', False)
        # Obtaining the member 'pop' of a type (line 184)
        pop_270575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), data_270574, 'pop')
        # Calling pop(args, kwargs) (line 184)
        pop_call_result_270578 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), pop_270575, *[int_270576], **kwargs_270577)
        
        # Assigning a type to the variable 'general' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'general', pop_call_result_270578)
        
        # Assigning a IfExp to a Name (line 185):
        
        # Assigning a IfExp to a Name (line 185):
        
        # Getting the type of 'has_curve' (line 185)
        has_curve_270579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'has_curve')
        # Testing the type of an if expression (line 185)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 17), has_curve_270579)
        # SSA begins for if expression (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to pop(...): (line 185)
        # Processing the call arguments (line 185)
        int_270582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 26), 'int')
        # Processing the call keyword arguments (line 185)
        kwargs_270583 = {}
        # Getting the type of 'data' (line 185)
        data_270580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'data', False)
        # Obtaining the member 'pop' of a type (line 185)
        pop_270581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), data_270580, 'pop')
        # Calling pop(args, kwargs) (line 185)
        pop_call_result_270584 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), pop_270581, *[int_270582], **kwargs_270583)
        
        # SSA branch for the else part of an if expression (line 185)
        module_type_store.open_ssa_branch('if expression else')
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_270585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        
        # SSA join for if expression (line 185)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_270586 = union_type.UnionType.add(pop_call_result_270584, list_270585)
        
        # Assigning a type to the variable 'curves' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'curves', if_exp_270586)
        
        # Assigning a IfExp to a Name (line 186):
        
        # Assigning a IfExp to a Name (line 186):
        
        # Getting the type of 'has_image' (line 186)
        has_image_270587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 32), 'has_image')
        # Testing the type of an if expression (line 186)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 17), has_image_270587)
        # SSA begins for if expression (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to pop(...): (line 186)
        # Processing the call arguments (line 186)
        int_270590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 26), 'int')
        # Processing the call keyword arguments (line 186)
        kwargs_270591 = {}
        # Getting the type of 'data' (line 186)
        data_270588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'data', False)
        # Obtaining the member 'pop' of a type (line 186)
        pop_270589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 17), data_270588, 'pop')
        # Calling pop(args, kwargs) (line 186)
        pop_call_result_270592 = invoke(stypy.reporting.localization.Localization(__file__, 186, 17), pop_270589, *[int_270590], **kwargs_270591)
        
        # SSA branch for the else part of an if expression (line 186)
        module_type_store.open_ssa_branch('if expression else')
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_270593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        
        # SSA join for if expression (line 186)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_270594 = union_type.UnionType.add(pop_call_result_270592, list_270593)
        
        # Assigning a type to the variable 'images' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'images', if_exp_270594)
        
        # Getting the type of 'data' (line 187)
        data_270595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'data')
        # Testing the type of an if condition (line 187)
        if_condition_270596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), data_270595)
        # Assigning a type to the variable 'if_condition_270596' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_270596', if_condition_270596)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 188)
        # Processing the call arguments (line 188)
        unicode_270598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'unicode', u'Unexpected field')
        # Processing the call keyword arguments (line 188)
        kwargs_270599 = {}
        # Getting the type of 'ValueError' (line 188)
        ValueError_270597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 188)
        ValueError_call_result_270600 = invoke(stypy.reporting.localization.Localization(__file__, 188, 18), ValueError_270597, *[unicode_270598], **kwargs_270599)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 188, 12), ValueError_call_result_270600, 'raise parameter', BaseException)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 191):
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270604 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270603, int_270601)
        
        # Assigning a type to the variable 'tuple_var_assignment_269981' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269981', subscript_call_result_270604)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270608 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270607, int_270605)
        
        # Assigning a type to the variable 'tuple_var_assignment_269982' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269982', subscript_call_result_270608)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270610, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270612 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270611, int_270609)
        
        # Assigning a type to the variable 'tuple_var_assignment_269983' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269983', subscript_call_result_270612)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270614, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270616 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270615, int_270613)
        
        # Assigning a type to the variable 'tuple_var_assignment_269984' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269984', subscript_call_result_270616)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270620 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270619, int_270617)
        
        # Assigning a type to the variable 'tuple_var_assignment_269985' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269985', subscript_call_result_270620)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270624 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270623, int_270621)
        
        # Assigning a type to the variable 'tuple_var_assignment_269986' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269986', subscript_call_result_270624)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270626, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270628 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270627, int_270625)
        
        # Assigning a type to the variable 'tuple_var_assignment_269987' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269987', subscript_call_result_270628)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270632 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270631, int_270629)
        
        # Assigning a type to the variable 'tuple_var_assignment_269988' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269988', subscript_call_result_270632)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270634, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270636 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270635, int_270633)
        
        # Assigning a type to the variable 'tuple_var_assignment_269989' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269989', subscript_call_result_270636)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        int_270637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'int')
        # Getting the type of 'general' (line 192)
        general_270638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'general')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___270639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), general_270638, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_270640 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), getitem___270639, int_270637)
        
        # Assigning a type to the variable 'tuple_var_assignment_269990' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269990', subscript_call_result_270640)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269981' (line 191)
        tuple_var_assignment_269981_270641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269981')
        # Assigning a type to the variable 'title' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'title', tuple_var_assignment_269981_270641)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269982' (line 191)
        tuple_var_assignment_269982_270642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269982')
        # Assigning a type to the variable 'xmin' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'xmin', tuple_var_assignment_269982_270642)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269983' (line 191)
        tuple_var_assignment_269983_270643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269983')
        # Assigning a type to the variable 'xmax' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'xmax', tuple_var_assignment_269983_270643)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269984' (line 191)
        tuple_var_assignment_269984_270644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269984')
        # Assigning a type to the variable 'xlabel' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'xlabel', tuple_var_assignment_269984_270644)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269985' (line 191)
        tuple_var_assignment_269985_270645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269985')
        # Assigning a type to the variable 'xscale' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 'xscale', tuple_var_assignment_269985_270645)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269986' (line 191)
        tuple_var_assignment_269986_270646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269986')
        # Assigning a type to the variable 'ymin' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 44), 'ymin', tuple_var_assignment_269986_270646)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269987' (line 191)
        tuple_var_assignment_269987_270647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269987')
        # Assigning a type to the variable 'ymax' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 50), 'ymax', tuple_var_assignment_269987_270647)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269988' (line 191)
        tuple_var_assignment_269988_270648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269988')
        # Assigning a type to the variable 'ylabel' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 56), 'ylabel', tuple_var_assignment_269988_270648)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269989' (line 191)
        tuple_var_assignment_269989_270649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269989')
        # Assigning a type to the variable 'yscale' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 64), 'yscale', tuple_var_assignment_269989_270649)
        
        # Assigning a Name to a Name (line 191):
        # Getting the type of 'tuple_var_assignment_269990' (line 191)
        tuple_var_assignment_269990_270650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'tuple_var_assignment_269990')
        # Assigning a type to the variable 'generate_legend' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 9), 'generate_legend', tuple_var_assignment_269990_270650)
        
        
        
        # Call to get_xscale(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_270653 = {}
        # Getting the type of 'axes' (line 194)
        axes_270651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'axes', False)
        # Obtaining the member 'get_xscale' of a type (line 194)
        get_xscale_270652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), axes_270651, 'get_xscale')
        # Calling get_xscale(args, kwargs) (line 194)
        get_xscale_call_result_270654 = invoke(stypy.reporting.localization.Localization(__file__, 194, 11), get_xscale_270652, *[], **kwargs_270653)
        
        # Getting the type of 'xscale' (line 194)
        xscale_270655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 32), 'xscale')
        # Applying the binary operator '!=' (line 194)
        result_ne_270656 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), '!=', get_xscale_call_result_270654, xscale_270655)
        
        # Testing the type of an if condition (line 194)
        if_condition_270657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_ne_270656)
        # Assigning a type to the variable 'if_condition_270657' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_270657', if_condition_270657)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_xscale(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'xscale' (line 195)
        xscale_270660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'xscale', False)
        # Processing the call keyword arguments (line 195)
        kwargs_270661 = {}
        # Getting the type of 'axes' (line 195)
        axes_270658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'axes', False)
        # Obtaining the member 'set_xscale' of a type (line 195)
        set_xscale_270659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), axes_270658, 'set_xscale')
        # Calling set_xscale(args, kwargs) (line 195)
        set_xscale_call_result_270662 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), set_xscale_270659, *[xscale_270660], **kwargs_270661)
        
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to get_yscale(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_270665 = {}
        # Getting the type of 'axes' (line 196)
        axes_270663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'axes', False)
        # Obtaining the member 'get_yscale' of a type (line 196)
        get_yscale_270664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 11), axes_270663, 'get_yscale')
        # Calling get_yscale(args, kwargs) (line 196)
        get_yscale_call_result_270666 = invoke(stypy.reporting.localization.Localization(__file__, 196, 11), get_yscale_270664, *[], **kwargs_270665)
        
        # Getting the type of 'yscale' (line 196)
        yscale_270667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 32), 'yscale')
        # Applying the binary operator '!=' (line 196)
        result_ne_270668 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), '!=', get_yscale_call_result_270666, yscale_270667)
        
        # Testing the type of an if condition (line 196)
        if_condition_270669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), result_ne_270668)
        # Assigning a type to the variable 'if_condition_270669' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_270669', if_condition_270669)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_yscale(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'yscale' (line 197)
        yscale_270672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'yscale', False)
        # Processing the call keyword arguments (line 197)
        kwargs_270673 = {}
        # Getting the type of 'axes' (line 197)
        axes_270670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'axes', False)
        # Obtaining the member 'set_yscale' of a type (line 197)
        set_yscale_270671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), axes_270670, 'set_yscale')
        # Calling set_yscale(args, kwargs) (line 197)
        set_yscale_call_result_270674 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), set_yscale_270671, *[yscale_270672], **kwargs_270673)
        
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_title(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'title' (line 199)
        title_270677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'title', False)
        # Processing the call keyword arguments (line 199)
        kwargs_270678 = {}
        # Getting the type of 'axes' (line 199)
        axes_270675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'axes', False)
        # Obtaining the member 'set_title' of a type (line 199)
        set_title_270676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), axes_270675, 'set_title')
        # Calling set_title(args, kwargs) (line 199)
        set_title_call_result_270679 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), set_title_270676, *[title_270677], **kwargs_270678)
        
        
        # Call to set_xlim(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'xmin' (line 200)
        xmin_270682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'xmin', False)
        # Getting the type of 'xmax' (line 200)
        xmax_270683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'xmax', False)
        # Processing the call keyword arguments (line 200)
        kwargs_270684 = {}
        # Getting the type of 'axes' (line 200)
        axes_270680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'axes', False)
        # Obtaining the member 'set_xlim' of a type (line 200)
        set_xlim_270681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), axes_270680, 'set_xlim')
        # Calling set_xlim(args, kwargs) (line 200)
        set_xlim_call_result_270685 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), set_xlim_270681, *[xmin_270682, xmax_270683], **kwargs_270684)
        
        
        # Call to set_xlabel(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'xlabel' (line 201)
        xlabel_270688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'xlabel', False)
        # Processing the call keyword arguments (line 201)
        kwargs_270689 = {}
        # Getting the type of 'axes' (line 201)
        axes_270686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'axes', False)
        # Obtaining the member 'set_xlabel' of a type (line 201)
        set_xlabel_270687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), axes_270686, 'set_xlabel')
        # Calling set_xlabel(args, kwargs) (line 201)
        set_xlabel_call_result_270690 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), set_xlabel_270687, *[xlabel_270688], **kwargs_270689)
        
        
        # Call to set_ylim(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'ymin' (line 202)
        ymin_270693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'ymin', False)
        # Getting the type of 'ymax' (line 202)
        ymax_270694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'ymax', False)
        # Processing the call keyword arguments (line 202)
        kwargs_270695 = {}
        # Getting the type of 'axes' (line 202)
        axes_270691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'axes', False)
        # Obtaining the member 'set_ylim' of a type (line 202)
        set_ylim_270692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), axes_270691, 'set_ylim')
        # Calling set_ylim(args, kwargs) (line 202)
        set_ylim_call_result_270696 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), set_ylim_270692, *[ymin_270693, ymax_270694], **kwargs_270695)
        
        
        # Call to set_ylabel(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'ylabel' (line 203)
        ylabel_270699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'ylabel', False)
        # Processing the call keyword arguments (line 203)
        kwargs_270700 = {}
        # Getting the type of 'axes' (line 203)
        axes_270697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'axes', False)
        # Obtaining the member 'set_ylabel' of a type (line 203)
        set_ylabel_270698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), axes_270697, 'set_ylabel')
        # Calling set_ylabel(args, kwargs) (line 203)
        set_ylabel_call_result_270701 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), set_ylabel_270698, *[ylabel_270699], **kwargs_270700)
        
        
        # Assigning a Name to a Attribute (line 206):
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'xconverter' (line 206)
        xconverter_270702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'xconverter')
        # Getting the type of 'axes' (line 206)
        axes_270703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'axes')
        # Obtaining the member 'xaxis' of a type (line 206)
        xaxis_270704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), axes_270703, 'xaxis')
        # Setting the type of the member 'converter' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), xaxis_270704, 'converter', xconverter_270702)
        
        # Assigning a Name to a Attribute (line 207):
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'yconverter' (line 207)
        yconverter_270705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'yconverter')
        # Getting the type of 'axes' (line 207)
        axes_270706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'axes')
        # Obtaining the member 'yaxis' of a type (line 207)
        yaxis_270707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), axes_270706, 'yaxis')
        # Setting the type of the member 'converter' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), yaxis_270707, 'converter', yconverter_270705)
        
        # Call to set_units(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'xunits' (line 208)
        xunits_270711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'xunits', False)
        # Processing the call keyword arguments (line 208)
        kwargs_270712 = {}
        # Getting the type of 'axes' (line 208)
        axes_270708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'axes', False)
        # Obtaining the member 'xaxis' of a type (line 208)
        xaxis_270709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), axes_270708, 'xaxis')
        # Obtaining the member 'set_units' of a type (line 208)
        set_units_270710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), xaxis_270709, 'set_units')
        # Calling set_units(args, kwargs) (line 208)
        set_units_call_result_270713 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), set_units_270710, *[xunits_270711], **kwargs_270712)
        
        
        # Call to set_units(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'yunits' (line 209)
        yunits_270717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'yunits', False)
        # Processing the call keyword arguments (line 209)
        kwargs_270718 = {}
        # Getting the type of 'axes' (line 209)
        axes_270714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'axes', False)
        # Obtaining the member 'yaxis' of a type (line 209)
        yaxis_270715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), axes_270714, 'yaxis')
        # Obtaining the member 'set_units' of a type (line 209)
        set_units_270716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), yaxis_270715, 'set_units')
        # Calling set_units(args, kwargs) (line 209)
        set_units_call_result_270719 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), set_units_270716, *[yunits_270717], **kwargs_270718)
        
        
        # Call to _update_axisinfo(...): (line 210)
        # Processing the call keyword arguments (line 210)
        kwargs_270723 = {}
        # Getting the type of 'axes' (line 210)
        axes_270720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'axes', False)
        # Obtaining the member 'xaxis' of a type (line 210)
        xaxis_270721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), axes_270720, 'xaxis')
        # Obtaining the member '_update_axisinfo' of a type (line 210)
        _update_axisinfo_270722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), xaxis_270721, '_update_axisinfo')
        # Calling _update_axisinfo(args, kwargs) (line 210)
        _update_axisinfo_call_result_270724 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), _update_axisinfo_270722, *[], **kwargs_270723)
        
        
        # Call to _update_axisinfo(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_270728 = {}
        # Getting the type of 'axes' (line 211)
        axes_270725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'axes', False)
        # Obtaining the member 'yaxis' of a type (line 211)
        yaxis_270726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), axes_270725, 'yaxis')
        # Obtaining the member '_update_axisinfo' of a type (line 211)
        _update_axisinfo_270727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), yaxis_270726, '_update_axisinfo')
        # Calling _update_axisinfo(args, kwargs) (line 211)
        _update_axisinfo_call_result_270729 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), _update_axisinfo_270727, *[], **kwargs_270728)
        
        
        
        # Call to enumerate(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'curves' (line 214)
        curves_270731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'curves', False)
        # Processing the call keyword arguments (line 214)
        kwargs_270732 = {}
        # Getting the type of 'enumerate' (line 214)
        enumerate_270730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 214)
        enumerate_call_result_270733 = invoke(stypy.reporting.localization.Localization(__file__, 214, 28), enumerate_270730, *[curves_270731], **kwargs_270732)
        
        # Testing the type of a for loop iterable (line 214)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 214, 8), enumerate_call_result_270733)
        # Getting the type of the for loop variable (line 214)
        for_loop_var_270734 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 214, 8), enumerate_call_result_270733)
        # Assigning a type to the variable 'index' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 8), for_loop_var_270734))
        # Assigning a type to the variable 'curve' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'curve', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 8), for_loop_var_270734))
        # SSA begins for a for statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 215):
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 215)
        index_270735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 40), 'index')
        # Getting the type of 'curvelabels' (line 215)
        curvelabels_270736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 28), 'curvelabels')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___270737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 28), curvelabels_270736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_270738 = invoke(stypy.reporting.localization.Localization(__file__, 215, 28), getitem___270737, index_270735)
        
        # Getting the type of 'linedict' (line 215)
        linedict_270739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'linedict')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___270740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 19), linedict_270739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_270741 = invoke(stypy.reporting.localization.Localization(__file__, 215, 19), getitem___270740, subscript_call_result_270738)
        
        # Assigning a type to the variable 'line' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'line', subscript_call_result_270741)
        
        # Assigning a Name to a Tuple (line 216):
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270745 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270744, int_270742)
        
        # Assigning a type to the variable 'tuple_var_assignment_269991' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269991', subscript_call_result_270745)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270749 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270748, int_270746)
        
        # Assigning a type to the variable 'tuple_var_assignment_269992' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269992', subscript_call_result_270749)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270753 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270752, int_270750)
        
        # Assigning a type to the variable 'tuple_var_assignment_269993' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269993', subscript_call_result_270753)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270757 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270756, int_270754)
        
        # Assigning a type to the variable 'tuple_var_assignment_269994' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269994', subscript_call_result_270757)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270761 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270760, int_270758)
        
        # Assigning a type to the variable 'tuple_var_assignment_269995' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269995', subscript_call_result_270761)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270765 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270764, int_270762)
        
        # Assigning a type to the variable 'tuple_var_assignment_269996' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269996', subscript_call_result_270765)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270769 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270768, int_270766)
        
        # Assigning a type to the variable 'tuple_var_assignment_269997' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269997', subscript_call_result_270769)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270773 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270772, int_270770)
        
        # Assigning a type to the variable 'tuple_var_assignment_269998' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269998', subscript_call_result_270773)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_270774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Getting the type of 'curve' (line 217)
        curve_270775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'curve')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___270776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), curve_270775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_270777 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___270776, int_270774)
        
        # Assigning a type to the variable 'tuple_var_assignment_269999' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269999', subscript_call_result_270777)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269991' (line 216)
        tuple_var_assignment_269991_270778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269991')
        # Assigning a type to the variable 'label' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'label', tuple_var_assignment_269991_270778)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269992' (line 216)
        tuple_var_assignment_269992_270779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269992')
        # Assigning a type to the variable 'linestyle' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'linestyle', tuple_var_assignment_269992_270779)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269993' (line 216)
        tuple_var_assignment_269993_270780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269993')
        # Assigning a type to the variable 'drawstyle' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'drawstyle', tuple_var_assignment_269993_270780)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269994' (line 216)
        tuple_var_assignment_269994_270781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269994')
        # Assigning a type to the variable 'linewidth' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 42), 'linewidth', tuple_var_assignment_269994_270781)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269995' (line 216)
        tuple_var_assignment_269995_270782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269995')
        # Assigning a type to the variable 'color' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 53), 'color', tuple_var_assignment_269995_270782)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269996' (line 216)
        tuple_var_assignment_269996_270783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269996')
        # Assigning a type to the variable 'marker' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 60), 'marker', tuple_var_assignment_269996_270783)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269997' (line 216)
        tuple_var_assignment_269997_270784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269997')
        # Assigning a type to the variable 'markersize' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 68), 'markersize', tuple_var_assignment_269997_270784)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269998' (line 216)
        tuple_var_assignment_269998_270785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269998')
        # Assigning a type to the variable 'markerfacecolor' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'markerfacecolor', tuple_var_assignment_269998_270785)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_269999' (line 216)
        tuple_var_assignment_269999_270786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tuple_var_assignment_269999')
        # Assigning a type to the variable 'markeredgecolor' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 30), 'markeredgecolor', tuple_var_assignment_269999_270786)
        
        # Call to set_label(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'label' (line 218)
        label_270789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'label', False)
        # Processing the call keyword arguments (line 218)
        kwargs_270790 = {}
        # Getting the type of 'line' (line 218)
        line_270787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'line', False)
        # Obtaining the member 'set_label' of a type (line 218)
        set_label_270788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), line_270787, 'set_label')
        # Calling set_label(args, kwargs) (line 218)
        set_label_call_result_270791 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), set_label_270788, *[label_270789], **kwargs_270790)
        
        
        # Call to set_linestyle(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'linestyle' (line 219)
        linestyle_270794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'linestyle', False)
        # Processing the call keyword arguments (line 219)
        kwargs_270795 = {}
        # Getting the type of 'line' (line 219)
        line_270792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'line', False)
        # Obtaining the member 'set_linestyle' of a type (line 219)
        set_linestyle_270793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), line_270792, 'set_linestyle')
        # Calling set_linestyle(args, kwargs) (line 219)
        set_linestyle_call_result_270796 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), set_linestyle_270793, *[linestyle_270794], **kwargs_270795)
        
        
        # Call to set_drawstyle(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'drawstyle' (line 220)
        drawstyle_270799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'drawstyle', False)
        # Processing the call keyword arguments (line 220)
        kwargs_270800 = {}
        # Getting the type of 'line' (line 220)
        line_270797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'line', False)
        # Obtaining the member 'set_drawstyle' of a type (line 220)
        set_drawstyle_270798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), line_270797, 'set_drawstyle')
        # Calling set_drawstyle(args, kwargs) (line 220)
        set_drawstyle_call_result_270801 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), set_drawstyle_270798, *[drawstyle_270799], **kwargs_270800)
        
        
        # Call to set_linewidth(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'linewidth' (line 221)
        linewidth_270804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'linewidth', False)
        # Processing the call keyword arguments (line 221)
        kwargs_270805 = {}
        # Getting the type of 'line' (line 221)
        line_270802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'line', False)
        # Obtaining the member 'set_linewidth' of a type (line 221)
        set_linewidth_270803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), line_270802, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 221)
        set_linewidth_call_result_270806 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), set_linewidth_270803, *[linewidth_270804], **kwargs_270805)
        
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to to_rgba(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'color' (line 222)
        color_270809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'color', False)
        # Processing the call keyword arguments (line 222)
        kwargs_270810 = {}
        # Getting the type of 'mcolors' (line 222)
        mcolors_270807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'mcolors', False)
        # Obtaining the member 'to_rgba' of a type (line 222)
        to_rgba_270808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), mcolors_270807, 'to_rgba')
        # Calling to_rgba(args, kwargs) (line 222)
        to_rgba_call_result_270811 = invoke(stypy.reporting.localization.Localization(__file__, 222, 19), to_rgba_270808, *[color_270809], **kwargs_270810)
        
        # Assigning a type to the variable 'rgba' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'rgba', to_rgba_call_result_270811)
        
        # Call to set_alpha(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'None' (line 223)
        None_270814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'None', False)
        # Processing the call keyword arguments (line 223)
        kwargs_270815 = {}
        # Getting the type of 'line' (line 223)
        line_270812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'line', False)
        # Obtaining the member 'set_alpha' of a type (line 223)
        set_alpha_270813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), line_270812, 'set_alpha')
        # Calling set_alpha(args, kwargs) (line 223)
        set_alpha_call_result_270816 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), set_alpha_270813, *[None_270814], **kwargs_270815)
        
        
        # Call to set_color(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'rgba' (line 224)
        rgba_270819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'rgba', False)
        # Processing the call keyword arguments (line 224)
        kwargs_270820 = {}
        # Getting the type of 'line' (line 224)
        line_270817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'line', False)
        # Obtaining the member 'set_color' of a type (line 224)
        set_color_270818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), line_270817, 'set_color')
        # Calling set_color(args, kwargs) (line 224)
        set_color_call_result_270821 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), set_color_270818, *[rgba_270819], **kwargs_270820)
        
        
        
        # Getting the type of 'marker' (line 225)
        marker_270822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'marker')
        unicode_270823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'unicode', u'none')
        # Applying the binary operator 'isnot' (line 225)
        result_is_not_270824 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 15), 'isnot', marker_270822, unicode_270823)
        
        # Testing the type of an if condition (line 225)
        if_condition_270825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 12), result_is_not_270824)
        # Assigning a type to the variable 'if_condition_270825' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'if_condition_270825', if_condition_270825)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_marker(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'marker' (line 226)
        marker_270828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'marker', False)
        # Processing the call keyword arguments (line 226)
        kwargs_270829 = {}
        # Getting the type of 'line' (line 226)
        line_270826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'line', False)
        # Obtaining the member 'set_marker' of a type (line 226)
        set_marker_270827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), line_270826, 'set_marker')
        # Calling set_marker(args, kwargs) (line 226)
        set_marker_call_result_270830 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), set_marker_270827, *[marker_270828], **kwargs_270829)
        
        
        # Call to set_markersize(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'markersize' (line 227)
        markersize_270833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'markersize', False)
        # Processing the call keyword arguments (line 227)
        kwargs_270834 = {}
        # Getting the type of 'line' (line 227)
        line_270831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'line', False)
        # Obtaining the member 'set_markersize' of a type (line 227)
        set_markersize_270832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), line_270831, 'set_markersize')
        # Calling set_markersize(args, kwargs) (line 227)
        set_markersize_call_result_270835 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), set_markersize_270832, *[markersize_270833], **kwargs_270834)
        
        
        # Call to set_markerfacecolor(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'markerfacecolor' (line 228)
        markerfacecolor_270838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 41), 'markerfacecolor', False)
        # Processing the call keyword arguments (line 228)
        kwargs_270839 = {}
        # Getting the type of 'line' (line 228)
        line_270836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'line', False)
        # Obtaining the member 'set_markerfacecolor' of a type (line 228)
        set_markerfacecolor_270837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), line_270836, 'set_markerfacecolor')
        # Calling set_markerfacecolor(args, kwargs) (line 228)
        set_markerfacecolor_call_result_270840 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), set_markerfacecolor_270837, *[markerfacecolor_270838], **kwargs_270839)
        
        
        # Call to set_markeredgecolor(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'markeredgecolor' (line 229)
        markeredgecolor_270843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 41), 'markeredgecolor', False)
        # Processing the call keyword arguments (line 229)
        kwargs_270844 = {}
        # Getting the type of 'line' (line 229)
        line_270841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'line', False)
        # Obtaining the member 'set_markeredgecolor' of a type (line 229)
        set_markeredgecolor_270842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), line_270841, 'set_markeredgecolor')
        # Calling set_markeredgecolor(args, kwargs) (line 229)
        set_markeredgecolor_call_result_270845 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), set_markeredgecolor_270842, *[markeredgecolor_270843], **kwargs_270844)
        
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'images' (line 232)
        images_270847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 47), 'images', False)
        # Processing the call keyword arguments (line 232)
        kwargs_270848 = {}
        # Getting the type of 'enumerate' (line 232)
        enumerate_270846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 37), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 232)
        enumerate_call_result_270849 = invoke(stypy.reporting.localization.Localization(__file__, 232, 37), enumerate_270846, *[images_270847], **kwargs_270848)
        
        # Testing the type of a for loop iterable (line 232)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 232, 8), enumerate_call_result_270849)
        # Getting the type of the for loop variable (line 232)
        for_loop_var_270850 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 232, 8), enumerate_call_result_270849)
        # Assigning a type to the variable 'index' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 8), for_loop_var_270850))
        # Assigning a type to the variable 'image_settings' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'image_settings', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 8), for_loop_var_270850))
        # SSA begins for a for statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 233):
        
        # Assigning a Subscript to a Name (line 233):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 233)
        index_270851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 42), 'index')
        # Getting the type of 'imagelabels' (line 233)
        imagelabels_270852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'imagelabels')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___270853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 30), imagelabels_270852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_270854 = invoke(stypy.reporting.localization.Localization(__file__, 233, 30), getitem___270853, index_270851)
        
        # Getting the type of 'imagedict' (line 233)
        imagedict_270855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'imagedict')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___270856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), imagedict_270855, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_270857 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), getitem___270856, subscript_call_result_270854)
        
        # Assigning a type to the variable 'image' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'image', subscript_call_result_270857)
        
        # Assigning a Name to a Tuple (line 234):
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_270858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'int')
        # Getting the type of 'image_settings' (line 234)
        image_settings_270859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'image_settings')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___270860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), image_settings_270859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_270861 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), getitem___270860, int_270858)
        
        # Assigning a type to the variable 'tuple_var_assignment_270000' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270000', subscript_call_result_270861)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_270862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'int')
        # Getting the type of 'image_settings' (line 234)
        image_settings_270863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'image_settings')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___270864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), image_settings_270863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_270865 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), getitem___270864, int_270862)
        
        # Assigning a type to the variable 'tuple_var_assignment_270001' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270001', subscript_call_result_270865)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_270866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'int')
        # Getting the type of 'image_settings' (line 234)
        image_settings_270867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'image_settings')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___270868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), image_settings_270867, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_270869 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), getitem___270868, int_270866)
        
        # Assigning a type to the variable 'tuple_var_assignment_270002' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270002', subscript_call_result_270869)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_270870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'int')
        # Getting the type of 'image_settings' (line 234)
        image_settings_270871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'image_settings')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___270872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), image_settings_270871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_270873 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), getitem___270872, int_270870)
        
        # Assigning a type to the variable 'tuple_var_assignment_270003' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270003', subscript_call_result_270873)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_270874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'int')
        # Getting the type of 'image_settings' (line 234)
        image_settings_270875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'image_settings')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___270876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), image_settings_270875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_270877 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), getitem___270876, int_270874)
        
        # Assigning a type to the variable 'tuple_var_assignment_270004' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270004', subscript_call_result_270877)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_270000' (line 234)
        tuple_var_assignment_270000_270878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270000')
        # Assigning a type to the variable 'label' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'label', tuple_var_assignment_270000_270878)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_270001' (line 234)
        tuple_var_assignment_270001_270879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270001')
        # Assigning a type to the variable 'cmap' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'cmap', tuple_var_assignment_270001_270879)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_270002' (line 234)
        tuple_var_assignment_270002_270880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270002')
        # Assigning a type to the variable 'low' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'low', tuple_var_assignment_270002_270880)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_270003' (line 234)
        tuple_var_assignment_270003_270881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270003')
        # Assigning a type to the variable 'high' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 30), 'high', tuple_var_assignment_270003_270881)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_270004' (line 234)
        tuple_var_assignment_270004_270882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'tuple_var_assignment_270004')
        # Assigning a type to the variable 'interpolation' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 36), 'interpolation', tuple_var_assignment_270004_270882)
        
        # Call to set_label(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'label' (line 235)
        label_270885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'label', False)
        # Processing the call keyword arguments (line 235)
        kwargs_270886 = {}
        # Getting the type of 'image' (line 235)
        image_270883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'image', False)
        # Obtaining the member 'set_label' of a type (line 235)
        set_label_270884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), image_270883, 'set_label')
        # Calling set_label(args, kwargs) (line 235)
        set_label_call_result_270887 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), set_label_270884, *[label_270885], **kwargs_270886)
        
        
        # Call to set_cmap(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to get_cmap(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'cmap' (line 236)
        cmap_270892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 39), 'cmap', False)
        # Processing the call keyword arguments (line 236)
        kwargs_270893 = {}
        # Getting the type of 'cm' (line 236)
        cm_270890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'cm', False)
        # Obtaining the member 'get_cmap' of a type (line 236)
        get_cmap_270891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 27), cm_270890, 'get_cmap')
        # Calling get_cmap(args, kwargs) (line 236)
        get_cmap_call_result_270894 = invoke(stypy.reporting.localization.Localization(__file__, 236, 27), get_cmap_270891, *[cmap_270892], **kwargs_270893)
        
        # Processing the call keyword arguments (line 236)
        kwargs_270895 = {}
        # Getting the type of 'image' (line 236)
        image_270888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'image', False)
        # Obtaining the member 'set_cmap' of a type (line 236)
        set_cmap_270889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), image_270888, 'set_cmap')
        # Calling set_cmap(args, kwargs) (line 236)
        set_cmap_call_result_270896 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), set_cmap_270889, *[get_cmap_call_result_270894], **kwargs_270895)
        
        
        # Call to set_clim(...): (line 237)
        
        # Call to sorted(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_270900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        # Getting the type of 'low' (line 237)
        low_270901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 36), 'low', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 35), list_270900, low_270901)
        # Adding element type (line 237)
        # Getting the type of 'high' (line 237)
        high_270902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 41), 'high', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 35), list_270900, high_270902)
        
        # Processing the call keyword arguments (line 237)
        kwargs_270903 = {}
        # Getting the type of 'sorted' (line 237)
        sorted_270899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 28), 'sorted', False)
        # Calling sorted(args, kwargs) (line 237)
        sorted_call_result_270904 = invoke(stypy.reporting.localization.Localization(__file__, 237, 28), sorted_270899, *[list_270900], **kwargs_270903)
        
        # Processing the call keyword arguments (line 237)
        kwargs_270905 = {}
        # Getting the type of 'image' (line 237)
        image_270897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'image', False)
        # Obtaining the member 'set_clim' of a type (line 237)
        set_clim_270898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), image_270897, 'set_clim')
        # Calling set_clim(args, kwargs) (line 237)
        set_clim_call_result_270906 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), set_clim_270898, *[sorted_call_result_270904], **kwargs_270905)
        
        
        # Call to set_interpolation(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'interpolation' (line 238)
        interpolation_270909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'interpolation', False)
        # Processing the call keyword arguments (line 238)
        kwargs_270910 = {}
        # Getting the type of 'image' (line 238)
        image_270907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'image', False)
        # Obtaining the member 'set_interpolation' of a type (line 238)
        set_interpolation_270908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), image_270907, 'set_interpolation')
        # Calling set_interpolation(args, kwargs) (line 238)
        set_interpolation_call_result_270911 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), set_interpolation_270908, *[interpolation_270909], **kwargs_270910)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'generate_legend' (line 241)
        generate_legend_270912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'generate_legend')
        # Testing the type of an if condition (line 241)
        if_condition_270913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), generate_legend_270912)
        # Assigning a type to the variable 'if_condition_270913' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_270913', if_condition_270913)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 242):
        
        # Assigning a Name to a Name (line 242):
        # Getting the type of 'None' (line 242)
        None_270914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'None')
        # Assigning a type to the variable 'draggable' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'draggable', None_270914)
        
        # Assigning a Num to a Name (line 243):
        
        # Assigning a Num to a Name (line 243):
        int_270915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 19), 'int')
        # Assigning a type to the variable 'ncol' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'ncol', int_270915)
        
        
        # Getting the type of 'axes' (line 244)
        axes_270916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'axes')
        # Obtaining the member 'legend_' of a type (line 244)
        legend__270917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 15), axes_270916, 'legend_')
        # Getting the type of 'None' (line 244)
        None_270918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 35), 'None')
        # Applying the binary operator 'isnot' (line 244)
        result_is_not_270919 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 15), 'isnot', legend__270917, None_270918)
        
        # Testing the type of an if condition (line 244)
        if_condition_270920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 12), result_is_not_270919)
        # Assigning a type to the variable 'if_condition_270920' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'if_condition_270920', if_condition_270920)
        # SSA begins for if statement (line 244)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to get_legend(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_270923 = {}
        # Getting the type of 'axes' (line 245)
        axes_270921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'axes', False)
        # Obtaining the member 'get_legend' of a type (line 245)
        get_legend_270922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 29), axes_270921, 'get_legend')
        # Calling get_legend(args, kwargs) (line 245)
        get_legend_call_result_270924 = invoke(stypy.reporting.localization.Localization(__file__, 245, 29), get_legend_270922, *[], **kwargs_270923)
        
        # Assigning a type to the variable 'old_legend' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'old_legend', get_legend_call_result_270924)
        
        # Assigning a Compare to a Name (line 246):
        
        # Assigning a Compare to a Name (line 246):
        
        # Getting the type of 'old_legend' (line 246)
        old_legend_270925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'old_legend')
        # Obtaining the member '_draggable' of a type (line 246)
        _draggable_270926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 28), old_legend_270925, '_draggable')
        # Getting the type of 'None' (line 246)
        None_270927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 57), 'None')
        # Applying the binary operator 'isnot' (line 246)
        result_is_not_270928 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 28), 'isnot', _draggable_270926, None_270927)
        
        # Assigning a type to the variable 'draggable' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'draggable', result_is_not_270928)
        
        # Assigning a Attribute to a Name (line 247):
        
        # Assigning a Attribute to a Name (line 247):
        # Getting the type of 'old_legend' (line 247)
        old_legend_270929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'old_legend')
        # Obtaining the member '_ncol' of a type (line 247)
        _ncol_270930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 23), old_legend_270929, '_ncol')
        # Assigning a type to the variable 'ncol' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'ncol', _ncol_270930)
        # SSA join for if statement (line 244)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to legend(...): (line 248)
        # Processing the call keyword arguments (line 248)
        # Getting the type of 'ncol' (line 248)
        ncol_270933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'ncol', False)
        keyword_270934 = ncol_270933
        kwargs_270935 = {'ncol': keyword_270934}
        # Getting the type of 'axes' (line 248)
        axes_270931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'axes', False)
        # Obtaining the member 'legend' of a type (line 248)
        legend_270932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), axes_270931, 'legend')
        # Calling legend(args, kwargs) (line 248)
        legend_call_result_270936 = invoke(stypy.reporting.localization.Localization(__file__, 248, 25), legend_270932, *[], **kwargs_270935)
        
        # Assigning a type to the variable 'new_legend' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'new_legend', legend_call_result_270936)
        
        # Getting the type of 'new_legend' (line 249)
        new_legend_270937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 'new_legend')
        # Testing the type of an if condition (line 249)
        if_condition_270938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 12), new_legend_270937)
        # Assigning a type to the variable 'if_condition_270938' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'if_condition_270938', if_condition_270938)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draggable(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'draggable' (line 250)
        draggable_270941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 37), 'draggable', False)
        # Processing the call keyword arguments (line 250)
        kwargs_270942 = {}
        # Getting the type of 'new_legend' (line 250)
        new_legend_270939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'new_legend', False)
        # Obtaining the member 'draggable' of a type (line 250)
        draggable_270940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), new_legend_270939, 'draggable')
        # Calling draggable(args, kwargs) (line 250)
        draggable_call_result_270943 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), draggable_270940, *[draggable_270941], **kwargs_270942)
        
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to get_figure(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_270946 = {}
        # Getting the type of 'axes' (line 253)
        axes_270944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'axes', False)
        # Obtaining the member 'get_figure' of a type (line 253)
        get_figure_270945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 17), axes_270944, 'get_figure')
        # Calling get_figure(args, kwargs) (line 253)
        get_figure_call_result_270947 = invoke(stypy.reporting.localization.Localization(__file__, 253, 17), get_figure_270945, *[], **kwargs_270946)
        
        # Assigning a type to the variable 'figure' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'figure', get_figure_call_result_270947)
        
        # Call to draw(...): (line 254)
        # Processing the call keyword arguments (line 254)
        kwargs_270951 = {}
        # Getting the type of 'figure' (line 254)
        figure_270948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'figure', False)
        # Obtaining the member 'canvas' of a type (line 254)
        canvas_270949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), figure_270948, 'canvas')
        # Obtaining the member 'draw' of a type (line 254)
        draw_270950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), canvas_270949, 'draw')
        # Calling draw(args, kwargs) (line 254)
        draw_call_result_270952 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), draw_270950, *[], **kwargs_270951)
        
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to get_xlim(...): (line 255)
        # Processing the call keyword arguments (line 255)
        kwargs_270955 = {}
        # Getting the type of 'axes' (line 255)
        axes_270953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'axes', False)
        # Obtaining the member 'get_xlim' of a type (line 255)
        get_xlim_270954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 16), axes_270953, 'get_xlim')
        # Calling get_xlim(args, kwargs) (line 255)
        get_xlim_call_result_270956 = invoke(stypy.reporting.localization.Localization(__file__, 255, 16), get_xlim_270954, *[], **kwargs_270955)
        
        # Getting the type of 'orig_xlim' (line 255)
        orig_xlim_270957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 35), 'orig_xlim')
        # Applying the binary operator '==' (line 255)
        result_eq_270958 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), '==', get_xlim_call_result_270956, orig_xlim_270957)
        
        
        
        # Call to get_ylim(...): (line 255)
        # Processing the call keyword arguments (line 255)
        kwargs_270961 = {}
        # Getting the type of 'axes' (line 255)
        axes_270959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 49), 'axes', False)
        # Obtaining the member 'get_ylim' of a type (line 255)
        get_ylim_270960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 49), axes_270959, 'get_ylim')
        # Calling get_ylim(args, kwargs) (line 255)
        get_ylim_call_result_270962 = invoke(stypy.reporting.localization.Localization(__file__, 255, 49), get_ylim_270960, *[], **kwargs_270961)
        
        # Getting the type of 'orig_ylim' (line 255)
        orig_ylim_270963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 68), 'orig_ylim')
        # Applying the binary operator '==' (line 255)
        result_eq_270964 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 49), '==', get_ylim_call_result_270962, orig_ylim_270963)
        
        # Applying the binary operator 'and' (line 255)
        result_and_keyword_270965 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), 'and', result_eq_270958, result_eq_270964)
        
        # Applying the 'not' unary operator (line 255)
        result_not__270966 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), 'not', result_and_keyword_270965)
        
        # Testing the type of an if condition (line 255)
        if_condition_270967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_not__270966)
        # Assigning a type to the variable 'if_condition_270967' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_270967', if_condition_270967)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to push_current(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_270972 = {}
        # Getting the type of 'figure' (line 256)
        figure_270968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'figure', False)
        # Obtaining the member 'canvas' of a type (line 256)
        canvas_270969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), figure_270968, 'canvas')
        # Obtaining the member 'toolbar' of a type (line 256)
        toolbar_270970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), canvas_270969, 'toolbar')
        # Obtaining the member 'push_current' of a type (line 256)
        push_current_270971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), toolbar_270970, 'push_current')
        # Calling push_current(args, kwargs) (line 256)
        push_current_call_result_270973 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), push_current_270971, *[], **kwargs_270972)
        
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'apply_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'apply_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_270974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_270974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'apply_callback'
        return stypy_return_type_270974

    # Assigning a type to the variable 'apply_callback' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'apply_callback', apply_callback)
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to fedit(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'datalist' (line 258)
    datalist_270977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'datalist', False)
    # Processing the call keyword arguments (line 258)
    unicode_270978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 44), 'unicode', u'Figure options')
    keyword_270979 = unicode_270978
    # Getting the type of 'parent' (line 258)
    parent_270980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 69), 'parent', False)
    keyword_270981 = parent_270980
    
    # Call to get_icon(...): (line 259)
    # Processing the call arguments (line 259)
    unicode_270983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 42), 'unicode', u'qt4_editor_options.svg')
    # Processing the call keyword arguments (line 259)
    kwargs_270984 = {}
    # Getting the type of 'get_icon' (line 259)
    get_icon_270982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 33), 'get_icon', False)
    # Calling get_icon(args, kwargs) (line 259)
    get_icon_call_result_270985 = invoke(stypy.reporting.localization.Localization(__file__, 259, 33), get_icon_270982, *[unicode_270983], **kwargs_270984)
    
    keyword_270986 = get_icon_call_result_270985
    # Getting the type of 'apply_callback' (line 260)
    apply_callback_270987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'apply_callback', False)
    keyword_270988 = apply_callback_270987
    kwargs_270989 = {'apply': keyword_270988, 'icon': keyword_270986, 'parent': keyword_270981, 'title': keyword_270979}
    # Getting the type of 'formlayout' (line 258)
    formlayout_270975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'formlayout', False)
    # Obtaining the member 'fedit' of a type (line 258)
    fedit_270976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 11), formlayout_270975, 'fedit')
    # Calling fedit(args, kwargs) (line 258)
    fedit_call_result_270990 = invoke(stypy.reporting.localization.Localization(__file__, 258, 11), fedit_270976, *[datalist_270977], **kwargs_270989)
    
    # Assigning a type to the variable 'data' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'data', fedit_call_result_270990)
    
    # Type idiom detected: calculating its left and rigth part (line 261)
    # Getting the type of 'data' (line 261)
    data_270991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'data')
    # Getting the type of 'None' (line 261)
    None_270992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'None')
    
    (may_be_270993, more_types_in_union_270994) = may_not_be_none(data_270991, None_270992)

    if may_be_270993:

        if more_types_in_union_270994:
            # Runtime conditional SSA (line 261)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to apply_callback(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'data' (line 262)
        data_270996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'data', False)
        # Processing the call keyword arguments (line 262)
        kwargs_270997 = {}
        # Getting the type of 'apply_callback' (line 262)
        apply_callback_270995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'apply_callback', False)
        # Calling apply_callback(args, kwargs) (line 262)
        apply_callback_call_result_270998 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), apply_callback_270995, *[data_270996], **kwargs_270997)
        

        if more_types_in_union_270994:
            # SSA join for if statement (line 261)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'figure_edit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'figure_edit' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_270999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_270999)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'figure_edit'
    return stypy_return_type_270999

# Assigning a type to the variable 'figure_edit' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'figure_edit', figure_edit)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module provides routines to adjust subplot params so that subplots are
3: nicely fit in the figure. In doing so, only axis labels, tick labels, axes
4: titles and offsetboxes that are anchored to axes are currently considered.
5: 
6: Internally, it assumes that the margins (left_margin, etc.) which are
7: differences between ax.get_tightbbox and ax.bbox are independent of axes
8: position. This may fail if Axes.adjustable is datalim. Also, This will fail
9: for some cases (for example, left or right margin is affected by xlabel).
10: '''
11: 
12: import warnings
13: 
14: import matplotlib
15: from matplotlib.transforms import TransformedBbox, Bbox
16: 
17: from matplotlib.font_manager import FontProperties
18: rcParams = matplotlib.rcParams
19: 
20: 
21: def _get_left(tight_bbox, axes_bbox):
22:     return axes_bbox.xmin - tight_bbox.xmin
23: 
24: 
25: def _get_right(tight_bbox, axes_bbox):
26:     return tight_bbox.xmax - axes_bbox.xmax
27: 
28: 
29: def _get_bottom(tight_bbox, axes_bbox):
30:     return axes_bbox.ymin - tight_bbox.ymin
31: 
32: 
33: def _get_top(tight_bbox, axes_bbox):
34:     return tight_bbox.ymax - axes_bbox.ymax
35: 
36: 
37: def auto_adjust_subplotpars(fig, renderer,
38:                             nrows_ncols,
39:                             num1num2_list,
40:                             subplot_list,
41:                             ax_bbox_list=None,
42:                             pad=1.08, h_pad=None, w_pad=None,
43:                             rect=None):
44:     '''
45:     Return a dictionary of subplot parameters so that spacing between
46:     subplots are adjusted. Note that this function ignore geometry
47:     information of subplot itself, but uses what is given by
48:     *nrows_ncols* and *num1num2_list* parameteres. Also, the results could be
49:     incorrect if some subplots have ``adjustable=datalim``.
50: 
51:     Parameters:
52: 
53:     nrows_ncols
54:       number of rows and number of columns of the grid.
55: 
56:     num1num2_list
57:       list of numbers specifying the area occupied by the subplot
58: 
59:     subplot_list
60:       list of subplots that will be used to calcuate optimal subplot_params.
61: 
62:     pad : float
63:       padding between the figure edge and the edges of subplots, as a fraction
64:       of the font-size.
65:     h_pad, w_pad : float
66:       padding (height/width) between edges of adjacent subplots.
67:         Defaults to `pad_inches`.
68: 
69:     rect
70:       [left, bottom, right, top] in normalized (0, 1) figure coordinates.
71:     '''
72:     rows, cols = nrows_ncols
73: 
74:     pad_inches = pad * FontProperties(
75:                     size=rcParams["font.size"]).get_size_in_points() / 72.
76: 
77:     if h_pad is not None:
78:         vpad_inches = h_pad * FontProperties(
79:                         size=rcParams["font.size"]).get_size_in_points() / 72.
80:     else:
81:         vpad_inches = pad_inches
82: 
83:     if w_pad is not None:
84:         hpad_inches = w_pad * FontProperties(
85:                         size=rcParams["font.size"]).get_size_in_points() / 72.
86:     else:
87:         hpad_inches = pad_inches
88: 
89:     if len(subplot_list) == 0:
90:         raise RuntimeError("")
91: 
92:     if len(num1num2_list) != len(subplot_list):
93:         raise RuntimeError("")
94: 
95:     if rect is None:
96:         margin_left = None
97:         margin_bottom = None
98:         margin_right = None
99:         margin_top = None
100:     else:
101:         margin_left, margin_bottom, _right, _top = rect
102:         if _right:
103:             margin_right = 1. - _right
104:         else:
105:             margin_right = None
106:         if _top:
107:             margin_top = 1. - _top
108:         else:
109:             margin_top = None
110: 
111:     vspaces = [[] for i in range((rows + 1) * cols)]
112:     hspaces = [[] for i in range(rows * (cols + 1))]
113: 
114:     union = Bbox.union
115: 
116:     if ax_bbox_list is None:
117:         ax_bbox_list = []
118:         for subplots in subplot_list:
119:             ax_bbox = union([ax.get_position(original=True)
120:                              for ax in subplots])
121:             ax_bbox_list.append(ax_bbox)
122: 
123:     for subplots, ax_bbox, (num1, num2) in zip(subplot_list,
124:                                                ax_bbox_list,
125:                                                num1num2_list):
126:         if all([not ax.get_visible() for ax in subplots]):
127:             continue
128: 
129:         tight_bbox_raw = union([ax.get_tightbbox(renderer) for ax in subplots
130:                                 if ax.get_visible()])
131:         tight_bbox = TransformedBbox(tight_bbox_raw,
132:                                      fig.transFigure.inverted())
133: 
134:         row1, col1 = divmod(num1, cols)
135: 
136:         if num2 is None:
137:             # left
138:             hspaces[row1 * (cols + 1) + col1].append(
139:                                         _get_left(tight_bbox, ax_bbox))
140:             # right
141:             hspaces[row1 * (cols + 1) + (col1 + 1)].append(
142:                                         _get_right(tight_bbox, ax_bbox))
143:             # top
144:             vspaces[row1 * cols + col1].append(
145:                                         _get_top(tight_bbox, ax_bbox))
146:             # bottom
147:             vspaces[(row1 + 1) * cols + col1].append(
148:                                         _get_bottom(tight_bbox, ax_bbox))
149: 
150:         else:
151:             row2, col2 = divmod(num2, cols)
152: 
153:             for row_i in range(row1, row2 + 1):
154:                 # left
155:                 hspaces[row_i * (cols + 1) + col1].append(
156:                                     _get_left(tight_bbox, ax_bbox))
157:                 # right
158:                 hspaces[row_i * (cols + 1) + (col2 + 1)].append(
159:                                     _get_right(tight_bbox, ax_bbox))
160:             for col_i in range(col1, col2 + 1):
161:                 # top
162:                 vspaces[row1 * cols + col_i].append(
163:                                     _get_top(tight_bbox, ax_bbox))
164:                 # bottom
165:                 vspaces[(row2 + 1) * cols + col_i].append(
166:                                     _get_bottom(tight_bbox, ax_bbox))
167: 
168:     fig_width_inch, fig_height_inch = fig.get_size_inches()
169: 
170:     # margins can be negative for axes with aspect applied. And we
171:     # append + [0] to make minimum margins 0
172: 
173:     if not margin_left:
174:         margin_left = max([sum(s) for s in hspaces[::cols + 1]] + [0])
175:         margin_left += pad_inches / fig_width_inch
176: 
177:     if not margin_right:
178:         margin_right = max([sum(s) for s in hspaces[cols::cols + 1]] + [0])
179:         margin_right += pad_inches / fig_width_inch
180: 
181:     if not margin_top:
182:         margin_top = max([sum(s) for s in vspaces[:cols]] + [0])
183:         margin_top += pad_inches / fig_height_inch
184: 
185:     if not margin_bottom:
186:         margin_bottom = max([sum(s) for s in vspaces[-cols:]] + [0])
187:         margin_bottom += pad_inches / fig_height_inch
188: 
189:     kwargs = dict(left=margin_left,
190:                   right=1 - margin_right,
191:                   bottom=margin_bottom,
192:                   top=1 - margin_top)
193: 
194:     if cols > 1:
195:         hspace = (
196:             max(sum(s)
197:                 for i in range(rows)
198:                 for s in hspaces[i * (cols + 1) + 1:(i + 1) * (cols + 1) - 1])
199:             + hpad_inches / fig_width_inch)
200:         h_axes = (1 - margin_right - margin_left - hspace * (cols - 1)) / cols
201:         kwargs["wspace"] = hspace / h_axes
202: 
203:     if rows > 1:
204:         vspace = (max(sum(s) for s in vspaces[cols:-cols])
205:                   + vpad_inches / fig_height_inch)
206:         v_axes = (1 - margin_top - margin_bottom - vspace * (rows - 1)) / rows
207:         kwargs["hspace"] = vspace / v_axes
208: 
209:     return kwargs
210: 
211: 
212: def get_renderer(fig):
213:     if fig._cachedRenderer:
214:         renderer = fig._cachedRenderer
215:     else:
216:         canvas = fig.canvas
217: 
218:         if canvas and hasattr(canvas, "get_renderer"):
219:             renderer = canvas.get_renderer()
220:         else:
221:             # not sure if this can happen
222:             warnings.warn("tight_layout : falling back to Agg renderer")
223:             from matplotlib.backends.backend_agg import FigureCanvasAgg
224:             canvas = FigureCanvasAgg(fig)
225:             renderer = canvas.get_renderer()
226: 
227:     return renderer
228: 
229: 
230: def get_subplotspec_list(axes_list, grid_spec=None):
231:     '''Return a list of subplotspec from the given list of axes.
232: 
233:     For an instance of axes that does not support subplotspec, None is inserted
234:     in the list.
235: 
236:     If grid_spec is given, None is inserted for those not from the given
237:     grid_spec.
238:     '''
239:     subplotspec_list = []
240:     for ax in axes_list:
241:         axes_or_locator = ax.get_axes_locator()
242:         if axes_or_locator is None:
243:             axes_or_locator = ax
244: 
245:         if hasattr(axes_or_locator, "get_subplotspec"):
246:             subplotspec = axes_or_locator.get_subplotspec()
247:             subplotspec = subplotspec.get_topmost_subplotspec()
248:             gs = subplotspec.get_gridspec()
249:             if grid_spec is not None:
250:                 if gs != grid_spec:
251:                     subplotspec = None
252:             elif gs.locally_modified_subplot_params():
253:                 subplotspec = None
254:         else:
255:             subplotspec = None
256: 
257:         subplotspec_list.append(subplotspec)
258: 
259:     return subplotspec_list
260: 
261: 
262: def get_tight_layout_figure(fig, axes_list, subplotspec_list, renderer,
263:                             pad=1.08, h_pad=None, w_pad=None, rect=None):
264:     '''
265:     Return subplot parameters for tight-layouted-figure with specified
266:     padding.
267: 
268:     Parameters:
269: 
270:       *fig* : figure instance
271: 
272:       *axes_list* : a list of axes
273: 
274:       *subplotspec_list* : a list of subplotspec associated with each
275:         axes in axes_list
276: 
277:       *renderer* : renderer instance
278: 
279:       *pad* : float
280:         padding between the figure edge and the edges of subplots,
281:         as a fraction of the font-size.
282: 
283:       *h_pad*, *w_pad* : float
284:         padding (height/width) between edges of adjacent subplots.
285:         Defaults to `pad_inches`.
286: 
287:       *rect* : if rect is given, it is interpreted as a rectangle
288:         (left, bottom, right, top) in the normalized figure
289:         coordinate that the whole subplots area (including
290:         labels) will fit into. Default is (0, 0, 1, 1).
291:     '''
292: 
293:     subplot_list = []
294:     nrows_list = []
295:     ncols_list = []
296:     ax_bbox_list = []
297: 
298:     subplot_dict = {}  # multiple axes can share
299:                        # same subplot_interface (e.g., axes_grid1). Thus
300:                        # we need to join them together.
301: 
302:     subplotspec_list2 = []
303: 
304:     for ax, subplotspec in zip(axes_list,
305:                                subplotspec_list):
306:         if subplotspec is None:
307:             continue
308: 
309:         subplots = subplot_dict.setdefault(subplotspec, [])
310: 
311:         if not subplots:
312:             myrows, mycols, _, _ = subplotspec.get_geometry()
313:             nrows_list.append(myrows)
314:             ncols_list.append(mycols)
315:             subplotspec_list2.append(subplotspec)
316:             subplot_list.append(subplots)
317:             ax_bbox_list.append(subplotspec.get_position(fig))
318: 
319:         subplots.append(ax)
320: 
321:     if (len(nrows_list) == 0) or (len(ncols_list) == 0):
322:         return {}
323: 
324:     max_nrows = max(nrows_list)
325:     max_ncols = max(ncols_list)
326: 
327:     num1num2_list = []
328:     for subplotspec in subplotspec_list2:
329:         rows, cols, num1, num2 = subplotspec.get_geometry()
330:         div_row, mod_row = divmod(max_nrows, rows)
331:         div_col, mod_col = divmod(max_ncols, cols)
332:         if (mod_row != 0) or (mod_col != 0):
333:             raise RuntimeError("")
334: 
335:         rowNum1, colNum1 = divmod(num1, cols)
336:         if num2 is None:
337:             rowNum2, colNum2 = rowNum1, colNum1
338:         else:
339:             rowNum2, colNum2 = divmod(num2, cols)
340: 
341:         num1num2_list.append((rowNum1 * div_row * max_ncols +
342:                               colNum1 * div_col,
343:                               ((rowNum2 + 1) * div_row - 1) * max_ncols +
344:                               (colNum2 + 1) * div_col - 1))
345: 
346:     kwargs = auto_adjust_subplotpars(fig, renderer,
347:                                      nrows_ncols=(max_nrows, max_ncols),
348:                                      num1num2_list=num1num2_list,
349:                                      subplot_list=subplot_list,
350:                                      ax_bbox_list=ax_bbox_list,
351:                                      pad=pad, h_pad=h_pad, w_pad=w_pad)
352: 
353:     if rect is not None:
354:         # if rect is given, the whole subplots area (including
355:         # labels) will fit into the rect instead of the
356:         # figure. Note that the rect argument of
357:         # *auto_adjust_subplotpars* specify the area that will be
358:         # covered by the total area of axes.bbox. Thus we call
359:         # auto_adjust_subplotpars twice, where the second run
360:         # with adjusted rect parameters.
361: 
362:         left, bottom, right, top = rect
363:         if left is not None:
364:             left += kwargs["left"]
365:         if bottom is not None:
366:             bottom += kwargs["bottom"]
367:         if right is not None:
368:             right -= (1 - kwargs["right"])
369:         if top is not None:
370:             top -= (1 - kwargs["top"])
371: 
372:         #if h_pad is None: h_pad = pad
373:         #if w_pad is None: w_pad = pad
374: 
375:         kwargs = auto_adjust_subplotpars(fig, renderer,
376:                                          nrows_ncols=(max_nrows, max_ncols),
377:                                          num1num2_list=num1num2_list,
378:                                          subplot_list=subplot_list,
379:                                          ax_bbox_list=ax_bbox_list,
380:                                          pad=pad, h_pad=h_pad, w_pad=w_pad,
381:                                          rect=(left, bottom, right, top))
382: 
383:     return kwargs
384: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_153046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', '\nThis module provides routines to adjust subplot params so that subplots are\nnicely fit in the figure. In doing so, only axis labels, tick labels, axes\ntitles and offsetboxes that are anchored to axes are currently considered.\n\nInternally, it assumes that the margins (left_margin, etc.) which are\ndifferences between ax.get_tightbbox and ax.bbox are independent of axes\nposition. This may fail if Axes.adjustable is datalim. Also, This will fail\nfor some cases (for example, left or right margin is affected by xlabel).\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import matplotlib' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_153047 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_153047) is not StypyTypeError):

    if (import_153047 != 'pyd_module'):
        __import__(import_153047)
        sys_modules_153048 = sys.modules[import_153047]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_153048.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_153047)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.transforms import TransformedBbox, Bbox' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_153049 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.transforms')

if (type(import_153049) is not StypyTypeError):

    if (import_153049 != 'pyd_module'):
        __import__(import_153049)
        sys_modules_153050 = sys.modules[import_153049]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.transforms', sys_modules_153050.module_type_store, module_type_store, ['TransformedBbox', 'Bbox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_153050, sys_modules_153050.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import TransformedBbox, Bbox

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.transforms', None, module_type_store, ['TransformedBbox', 'Bbox'], [TransformedBbox, Bbox])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.transforms', import_153049)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from matplotlib.font_manager import FontProperties' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_153051 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.font_manager')

if (type(import_153051) is not StypyTypeError):

    if (import_153051 != 'pyd_module'):
        __import__(import_153051)
        sys_modules_153052 = sys.modules[import_153051]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.font_manager', sys_modules_153052.module_type_store, module_type_store, ['FontProperties'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_153052, sys_modules_153052.module_type_store, module_type_store)
    else:
        from matplotlib.font_manager import FontProperties

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.font_manager', None, module_type_store, ['FontProperties'], [FontProperties])

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.font_manager', import_153051)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Assigning a Attribute to a Name (line 18):

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'matplotlib' (line 18)
matplotlib_153053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'matplotlib')
# Obtaining the member 'rcParams' of a type (line 18)
rcParams_153054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), matplotlib_153053, 'rcParams')
# Assigning a type to the variable 'rcParams' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'rcParams', rcParams_153054)

@norecursion
def _get_left(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_left'
    module_type_store = module_type_store.open_function_context('_get_left', 21, 0, False)
    
    # Passed parameters checking function
    _get_left.stypy_localization = localization
    _get_left.stypy_type_of_self = None
    _get_left.stypy_type_store = module_type_store
    _get_left.stypy_function_name = '_get_left'
    _get_left.stypy_param_names_list = ['tight_bbox', 'axes_bbox']
    _get_left.stypy_varargs_param_name = None
    _get_left.stypy_kwargs_param_name = None
    _get_left.stypy_call_defaults = defaults
    _get_left.stypy_call_varargs = varargs
    _get_left.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_left', ['tight_bbox', 'axes_bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_left', localization, ['tight_bbox', 'axes_bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_left(...)' code ##################

    # Getting the type of 'axes_bbox' (line 22)
    axes_bbox_153055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'axes_bbox')
    # Obtaining the member 'xmin' of a type (line 22)
    xmin_153056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), axes_bbox_153055, 'xmin')
    # Getting the type of 'tight_bbox' (line 22)
    tight_bbox_153057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'tight_bbox')
    # Obtaining the member 'xmin' of a type (line 22)
    xmin_153058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 28), tight_bbox_153057, 'xmin')
    # Applying the binary operator '-' (line 22)
    result_sub_153059 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '-', xmin_153056, xmin_153058)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', result_sub_153059)
    
    # ################# End of '_get_left(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_left' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_153060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153060)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_left'
    return stypy_return_type_153060

# Assigning a type to the variable '_get_left' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '_get_left', _get_left)

@norecursion
def _get_right(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_right'
    module_type_store = module_type_store.open_function_context('_get_right', 25, 0, False)
    
    # Passed parameters checking function
    _get_right.stypy_localization = localization
    _get_right.stypy_type_of_self = None
    _get_right.stypy_type_store = module_type_store
    _get_right.stypy_function_name = '_get_right'
    _get_right.stypy_param_names_list = ['tight_bbox', 'axes_bbox']
    _get_right.stypy_varargs_param_name = None
    _get_right.stypy_kwargs_param_name = None
    _get_right.stypy_call_defaults = defaults
    _get_right.stypy_call_varargs = varargs
    _get_right.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_right', ['tight_bbox', 'axes_bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_right', localization, ['tight_bbox', 'axes_bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_right(...)' code ##################

    # Getting the type of 'tight_bbox' (line 26)
    tight_bbox_153061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'tight_bbox')
    # Obtaining the member 'xmax' of a type (line 26)
    xmax_153062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), tight_bbox_153061, 'xmax')
    # Getting the type of 'axes_bbox' (line 26)
    axes_bbox_153063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'axes_bbox')
    # Obtaining the member 'xmax' of a type (line 26)
    xmax_153064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 29), axes_bbox_153063, 'xmax')
    # Applying the binary operator '-' (line 26)
    result_sub_153065 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), '-', xmax_153062, xmax_153064)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', result_sub_153065)
    
    # ################# End of '_get_right(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_right' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_153066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_right'
    return stypy_return_type_153066

# Assigning a type to the variable '_get_right' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_get_right', _get_right)

@norecursion
def _get_bottom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_bottom'
    module_type_store = module_type_store.open_function_context('_get_bottom', 29, 0, False)
    
    # Passed parameters checking function
    _get_bottom.stypy_localization = localization
    _get_bottom.stypy_type_of_self = None
    _get_bottom.stypy_type_store = module_type_store
    _get_bottom.stypy_function_name = '_get_bottom'
    _get_bottom.stypy_param_names_list = ['tight_bbox', 'axes_bbox']
    _get_bottom.stypy_varargs_param_name = None
    _get_bottom.stypy_kwargs_param_name = None
    _get_bottom.stypy_call_defaults = defaults
    _get_bottom.stypy_call_varargs = varargs
    _get_bottom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_bottom', ['tight_bbox', 'axes_bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_bottom', localization, ['tight_bbox', 'axes_bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_bottom(...)' code ##################

    # Getting the type of 'axes_bbox' (line 30)
    axes_bbox_153067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'axes_bbox')
    # Obtaining the member 'ymin' of a type (line 30)
    ymin_153068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), axes_bbox_153067, 'ymin')
    # Getting the type of 'tight_bbox' (line 30)
    tight_bbox_153069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'tight_bbox')
    # Obtaining the member 'ymin' of a type (line 30)
    ymin_153070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 28), tight_bbox_153069, 'ymin')
    # Applying the binary operator '-' (line 30)
    result_sub_153071 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), '-', ymin_153068, ymin_153070)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', result_sub_153071)
    
    # ################# End of '_get_bottom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_bottom' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_153072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153072)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_bottom'
    return stypy_return_type_153072

# Assigning a type to the variable '_get_bottom' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '_get_bottom', _get_bottom)

@norecursion
def _get_top(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_top'
    module_type_store = module_type_store.open_function_context('_get_top', 33, 0, False)
    
    # Passed parameters checking function
    _get_top.stypy_localization = localization
    _get_top.stypy_type_of_self = None
    _get_top.stypy_type_store = module_type_store
    _get_top.stypy_function_name = '_get_top'
    _get_top.stypy_param_names_list = ['tight_bbox', 'axes_bbox']
    _get_top.stypy_varargs_param_name = None
    _get_top.stypy_kwargs_param_name = None
    _get_top.stypy_call_defaults = defaults
    _get_top.stypy_call_varargs = varargs
    _get_top.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_top', ['tight_bbox', 'axes_bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_top', localization, ['tight_bbox', 'axes_bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_top(...)' code ##################

    # Getting the type of 'tight_bbox' (line 34)
    tight_bbox_153073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'tight_bbox')
    # Obtaining the member 'ymax' of a type (line 34)
    ymax_153074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), tight_bbox_153073, 'ymax')
    # Getting the type of 'axes_bbox' (line 34)
    axes_bbox_153075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'axes_bbox')
    # Obtaining the member 'ymax' of a type (line 34)
    ymax_153076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 29), axes_bbox_153075, 'ymax')
    # Applying the binary operator '-' (line 34)
    result_sub_153077 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), '-', ymax_153074, ymax_153076)
    
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', result_sub_153077)
    
    # ################# End of '_get_top(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_top' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_153078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153078)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_top'
    return stypy_return_type_153078

# Assigning a type to the variable '_get_top' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_get_top', _get_top)

@norecursion
def auto_adjust_subplotpars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 41)
    None_153079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'None')
    float_153080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'float')
    # Getting the type of 'None' (line 42)
    None_153081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 44), 'None')
    # Getting the type of 'None' (line 42)
    None_153082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 56), 'None')
    # Getting the type of 'None' (line 43)
    None_153083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'None')
    defaults = [None_153079, float_153080, None_153081, None_153082, None_153083]
    # Create a new context for function 'auto_adjust_subplotpars'
    module_type_store = module_type_store.open_function_context('auto_adjust_subplotpars', 37, 0, False)
    
    # Passed parameters checking function
    auto_adjust_subplotpars.stypy_localization = localization
    auto_adjust_subplotpars.stypy_type_of_self = None
    auto_adjust_subplotpars.stypy_type_store = module_type_store
    auto_adjust_subplotpars.stypy_function_name = 'auto_adjust_subplotpars'
    auto_adjust_subplotpars.stypy_param_names_list = ['fig', 'renderer', 'nrows_ncols', 'num1num2_list', 'subplot_list', 'ax_bbox_list', 'pad', 'h_pad', 'w_pad', 'rect']
    auto_adjust_subplotpars.stypy_varargs_param_name = None
    auto_adjust_subplotpars.stypy_kwargs_param_name = None
    auto_adjust_subplotpars.stypy_call_defaults = defaults
    auto_adjust_subplotpars.stypy_call_varargs = varargs
    auto_adjust_subplotpars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'auto_adjust_subplotpars', ['fig', 'renderer', 'nrows_ncols', 'num1num2_list', 'subplot_list', 'ax_bbox_list', 'pad', 'h_pad', 'w_pad', 'rect'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'auto_adjust_subplotpars', localization, ['fig', 'renderer', 'nrows_ncols', 'num1num2_list', 'subplot_list', 'ax_bbox_list', 'pad', 'h_pad', 'w_pad', 'rect'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'auto_adjust_subplotpars(...)' code ##################

    str_153084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n    Return a dictionary of subplot parameters so that spacing between\n    subplots are adjusted. Note that this function ignore geometry\n    information of subplot itself, but uses what is given by\n    *nrows_ncols* and *num1num2_list* parameteres. Also, the results could be\n    incorrect if some subplots have ``adjustable=datalim``.\n\n    Parameters:\n\n    nrows_ncols\n      number of rows and number of columns of the grid.\n\n    num1num2_list\n      list of numbers specifying the area occupied by the subplot\n\n    subplot_list\n      list of subplots that will be used to calcuate optimal subplot_params.\n\n    pad : float\n      padding between the figure edge and the edges of subplots, as a fraction\n      of the font-size.\n    h_pad, w_pad : float\n      padding (height/width) between edges of adjacent subplots.\n        Defaults to `pad_inches`.\n\n    rect\n      [left, bottom, right, top] in normalized (0, 1) figure coordinates.\n    ')
    
    # Assigning a Name to a Tuple (line 72):
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_153085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'nrows_ncols' (line 72)
    nrows_ncols_153086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'nrows_ncols')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___153087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), nrows_ncols_153086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_153088 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___153087, int_153085)
    
    # Assigning a type to the variable 'tuple_var_assignment_153003' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_153003', subscript_call_result_153088)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_153089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'nrows_ncols' (line 72)
    nrows_ncols_153090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'nrows_ncols')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___153091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), nrows_ncols_153090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_153092 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___153091, int_153089)
    
    # Assigning a type to the variable 'tuple_var_assignment_153004' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_153004', subscript_call_result_153092)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_153003' (line 72)
    tuple_var_assignment_153003_153093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_153003')
    # Assigning a type to the variable 'rows' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'rows', tuple_var_assignment_153003_153093)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_153004' (line 72)
    tuple_var_assignment_153004_153094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_153004')
    # Assigning a type to the variable 'cols' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 10), 'cols', tuple_var_assignment_153004_153094)
    
    # Assigning a BinOp to a Name (line 74):
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'pad' (line 74)
    pad_153095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'pad')
    
    # Call to get_size_in_points(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_153105 = {}
    
    # Call to FontProperties(...): (line 74)
    # Processing the call keyword arguments (line 74)
    
    # Obtaining the type of the subscript
    str_153097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'str', 'font.size')
    # Getting the type of 'rcParams' (line 75)
    rcParams_153098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'rcParams', False)
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___153099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), rcParams_153098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_153100 = invoke(stypy.reporting.localization.Localization(__file__, 75, 25), getitem___153099, str_153097)
    
    keyword_153101 = subscript_call_result_153100
    kwargs_153102 = {'size': keyword_153101}
    # Getting the type of 'FontProperties' (line 74)
    FontProperties_153096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'FontProperties', False)
    # Calling FontProperties(args, kwargs) (line 74)
    FontProperties_call_result_153103 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), FontProperties_153096, *[], **kwargs_153102)
    
    # Obtaining the member 'get_size_in_points' of a type (line 74)
    get_size_in_points_153104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 23), FontProperties_call_result_153103, 'get_size_in_points')
    # Calling get_size_in_points(args, kwargs) (line 74)
    get_size_in_points_call_result_153106 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), get_size_in_points_153104, *[], **kwargs_153105)
    
    # Applying the binary operator '*' (line 74)
    result_mul_153107 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 17), '*', pad_153095, get_size_in_points_call_result_153106)
    
    float_153108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 71), 'float')
    # Applying the binary operator 'div' (line 75)
    result_div_153109 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 69), 'div', result_mul_153107, float_153108)
    
    # Assigning a type to the variable 'pad_inches' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'pad_inches', result_div_153109)
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    # Getting the type of 'h_pad' (line 77)
    h_pad_153110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'h_pad')
    # Getting the type of 'None' (line 77)
    None_153111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'None')
    
    (may_be_153112, more_types_in_union_153113) = may_not_be_none(h_pad_153110, None_153111)

    if may_be_153112:

        if more_types_in_union_153113:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 78):
        
        # Assigning a BinOp to a Name (line 78):
        # Getting the type of 'h_pad' (line 78)
        h_pad_153114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'h_pad')
        
        # Call to get_size_in_points(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_153124 = {}
        
        # Call to FontProperties(...): (line 78)
        # Processing the call keyword arguments (line 78)
        
        # Obtaining the type of the subscript
        str_153116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 38), 'str', 'font.size')
        # Getting the type of 'rcParams' (line 79)
        rcParams_153117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___153118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 29), rcParams_153117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_153119 = invoke(stypy.reporting.localization.Localization(__file__, 79, 29), getitem___153118, str_153116)
        
        keyword_153120 = subscript_call_result_153119
        kwargs_153121 = {'size': keyword_153120}
        # Getting the type of 'FontProperties' (line 78)
        FontProperties_153115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'FontProperties', False)
        # Calling FontProperties(args, kwargs) (line 78)
        FontProperties_call_result_153122 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), FontProperties_153115, *[], **kwargs_153121)
        
        # Obtaining the member 'get_size_in_points' of a type (line 78)
        get_size_in_points_153123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), FontProperties_call_result_153122, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 78)
        get_size_in_points_call_result_153125 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), get_size_in_points_153123, *[], **kwargs_153124)
        
        # Applying the binary operator '*' (line 78)
        result_mul_153126 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 22), '*', h_pad_153114, get_size_in_points_call_result_153125)
        
        float_153127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 75), 'float')
        # Applying the binary operator 'div' (line 79)
        result_div_153128 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 73), 'div', result_mul_153126, float_153127)
        
        # Assigning a type to the variable 'vpad_inches' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'vpad_inches', result_div_153128)

        if more_types_in_union_153113:
            # Runtime conditional SSA for else branch (line 77)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_153112) or more_types_in_union_153113):
        
        # Assigning a Name to a Name (line 81):
        
        # Assigning a Name to a Name (line 81):
        # Getting the type of 'pad_inches' (line 81)
        pad_inches_153129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'pad_inches')
        # Assigning a type to the variable 'vpad_inches' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'vpad_inches', pad_inches_153129)

        if (may_be_153112 and more_types_in_union_153113):
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 83)
    # Getting the type of 'w_pad' (line 83)
    w_pad_153130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'w_pad')
    # Getting the type of 'None' (line 83)
    None_153131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'None')
    
    (may_be_153132, more_types_in_union_153133) = may_not_be_none(w_pad_153130, None_153131)

    if may_be_153132:

        if more_types_in_union_153133:
            # Runtime conditional SSA (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 84):
        
        # Assigning a BinOp to a Name (line 84):
        # Getting the type of 'w_pad' (line 84)
        w_pad_153134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'w_pad')
        
        # Call to get_size_in_points(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_153144 = {}
        
        # Call to FontProperties(...): (line 84)
        # Processing the call keyword arguments (line 84)
        
        # Obtaining the type of the subscript
        str_153136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 38), 'str', 'font.size')
        # Getting the type of 'rcParams' (line 85)
        rcParams_153137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___153138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 29), rcParams_153137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_153139 = invoke(stypy.reporting.localization.Localization(__file__, 85, 29), getitem___153138, str_153136)
        
        keyword_153140 = subscript_call_result_153139
        kwargs_153141 = {'size': keyword_153140}
        # Getting the type of 'FontProperties' (line 84)
        FontProperties_153135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'FontProperties', False)
        # Calling FontProperties(args, kwargs) (line 84)
        FontProperties_call_result_153142 = invoke(stypy.reporting.localization.Localization(__file__, 84, 30), FontProperties_153135, *[], **kwargs_153141)
        
        # Obtaining the member 'get_size_in_points' of a type (line 84)
        get_size_in_points_153143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), FontProperties_call_result_153142, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 84)
        get_size_in_points_call_result_153145 = invoke(stypy.reporting.localization.Localization(__file__, 84, 30), get_size_in_points_153143, *[], **kwargs_153144)
        
        # Applying the binary operator '*' (line 84)
        result_mul_153146 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 22), '*', w_pad_153134, get_size_in_points_call_result_153145)
        
        float_153147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 75), 'float')
        # Applying the binary operator 'div' (line 85)
        result_div_153148 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 73), 'div', result_mul_153146, float_153147)
        
        # Assigning a type to the variable 'hpad_inches' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'hpad_inches', result_div_153148)

        if more_types_in_union_153133:
            # Runtime conditional SSA for else branch (line 83)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_153132) or more_types_in_union_153133):
        
        # Assigning a Name to a Name (line 87):
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'pad_inches' (line 87)
        pad_inches_153149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'pad_inches')
        # Assigning a type to the variable 'hpad_inches' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'hpad_inches', pad_inches_153149)

        if (may_be_153132 and more_types_in_union_153133):
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'subplot_list' (line 89)
    subplot_list_153151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'subplot_list', False)
    # Processing the call keyword arguments (line 89)
    kwargs_153152 = {}
    # Getting the type of 'len' (line 89)
    len_153150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'len', False)
    # Calling len(args, kwargs) (line 89)
    len_call_result_153153 = invoke(stypy.reporting.localization.Localization(__file__, 89, 7), len_153150, *[subplot_list_153151], **kwargs_153152)
    
    int_153154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 28), 'int')
    # Applying the binary operator '==' (line 89)
    result_eq_153155 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 7), '==', len_call_result_153153, int_153154)
    
    # Testing the type of an if condition (line 89)
    if_condition_153156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), result_eq_153155)
    # Assigning a type to the variable 'if_condition_153156' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_153156', if_condition_153156)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 90)
    # Processing the call arguments (line 90)
    str_153158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'str', '')
    # Processing the call keyword arguments (line 90)
    kwargs_153159 = {}
    # Getting the type of 'RuntimeError' (line 90)
    RuntimeError_153157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 90)
    RuntimeError_call_result_153160 = invoke(stypy.reporting.localization.Localization(__file__, 90, 14), RuntimeError_153157, *[str_153158], **kwargs_153159)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 90, 8), RuntimeError_call_result_153160, 'raise parameter', BaseException)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'num1num2_list' (line 92)
    num1num2_list_153162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'num1num2_list', False)
    # Processing the call keyword arguments (line 92)
    kwargs_153163 = {}
    # Getting the type of 'len' (line 92)
    len_153161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'len', False)
    # Calling len(args, kwargs) (line 92)
    len_call_result_153164 = invoke(stypy.reporting.localization.Localization(__file__, 92, 7), len_153161, *[num1num2_list_153162], **kwargs_153163)
    
    
    # Call to len(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'subplot_list' (line 92)
    subplot_list_153166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'subplot_list', False)
    # Processing the call keyword arguments (line 92)
    kwargs_153167 = {}
    # Getting the type of 'len' (line 92)
    len_153165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'len', False)
    # Calling len(args, kwargs) (line 92)
    len_call_result_153168 = invoke(stypy.reporting.localization.Localization(__file__, 92, 29), len_153165, *[subplot_list_153166], **kwargs_153167)
    
    # Applying the binary operator '!=' (line 92)
    result_ne_153169 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 7), '!=', len_call_result_153164, len_call_result_153168)
    
    # Testing the type of an if condition (line 92)
    if_condition_153170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), result_ne_153169)
    # Assigning a type to the variable 'if_condition_153170' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_153170', if_condition_153170)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 93)
    # Processing the call arguments (line 93)
    str_153172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 27), 'str', '')
    # Processing the call keyword arguments (line 93)
    kwargs_153173 = {}
    # Getting the type of 'RuntimeError' (line 93)
    RuntimeError_153171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 93)
    RuntimeError_call_result_153174 = invoke(stypy.reporting.localization.Localization(__file__, 93, 14), RuntimeError_153171, *[str_153172], **kwargs_153173)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 93, 8), RuntimeError_call_result_153174, 'raise parameter', BaseException)
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 95)
    # Getting the type of 'rect' (line 95)
    rect_153175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'rect')
    # Getting the type of 'None' (line 95)
    None_153176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'None')
    
    (may_be_153177, more_types_in_union_153178) = may_be_none(rect_153175, None_153176)

    if may_be_153177:

        if more_types_in_union_153178:
            # Runtime conditional SSA (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 96):
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'None' (line 96)
        None_153179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'None')
        # Assigning a type to the variable 'margin_left' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'margin_left', None_153179)
        
        # Assigning a Name to a Name (line 97):
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'None' (line 97)
        None_153180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'None')
        # Assigning a type to the variable 'margin_bottom' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'margin_bottom', None_153180)
        
        # Assigning a Name to a Name (line 98):
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'None' (line 98)
        None_153181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'None')
        # Assigning a type to the variable 'margin_right' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'margin_right', None_153181)
        
        # Assigning a Name to a Name (line 99):
        
        # Assigning a Name to a Name (line 99):
        # Getting the type of 'None' (line 99)
        None_153182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'None')
        # Assigning a type to the variable 'margin_top' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'margin_top', None_153182)

        if more_types_in_union_153178:
            # Runtime conditional SSA for else branch (line 95)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_153177) or more_types_in_union_153178):
        
        # Assigning a Name to a Tuple (line 101):
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_153183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'rect' (line 101)
        rect_153184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'rect')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___153185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), rect_153184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_153186 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___153185, int_153183)
        
        # Assigning a type to the variable 'tuple_var_assignment_153005' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153005', subscript_call_result_153186)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_153187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'rect' (line 101)
        rect_153188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'rect')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___153189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), rect_153188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_153190 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___153189, int_153187)
        
        # Assigning a type to the variable 'tuple_var_assignment_153006' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153006', subscript_call_result_153190)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_153191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'rect' (line 101)
        rect_153192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'rect')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___153193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), rect_153192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_153194 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___153193, int_153191)
        
        # Assigning a type to the variable 'tuple_var_assignment_153007' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153007', subscript_call_result_153194)
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        int_153195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'int')
        # Getting the type of 'rect' (line 101)
        rect_153196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'rect')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___153197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), rect_153196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_153198 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), getitem___153197, int_153195)
        
        # Assigning a type to the variable 'tuple_var_assignment_153008' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153008', subscript_call_result_153198)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_153005' (line 101)
        tuple_var_assignment_153005_153199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153005')
        # Assigning a type to the variable 'margin_left' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'margin_left', tuple_var_assignment_153005_153199)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_153006' (line 101)
        tuple_var_assignment_153006_153200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153006')
        # Assigning a type to the variable 'margin_bottom' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'margin_bottom', tuple_var_assignment_153006_153200)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_153007' (line 101)
        tuple_var_assignment_153007_153201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153007')
        # Assigning a type to the variable '_right' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 36), '_right', tuple_var_assignment_153007_153201)
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'tuple_var_assignment_153008' (line 101)
        tuple_var_assignment_153008_153202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tuple_var_assignment_153008')
        # Assigning a type to the variable '_top' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), '_top', tuple_var_assignment_153008_153202)
        
        # Getting the type of '_right' (line 102)
        _right_153203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), '_right')
        # Testing the type of an if condition (line 102)
        if_condition_153204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), _right_153203)
        # Assigning a type to the variable 'if_condition_153204' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_153204', if_condition_153204)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 103):
        
        # Assigning a BinOp to a Name (line 103):
        float_153205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'float')
        # Getting the type of '_right' (line 103)
        _right_153206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), '_right')
        # Applying the binary operator '-' (line 103)
        result_sub_153207 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 27), '-', float_153205, _right_153206)
        
        # Assigning a type to the variable 'margin_right' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'margin_right', result_sub_153207)
        # SSA branch for the else part of an if statement (line 102)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 105):
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'None' (line 105)
        None_153208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'None')
        # Assigning a type to the variable 'margin_right' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'margin_right', None_153208)
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of '_top' (line 106)
        _top_153209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), '_top')
        # Testing the type of an if condition (line 106)
        if_condition_153210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), _top_153209)
        # Assigning a type to the variable 'if_condition_153210' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_153210', if_condition_153210)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 107):
        
        # Assigning a BinOp to a Name (line 107):
        float_153211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'float')
        # Getting the type of '_top' (line 107)
        _top_153212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), '_top')
        # Applying the binary operator '-' (line 107)
        result_sub_153213 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 25), '-', float_153211, _top_153212)
        
        # Assigning a type to the variable 'margin_top' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'margin_top', result_sub_153213)
        # SSA branch for the else part of an if statement (line 106)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 109):
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'None' (line 109)
        None_153214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'None')
        # Assigning a type to the variable 'margin_top' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'margin_top', None_153214)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_153177 and more_types_in_union_153178):
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a ListComp to a Name (line 111):
    
    # Assigning a ListComp to a Name (line 111):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'rows' (line 111)
    rows_153217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'rows', False)
    int_153218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 41), 'int')
    # Applying the binary operator '+' (line 111)
    result_add_153219 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 34), '+', rows_153217, int_153218)
    
    # Getting the type of 'cols' (line 111)
    cols_153220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 46), 'cols', False)
    # Applying the binary operator '*' (line 111)
    result_mul_153221 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 33), '*', result_add_153219, cols_153220)
    
    # Processing the call keyword arguments (line 111)
    kwargs_153222 = {}
    # Getting the type of 'range' (line 111)
    range_153216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'range', False)
    # Calling range(args, kwargs) (line 111)
    range_call_result_153223 = invoke(stypy.reporting.localization.Localization(__file__, 111, 27), range_153216, *[result_mul_153221], **kwargs_153222)
    
    comprehension_153224 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 15), range_call_result_153223)
    # Assigning a type to the variable 'i' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'i', comprehension_153224)
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_153215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    
    list_153225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 15), list_153225, list_153215)
    # Assigning a type to the variable 'vspaces' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'vspaces', list_153225)
    
    # Assigning a ListComp to a Name (line 112):
    
    # Assigning a ListComp to a Name (line 112):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'rows' (line 112)
    rows_153228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 33), 'rows', False)
    # Getting the type of 'cols' (line 112)
    cols_153229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 41), 'cols', False)
    int_153230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 48), 'int')
    # Applying the binary operator '+' (line 112)
    result_add_153231 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 41), '+', cols_153229, int_153230)
    
    # Applying the binary operator '*' (line 112)
    result_mul_153232 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 33), '*', rows_153228, result_add_153231)
    
    # Processing the call keyword arguments (line 112)
    kwargs_153233 = {}
    # Getting the type of 'range' (line 112)
    range_153227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'range', False)
    # Calling range(args, kwargs) (line 112)
    range_call_result_153234 = invoke(stypy.reporting.localization.Localization(__file__, 112, 27), range_153227, *[result_mul_153232], **kwargs_153233)
    
    comprehension_153235 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), range_call_result_153234)
    # Assigning a type to the variable 'i' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'i', comprehension_153235)
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_153226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    
    list_153236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_153236, list_153226)
    # Assigning a type to the variable 'hspaces' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'hspaces', list_153236)
    
    # Assigning a Attribute to a Name (line 114):
    
    # Assigning a Attribute to a Name (line 114):
    # Getting the type of 'Bbox' (line 114)
    Bbox_153237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'Bbox')
    # Obtaining the member 'union' of a type (line 114)
    union_153238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), Bbox_153237, 'union')
    # Assigning a type to the variable 'union' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'union', union_153238)
    
    # Type idiom detected: calculating its left and rigth part (line 116)
    # Getting the type of 'ax_bbox_list' (line 116)
    ax_bbox_list_153239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'ax_bbox_list')
    # Getting the type of 'None' (line 116)
    None_153240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'None')
    
    (may_be_153241, more_types_in_union_153242) = may_be_none(ax_bbox_list_153239, None_153240)

    if may_be_153241:

        if more_types_in_union_153242:
            # Runtime conditional SSA (line 116)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 117):
        
        # Assigning a List to a Name (line 117):
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_153243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        
        # Assigning a type to the variable 'ax_bbox_list' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'ax_bbox_list', list_153243)
        
        # Getting the type of 'subplot_list' (line 118)
        subplot_list_153244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'subplot_list')
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 8), subplot_list_153244)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_153245 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 8), subplot_list_153244)
        # Assigning a type to the variable 'subplots' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'subplots', for_loop_var_153245)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to union(...): (line 119)
        # Processing the call arguments (line 119)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'subplots' (line 120)
        subplots_153253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 39), 'subplots', False)
        comprehension_153254 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 29), subplots_153253)
        # Assigning a type to the variable 'ax' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'ax', comprehension_153254)
        
        # Call to get_position(...): (line 119)
        # Processing the call keyword arguments (line 119)
        # Getting the type of 'True' (line 119)
        True_153249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 54), 'True', False)
        keyword_153250 = True_153249
        kwargs_153251 = {'original': keyword_153250}
        # Getting the type of 'ax' (line 119)
        ax_153247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'ax', False)
        # Obtaining the member 'get_position' of a type (line 119)
        get_position_153248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 29), ax_153247, 'get_position')
        # Calling get_position(args, kwargs) (line 119)
        get_position_call_result_153252 = invoke(stypy.reporting.localization.Localization(__file__, 119, 29), get_position_153248, *[], **kwargs_153251)
        
        list_153255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 29), list_153255, get_position_call_result_153252)
        # Processing the call keyword arguments (line 119)
        kwargs_153256 = {}
        # Getting the type of 'union' (line 119)
        union_153246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'union', False)
        # Calling union(args, kwargs) (line 119)
        union_call_result_153257 = invoke(stypy.reporting.localization.Localization(__file__, 119, 22), union_153246, *[list_153255], **kwargs_153256)
        
        # Assigning a type to the variable 'ax_bbox' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'ax_bbox', union_call_result_153257)
        
        # Call to append(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'ax_bbox' (line 121)
        ax_bbox_153260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'ax_bbox', False)
        # Processing the call keyword arguments (line 121)
        kwargs_153261 = {}
        # Getting the type of 'ax_bbox_list' (line 121)
        ax_bbox_list_153258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'ax_bbox_list', False)
        # Obtaining the member 'append' of a type (line 121)
        append_153259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), ax_bbox_list_153258, 'append')
        # Calling append(args, kwargs) (line 121)
        append_call_result_153262 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), append_153259, *[ax_bbox_153260], **kwargs_153261)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_153242:
            # SSA join for if statement (line 116)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to zip(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'subplot_list' (line 123)
    subplot_list_153264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'subplot_list', False)
    # Getting the type of 'ax_bbox_list' (line 124)
    ax_bbox_list_153265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 47), 'ax_bbox_list', False)
    # Getting the type of 'num1num2_list' (line 125)
    num1num2_list_153266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'num1num2_list', False)
    # Processing the call keyword arguments (line 123)
    kwargs_153267 = {}
    # Getting the type of 'zip' (line 123)
    zip_153263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'zip', False)
    # Calling zip(args, kwargs) (line 123)
    zip_call_result_153268 = invoke(stypy.reporting.localization.Localization(__file__, 123, 43), zip_153263, *[subplot_list_153264, ax_bbox_list_153265, num1num2_list_153266], **kwargs_153267)
    
    # Testing the type of a for loop iterable (line 123)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 4), zip_call_result_153268)
    # Getting the type of the for loop variable (line 123)
    for_loop_var_153269 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 4), zip_call_result_153268)
    # Assigning a type to the variable 'subplots' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'subplots', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), for_loop_var_153269))
    # Assigning a type to the variable 'ax_bbox' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'ax_bbox', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), for_loop_var_153269))
    # Assigning a type to the variable 'num1' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'num1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), for_loop_var_153269))
    # Assigning a type to the variable 'num2' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'num2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), for_loop_var_153269))
    # SSA begins for a for statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to all(...): (line 126)
    # Processing the call arguments (line 126)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'subplots' (line 126)
    subplots_153276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 47), 'subplots', False)
    comprehension_153277 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), subplots_153276)
    # Assigning a type to the variable 'ax' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'ax', comprehension_153277)
    
    
    # Call to get_visible(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_153273 = {}
    # Getting the type of 'ax' (line 126)
    ax_153271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'ax', False)
    # Obtaining the member 'get_visible' of a type (line 126)
    get_visible_153272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 20), ax_153271, 'get_visible')
    # Calling get_visible(args, kwargs) (line 126)
    get_visible_call_result_153274 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), get_visible_153272, *[], **kwargs_153273)
    
    # Applying the 'not' unary operator (line 126)
    result_not__153275 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), 'not', get_visible_call_result_153274)
    
    list_153278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), list_153278, result_not__153275)
    # Processing the call keyword arguments (line 126)
    kwargs_153279 = {}
    # Getting the type of 'all' (line 126)
    all_153270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'all', False)
    # Calling all(args, kwargs) (line 126)
    all_call_result_153280 = invoke(stypy.reporting.localization.Localization(__file__, 126, 11), all_153270, *[list_153278], **kwargs_153279)
    
    # Testing the type of an if condition (line 126)
    if_condition_153281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 8), all_call_result_153280)
    # Assigning a type to the variable 'if_condition_153281' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'if_condition_153281', if_condition_153281)
    # SSA begins for if statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to union(...): (line 129)
    # Processing the call arguments (line 129)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'subplots' (line 129)
    subplots_153292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 69), 'subplots', False)
    comprehension_153293 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 32), subplots_153292)
    # Assigning a type to the variable 'ax' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'ax', comprehension_153293)
    
    # Call to get_visible(...): (line 130)
    # Processing the call keyword arguments (line 130)
    kwargs_153290 = {}
    # Getting the type of 'ax' (line 130)
    ax_153288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'ax', False)
    # Obtaining the member 'get_visible' of a type (line 130)
    get_visible_153289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), ax_153288, 'get_visible')
    # Calling get_visible(args, kwargs) (line 130)
    get_visible_call_result_153291 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), get_visible_153289, *[], **kwargs_153290)
    
    
    # Call to get_tightbbox(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'renderer' (line 129)
    renderer_153285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'renderer', False)
    # Processing the call keyword arguments (line 129)
    kwargs_153286 = {}
    # Getting the type of 'ax' (line 129)
    ax_153283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'ax', False)
    # Obtaining the member 'get_tightbbox' of a type (line 129)
    get_tightbbox_153284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 32), ax_153283, 'get_tightbbox')
    # Calling get_tightbbox(args, kwargs) (line 129)
    get_tightbbox_call_result_153287 = invoke(stypy.reporting.localization.Localization(__file__, 129, 32), get_tightbbox_153284, *[renderer_153285], **kwargs_153286)
    
    list_153294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 32), list_153294, get_tightbbox_call_result_153287)
    # Processing the call keyword arguments (line 129)
    kwargs_153295 = {}
    # Getting the type of 'union' (line 129)
    union_153282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'union', False)
    # Calling union(args, kwargs) (line 129)
    union_call_result_153296 = invoke(stypy.reporting.localization.Localization(__file__, 129, 25), union_153282, *[list_153294], **kwargs_153295)
    
    # Assigning a type to the variable 'tight_bbox_raw' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'tight_bbox_raw', union_call_result_153296)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to TransformedBbox(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'tight_bbox_raw' (line 131)
    tight_bbox_raw_153298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'tight_bbox_raw', False)
    
    # Call to inverted(...): (line 132)
    # Processing the call keyword arguments (line 132)
    kwargs_153302 = {}
    # Getting the type of 'fig' (line 132)
    fig_153299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 37), 'fig', False)
    # Obtaining the member 'transFigure' of a type (line 132)
    transFigure_153300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 37), fig_153299, 'transFigure')
    # Obtaining the member 'inverted' of a type (line 132)
    inverted_153301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 37), transFigure_153300, 'inverted')
    # Calling inverted(args, kwargs) (line 132)
    inverted_call_result_153303 = invoke(stypy.reporting.localization.Localization(__file__, 132, 37), inverted_153301, *[], **kwargs_153302)
    
    # Processing the call keyword arguments (line 131)
    kwargs_153304 = {}
    # Getting the type of 'TransformedBbox' (line 131)
    TransformedBbox_153297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'TransformedBbox', False)
    # Calling TransformedBbox(args, kwargs) (line 131)
    TransformedBbox_call_result_153305 = invoke(stypy.reporting.localization.Localization(__file__, 131, 21), TransformedBbox_153297, *[tight_bbox_raw_153298, inverted_call_result_153303], **kwargs_153304)
    
    # Assigning a type to the variable 'tight_bbox' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'tight_bbox', TransformedBbox_call_result_153305)
    
    # Assigning a Call to a Tuple (line 134):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'num1' (line 134)
    num1_153307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'num1', False)
    # Getting the type of 'cols' (line 134)
    cols_153308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 34), 'cols', False)
    # Processing the call keyword arguments (line 134)
    kwargs_153309 = {}
    # Getting the type of 'divmod' (line 134)
    divmod_153306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'divmod', False)
    # Calling divmod(args, kwargs) (line 134)
    divmod_call_result_153310 = invoke(stypy.reporting.localization.Localization(__file__, 134, 21), divmod_153306, *[num1_153307, cols_153308], **kwargs_153309)
    
    # Assigning a type to the variable 'call_assignment_153009' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153009', divmod_call_result_153310)
    
    # Assigning a Call to a Name (line 134):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153314 = {}
    # Getting the type of 'call_assignment_153009' (line 134)
    call_assignment_153009_153311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153009', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___153312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), call_assignment_153009_153311, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153315 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153312, *[int_153313], **kwargs_153314)
    
    # Assigning a type to the variable 'call_assignment_153010' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153010', getitem___call_result_153315)
    
    # Assigning a Name to a Name (line 134):
    # Getting the type of 'call_assignment_153010' (line 134)
    call_assignment_153010_153316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153010')
    # Assigning a type to the variable 'row1' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'row1', call_assignment_153010_153316)
    
    # Assigning a Call to a Name (line 134):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153320 = {}
    # Getting the type of 'call_assignment_153009' (line 134)
    call_assignment_153009_153317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153009', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___153318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), call_assignment_153009_153317, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153321 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153318, *[int_153319], **kwargs_153320)
    
    # Assigning a type to the variable 'call_assignment_153011' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153011', getitem___call_result_153321)
    
    # Assigning a Name to a Name (line 134):
    # Getting the type of 'call_assignment_153011' (line 134)
    call_assignment_153011_153322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'call_assignment_153011')
    # Assigning a type to the variable 'col1' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 14), 'col1', call_assignment_153011_153322)
    
    # Type idiom detected: calculating its left and rigth part (line 136)
    # Getting the type of 'num2' (line 136)
    num2_153323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'num2')
    # Getting the type of 'None' (line 136)
    None_153324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'None')
    
    (may_be_153325, more_types_in_union_153326) = may_be_none(num2_153323, None_153324)

    if may_be_153325:

        if more_types_in_union_153326:
            # Runtime conditional SSA (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to _get_left(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'tight_bbox' (line 139)
        tight_bbox_153339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 139)
        ax_bbox_153340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 62), 'ax_bbox', False)
        # Processing the call keyword arguments (line 139)
        kwargs_153341 = {}
        # Getting the type of '_get_left' (line 139)
        _get_left_153338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 40), '_get_left', False)
        # Calling _get_left(args, kwargs) (line 139)
        _get_left_call_result_153342 = invoke(stypy.reporting.localization.Localization(__file__, 139, 40), _get_left_153338, *[tight_bbox_153339, ax_bbox_153340], **kwargs_153341)
        
        # Processing the call keyword arguments (line 138)
        kwargs_153343 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 138)
        row1_153327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'row1', False)
        # Getting the type of 'cols' (line 138)
        cols_153328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'cols', False)
        int_153329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'int')
        # Applying the binary operator '+' (line 138)
        result_add_153330 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 28), '+', cols_153328, int_153329)
        
        # Applying the binary operator '*' (line 138)
        result_mul_153331 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '*', row1_153327, result_add_153330)
        
        # Getting the type of 'col1' (line 138)
        col1_153332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'col1', False)
        # Applying the binary operator '+' (line 138)
        result_add_153333 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '+', result_mul_153331, col1_153332)
        
        # Getting the type of 'hspaces' (line 138)
        hspaces_153334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'hspaces', False)
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___153335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), hspaces_153334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_153336 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), getitem___153335, result_add_153333)
        
        # Obtaining the member 'append' of a type (line 138)
        append_153337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), subscript_call_result_153336, 'append')
        # Calling append(args, kwargs) (line 138)
        append_call_result_153344 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), append_153337, *[_get_left_call_result_153342], **kwargs_153343)
        
        
        # Call to append(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Call to _get_right(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'tight_bbox' (line 142)
        tight_bbox_153359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 51), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 142)
        ax_bbox_153360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 63), 'ax_bbox', False)
        # Processing the call keyword arguments (line 142)
        kwargs_153361 = {}
        # Getting the type of '_get_right' (line 142)
        _get_right_153358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 40), '_get_right', False)
        # Calling _get_right(args, kwargs) (line 142)
        _get_right_call_result_153362 = invoke(stypy.reporting.localization.Localization(__file__, 142, 40), _get_right_153358, *[tight_bbox_153359, ax_bbox_153360], **kwargs_153361)
        
        # Processing the call keyword arguments (line 141)
        kwargs_153363 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 141)
        row1_153345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'row1', False)
        # Getting the type of 'cols' (line 141)
        cols_153346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'cols', False)
        int_153347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'int')
        # Applying the binary operator '+' (line 141)
        result_add_153348 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 28), '+', cols_153346, int_153347)
        
        # Applying the binary operator '*' (line 141)
        result_mul_153349 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 20), '*', row1_153345, result_add_153348)
        
        # Getting the type of 'col1' (line 141)
        col1_153350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'col1', False)
        int_153351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 48), 'int')
        # Applying the binary operator '+' (line 141)
        result_add_153352 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 41), '+', col1_153350, int_153351)
        
        # Applying the binary operator '+' (line 141)
        result_add_153353 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 20), '+', result_mul_153349, result_add_153352)
        
        # Getting the type of 'hspaces' (line 141)
        hspaces_153354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'hspaces', False)
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___153355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), hspaces_153354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_153356 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), getitem___153355, result_add_153353)
        
        # Obtaining the member 'append' of a type (line 141)
        append_153357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), subscript_call_result_153356, 'append')
        # Calling append(args, kwargs) (line 141)
        append_call_result_153364 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), append_153357, *[_get_right_call_result_153362], **kwargs_153363)
        
        
        # Call to append(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to _get_top(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'tight_bbox' (line 145)
        tight_bbox_153375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 49), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 145)
        ax_bbox_153376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 61), 'ax_bbox', False)
        # Processing the call keyword arguments (line 145)
        kwargs_153377 = {}
        # Getting the type of '_get_top' (line 145)
        _get_top_153374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 40), '_get_top', False)
        # Calling _get_top(args, kwargs) (line 145)
        _get_top_call_result_153378 = invoke(stypy.reporting.localization.Localization(__file__, 145, 40), _get_top_153374, *[tight_bbox_153375, ax_bbox_153376], **kwargs_153377)
        
        # Processing the call keyword arguments (line 144)
        kwargs_153379 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 144)
        row1_153365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'row1', False)
        # Getting the type of 'cols' (line 144)
        cols_153366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'cols', False)
        # Applying the binary operator '*' (line 144)
        result_mul_153367 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), '*', row1_153365, cols_153366)
        
        # Getting the type of 'col1' (line 144)
        col1_153368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'col1', False)
        # Applying the binary operator '+' (line 144)
        result_add_153369 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), '+', result_mul_153367, col1_153368)
        
        # Getting the type of 'vspaces' (line 144)
        vspaces_153370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'vspaces', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___153371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), vspaces_153370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_153372 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), getitem___153371, result_add_153369)
        
        # Obtaining the member 'append' of a type (line 144)
        append_153373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), subscript_call_result_153372, 'append')
        # Calling append(args, kwargs) (line 144)
        append_call_result_153380 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), append_153373, *[_get_top_call_result_153378], **kwargs_153379)
        
        
        # Call to append(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to _get_bottom(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'tight_bbox' (line 148)
        tight_bbox_153393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 52), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 148)
        ax_bbox_153394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 64), 'ax_bbox', False)
        # Processing the call keyword arguments (line 148)
        kwargs_153395 = {}
        # Getting the type of '_get_bottom' (line 148)
        _get_bottom_153392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), '_get_bottom', False)
        # Calling _get_bottom(args, kwargs) (line 148)
        _get_bottom_call_result_153396 = invoke(stypy.reporting.localization.Localization(__file__, 148, 40), _get_bottom_153392, *[tight_bbox_153393, ax_bbox_153394], **kwargs_153395)
        
        # Processing the call keyword arguments (line 147)
        kwargs_153397 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 147)
        row1_153381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'row1', False)
        int_153382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 28), 'int')
        # Applying the binary operator '+' (line 147)
        result_add_153383 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 21), '+', row1_153381, int_153382)
        
        # Getting the type of 'cols' (line 147)
        cols_153384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'cols', False)
        # Applying the binary operator '*' (line 147)
        result_mul_153385 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 20), '*', result_add_153383, cols_153384)
        
        # Getting the type of 'col1' (line 147)
        col1_153386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 40), 'col1', False)
        # Applying the binary operator '+' (line 147)
        result_add_153387 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 20), '+', result_mul_153385, col1_153386)
        
        # Getting the type of 'vspaces' (line 147)
        vspaces_153388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'vspaces', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___153389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), vspaces_153388, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_153390 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), getitem___153389, result_add_153387)
        
        # Obtaining the member 'append' of a type (line 147)
        append_153391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), subscript_call_result_153390, 'append')
        # Calling append(args, kwargs) (line 147)
        append_call_result_153398 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), append_153391, *[_get_bottom_call_result_153396], **kwargs_153397)
        

        if more_types_in_union_153326:
            # Runtime conditional SSA for else branch (line 136)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_153325) or more_types_in_union_153326):
        
        # Assigning a Call to a Tuple (line 151):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'num2' (line 151)
        num2_153400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'num2', False)
        # Getting the type of 'cols' (line 151)
        cols_153401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 38), 'cols', False)
        # Processing the call keyword arguments (line 151)
        kwargs_153402 = {}
        # Getting the type of 'divmod' (line 151)
        divmod_153399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'divmod', False)
        # Calling divmod(args, kwargs) (line 151)
        divmod_call_result_153403 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), divmod_153399, *[num2_153400, cols_153401], **kwargs_153402)
        
        # Assigning a type to the variable 'call_assignment_153012' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153012', divmod_call_result_153403)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_153406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
        # Processing the call keyword arguments
        kwargs_153407 = {}
        # Getting the type of 'call_assignment_153012' (line 151)
        call_assignment_153012_153404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153012', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___153405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), call_assignment_153012_153404, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_153408 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153405, *[int_153406], **kwargs_153407)
        
        # Assigning a type to the variable 'call_assignment_153013' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153013', getitem___call_result_153408)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_153013' (line 151)
        call_assignment_153013_153409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153013')
        # Assigning a type to the variable 'row2' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'row2', call_assignment_153013_153409)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_153412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
        # Processing the call keyword arguments
        kwargs_153413 = {}
        # Getting the type of 'call_assignment_153012' (line 151)
        call_assignment_153012_153410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153012', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___153411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), call_assignment_153012_153410, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_153414 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153411, *[int_153412], **kwargs_153413)
        
        # Assigning a type to the variable 'call_assignment_153014' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153014', getitem___call_result_153414)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'call_assignment_153014' (line 151)
        call_assignment_153014_153415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'call_assignment_153014')
        # Assigning a type to the variable 'col2' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'col2', call_assignment_153014_153415)
        
        
        # Call to range(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'row1' (line 153)
        row1_153417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'row1', False)
        # Getting the type of 'row2' (line 153)
        row2_153418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'row2', False)
        int_153419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 44), 'int')
        # Applying the binary operator '+' (line 153)
        result_add_153420 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 37), '+', row2_153418, int_153419)
        
        # Processing the call keyword arguments (line 153)
        kwargs_153421 = {}
        # Getting the type of 'range' (line 153)
        range_153416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'range', False)
        # Calling range(args, kwargs) (line 153)
        range_call_result_153422 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), range_153416, *[row1_153417, result_add_153420], **kwargs_153421)
        
        # Testing the type of a for loop iterable (line 153)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 12), range_call_result_153422)
        # Getting the type of the for loop variable (line 153)
        for_loop_var_153423 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 12), range_call_result_153422)
        # Assigning a type to the variable 'row_i' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'row_i', for_loop_var_153423)
        # SSA begins for a for statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to _get_left(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'tight_bbox' (line 156)
        tight_bbox_153436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 46), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 156)
        ax_bbox_153437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 58), 'ax_bbox', False)
        # Processing the call keyword arguments (line 156)
        kwargs_153438 = {}
        # Getting the type of '_get_left' (line 156)
        _get_left_153435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 36), '_get_left', False)
        # Calling _get_left(args, kwargs) (line 156)
        _get_left_call_result_153439 = invoke(stypy.reporting.localization.Localization(__file__, 156, 36), _get_left_153435, *[tight_bbox_153436, ax_bbox_153437], **kwargs_153438)
        
        # Processing the call keyword arguments (line 155)
        kwargs_153440 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row_i' (line 155)
        row_i_153424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'row_i', False)
        # Getting the type of 'cols' (line 155)
        cols_153425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'cols', False)
        int_153426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 40), 'int')
        # Applying the binary operator '+' (line 155)
        result_add_153427 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 33), '+', cols_153425, int_153426)
        
        # Applying the binary operator '*' (line 155)
        result_mul_153428 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 24), '*', row_i_153424, result_add_153427)
        
        # Getting the type of 'col1' (line 155)
        col1_153429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 45), 'col1', False)
        # Applying the binary operator '+' (line 155)
        result_add_153430 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 24), '+', result_mul_153428, col1_153429)
        
        # Getting the type of 'hspaces' (line 155)
        hspaces_153431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'hspaces', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___153432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), hspaces_153431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_153433 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), getitem___153432, result_add_153430)
        
        # Obtaining the member 'append' of a type (line 155)
        append_153434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), subscript_call_result_153433, 'append')
        # Calling append(args, kwargs) (line 155)
        append_call_result_153441 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), append_153434, *[_get_left_call_result_153439], **kwargs_153440)
        
        
        # Call to append(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Call to _get_right(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'tight_bbox' (line 159)
        tight_bbox_153456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 47), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 159)
        ax_bbox_153457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'ax_bbox', False)
        # Processing the call keyword arguments (line 159)
        kwargs_153458 = {}
        # Getting the type of '_get_right' (line 159)
        _get_right_153455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), '_get_right', False)
        # Calling _get_right(args, kwargs) (line 159)
        _get_right_call_result_153459 = invoke(stypy.reporting.localization.Localization(__file__, 159, 36), _get_right_153455, *[tight_bbox_153456, ax_bbox_153457], **kwargs_153458)
        
        # Processing the call keyword arguments (line 158)
        kwargs_153460 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row_i' (line 158)
        row_i_153442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'row_i', False)
        # Getting the type of 'cols' (line 158)
        cols_153443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'cols', False)
        int_153444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 40), 'int')
        # Applying the binary operator '+' (line 158)
        result_add_153445 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 33), '+', cols_153443, int_153444)
        
        # Applying the binary operator '*' (line 158)
        result_mul_153446 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 24), '*', row_i_153442, result_add_153445)
        
        # Getting the type of 'col2' (line 158)
        col2_153447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'col2', False)
        int_153448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 53), 'int')
        # Applying the binary operator '+' (line 158)
        result_add_153449 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 46), '+', col2_153447, int_153448)
        
        # Applying the binary operator '+' (line 158)
        result_add_153450 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 24), '+', result_mul_153446, result_add_153449)
        
        # Getting the type of 'hspaces' (line 158)
        hspaces_153451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'hspaces', False)
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___153452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), hspaces_153451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_153453 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), getitem___153452, result_add_153450)
        
        # Obtaining the member 'append' of a type (line 158)
        append_153454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), subscript_call_result_153453, 'append')
        # Calling append(args, kwargs) (line 158)
        append_call_result_153461 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), append_153454, *[_get_right_call_result_153459], **kwargs_153460)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'col1' (line 160)
        col1_153463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'col1', False)
        # Getting the type of 'col2' (line 160)
        col2_153464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'col2', False)
        int_153465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 44), 'int')
        # Applying the binary operator '+' (line 160)
        result_add_153466 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 37), '+', col2_153464, int_153465)
        
        # Processing the call keyword arguments (line 160)
        kwargs_153467 = {}
        # Getting the type of 'range' (line 160)
        range_153462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'range', False)
        # Calling range(args, kwargs) (line 160)
        range_call_result_153468 = invoke(stypy.reporting.localization.Localization(__file__, 160, 25), range_153462, *[col1_153463, result_add_153466], **kwargs_153467)
        
        # Testing the type of a for loop iterable (line 160)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 12), range_call_result_153468)
        # Getting the type of the for loop variable (line 160)
        for_loop_var_153469 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 12), range_call_result_153468)
        # Assigning a type to the variable 'col_i' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'col_i', for_loop_var_153469)
        # SSA begins for a for statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Call to _get_top(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'tight_bbox' (line 163)
        tight_bbox_153480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 45), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 163)
        ax_bbox_153481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 57), 'ax_bbox', False)
        # Processing the call keyword arguments (line 163)
        kwargs_153482 = {}
        # Getting the type of '_get_top' (line 163)
        _get_top_153479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), '_get_top', False)
        # Calling _get_top(args, kwargs) (line 163)
        _get_top_call_result_153483 = invoke(stypy.reporting.localization.Localization(__file__, 163, 36), _get_top_153479, *[tight_bbox_153480, ax_bbox_153481], **kwargs_153482)
        
        # Processing the call keyword arguments (line 162)
        kwargs_153484 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row1' (line 162)
        row1_153470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'row1', False)
        # Getting the type of 'cols' (line 162)
        cols_153471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'cols', False)
        # Applying the binary operator '*' (line 162)
        result_mul_153472 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 24), '*', row1_153470, cols_153471)
        
        # Getting the type of 'col_i' (line 162)
        col_i_153473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'col_i', False)
        # Applying the binary operator '+' (line 162)
        result_add_153474 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 24), '+', result_mul_153472, col_i_153473)
        
        # Getting the type of 'vspaces' (line 162)
        vspaces_153475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'vspaces', False)
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___153476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), vspaces_153475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_153477 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), getitem___153476, result_add_153474)
        
        # Obtaining the member 'append' of a type (line 162)
        append_153478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), subscript_call_result_153477, 'append')
        # Calling append(args, kwargs) (line 162)
        append_call_result_153485 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), append_153478, *[_get_top_call_result_153483], **kwargs_153484)
        
        
        # Call to append(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to _get_bottom(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'tight_bbox' (line 166)
        tight_bbox_153498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'tight_bbox', False)
        # Getting the type of 'ax_bbox' (line 166)
        ax_bbox_153499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 60), 'ax_bbox', False)
        # Processing the call keyword arguments (line 166)
        kwargs_153500 = {}
        # Getting the type of '_get_bottom' (line 166)
        _get_bottom_153497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), '_get_bottom', False)
        # Calling _get_bottom(args, kwargs) (line 166)
        _get_bottom_call_result_153501 = invoke(stypy.reporting.localization.Localization(__file__, 166, 36), _get_bottom_153497, *[tight_bbox_153498, ax_bbox_153499], **kwargs_153500)
        
        # Processing the call keyword arguments (line 165)
        kwargs_153502 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'row2' (line 165)
        row2_153486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'row2', False)
        int_153487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
        # Applying the binary operator '+' (line 165)
        result_add_153488 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 25), '+', row2_153486, int_153487)
        
        # Getting the type of 'cols' (line 165)
        cols_153489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'cols', False)
        # Applying the binary operator '*' (line 165)
        result_mul_153490 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '*', result_add_153488, cols_153489)
        
        # Getting the type of 'col_i' (line 165)
        col_i_153491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 44), 'col_i', False)
        # Applying the binary operator '+' (line 165)
        result_add_153492 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '+', result_mul_153490, col_i_153491)
        
        # Getting the type of 'vspaces' (line 165)
        vspaces_153493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'vspaces', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___153494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), vspaces_153493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_153495 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), getitem___153494, result_add_153492)
        
        # Obtaining the member 'append' of a type (line 165)
        append_153496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), subscript_call_result_153495, 'append')
        # Calling append(args, kwargs) (line 165)
        append_call_result_153503 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), append_153496, *[_get_bottom_call_result_153501], **kwargs_153502)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_153325 and more_types_in_union_153326):
            # SSA join for if statement (line 136)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 168):
    
    # Assigning a Call to a Name:
    
    # Call to get_size_inches(...): (line 168)
    # Processing the call keyword arguments (line 168)
    kwargs_153506 = {}
    # Getting the type of 'fig' (line 168)
    fig_153504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'fig', False)
    # Obtaining the member 'get_size_inches' of a type (line 168)
    get_size_inches_153505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 38), fig_153504, 'get_size_inches')
    # Calling get_size_inches(args, kwargs) (line 168)
    get_size_inches_call_result_153507 = invoke(stypy.reporting.localization.Localization(__file__, 168, 38), get_size_inches_153505, *[], **kwargs_153506)
    
    # Assigning a type to the variable 'call_assignment_153015' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153015', get_size_inches_call_result_153507)
    
    # Assigning a Call to a Name (line 168):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')
    # Processing the call keyword arguments
    kwargs_153511 = {}
    # Getting the type of 'call_assignment_153015' (line 168)
    call_assignment_153015_153508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153015', False)
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___153509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), call_assignment_153015_153508, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153512 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153509, *[int_153510], **kwargs_153511)
    
    # Assigning a type to the variable 'call_assignment_153016' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153016', getitem___call_result_153512)
    
    # Assigning a Name to a Name (line 168):
    # Getting the type of 'call_assignment_153016' (line 168)
    call_assignment_153016_153513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153016')
    # Assigning a type to the variable 'fig_width_inch' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'fig_width_inch', call_assignment_153016_153513)
    
    # Assigning a Call to a Name (line 168):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')
    # Processing the call keyword arguments
    kwargs_153517 = {}
    # Getting the type of 'call_assignment_153015' (line 168)
    call_assignment_153015_153514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153015', False)
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___153515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), call_assignment_153015_153514, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153518 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153515, *[int_153516], **kwargs_153517)
    
    # Assigning a type to the variable 'call_assignment_153017' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153017', getitem___call_result_153518)
    
    # Assigning a Name to a Name (line 168):
    # Getting the type of 'call_assignment_153017' (line 168)
    call_assignment_153017_153519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'call_assignment_153017')
    # Assigning a type to the variable 'fig_height_inch' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'fig_height_inch', call_assignment_153017_153519)
    
    
    # Getting the type of 'margin_left' (line 173)
    margin_left_153520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'margin_left')
    # Applying the 'not' unary operator (line 173)
    result_not__153521 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 7), 'not', margin_left_153520)
    
    # Testing the type of an if condition (line 173)
    if_condition_153522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 4), result_not__153521)
    # Assigning a type to the variable 'if_condition_153522' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'if_condition_153522', if_condition_153522)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to max(...): (line 174)
    # Processing the call arguments (line 174)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'cols' (line 174)
    cols_153528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 53), 'cols', False)
    int_153529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 60), 'int')
    # Applying the binary operator '+' (line 174)
    result_add_153530 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 53), '+', cols_153528, int_153529)
    
    slice_153531 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 43), None, None, result_add_153530)
    # Getting the type of 'hspaces' (line 174)
    hspaces_153532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'hspaces', False)
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___153533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 43), hspaces_153532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_153534 = invoke(stypy.reporting.localization.Localization(__file__, 174, 43), getitem___153533, slice_153531)
    
    comprehension_153535 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), subscript_call_result_153534)
    # Assigning a type to the variable 's' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 's', comprehension_153535)
    
    # Call to sum(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 's' (line 174)
    s_153525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 's', False)
    # Processing the call keyword arguments (line 174)
    kwargs_153526 = {}
    # Getting the type of 'sum' (line 174)
    sum_153524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'sum', False)
    # Calling sum(args, kwargs) (line 174)
    sum_call_result_153527 = invoke(stypy.reporting.localization.Localization(__file__, 174, 27), sum_153524, *[s_153525], **kwargs_153526)
    
    list_153536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), list_153536, sum_call_result_153527)
    
    # Obtaining an instance of the builtin type 'list' (line 174)
    list_153537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 66), 'list')
    # Adding type elements to the builtin type 'list' instance (line 174)
    # Adding element type (line 174)
    int_153538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 66), list_153537, int_153538)
    
    # Applying the binary operator '+' (line 174)
    result_add_153539 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 26), '+', list_153536, list_153537)
    
    # Processing the call keyword arguments (line 174)
    kwargs_153540 = {}
    # Getting the type of 'max' (line 174)
    max_153523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'max', False)
    # Calling max(args, kwargs) (line 174)
    max_call_result_153541 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), max_153523, *[result_add_153539], **kwargs_153540)
    
    # Assigning a type to the variable 'margin_left' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'margin_left', max_call_result_153541)
    
    # Getting the type of 'margin_left' (line 175)
    margin_left_153542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'margin_left')
    # Getting the type of 'pad_inches' (line 175)
    pad_inches_153543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'pad_inches')
    # Getting the type of 'fig_width_inch' (line 175)
    fig_width_inch_153544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'fig_width_inch')
    # Applying the binary operator 'div' (line 175)
    result_div_153545 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 23), 'div', pad_inches_153543, fig_width_inch_153544)
    
    # Applying the binary operator '+=' (line 175)
    result_iadd_153546 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 8), '+=', margin_left_153542, result_div_153545)
    # Assigning a type to the variable 'margin_left' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'margin_left', result_iadd_153546)
    
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'margin_right' (line 177)
    margin_right_153547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'margin_right')
    # Applying the 'not' unary operator (line 177)
    result_not__153548 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 7), 'not', margin_right_153547)
    
    # Testing the type of an if condition (line 177)
    if_condition_153549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), result_not__153548)
    # Assigning a type to the variable 'if_condition_153549' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_153549', if_condition_153549)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to max(...): (line 178)
    # Processing the call arguments (line 178)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'cols' (line 178)
    cols_153555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 52), 'cols', False)
    # Getting the type of 'cols' (line 178)
    cols_153556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 58), 'cols', False)
    int_153557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 65), 'int')
    # Applying the binary operator '+' (line 178)
    result_add_153558 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 58), '+', cols_153556, int_153557)
    
    slice_153559 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 44), cols_153555, None, result_add_153558)
    # Getting the type of 'hspaces' (line 178)
    hspaces_153560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'hspaces', False)
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___153561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 44), hspaces_153560, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_153562 = invoke(stypy.reporting.localization.Localization(__file__, 178, 44), getitem___153561, slice_153559)
    
    comprehension_153563 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 28), subscript_call_result_153562)
    # Assigning a type to the variable 's' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 's', comprehension_153563)
    
    # Call to sum(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 's' (line 178)
    s_153552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 's', False)
    # Processing the call keyword arguments (line 178)
    kwargs_153553 = {}
    # Getting the type of 'sum' (line 178)
    sum_153551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'sum', False)
    # Calling sum(args, kwargs) (line 178)
    sum_call_result_153554 = invoke(stypy.reporting.localization.Localization(__file__, 178, 28), sum_153551, *[s_153552], **kwargs_153553)
    
    list_153564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 28), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 28), list_153564, sum_call_result_153554)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_153565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 71), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_153566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 71), list_153565, int_153566)
    
    # Applying the binary operator '+' (line 178)
    result_add_153567 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 27), '+', list_153564, list_153565)
    
    # Processing the call keyword arguments (line 178)
    kwargs_153568 = {}
    # Getting the type of 'max' (line 178)
    max_153550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'max', False)
    # Calling max(args, kwargs) (line 178)
    max_call_result_153569 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), max_153550, *[result_add_153567], **kwargs_153568)
    
    # Assigning a type to the variable 'margin_right' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'margin_right', max_call_result_153569)
    
    # Getting the type of 'margin_right' (line 179)
    margin_right_153570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'margin_right')
    # Getting the type of 'pad_inches' (line 179)
    pad_inches_153571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'pad_inches')
    # Getting the type of 'fig_width_inch' (line 179)
    fig_width_inch_153572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 37), 'fig_width_inch')
    # Applying the binary operator 'div' (line 179)
    result_div_153573 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 24), 'div', pad_inches_153571, fig_width_inch_153572)
    
    # Applying the binary operator '+=' (line 179)
    result_iadd_153574 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 8), '+=', margin_right_153570, result_div_153573)
    # Assigning a type to the variable 'margin_right' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'margin_right', result_iadd_153574)
    
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'margin_top' (line 181)
    margin_top_153575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'margin_top')
    # Applying the 'not' unary operator (line 181)
    result_not__153576 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), 'not', margin_top_153575)
    
    # Testing the type of an if condition (line 181)
    if_condition_153577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_not__153576)
    # Assigning a type to the variable 'if_condition_153577' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_153577', if_condition_153577)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to max(...): (line 182)
    # Processing the call arguments (line 182)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'cols' (line 182)
    cols_153583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 51), 'cols', False)
    slice_153584 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 42), None, cols_153583, None)
    # Getting the type of 'vspaces' (line 182)
    vspaces_153585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 42), 'vspaces', False)
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___153586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 42), vspaces_153585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_153587 = invoke(stypy.reporting.localization.Localization(__file__, 182, 42), getitem___153586, slice_153584)
    
    comprehension_153588 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 26), subscript_call_result_153587)
    # Assigning a type to the variable 's' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 's', comprehension_153588)
    
    # Call to sum(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 's' (line 182)
    s_153580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 's', False)
    # Processing the call keyword arguments (line 182)
    kwargs_153581 = {}
    # Getting the type of 'sum' (line 182)
    sum_153579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'sum', False)
    # Calling sum(args, kwargs) (line 182)
    sum_call_result_153582 = invoke(stypy.reporting.localization.Localization(__file__, 182, 26), sum_153579, *[s_153580], **kwargs_153581)
    
    list_153589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 26), list_153589, sum_call_result_153582)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_153590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 60), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    int_153591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 60), list_153590, int_153591)
    
    # Applying the binary operator '+' (line 182)
    result_add_153592 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 25), '+', list_153589, list_153590)
    
    # Processing the call keyword arguments (line 182)
    kwargs_153593 = {}
    # Getting the type of 'max' (line 182)
    max_153578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'max', False)
    # Calling max(args, kwargs) (line 182)
    max_call_result_153594 = invoke(stypy.reporting.localization.Localization(__file__, 182, 21), max_153578, *[result_add_153592], **kwargs_153593)
    
    # Assigning a type to the variable 'margin_top' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'margin_top', max_call_result_153594)
    
    # Getting the type of 'margin_top' (line 183)
    margin_top_153595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'margin_top')
    # Getting the type of 'pad_inches' (line 183)
    pad_inches_153596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'pad_inches')
    # Getting the type of 'fig_height_inch' (line 183)
    fig_height_inch_153597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 35), 'fig_height_inch')
    # Applying the binary operator 'div' (line 183)
    result_div_153598 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 22), 'div', pad_inches_153596, fig_height_inch_153597)
    
    # Applying the binary operator '+=' (line 183)
    result_iadd_153599 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 8), '+=', margin_top_153595, result_div_153598)
    # Assigning a type to the variable 'margin_top' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'margin_top', result_iadd_153599)
    
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'margin_bottom' (line 185)
    margin_bottom_153600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'margin_bottom')
    # Applying the 'not' unary operator (line 185)
    result_not__153601 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 7), 'not', margin_bottom_153600)
    
    # Testing the type of an if condition (line 185)
    if_condition_153602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 4), result_not__153601)
    # Assigning a type to the variable 'if_condition_153602' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'if_condition_153602', if_condition_153602)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to max(...): (line 186)
    # Processing the call arguments (line 186)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'cols' (line 186)
    cols_153608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 54), 'cols', False)
    # Applying the 'usub' unary operator (line 186)
    result___neg___153609 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 53), 'usub', cols_153608)
    
    slice_153610 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 186, 45), result___neg___153609, None, None)
    # Getting the type of 'vspaces' (line 186)
    vspaces_153611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'vspaces', False)
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___153612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), vspaces_153611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_153613 = invoke(stypy.reporting.localization.Localization(__file__, 186, 45), getitem___153612, slice_153610)
    
    comprehension_153614 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 29), subscript_call_result_153613)
    # Assigning a type to the variable 's' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 's', comprehension_153614)
    
    # Call to sum(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 's' (line 186)
    s_153605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 's', False)
    # Processing the call keyword arguments (line 186)
    kwargs_153606 = {}
    # Getting the type of 'sum' (line 186)
    sum_153604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'sum', False)
    # Calling sum(args, kwargs) (line 186)
    sum_call_result_153607 = invoke(stypy.reporting.localization.Localization(__file__, 186, 29), sum_153604, *[s_153605], **kwargs_153606)
    
    list_153615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 29), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 29), list_153615, sum_call_result_153607)
    
    # Obtaining an instance of the builtin type 'list' (line 186)
    list_153616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 64), 'list')
    # Adding type elements to the builtin type 'list' instance (line 186)
    # Adding element type (line 186)
    int_153617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 64), list_153616, int_153617)
    
    # Applying the binary operator '+' (line 186)
    result_add_153618 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 28), '+', list_153615, list_153616)
    
    # Processing the call keyword arguments (line 186)
    kwargs_153619 = {}
    # Getting the type of 'max' (line 186)
    max_153603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'max', False)
    # Calling max(args, kwargs) (line 186)
    max_call_result_153620 = invoke(stypy.reporting.localization.Localization(__file__, 186, 24), max_153603, *[result_add_153618], **kwargs_153619)
    
    # Assigning a type to the variable 'margin_bottom' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'margin_bottom', max_call_result_153620)
    
    # Getting the type of 'margin_bottom' (line 187)
    margin_bottom_153621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'margin_bottom')
    # Getting the type of 'pad_inches' (line 187)
    pad_inches_153622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'pad_inches')
    # Getting the type of 'fig_height_inch' (line 187)
    fig_height_inch_153623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'fig_height_inch')
    # Applying the binary operator 'div' (line 187)
    result_div_153624 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 25), 'div', pad_inches_153622, fig_height_inch_153623)
    
    # Applying the binary operator '+=' (line 187)
    result_iadd_153625 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 8), '+=', margin_bottom_153621, result_div_153624)
    # Assigning a type to the variable 'margin_bottom' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'margin_bottom', result_iadd_153625)
    
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to dict(...): (line 189)
    # Processing the call keyword arguments (line 189)
    # Getting the type of 'margin_left' (line 189)
    margin_left_153627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'margin_left', False)
    keyword_153628 = margin_left_153627
    int_153629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 24), 'int')
    # Getting the type of 'margin_right' (line 190)
    margin_right_153630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'margin_right', False)
    # Applying the binary operator '-' (line 190)
    result_sub_153631 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 24), '-', int_153629, margin_right_153630)
    
    keyword_153632 = result_sub_153631
    # Getting the type of 'margin_bottom' (line 191)
    margin_bottom_153633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'margin_bottom', False)
    keyword_153634 = margin_bottom_153633
    int_153635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 22), 'int')
    # Getting the type of 'margin_top' (line 192)
    margin_top_153636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 26), 'margin_top', False)
    # Applying the binary operator '-' (line 192)
    result_sub_153637 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 22), '-', int_153635, margin_top_153636)
    
    keyword_153638 = result_sub_153637
    kwargs_153639 = {'top': keyword_153638, 'right': keyword_153632, 'bottom': keyword_153634, 'left': keyword_153628}
    # Getting the type of 'dict' (line 189)
    dict_153626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), 'dict', False)
    # Calling dict(args, kwargs) (line 189)
    dict_call_result_153640 = invoke(stypy.reporting.localization.Localization(__file__, 189, 13), dict_153626, *[], **kwargs_153639)
    
    # Assigning a type to the variable 'kwargs' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'kwargs', dict_call_result_153640)
    
    
    # Getting the type of 'cols' (line 194)
    cols_153641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 7), 'cols')
    int_153642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 14), 'int')
    # Applying the binary operator '>' (line 194)
    result_gt_153643 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), '>', cols_153641, int_153642)
    
    # Testing the type of an if condition (line 194)
    if_condition_153644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), result_gt_153643)
    # Assigning a type to the variable 'if_condition_153644' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_153644', if_condition_153644)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 195):
    
    # Assigning a BinOp to a Name (line 195):
    
    # Call to max(...): (line 196)
    # Processing the call arguments (line 196)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 196, 16, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'rows' (line 197)
    rows_153651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 31), 'rows', False)
    # Processing the call keyword arguments (line 197)
    kwargs_153652 = {}
    # Getting the type of 'range' (line 197)
    range_153650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'range', False)
    # Calling range(args, kwargs) (line 197)
    range_call_result_153653 = invoke(stypy.reporting.localization.Localization(__file__, 197, 25), range_153650, *[rows_153651], **kwargs_153652)
    
    comprehension_153654 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), range_call_result_153653)
    # Assigning a type to the variable 'i' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'i', comprehension_153654)
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 198)
    i_153655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'i', False)
    # Getting the type of 'cols' (line 198)
    cols_153656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 38), 'cols', False)
    int_153657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 45), 'int')
    # Applying the binary operator '+' (line 198)
    result_add_153658 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 38), '+', cols_153656, int_153657)
    
    # Applying the binary operator '*' (line 198)
    result_mul_153659 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 33), '*', i_153655, result_add_153658)
    
    int_153660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 50), 'int')
    # Applying the binary operator '+' (line 198)
    result_add_153661 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 33), '+', result_mul_153659, int_153660)
    
    # Getting the type of 'i' (line 198)
    i_153662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 53), 'i', False)
    int_153663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 57), 'int')
    # Applying the binary operator '+' (line 198)
    result_add_153664 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 53), '+', i_153662, int_153663)
    
    # Getting the type of 'cols' (line 198)
    cols_153665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 63), 'cols', False)
    int_153666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 70), 'int')
    # Applying the binary operator '+' (line 198)
    result_add_153667 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 63), '+', cols_153665, int_153666)
    
    # Applying the binary operator '*' (line 198)
    result_mul_153668 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 52), '*', result_add_153664, result_add_153667)
    
    int_153669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 75), 'int')
    # Applying the binary operator '-' (line 198)
    result_sub_153670 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 52), '-', result_mul_153668, int_153669)
    
    slice_153671 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 198, 25), result_add_153661, result_sub_153670, None)
    # Getting the type of 'hspaces' (line 198)
    hspaces_153672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 25), 'hspaces', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___153673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 25), hspaces_153672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_153674 = invoke(stypy.reporting.localization.Localization(__file__, 198, 25), getitem___153673, slice_153671)
    
    comprehension_153675 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), subscript_call_result_153674)
    # Assigning a type to the variable 's' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 's', comprehension_153675)
    
    # Call to sum(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 's' (line 196)
    s_153647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 's', False)
    # Processing the call keyword arguments (line 196)
    kwargs_153648 = {}
    # Getting the type of 'sum' (line 196)
    sum_153646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'sum', False)
    # Calling sum(args, kwargs) (line 196)
    sum_call_result_153649 = invoke(stypy.reporting.localization.Localization(__file__, 196, 16), sum_153646, *[s_153647], **kwargs_153648)
    
    list_153676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 16), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), list_153676, sum_call_result_153649)
    # Processing the call keyword arguments (line 196)
    kwargs_153677 = {}
    # Getting the type of 'max' (line 196)
    max_153645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'max', False)
    # Calling max(args, kwargs) (line 196)
    max_call_result_153678 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), max_153645, *[list_153676], **kwargs_153677)
    
    # Getting the type of 'hpad_inches' (line 199)
    hpad_inches_153679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'hpad_inches')
    # Getting the type of 'fig_width_inch' (line 199)
    fig_width_inch_153680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'fig_width_inch')
    # Applying the binary operator 'div' (line 199)
    result_div_153681 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 14), 'div', hpad_inches_153679, fig_width_inch_153680)
    
    # Applying the binary operator '+' (line 196)
    result_add_153682 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 12), '+', max_call_result_153678, result_div_153681)
    
    # Assigning a type to the variable 'hspace' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'hspace', result_add_153682)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    int_153683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'int')
    # Getting the type of 'margin_right' (line 200)
    margin_right_153684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'margin_right')
    # Applying the binary operator '-' (line 200)
    result_sub_153685 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 18), '-', int_153683, margin_right_153684)
    
    # Getting the type of 'margin_left' (line 200)
    margin_left_153686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'margin_left')
    # Applying the binary operator '-' (line 200)
    result_sub_153687 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 35), '-', result_sub_153685, margin_left_153686)
    
    # Getting the type of 'hspace' (line 200)
    hspace_153688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 51), 'hspace')
    # Getting the type of 'cols' (line 200)
    cols_153689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 61), 'cols')
    int_153690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 68), 'int')
    # Applying the binary operator '-' (line 200)
    result_sub_153691 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 61), '-', cols_153689, int_153690)
    
    # Applying the binary operator '*' (line 200)
    result_mul_153692 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 51), '*', hspace_153688, result_sub_153691)
    
    # Applying the binary operator '-' (line 200)
    result_sub_153693 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 49), '-', result_sub_153687, result_mul_153692)
    
    # Getting the type of 'cols' (line 200)
    cols_153694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 74), 'cols')
    # Applying the binary operator 'div' (line 200)
    result_div_153695 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 17), 'div', result_sub_153693, cols_153694)
    
    # Assigning a type to the variable 'h_axes' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'h_axes', result_div_153695)
    
    # Assigning a BinOp to a Subscript (line 201):
    
    # Assigning a BinOp to a Subscript (line 201):
    # Getting the type of 'hspace' (line 201)
    hspace_153696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'hspace')
    # Getting the type of 'h_axes' (line 201)
    h_axes_153697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'h_axes')
    # Applying the binary operator 'div' (line 201)
    result_div_153698 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 27), 'div', hspace_153696, h_axes_153697)
    
    # Getting the type of 'kwargs' (line 201)
    kwargs_153699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'kwargs')
    str_153700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 15), 'str', 'wspace')
    # Storing an element on a container (line 201)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 8), kwargs_153699, (str_153700, result_div_153698))
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rows' (line 203)
    rows_153701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 7), 'rows')
    int_153702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 14), 'int')
    # Applying the binary operator '>' (line 203)
    result_gt_153703 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 7), '>', rows_153701, int_153702)
    
    # Testing the type of an if condition (line 203)
    if_condition_153704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 4), result_gt_153703)
    # Assigning a type to the variable 'if_condition_153704' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'if_condition_153704', if_condition_153704)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 204):
    
    # Assigning a BinOp to a Name (line 204):
    
    # Call to max(...): (line 204)
    # Processing the call arguments (line 204)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 204, 22, True)
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'cols' (line 204)
    cols_153710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 46), 'cols', False)
    
    # Getting the type of 'cols' (line 204)
    cols_153711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 52), 'cols', False)
    # Applying the 'usub' unary operator (line 204)
    result___neg___153712 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 51), 'usub', cols_153711)
    
    slice_153713 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 204, 38), cols_153710, result___neg___153712, None)
    # Getting the type of 'vspaces' (line 204)
    vspaces_153714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 'vspaces', False)
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___153715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 38), vspaces_153714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_153716 = invoke(stypy.reporting.localization.Localization(__file__, 204, 38), getitem___153715, slice_153713)
    
    comprehension_153717 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 22), subscript_call_result_153716)
    # Assigning a type to the variable 's' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 's', comprehension_153717)
    
    # Call to sum(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 's' (line 204)
    s_153707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 's', False)
    # Processing the call keyword arguments (line 204)
    kwargs_153708 = {}
    # Getting the type of 'sum' (line 204)
    sum_153706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'sum', False)
    # Calling sum(args, kwargs) (line 204)
    sum_call_result_153709 = invoke(stypy.reporting.localization.Localization(__file__, 204, 22), sum_153706, *[s_153707], **kwargs_153708)
    
    list_153718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 22), list_153718, sum_call_result_153709)
    # Processing the call keyword arguments (line 204)
    kwargs_153719 = {}
    # Getting the type of 'max' (line 204)
    max_153705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'max', False)
    # Calling max(args, kwargs) (line 204)
    max_call_result_153720 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), max_153705, *[list_153718], **kwargs_153719)
    
    # Getting the type of 'vpad_inches' (line 205)
    vpad_inches_153721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'vpad_inches')
    # Getting the type of 'fig_height_inch' (line 205)
    fig_height_inch_153722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'fig_height_inch')
    # Applying the binary operator 'div' (line 205)
    result_div_153723 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 20), 'div', vpad_inches_153721, fig_height_inch_153722)
    
    # Applying the binary operator '+' (line 204)
    result_add_153724 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 18), '+', max_call_result_153720, result_div_153723)
    
    # Assigning a type to the variable 'vspace' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'vspace', result_add_153724)
    
    # Assigning a BinOp to a Name (line 206):
    
    # Assigning a BinOp to a Name (line 206):
    int_153725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 18), 'int')
    # Getting the type of 'margin_top' (line 206)
    margin_top_153726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'margin_top')
    # Applying the binary operator '-' (line 206)
    result_sub_153727 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 18), '-', int_153725, margin_top_153726)
    
    # Getting the type of 'margin_bottom' (line 206)
    margin_bottom_153728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 35), 'margin_bottom')
    # Applying the binary operator '-' (line 206)
    result_sub_153729 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 33), '-', result_sub_153727, margin_bottom_153728)
    
    # Getting the type of 'vspace' (line 206)
    vspace_153730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 51), 'vspace')
    # Getting the type of 'rows' (line 206)
    rows_153731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 61), 'rows')
    int_153732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 68), 'int')
    # Applying the binary operator '-' (line 206)
    result_sub_153733 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 61), '-', rows_153731, int_153732)
    
    # Applying the binary operator '*' (line 206)
    result_mul_153734 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 51), '*', vspace_153730, result_sub_153733)
    
    # Applying the binary operator '-' (line 206)
    result_sub_153735 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 49), '-', result_sub_153729, result_mul_153734)
    
    # Getting the type of 'rows' (line 206)
    rows_153736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 74), 'rows')
    # Applying the binary operator 'div' (line 206)
    result_div_153737 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 17), 'div', result_sub_153735, rows_153736)
    
    # Assigning a type to the variable 'v_axes' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'v_axes', result_div_153737)
    
    # Assigning a BinOp to a Subscript (line 207):
    
    # Assigning a BinOp to a Subscript (line 207):
    # Getting the type of 'vspace' (line 207)
    vspace_153738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'vspace')
    # Getting the type of 'v_axes' (line 207)
    v_axes_153739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 36), 'v_axes')
    # Applying the binary operator 'div' (line 207)
    result_div_153740 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 27), 'div', vspace_153738, v_axes_153739)
    
    # Getting the type of 'kwargs' (line 207)
    kwargs_153741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'kwargs')
    str_153742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 15), 'str', 'hspace')
    # Storing an element on a container (line 207)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 8), kwargs_153741, (str_153742, result_div_153740))
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'kwargs' (line 209)
    kwargs_153743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'kwargs')
    # Assigning a type to the variable 'stypy_return_type' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type', kwargs_153743)
    
    # ################# End of 'auto_adjust_subplotpars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'auto_adjust_subplotpars' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_153744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153744)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'auto_adjust_subplotpars'
    return stypy_return_type_153744

# Assigning a type to the variable 'auto_adjust_subplotpars' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'auto_adjust_subplotpars', auto_adjust_subplotpars)

@norecursion
def get_renderer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_renderer'
    module_type_store = module_type_store.open_function_context('get_renderer', 212, 0, False)
    
    # Passed parameters checking function
    get_renderer.stypy_localization = localization
    get_renderer.stypy_type_of_self = None
    get_renderer.stypy_type_store = module_type_store
    get_renderer.stypy_function_name = 'get_renderer'
    get_renderer.stypy_param_names_list = ['fig']
    get_renderer.stypy_varargs_param_name = None
    get_renderer.stypy_kwargs_param_name = None
    get_renderer.stypy_call_defaults = defaults
    get_renderer.stypy_call_varargs = varargs
    get_renderer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_renderer', ['fig'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_renderer', localization, ['fig'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_renderer(...)' code ##################

    
    # Getting the type of 'fig' (line 213)
    fig_153745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'fig')
    # Obtaining the member '_cachedRenderer' of a type (line 213)
    _cachedRenderer_153746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 7), fig_153745, '_cachedRenderer')
    # Testing the type of an if condition (line 213)
    if_condition_153747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 4), _cachedRenderer_153746)
    # Assigning a type to the variable 'if_condition_153747' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'if_condition_153747', if_condition_153747)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 214):
    
    # Assigning a Attribute to a Name (line 214):
    # Getting the type of 'fig' (line 214)
    fig_153748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'fig')
    # Obtaining the member '_cachedRenderer' of a type (line 214)
    _cachedRenderer_153749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), fig_153748, '_cachedRenderer')
    # Assigning a type to the variable 'renderer' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'renderer', _cachedRenderer_153749)
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 216):
    
    # Assigning a Attribute to a Name (line 216):
    # Getting the type of 'fig' (line 216)
    fig_153750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'fig')
    # Obtaining the member 'canvas' of a type (line 216)
    canvas_153751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 17), fig_153750, 'canvas')
    # Assigning a type to the variable 'canvas' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'canvas', canvas_153751)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'canvas' (line 218)
    canvas_153752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'canvas')
    
    # Call to hasattr(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'canvas' (line 218)
    canvas_153754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'canvas', False)
    str_153755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 38), 'str', 'get_renderer')
    # Processing the call keyword arguments (line 218)
    kwargs_153756 = {}
    # Getting the type of 'hasattr' (line 218)
    hasattr_153753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 218)
    hasattr_call_result_153757 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), hasattr_153753, *[canvas_153754, str_153755], **kwargs_153756)
    
    # Applying the binary operator 'and' (line 218)
    result_and_keyword_153758 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'and', canvas_153752, hasattr_call_result_153757)
    
    # Testing the type of an if condition (line 218)
    if_condition_153759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_and_keyword_153758)
    # Assigning a type to the variable 'if_condition_153759' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_153759', if_condition_153759)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to get_renderer(...): (line 219)
    # Processing the call keyword arguments (line 219)
    kwargs_153762 = {}
    # Getting the type of 'canvas' (line 219)
    canvas_153760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'canvas', False)
    # Obtaining the member 'get_renderer' of a type (line 219)
    get_renderer_153761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 23), canvas_153760, 'get_renderer')
    # Calling get_renderer(args, kwargs) (line 219)
    get_renderer_call_result_153763 = invoke(stypy.reporting.localization.Localization(__file__, 219, 23), get_renderer_153761, *[], **kwargs_153762)
    
    # Assigning a type to the variable 'renderer' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'renderer', get_renderer_call_result_153763)
    # SSA branch for the else part of an if statement (line 218)
    module_type_store.open_ssa_branch('else')
    
    # Call to warn(...): (line 222)
    # Processing the call arguments (line 222)
    str_153766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 26), 'str', 'tight_layout : falling back to Agg renderer')
    # Processing the call keyword arguments (line 222)
    kwargs_153767 = {}
    # Getting the type of 'warnings' (line 222)
    warnings_153764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 222)
    warn_153765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), warnings_153764, 'warn')
    # Calling warn(args, kwargs) (line 222)
    warn_call_result_153768 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), warn_153765, *[str_153766], **kwargs_153767)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 223, 12))
    
    # 'from matplotlib.backends.backend_agg import FigureCanvasAgg' statement (line 223)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
    import_153769 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 223, 12), 'matplotlib.backends.backend_agg')

    if (type(import_153769) is not StypyTypeError):

        if (import_153769 != 'pyd_module'):
            __import__(import_153769)
            sys_modules_153770 = sys.modules[import_153769]
            import_from_module(stypy.reporting.localization.Localization(__file__, 223, 12), 'matplotlib.backends.backend_agg', sys_modules_153770.module_type_store, module_type_store, ['FigureCanvasAgg'])
            nest_module(stypy.reporting.localization.Localization(__file__, 223, 12), __file__, sys_modules_153770, sys_modules_153770.module_type_store, module_type_store)
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            import_from_module(stypy.reporting.localization.Localization(__file__, 223, 12), 'matplotlib.backends.backend_agg', None, module_type_store, ['FigureCanvasAgg'], [FigureCanvasAgg])

    else:
        # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'matplotlib.backends.backend_agg', import_153769)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
    
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to FigureCanvasAgg(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'fig' (line 224)
    fig_153772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 37), 'fig', False)
    # Processing the call keyword arguments (line 224)
    kwargs_153773 = {}
    # Getting the type of 'FigureCanvasAgg' (line 224)
    FigureCanvasAgg_153771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'FigureCanvasAgg', False)
    # Calling FigureCanvasAgg(args, kwargs) (line 224)
    FigureCanvasAgg_call_result_153774 = invoke(stypy.reporting.localization.Localization(__file__, 224, 21), FigureCanvasAgg_153771, *[fig_153772], **kwargs_153773)
    
    # Assigning a type to the variable 'canvas' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'canvas', FigureCanvasAgg_call_result_153774)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to get_renderer(...): (line 225)
    # Processing the call keyword arguments (line 225)
    kwargs_153777 = {}
    # Getting the type of 'canvas' (line 225)
    canvas_153775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'canvas', False)
    # Obtaining the member 'get_renderer' of a type (line 225)
    get_renderer_153776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 23), canvas_153775, 'get_renderer')
    # Calling get_renderer(args, kwargs) (line 225)
    get_renderer_call_result_153778 = invoke(stypy.reporting.localization.Localization(__file__, 225, 23), get_renderer_153776, *[], **kwargs_153777)
    
    # Assigning a type to the variable 'renderer' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'renderer', get_renderer_call_result_153778)
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'renderer' (line 227)
    renderer_153779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'renderer')
    # Assigning a type to the variable 'stypy_return_type' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type', renderer_153779)
    
    # ################# End of 'get_renderer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_renderer' in the type store
    # Getting the type of 'stypy_return_type' (line 212)
    stypy_return_type_153780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_renderer'
    return stypy_return_type_153780

# Assigning a type to the variable 'get_renderer' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'get_renderer', get_renderer)

@norecursion
def get_subplotspec_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 230)
    None_153781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 46), 'None')
    defaults = [None_153781]
    # Create a new context for function 'get_subplotspec_list'
    module_type_store = module_type_store.open_function_context('get_subplotspec_list', 230, 0, False)
    
    # Passed parameters checking function
    get_subplotspec_list.stypy_localization = localization
    get_subplotspec_list.stypy_type_of_self = None
    get_subplotspec_list.stypy_type_store = module_type_store
    get_subplotspec_list.stypy_function_name = 'get_subplotspec_list'
    get_subplotspec_list.stypy_param_names_list = ['axes_list', 'grid_spec']
    get_subplotspec_list.stypy_varargs_param_name = None
    get_subplotspec_list.stypy_kwargs_param_name = None
    get_subplotspec_list.stypy_call_defaults = defaults
    get_subplotspec_list.stypy_call_varargs = varargs
    get_subplotspec_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_subplotspec_list', ['axes_list', 'grid_spec'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_subplotspec_list', localization, ['axes_list', 'grid_spec'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_subplotspec_list(...)' code ##################

    str_153782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'str', 'Return a list of subplotspec from the given list of axes.\n\n    For an instance of axes that does not support subplotspec, None is inserted\n    in the list.\n\n    If grid_spec is given, None is inserted for those not from the given\n    grid_spec.\n    ')
    
    # Assigning a List to a Name (line 239):
    
    # Assigning a List to a Name (line 239):
    
    # Obtaining an instance of the builtin type 'list' (line 239)
    list_153783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 239)
    
    # Assigning a type to the variable 'subplotspec_list' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'subplotspec_list', list_153783)
    
    # Getting the type of 'axes_list' (line 240)
    axes_list_153784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'axes_list')
    # Testing the type of a for loop iterable (line 240)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 4), axes_list_153784)
    # Getting the type of the for loop variable (line 240)
    for_loop_var_153785 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 4), axes_list_153784)
    # Assigning a type to the variable 'ax' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'ax', for_loop_var_153785)
    # SSA begins for a for statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 241):
    
    # Assigning a Call to a Name (line 241):
    
    # Call to get_axes_locator(...): (line 241)
    # Processing the call keyword arguments (line 241)
    kwargs_153788 = {}
    # Getting the type of 'ax' (line 241)
    ax_153786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'ax', False)
    # Obtaining the member 'get_axes_locator' of a type (line 241)
    get_axes_locator_153787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 26), ax_153786, 'get_axes_locator')
    # Calling get_axes_locator(args, kwargs) (line 241)
    get_axes_locator_call_result_153789 = invoke(stypy.reporting.localization.Localization(__file__, 241, 26), get_axes_locator_153787, *[], **kwargs_153788)
    
    # Assigning a type to the variable 'axes_or_locator' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'axes_or_locator', get_axes_locator_call_result_153789)
    
    # Type idiom detected: calculating its left and rigth part (line 242)
    # Getting the type of 'axes_or_locator' (line 242)
    axes_or_locator_153790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'axes_or_locator')
    # Getting the type of 'None' (line 242)
    None_153791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'None')
    
    (may_be_153792, more_types_in_union_153793) = may_be_none(axes_or_locator_153790, None_153791)

    if may_be_153792:

        if more_types_in_union_153793:
            # Runtime conditional SSA (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 243):
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'ax' (line 243)
        ax_153794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'ax')
        # Assigning a type to the variable 'axes_or_locator' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'axes_or_locator', ax_153794)

        if more_types_in_union_153793:
            # SSA join for if statement (line 242)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 245)
    str_153795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 36), 'str', 'get_subplotspec')
    # Getting the type of 'axes_or_locator' (line 245)
    axes_or_locator_153796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'axes_or_locator')
    
    (may_be_153797, more_types_in_union_153798) = may_provide_member(str_153795, axes_or_locator_153796)

    if may_be_153797:

        if more_types_in_union_153798:
            # Runtime conditional SSA (line 245)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'axes_or_locator' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'axes_or_locator', remove_not_member_provider_from_union(axes_or_locator_153796, 'get_subplotspec'))
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to get_subplotspec(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_153801 = {}
        # Getting the type of 'axes_or_locator' (line 246)
        axes_or_locator_153799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'axes_or_locator', False)
        # Obtaining the member 'get_subplotspec' of a type (line 246)
        get_subplotspec_153800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 26), axes_or_locator_153799, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 246)
        get_subplotspec_call_result_153802 = invoke(stypy.reporting.localization.Localization(__file__, 246, 26), get_subplotspec_153800, *[], **kwargs_153801)
        
        # Assigning a type to the variable 'subplotspec' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'subplotspec', get_subplotspec_call_result_153802)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to get_topmost_subplotspec(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_153805 = {}
        # Getting the type of 'subplotspec' (line 247)
        subplotspec_153803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 26), 'subplotspec', False)
        # Obtaining the member 'get_topmost_subplotspec' of a type (line 247)
        get_topmost_subplotspec_153804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 26), subplotspec_153803, 'get_topmost_subplotspec')
        # Calling get_topmost_subplotspec(args, kwargs) (line 247)
        get_topmost_subplotspec_call_result_153806 = invoke(stypy.reporting.localization.Localization(__file__, 247, 26), get_topmost_subplotspec_153804, *[], **kwargs_153805)
        
        # Assigning a type to the variable 'subplotspec' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'subplotspec', get_topmost_subplotspec_call_result_153806)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to get_gridspec(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_153809 = {}
        # Getting the type of 'subplotspec' (line 248)
        subplotspec_153807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'subplotspec', False)
        # Obtaining the member 'get_gridspec' of a type (line 248)
        get_gridspec_153808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 17), subplotspec_153807, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 248)
        get_gridspec_call_result_153810 = invoke(stypy.reporting.localization.Localization(__file__, 248, 17), get_gridspec_153808, *[], **kwargs_153809)
        
        # Assigning a type to the variable 'gs' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'gs', get_gridspec_call_result_153810)
        
        # Type idiom detected: calculating its left and rigth part (line 249)
        # Getting the type of 'grid_spec' (line 249)
        grid_spec_153811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'grid_spec')
        # Getting the type of 'None' (line 249)
        None_153812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 32), 'None')
        
        (may_be_153813, more_types_in_union_153814) = may_not_be_none(grid_spec_153811, None_153812)

        if may_be_153813:

            if more_types_in_union_153814:
                # Runtime conditional SSA (line 249)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'gs' (line 250)
            gs_153815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'gs')
            # Getting the type of 'grid_spec' (line 250)
            grid_spec_153816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'grid_spec')
            # Applying the binary operator '!=' (line 250)
            result_ne_153817 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 19), '!=', gs_153815, grid_spec_153816)
            
            # Testing the type of an if condition (line 250)
            if_condition_153818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 16), result_ne_153817)
            # Assigning a type to the variable 'if_condition_153818' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'if_condition_153818', if_condition_153818)
            # SSA begins for if statement (line 250)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 251):
            
            # Assigning a Name to a Name (line 251):
            # Getting the type of 'None' (line 251)
            None_153819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'None')
            # Assigning a type to the variable 'subplotspec' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'subplotspec', None_153819)
            # SSA join for if statement (line 250)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_153814:
                # Runtime conditional SSA for else branch (line 249)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_153813) or more_types_in_union_153814):
            
            
            # Call to locally_modified_subplot_params(...): (line 252)
            # Processing the call keyword arguments (line 252)
            kwargs_153822 = {}
            # Getting the type of 'gs' (line 252)
            gs_153820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'gs', False)
            # Obtaining the member 'locally_modified_subplot_params' of a type (line 252)
            locally_modified_subplot_params_153821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 17), gs_153820, 'locally_modified_subplot_params')
            # Calling locally_modified_subplot_params(args, kwargs) (line 252)
            locally_modified_subplot_params_call_result_153823 = invoke(stypy.reporting.localization.Localization(__file__, 252, 17), locally_modified_subplot_params_153821, *[], **kwargs_153822)
            
            # Testing the type of an if condition (line 252)
            if_condition_153824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 17), locally_modified_subplot_params_call_result_153823)
            # Assigning a type to the variable 'if_condition_153824' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'if_condition_153824', if_condition_153824)
            # SSA begins for if statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 253):
            
            # Assigning a Name to a Name (line 253):
            # Getting the type of 'None' (line 253)
            None_153825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 30), 'None')
            # Assigning a type to the variable 'subplotspec' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'subplotspec', None_153825)
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_153813 and more_types_in_union_153814):
                # SSA join for if statement (line 249)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_153798:
            # Runtime conditional SSA for else branch (line 245)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_153797) or more_types_in_union_153798):
        # Assigning a type to the variable 'axes_or_locator' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'axes_or_locator', remove_member_provider_from_union(axes_or_locator_153796, 'get_subplotspec'))
        
        # Assigning a Name to a Name (line 255):
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'None' (line 255)
        None_153826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 26), 'None')
        # Assigning a type to the variable 'subplotspec' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'subplotspec', None_153826)

        if (may_be_153797 and more_types_in_union_153798):
            # SSA join for if statement (line 245)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to append(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'subplotspec' (line 257)
    subplotspec_153829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 32), 'subplotspec', False)
    # Processing the call keyword arguments (line 257)
    kwargs_153830 = {}
    # Getting the type of 'subplotspec_list' (line 257)
    subplotspec_list_153827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'subplotspec_list', False)
    # Obtaining the member 'append' of a type (line 257)
    append_153828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), subplotspec_list_153827, 'append')
    # Calling append(args, kwargs) (line 257)
    append_call_result_153831 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), append_153828, *[subplotspec_153829], **kwargs_153830)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'subplotspec_list' (line 259)
    subplotspec_list_153832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'subplotspec_list')
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type', subplotspec_list_153832)
    
    # ################# End of 'get_subplotspec_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_subplotspec_list' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_153833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153833)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_subplotspec_list'
    return stypy_return_type_153833

# Assigning a type to the variable 'get_subplotspec_list' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'get_subplotspec_list', get_subplotspec_list)

@norecursion
def get_tight_layout_figure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_153834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 32), 'float')
    # Getting the type of 'None' (line 263)
    None_153835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 44), 'None')
    # Getting the type of 'None' (line 263)
    None_153836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 56), 'None')
    # Getting the type of 'None' (line 263)
    None_153837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 67), 'None')
    defaults = [float_153834, None_153835, None_153836, None_153837]
    # Create a new context for function 'get_tight_layout_figure'
    module_type_store = module_type_store.open_function_context('get_tight_layout_figure', 262, 0, False)
    
    # Passed parameters checking function
    get_tight_layout_figure.stypy_localization = localization
    get_tight_layout_figure.stypy_type_of_self = None
    get_tight_layout_figure.stypy_type_store = module_type_store
    get_tight_layout_figure.stypy_function_name = 'get_tight_layout_figure'
    get_tight_layout_figure.stypy_param_names_list = ['fig', 'axes_list', 'subplotspec_list', 'renderer', 'pad', 'h_pad', 'w_pad', 'rect']
    get_tight_layout_figure.stypy_varargs_param_name = None
    get_tight_layout_figure.stypy_kwargs_param_name = None
    get_tight_layout_figure.stypy_call_defaults = defaults
    get_tight_layout_figure.stypy_call_varargs = varargs
    get_tight_layout_figure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_tight_layout_figure', ['fig', 'axes_list', 'subplotspec_list', 'renderer', 'pad', 'h_pad', 'w_pad', 'rect'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_tight_layout_figure', localization, ['fig', 'axes_list', 'subplotspec_list', 'renderer', 'pad', 'h_pad', 'w_pad', 'rect'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_tight_layout_figure(...)' code ##################

    str_153838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'str', '\n    Return subplot parameters for tight-layouted-figure with specified\n    padding.\n\n    Parameters:\n\n      *fig* : figure instance\n\n      *axes_list* : a list of axes\n\n      *subplotspec_list* : a list of subplotspec associated with each\n        axes in axes_list\n\n      *renderer* : renderer instance\n\n      *pad* : float\n        padding between the figure edge and the edges of subplots,\n        as a fraction of the font-size.\n\n      *h_pad*, *w_pad* : float\n        padding (height/width) between edges of adjacent subplots.\n        Defaults to `pad_inches`.\n\n      *rect* : if rect is given, it is interpreted as a rectangle\n        (left, bottom, right, top) in the normalized figure\n        coordinate that the whole subplots area (including\n        labels) will fit into. Default is (0, 0, 1, 1).\n    ')
    
    # Assigning a List to a Name (line 293):
    
    # Assigning a List to a Name (line 293):
    
    # Obtaining an instance of the builtin type 'list' (line 293)
    list_153839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 293)
    
    # Assigning a type to the variable 'subplot_list' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'subplot_list', list_153839)
    
    # Assigning a List to a Name (line 294):
    
    # Assigning a List to a Name (line 294):
    
    # Obtaining an instance of the builtin type 'list' (line 294)
    list_153840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 294)
    
    # Assigning a type to the variable 'nrows_list' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'nrows_list', list_153840)
    
    # Assigning a List to a Name (line 295):
    
    # Assigning a List to a Name (line 295):
    
    # Obtaining an instance of the builtin type 'list' (line 295)
    list_153841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 295)
    
    # Assigning a type to the variable 'ncols_list' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'ncols_list', list_153841)
    
    # Assigning a List to a Name (line 296):
    
    # Assigning a List to a Name (line 296):
    
    # Obtaining an instance of the builtin type 'list' (line 296)
    list_153842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 296)
    
    # Assigning a type to the variable 'ax_bbox_list' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'ax_bbox_list', list_153842)
    
    # Assigning a Dict to a Name (line 298):
    
    # Assigning a Dict to a Name (line 298):
    
    # Obtaining an instance of the builtin type 'dict' (line 298)
    dict_153843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 298)
    
    # Assigning a type to the variable 'subplot_dict' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'subplot_dict', dict_153843)
    
    # Assigning a List to a Name (line 302):
    
    # Assigning a List to a Name (line 302):
    
    # Obtaining an instance of the builtin type 'list' (line 302)
    list_153844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 302)
    
    # Assigning a type to the variable 'subplotspec_list2' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'subplotspec_list2', list_153844)
    
    
    # Call to zip(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'axes_list' (line 304)
    axes_list_153846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'axes_list', False)
    # Getting the type of 'subplotspec_list' (line 305)
    subplotspec_list_153847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 31), 'subplotspec_list', False)
    # Processing the call keyword arguments (line 304)
    kwargs_153848 = {}
    # Getting the type of 'zip' (line 304)
    zip_153845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'zip', False)
    # Calling zip(args, kwargs) (line 304)
    zip_call_result_153849 = invoke(stypy.reporting.localization.Localization(__file__, 304, 27), zip_153845, *[axes_list_153846, subplotspec_list_153847], **kwargs_153848)
    
    # Testing the type of a for loop iterable (line 304)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 304, 4), zip_call_result_153849)
    # Getting the type of the for loop variable (line 304)
    for_loop_var_153850 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 304, 4), zip_call_result_153849)
    # Assigning a type to the variable 'ax' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'ax', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 4), for_loop_var_153850))
    # Assigning a type to the variable 'subplotspec' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'subplotspec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 4), for_loop_var_153850))
    # SSA begins for a for statement (line 304)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'subplotspec' (line 306)
    subplotspec_153851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'subplotspec')
    # Getting the type of 'None' (line 306)
    None_153852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'None')
    
    (may_be_153853, more_types_in_union_153854) = may_be_none(subplotspec_153851, None_153852)

    if may_be_153853:

        if more_types_in_union_153854:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_153854:
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to setdefault(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'subplotspec' (line 309)
    subplotspec_153857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 43), 'subplotspec', False)
    
    # Obtaining an instance of the builtin type 'list' (line 309)
    list_153858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 309)
    
    # Processing the call keyword arguments (line 309)
    kwargs_153859 = {}
    # Getting the type of 'subplot_dict' (line 309)
    subplot_dict_153855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'subplot_dict', False)
    # Obtaining the member 'setdefault' of a type (line 309)
    setdefault_153856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), subplot_dict_153855, 'setdefault')
    # Calling setdefault(args, kwargs) (line 309)
    setdefault_call_result_153860 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), setdefault_153856, *[subplotspec_153857, list_153858], **kwargs_153859)
    
    # Assigning a type to the variable 'subplots' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'subplots', setdefault_call_result_153860)
    
    
    # Getting the type of 'subplots' (line 311)
    subplots_153861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'subplots')
    # Applying the 'not' unary operator (line 311)
    result_not__153862 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 11), 'not', subplots_153861)
    
    # Testing the type of an if condition (line 311)
    if_condition_153863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), result_not__153862)
    # Assigning a type to the variable 'if_condition_153863' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_153863', if_condition_153863)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 312):
    
    # Assigning a Call to a Name:
    
    # Call to get_geometry(...): (line 312)
    # Processing the call keyword arguments (line 312)
    kwargs_153866 = {}
    # Getting the type of 'subplotspec' (line 312)
    subplotspec_153864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 35), 'subplotspec', False)
    # Obtaining the member 'get_geometry' of a type (line 312)
    get_geometry_153865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 35), subplotspec_153864, 'get_geometry')
    # Calling get_geometry(args, kwargs) (line 312)
    get_geometry_call_result_153867 = invoke(stypy.reporting.localization.Localization(__file__, 312, 35), get_geometry_153865, *[], **kwargs_153866)
    
    # Assigning a type to the variable 'call_assignment_153018' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153018', get_geometry_call_result_153867)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'int')
    # Processing the call keyword arguments
    kwargs_153871 = {}
    # Getting the type of 'call_assignment_153018' (line 312)
    call_assignment_153018_153868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153018', False)
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___153869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), call_assignment_153018_153868, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153872 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153869, *[int_153870], **kwargs_153871)
    
    # Assigning a type to the variable 'call_assignment_153019' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153019', getitem___call_result_153872)
    
    # Assigning a Name to a Name (line 312):
    # Getting the type of 'call_assignment_153019' (line 312)
    call_assignment_153019_153873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153019')
    # Assigning a type to the variable 'myrows' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'myrows', call_assignment_153019_153873)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'int')
    # Processing the call keyword arguments
    kwargs_153877 = {}
    # Getting the type of 'call_assignment_153018' (line 312)
    call_assignment_153018_153874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153018', False)
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___153875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), call_assignment_153018_153874, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153878 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153875, *[int_153876], **kwargs_153877)
    
    # Assigning a type to the variable 'call_assignment_153020' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153020', getitem___call_result_153878)
    
    # Assigning a Name to a Name (line 312):
    # Getting the type of 'call_assignment_153020' (line 312)
    call_assignment_153020_153879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153020')
    # Assigning a type to the variable 'mycols' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'mycols', call_assignment_153020_153879)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'int')
    # Processing the call keyword arguments
    kwargs_153883 = {}
    # Getting the type of 'call_assignment_153018' (line 312)
    call_assignment_153018_153880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153018', False)
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___153881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), call_assignment_153018_153880, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153884 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153881, *[int_153882], **kwargs_153883)
    
    # Assigning a type to the variable 'call_assignment_153021' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153021', getitem___call_result_153884)
    
    # Assigning a Name to a Name (line 312):
    # Getting the type of 'call_assignment_153021' (line 312)
    call_assignment_153021_153885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153021')
    # Assigning a type to the variable '_' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), '_', call_assignment_153021_153885)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'int')
    # Processing the call keyword arguments
    kwargs_153889 = {}
    # Getting the type of 'call_assignment_153018' (line 312)
    call_assignment_153018_153886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153018', False)
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___153887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), call_assignment_153018_153886, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153890 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153887, *[int_153888], **kwargs_153889)
    
    # Assigning a type to the variable 'call_assignment_153022' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153022', getitem___call_result_153890)
    
    # Assigning a Name to a Name (line 312):
    # Getting the type of 'call_assignment_153022' (line 312)
    call_assignment_153022_153891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'call_assignment_153022')
    # Assigning a type to the variable '_' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 31), '_', call_assignment_153022_153891)
    
    # Call to append(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'myrows' (line 313)
    myrows_153894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'myrows', False)
    # Processing the call keyword arguments (line 313)
    kwargs_153895 = {}
    # Getting the type of 'nrows_list' (line 313)
    nrows_list_153892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'nrows_list', False)
    # Obtaining the member 'append' of a type (line 313)
    append_153893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), nrows_list_153892, 'append')
    # Calling append(args, kwargs) (line 313)
    append_call_result_153896 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), append_153893, *[myrows_153894], **kwargs_153895)
    
    
    # Call to append(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'mycols' (line 314)
    mycols_153899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 30), 'mycols', False)
    # Processing the call keyword arguments (line 314)
    kwargs_153900 = {}
    # Getting the type of 'ncols_list' (line 314)
    ncols_list_153897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'ncols_list', False)
    # Obtaining the member 'append' of a type (line 314)
    append_153898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), ncols_list_153897, 'append')
    # Calling append(args, kwargs) (line 314)
    append_call_result_153901 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), append_153898, *[mycols_153899], **kwargs_153900)
    
    
    # Call to append(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'subplotspec' (line 315)
    subplotspec_153904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'subplotspec', False)
    # Processing the call keyword arguments (line 315)
    kwargs_153905 = {}
    # Getting the type of 'subplotspec_list2' (line 315)
    subplotspec_list2_153902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'subplotspec_list2', False)
    # Obtaining the member 'append' of a type (line 315)
    append_153903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), subplotspec_list2_153902, 'append')
    # Calling append(args, kwargs) (line 315)
    append_call_result_153906 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), append_153903, *[subplotspec_153904], **kwargs_153905)
    
    
    # Call to append(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'subplots' (line 316)
    subplots_153909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 32), 'subplots', False)
    # Processing the call keyword arguments (line 316)
    kwargs_153910 = {}
    # Getting the type of 'subplot_list' (line 316)
    subplot_list_153907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'subplot_list', False)
    # Obtaining the member 'append' of a type (line 316)
    append_153908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), subplot_list_153907, 'append')
    # Calling append(args, kwargs) (line 316)
    append_call_result_153911 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), append_153908, *[subplots_153909], **kwargs_153910)
    
    
    # Call to append(...): (line 317)
    # Processing the call arguments (line 317)
    
    # Call to get_position(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'fig' (line 317)
    fig_153916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 57), 'fig', False)
    # Processing the call keyword arguments (line 317)
    kwargs_153917 = {}
    # Getting the type of 'subplotspec' (line 317)
    subplotspec_153914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'subplotspec', False)
    # Obtaining the member 'get_position' of a type (line 317)
    get_position_153915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 32), subplotspec_153914, 'get_position')
    # Calling get_position(args, kwargs) (line 317)
    get_position_call_result_153918 = invoke(stypy.reporting.localization.Localization(__file__, 317, 32), get_position_153915, *[fig_153916], **kwargs_153917)
    
    # Processing the call keyword arguments (line 317)
    kwargs_153919 = {}
    # Getting the type of 'ax_bbox_list' (line 317)
    ax_bbox_list_153912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'ax_bbox_list', False)
    # Obtaining the member 'append' of a type (line 317)
    append_153913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), ax_bbox_list_153912, 'append')
    # Calling append(args, kwargs) (line 317)
    append_call_result_153920 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), append_153913, *[get_position_call_result_153918], **kwargs_153919)
    
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'ax' (line 319)
    ax_153923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'ax', False)
    # Processing the call keyword arguments (line 319)
    kwargs_153924 = {}
    # Getting the type of 'subplots' (line 319)
    subplots_153921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'subplots', False)
    # Obtaining the member 'append' of a type (line 319)
    append_153922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), subplots_153921, 'append')
    # Calling append(args, kwargs) (line 319)
    append_call_result_153925 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), append_153922, *[ax_153923], **kwargs_153924)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'nrows_list' (line 321)
    nrows_list_153927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'nrows_list', False)
    # Processing the call keyword arguments (line 321)
    kwargs_153928 = {}
    # Getting the type of 'len' (line 321)
    len_153926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'len', False)
    # Calling len(args, kwargs) (line 321)
    len_call_result_153929 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), len_153926, *[nrows_list_153927], **kwargs_153928)
    
    int_153930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 27), 'int')
    # Applying the binary operator '==' (line 321)
    result_eq_153931 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), '==', len_call_result_153929, int_153930)
    
    
    
    # Call to len(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'ncols_list' (line 321)
    ncols_list_153933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 38), 'ncols_list', False)
    # Processing the call keyword arguments (line 321)
    kwargs_153934 = {}
    # Getting the type of 'len' (line 321)
    len_153932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 34), 'len', False)
    # Calling len(args, kwargs) (line 321)
    len_call_result_153935 = invoke(stypy.reporting.localization.Localization(__file__, 321, 34), len_153932, *[ncols_list_153933], **kwargs_153934)
    
    int_153936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 53), 'int')
    # Applying the binary operator '==' (line 321)
    result_eq_153937 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 34), '==', len_call_result_153935, int_153936)
    
    # Applying the binary operator 'or' (line 321)
    result_or_keyword_153938 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 7), 'or', result_eq_153931, result_eq_153937)
    
    # Testing the type of an if condition (line 321)
    if_condition_153939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 4), result_or_keyword_153938)
    # Assigning a type to the variable 'if_condition_153939' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_condition_153939', if_condition_153939)
    # SSA begins for if statement (line 321)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'dict' (line 322)
    dict_153940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 322)
    
    # Assigning a type to the variable 'stypy_return_type' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type', dict_153940)
    # SSA join for if statement (line 321)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to max(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'nrows_list' (line 324)
    nrows_list_153942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'nrows_list', False)
    # Processing the call keyword arguments (line 324)
    kwargs_153943 = {}
    # Getting the type of 'max' (line 324)
    max_153941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'max', False)
    # Calling max(args, kwargs) (line 324)
    max_call_result_153944 = invoke(stypy.reporting.localization.Localization(__file__, 324, 16), max_153941, *[nrows_list_153942], **kwargs_153943)
    
    # Assigning a type to the variable 'max_nrows' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'max_nrows', max_call_result_153944)
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to max(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'ncols_list' (line 325)
    ncols_list_153946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'ncols_list', False)
    # Processing the call keyword arguments (line 325)
    kwargs_153947 = {}
    # Getting the type of 'max' (line 325)
    max_153945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'max', False)
    # Calling max(args, kwargs) (line 325)
    max_call_result_153948 = invoke(stypy.reporting.localization.Localization(__file__, 325, 16), max_153945, *[ncols_list_153946], **kwargs_153947)
    
    # Assigning a type to the variable 'max_ncols' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'max_ncols', max_call_result_153948)
    
    # Assigning a List to a Name (line 327):
    
    # Assigning a List to a Name (line 327):
    
    # Obtaining an instance of the builtin type 'list' (line 327)
    list_153949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 327)
    
    # Assigning a type to the variable 'num1num2_list' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'num1num2_list', list_153949)
    
    # Getting the type of 'subplotspec_list2' (line 328)
    subplotspec_list2_153950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'subplotspec_list2')
    # Testing the type of a for loop iterable (line 328)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 328, 4), subplotspec_list2_153950)
    # Getting the type of the for loop variable (line 328)
    for_loop_var_153951 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 328, 4), subplotspec_list2_153950)
    # Assigning a type to the variable 'subplotspec' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'subplotspec', for_loop_var_153951)
    # SSA begins for a for statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 329):
    
    # Assigning a Call to a Name:
    
    # Call to get_geometry(...): (line 329)
    # Processing the call keyword arguments (line 329)
    kwargs_153954 = {}
    # Getting the type of 'subplotspec' (line 329)
    subplotspec_153952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'subplotspec', False)
    # Obtaining the member 'get_geometry' of a type (line 329)
    get_geometry_153953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 33), subplotspec_153952, 'get_geometry')
    # Calling get_geometry(args, kwargs) (line 329)
    get_geometry_call_result_153955 = invoke(stypy.reporting.localization.Localization(__file__, 329, 33), get_geometry_153953, *[], **kwargs_153954)
    
    # Assigning a type to the variable 'call_assignment_153023' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153023', get_geometry_call_result_153955)
    
    # Assigning a Call to a Name (line 329):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153959 = {}
    # Getting the type of 'call_assignment_153023' (line 329)
    call_assignment_153023_153956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153023', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___153957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_153023_153956, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153960 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153957, *[int_153958], **kwargs_153959)
    
    # Assigning a type to the variable 'call_assignment_153024' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153024', getitem___call_result_153960)
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'call_assignment_153024' (line 329)
    call_assignment_153024_153961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153024')
    # Assigning a type to the variable 'rows' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'rows', call_assignment_153024_153961)
    
    # Assigning a Call to a Name (line 329):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153965 = {}
    # Getting the type of 'call_assignment_153023' (line 329)
    call_assignment_153023_153962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153023', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___153963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_153023_153962, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153966 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153963, *[int_153964], **kwargs_153965)
    
    # Assigning a type to the variable 'call_assignment_153025' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153025', getitem___call_result_153966)
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'call_assignment_153025' (line 329)
    call_assignment_153025_153967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153025')
    # Assigning a type to the variable 'cols' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'cols', call_assignment_153025_153967)
    
    # Assigning a Call to a Name (line 329):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153971 = {}
    # Getting the type of 'call_assignment_153023' (line 329)
    call_assignment_153023_153968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153023', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___153969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_153023_153968, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153972 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153969, *[int_153970], **kwargs_153971)
    
    # Assigning a type to the variable 'call_assignment_153026' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153026', getitem___call_result_153972)
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'call_assignment_153026' (line 329)
    call_assignment_153026_153973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153026')
    # Assigning a type to the variable 'num1' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'num1', call_assignment_153026_153973)
    
    # Assigning a Call to a Name (line 329):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153977 = {}
    # Getting the type of 'call_assignment_153023' (line 329)
    call_assignment_153023_153974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153023', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___153975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_153023_153974, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153978 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153975, *[int_153976], **kwargs_153977)
    
    # Assigning a type to the variable 'call_assignment_153027' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153027', getitem___call_result_153978)
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'call_assignment_153027' (line 329)
    call_assignment_153027_153979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_153027')
    # Assigning a type to the variable 'num2' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 26), 'num2', call_assignment_153027_153979)
    
    # Assigning a Call to a Tuple (line 330):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'max_nrows' (line 330)
    max_nrows_153981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'max_nrows', False)
    # Getting the type of 'rows' (line 330)
    rows_153982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 45), 'rows', False)
    # Processing the call keyword arguments (line 330)
    kwargs_153983 = {}
    # Getting the type of 'divmod' (line 330)
    divmod_153980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'divmod', False)
    # Calling divmod(args, kwargs) (line 330)
    divmod_call_result_153984 = invoke(stypy.reporting.localization.Localization(__file__, 330, 27), divmod_153980, *[max_nrows_153981, rows_153982], **kwargs_153983)
    
    # Assigning a type to the variable 'call_assignment_153028' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153028', divmod_call_result_153984)
    
    # Assigning a Call to a Name (line 330):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153988 = {}
    # Getting the type of 'call_assignment_153028' (line 330)
    call_assignment_153028_153985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153028', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___153986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), call_assignment_153028_153985, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153989 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153986, *[int_153987], **kwargs_153988)
    
    # Assigning a type to the variable 'call_assignment_153029' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153029', getitem___call_result_153989)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'call_assignment_153029' (line 330)
    call_assignment_153029_153990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153029')
    # Assigning a type to the variable 'div_row' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'div_row', call_assignment_153029_153990)
    
    # Assigning a Call to a Name (line 330):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_153993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'int')
    # Processing the call keyword arguments
    kwargs_153994 = {}
    # Getting the type of 'call_assignment_153028' (line 330)
    call_assignment_153028_153991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153028', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___153992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), call_assignment_153028_153991, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_153995 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___153992, *[int_153993], **kwargs_153994)
    
    # Assigning a type to the variable 'call_assignment_153030' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153030', getitem___call_result_153995)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'call_assignment_153030' (line 330)
    call_assignment_153030_153996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'call_assignment_153030')
    # Assigning a type to the variable 'mod_row' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'mod_row', call_assignment_153030_153996)
    
    # Assigning a Call to a Tuple (line 331):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'max_ncols' (line 331)
    max_ncols_153998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 34), 'max_ncols', False)
    # Getting the type of 'cols' (line 331)
    cols_153999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 45), 'cols', False)
    # Processing the call keyword arguments (line 331)
    kwargs_154000 = {}
    # Getting the type of 'divmod' (line 331)
    divmod_153997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'divmod', False)
    # Calling divmod(args, kwargs) (line 331)
    divmod_call_result_154001 = invoke(stypy.reporting.localization.Localization(__file__, 331, 27), divmod_153997, *[max_ncols_153998, cols_153999], **kwargs_154000)
    
    # Assigning a type to the variable 'call_assignment_153031' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153031', divmod_call_result_154001)
    
    # Assigning a Call to a Name (line 331):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_154004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 8), 'int')
    # Processing the call keyword arguments
    kwargs_154005 = {}
    # Getting the type of 'call_assignment_153031' (line 331)
    call_assignment_153031_154002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153031', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___154003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), call_assignment_153031_154002, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_154006 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___154003, *[int_154004], **kwargs_154005)
    
    # Assigning a type to the variable 'call_assignment_153032' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153032', getitem___call_result_154006)
    
    # Assigning a Name to a Name (line 331):
    # Getting the type of 'call_assignment_153032' (line 331)
    call_assignment_153032_154007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153032')
    # Assigning a type to the variable 'div_col' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'div_col', call_assignment_153032_154007)
    
    # Assigning a Call to a Name (line 331):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_154010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 8), 'int')
    # Processing the call keyword arguments
    kwargs_154011 = {}
    # Getting the type of 'call_assignment_153031' (line 331)
    call_assignment_153031_154008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153031', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___154009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), call_assignment_153031_154008, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_154012 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___154009, *[int_154010], **kwargs_154011)
    
    # Assigning a type to the variable 'call_assignment_153033' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153033', getitem___call_result_154012)
    
    # Assigning a Name to a Name (line 331):
    # Getting the type of 'call_assignment_153033' (line 331)
    call_assignment_153033_154013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'call_assignment_153033')
    # Assigning a type to the variable 'mod_col' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 17), 'mod_col', call_assignment_153033_154013)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mod_row' (line 332)
    mod_row_154014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'mod_row')
    int_154015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 23), 'int')
    # Applying the binary operator '!=' (line 332)
    result_ne_154016 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 12), '!=', mod_row_154014, int_154015)
    
    
    # Getting the type of 'mod_col' (line 332)
    mod_col_154017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'mod_col')
    int_154018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 41), 'int')
    # Applying the binary operator '!=' (line 332)
    result_ne_154019 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 30), '!=', mod_col_154017, int_154018)
    
    # Applying the binary operator 'or' (line 332)
    result_or_keyword_154020 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 11), 'or', result_ne_154016, result_ne_154019)
    
    # Testing the type of an if condition (line 332)
    if_condition_154021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), result_or_keyword_154020)
    # Assigning a type to the variable 'if_condition_154021' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_154021', if_condition_154021)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 333)
    # Processing the call arguments (line 333)
    str_154023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 31), 'str', '')
    # Processing the call keyword arguments (line 333)
    kwargs_154024 = {}
    # Getting the type of 'RuntimeError' (line 333)
    RuntimeError_154022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 333)
    RuntimeError_call_result_154025 = invoke(stypy.reporting.localization.Localization(__file__, 333, 18), RuntimeError_154022, *[str_154023], **kwargs_154024)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 333, 12), RuntimeError_call_result_154025, 'raise parameter', BaseException)
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 335):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'num1' (line 335)
    num1_154027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'num1', False)
    # Getting the type of 'cols' (line 335)
    cols_154028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 40), 'cols', False)
    # Processing the call keyword arguments (line 335)
    kwargs_154029 = {}
    # Getting the type of 'divmod' (line 335)
    divmod_154026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 27), 'divmod', False)
    # Calling divmod(args, kwargs) (line 335)
    divmod_call_result_154030 = invoke(stypy.reporting.localization.Localization(__file__, 335, 27), divmod_154026, *[num1_154027, cols_154028], **kwargs_154029)
    
    # Assigning a type to the variable 'call_assignment_153034' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153034', divmod_call_result_154030)
    
    # Assigning a Call to a Name (line 335):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_154033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
    # Processing the call keyword arguments
    kwargs_154034 = {}
    # Getting the type of 'call_assignment_153034' (line 335)
    call_assignment_153034_154031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153034', False)
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___154032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), call_assignment_153034_154031, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_154035 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___154032, *[int_154033], **kwargs_154034)
    
    # Assigning a type to the variable 'call_assignment_153035' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153035', getitem___call_result_154035)
    
    # Assigning a Name to a Name (line 335):
    # Getting the type of 'call_assignment_153035' (line 335)
    call_assignment_153035_154036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153035')
    # Assigning a type to the variable 'rowNum1' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'rowNum1', call_assignment_153035_154036)
    
    # Assigning a Call to a Name (line 335):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_154039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
    # Processing the call keyword arguments
    kwargs_154040 = {}
    # Getting the type of 'call_assignment_153034' (line 335)
    call_assignment_153034_154037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153034', False)
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___154038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), call_assignment_153034_154037, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_154041 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___154038, *[int_154039], **kwargs_154040)
    
    # Assigning a type to the variable 'call_assignment_153036' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153036', getitem___call_result_154041)
    
    # Assigning a Name to a Name (line 335):
    # Getting the type of 'call_assignment_153036' (line 335)
    call_assignment_153036_154042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_153036')
    # Assigning a type to the variable 'colNum1' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'colNum1', call_assignment_153036_154042)
    
    # Type idiom detected: calculating its left and rigth part (line 336)
    # Getting the type of 'num2' (line 336)
    num2_154043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'num2')
    # Getting the type of 'None' (line 336)
    None_154044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'None')
    
    (may_be_154045, more_types_in_union_154046) = may_be_none(num2_154043, None_154044)

    if may_be_154045:

        if more_types_in_union_154046:
            # Runtime conditional SSA (line 336)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Tuple to a Tuple (line 337):
        
        # Assigning a Name to a Name (line 337):
        # Getting the type of 'rowNum1' (line 337)
        rowNum1_154047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'rowNum1')
        # Assigning a type to the variable 'tuple_assignment_153037' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_assignment_153037', rowNum1_154047)
        
        # Assigning a Name to a Name (line 337):
        # Getting the type of 'colNum1' (line 337)
        colNum1_154048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 40), 'colNum1')
        # Assigning a type to the variable 'tuple_assignment_153038' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_assignment_153038', colNum1_154048)
        
        # Assigning a Name to a Name (line 337):
        # Getting the type of 'tuple_assignment_153037' (line 337)
        tuple_assignment_153037_154049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_assignment_153037')
        # Assigning a type to the variable 'rowNum2' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'rowNum2', tuple_assignment_153037_154049)
        
        # Assigning a Name to a Name (line 337):
        # Getting the type of 'tuple_assignment_153038' (line 337)
        tuple_assignment_153038_154050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_assignment_153038')
        # Assigning a type to the variable 'colNum2' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'colNum2', tuple_assignment_153038_154050)

        if more_types_in_union_154046:
            # Runtime conditional SSA for else branch (line 336)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_154045) or more_types_in_union_154046):
        
        # Assigning a Call to a Tuple (line 339):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'num2' (line 339)
        num2_154052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 38), 'num2', False)
        # Getting the type of 'cols' (line 339)
        cols_154053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 44), 'cols', False)
        # Processing the call keyword arguments (line 339)
        kwargs_154054 = {}
        # Getting the type of 'divmod' (line 339)
        divmod_154051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'divmod', False)
        # Calling divmod(args, kwargs) (line 339)
        divmod_call_result_154055 = invoke(stypy.reporting.localization.Localization(__file__, 339, 31), divmod_154051, *[num2_154052, cols_154053], **kwargs_154054)
        
        # Assigning a type to the variable 'call_assignment_153039' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153039', divmod_call_result_154055)
        
        # Assigning a Call to a Name (line 339):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_154058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 12), 'int')
        # Processing the call keyword arguments
        kwargs_154059 = {}
        # Getting the type of 'call_assignment_153039' (line 339)
        call_assignment_153039_154056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153039', False)
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___154057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), call_assignment_153039_154056, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_154060 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___154057, *[int_154058], **kwargs_154059)
        
        # Assigning a type to the variable 'call_assignment_153040' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153040', getitem___call_result_154060)
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'call_assignment_153040' (line 339)
        call_assignment_153040_154061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153040')
        # Assigning a type to the variable 'rowNum2' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'rowNum2', call_assignment_153040_154061)
        
        # Assigning a Call to a Name (line 339):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_154064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 12), 'int')
        # Processing the call keyword arguments
        kwargs_154065 = {}
        # Getting the type of 'call_assignment_153039' (line 339)
        call_assignment_153039_154062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153039', False)
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___154063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), call_assignment_153039_154062, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_154066 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___154063, *[int_154064], **kwargs_154065)
        
        # Assigning a type to the variable 'call_assignment_153041' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153041', getitem___call_result_154066)
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'call_assignment_153041' (line 339)
        call_assignment_153041_154067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'call_assignment_153041')
        # Assigning a type to the variable 'colNum2' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 21), 'colNum2', call_assignment_153041_154067)

        if (may_be_154045 and more_types_in_union_154046):
            # SSA join for if statement (line 336)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to append(...): (line 341)
    # Processing the call arguments (line 341)
    
    # Obtaining an instance of the builtin type 'tuple' (line 341)
    tuple_154070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 341)
    # Adding element type (line 341)
    # Getting the type of 'rowNum1' (line 341)
    rowNum1_154071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'rowNum1', False)
    # Getting the type of 'div_row' (line 341)
    div_row_154072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 40), 'div_row', False)
    # Applying the binary operator '*' (line 341)
    result_mul_154073 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 30), '*', rowNum1_154071, div_row_154072)
    
    # Getting the type of 'max_ncols' (line 341)
    max_ncols_154074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 50), 'max_ncols', False)
    # Applying the binary operator '*' (line 341)
    result_mul_154075 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 48), '*', result_mul_154073, max_ncols_154074)
    
    # Getting the type of 'colNum1' (line 342)
    colNum1_154076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 30), 'colNum1', False)
    # Getting the type of 'div_col' (line 342)
    div_col_154077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 40), 'div_col', False)
    # Applying the binary operator '*' (line 342)
    result_mul_154078 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 30), '*', colNum1_154076, div_col_154077)
    
    # Applying the binary operator '+' (line 341)
    result_add_154079 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 30), '+', result_mul_154075, result_mul_154078)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 30), tuple_154070, result_add_154079)
    # Adding element type (line 341)
    # Getting the type of 'rowNum2' (line 343)
    rowNum2_154080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'rowNum2', False)
    int_154081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 42), 'int')
    # Applying the binary operator '+' (line 343)
    result_add_154082 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 32), '+', rowNum2_154080, int_154081)
    
    # Getting the type of 'div_row' (line 343)
    div_row_154083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'div_row', False)
    # Applying the binary operator '*' (line 343)
    result_mul_154084 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 31), '*', result_add_154082, div_row_154083)
    
    int_154085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 57), 'int')
    # Applying the binary operator '-' (line 343)
    result_sub_154086 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 31), '-', result_mul_154084, int_154085)
    
    # Getting the type of 'max_ncols' (line 343)
    max_ncols_154087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 62), 'max_ncols', False)
    # Applying the binary operator '*' (line 343)
    result_mul_154088 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 30), '*', result_sub_154086, max_ncols_154087)
    
    # Getting the type of 'colNum2' (line 344)
    colNum2_154089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 31), 'colNum2', False)
    int_154090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 41), 'int')
    # Applying the binary operator '+' (line 344)
    result_add_154091 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 31), '+', colNum2_154089, int_154090)
    
    # Getting the type of 'div_col' (line 344)
    div_col_154092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 46), 'div_col', False)
    # Applying the binary operator '*' (line 344)
    result_mul_154093 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 30), '*', result_add_154091, div_col_154092)
    
    # Applying the binary operator '+' (line 343)
    result_add_154094 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 30), '+', result_mul_154088, result_mul_154093)
    
    int_154095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 56), 'int')
    # Applying the binary operator '-' (line 344)
    result_sub_154096 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 54), '-', result_add_154094, int_154095)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 30), tuple_154070, result_sub_154096)
    
    # Processing the call keyword arguments (line 341)
    kwargs_154097 = {}
    # Getting the type of 'num1num2_list' (line 341)
    num1num2_list_154068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'num1num2_list', False)
    # Obtaining the member 'append' of a type (line 341)
    append_154069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), num1num2_list_154068, 'append')
    # Calling append(args, kwargs) (line 341)
    append_call_result_154098 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), append_154069, *[tuple_154070], **kwargs_154097)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 346):
    
    # Assigning a Call to a Name (line 346):
    
    # Call to auto_adjust_subplotpars(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'fig' (line 346)
    fig_154100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 37), 'fig', False)
    # Getting the type of 'renderer' (line 346)
    renderer_154101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 42), 'renderer', False)
    # Processing the call keyword arguments (line 346)
    
    # Obtaining an instance of the builtin type 'tuple' (line 347)
    tuple_154102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 347)
    # Adding element type (line 347)
    # Getting the type of 'max_nrows' (line 347)
    max_nrows_154103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 50), 'max_nrows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 50), tuple_154102, max_nrows_154103)
    # Adding element type (line 347)
    # Getting the type of 'max_ncols' (line 347)
    max_ncols_154104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 61), 'max_ncols', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 50), tuple_154102, max_ncols_154104)
    
    keyword_154105 = tuple_154102
    # Getting the type of 'num1num2_list' (line 348)
    num1num2_list_154106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 51), 'num1num2_list', False)
    keyword_154107 = num1num2_list_154106
    # Getting the type of 'subplot_list' (line 349)
    subplot_list_154108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 50), 'subplot_list', False)
    keyword_154109 = subplot_list_154108
    # Getting the type of 'ax_bbox_list' (line 350)
    ax_bbox_list_154110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 50), 'ax_bbox_list', False)
    keyword_154111 = ax_bbox_list_154110
    # Getting the type of 'pad' (line 351)
    pad_154112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 41), 'pad', False)
    keyword_154113 = pad_154112
    # Getting the type of 'h_pad' (line 351)
    h_pad_154114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 52), 'h_pad', False)
    keyword_154115 = h_pad_154114
    # Getting the type of 'w_pad' (line 351)
    w_pad_154116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 65), 'w_pad', False)
    keyword_154117 = w_pad_154116
    kwargs_154118 = {'nrows_ncols': keyword_154105, 'h_pad': keyword_154115, 'w_pad': keyword_154117, 'subplot_list': keyword_154109, 'pad': keyword_154113, 'num1num2_list': keyword_154107, 'ax_bbox_list': keyword_154111}
    # Getting the type of 'auto_adjust_subplotpars' (line 346)
    auto_adjust_subplotpars_154099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 13), 'auto_adjust_subplotpars', False)
    # Calling auto_adjust_subplotpars(args, kwargs) (line 346)
    auto_adjust_subplotpars_call_result_154119 = invoke(stypy.reporting.localization.Localization(__file__, 346, 13), auto_adjust_subplotpars_154099, *[fig_154100, renderer_154101], **kwargs_154118)
    
    # Assigning a type to the variable 'kwargs' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'kwargs', auto_adjust_subplotpars_call_result_154119)
    
    # Type idiom detected: calculating its left and rigth part (line 353)
    # Getting the type of 'rect' (line 353)
    rect_154120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'rect')
    # Getting the type of 'None' (line 353)
    None_154121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'None')
    
    (may_be_154122, more_types_in_union_154123) = may_not_be_none(rect_154120, None_154121)

    if may_be_154122:

        if more_types_in_union_154123:
            # Runtime conditional SSA (line 353)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Tuple (line 362):
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_154124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'int')
        # Getting the type of 'rect' (line 362)
        rect_154125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'rect')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___154126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), rect_154125, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_154127 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), getitem___154126, int_154124)
        
        # Assigning a type to the variable 'tuple_var_assignment_153042' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153042', subscript_call_result_154127)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_154128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'int')
        # Getting the type of 'rect' (line 362)
        rect_154129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'rect')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___154130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), rect_154129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_154131 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), getitem___154130, int_154128)
        
        # Assigning a type to the variable 'tuple_var_assignment_153043' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153043', subscript_call_result_154131)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_154132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'int')
        # Getting the type of 'rect' (line 362)
        rect_154133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'rect')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___154134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), rect_154133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_154135 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), getitem___154134, int_154132)
        
        # Assigning a type to the variable 'tuple_var_assignment_153044' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153044', subscript_call_result_154135)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_154136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'int')
        # Getting the type of 'rect' (line 362)
        rect_154137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'rect')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___154138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), rect_154137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_154139 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), getitem___154138, int_154136)
        
        # Assigning a type to the variable 'tuple_var_assignment_153045' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153045', subscript_call_result_154139)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_153042' (line 362)
        tuple_var_assignment_153042_154140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153042')
        # Assigning a type to the variable 'left' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'left', tuple_var_assignment_153042_154140)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_153043' (line 362)
        tuple_var_assignment_153043_154141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153043')
        # Assigning a type to the variable 'bottom' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 14), 'bottom', tuple_var_assignment_153043_154141)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_153044' (line 362)
        tuple_var_assignment_153044_154142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153044')
        # Assigning a type to the variable 'right' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'right', tuple_var_assignment_153044_154142)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_153045' (line 362)
        tuple_var_assignment_153045_154143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_153045')
        # Assigning a type to the variable 'top' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'top', tuple_var_assignment_153045_154143)
        
        # Type idiom detected: calculating its left and rigth part (line 363)
        # Getting the type of 'left' (line 363)
        left_154144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'left')
        # Getting the type of 'None' (line 363)
        None_154145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'None')
        
        (may_be_154146, more_types_in_union_154147) = may_not_be_none(left_154144, None_154145)

        if may_be_154146:

            if more_types_in_union_154147:
                # Runtime conditional SSA (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'left' (line 364)
            left_154148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'left')
            
            # Obtaining the type of the subscript
            str_154149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 27), 'str', 'left')
            # Getting the type of 'kwargs' (line 364)
            kwargs_154150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'kwargs')
            # Obtaining the member '__getitem__' of a type (line 364)
            getitem___154151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 20), kwargs_154150, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 364)
            subscript_call_result_154152 = invoke(stypy.reporting.localization.Localization(__file__, 364, 20), getitem___154151, str_154149)
            
            # Applying the binary operator '+=' (line 364)
            result_iadd_154153 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 12), '+=', left_154148, subscript_call_result_154152)
            # Assigning a type to the variable 'left' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'left', result_iadd_154153)
            

            if more_types_in_union_154147:
                # SSA join for if statement (line 363)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 365)
        # Getting the type of 'bottom' (line 365)
        bottom_154154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'bottom')
        # Getting the type of 'None' (line 365)
        None_154155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 25), 'None')
        
        (may_be_154156, more_types_in_union_154157) = may_not_be_none(bottom_154154, None_154155)

        if may_be_154156:

            if more_types_in_union_154157:
                # Runtime conditional SSA (line 365)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'bottom' (line 366)
            bottom_154158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'bottom')
            
            # Obtaining the type of the subscript
            str_154159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 29), 'str', 'bottom')
            # Getting the type of 'kwargs' (line 366)
            kwargs_154160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'kwargs')
            # Obtaining the member '__getitem__' of a type (line 366)
            getitem___154161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 22), kwargs_154160, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 366)
            subscript_call_result_154162 = invoke(stypy.reporting.localization.Localization(__file__, 366, 22), getitem___154161, str_154159)
            
            # Applying the binary operator '+=' (line 366)
            result_iadd_154163 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 12), '+=', bottom_154158, subscript_call_result_154162)
            # Assigning a type to the variable 'bottom' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'bottom', result_iadd_154163)
            

            if more_types_in_union_154157:
                # SSA join for if statement (line 365)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 367)
        # Getting the type of 'right' (line 367)
        right_154164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'right')
        # Getting the type of 'None' (line 367)
        None_154165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 24), 'None')
        
        (may_be_154166, more_types_in_union_154167) = may_not_be_none(right_154164, None_154165)

        if may_be_154166:

            if more_types_in_union_154167:
                # Runtime conditional SSA (line 367)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'right' (line 368)
            right_154168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'right')
            int_154169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 22), 'int')
            
            # Obtaining the type of the subscript
            str_154170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 33), 'str', 'right')
            # Getting the type of 'kwargs' (line 368)
            kwargs_154171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'kwargs')
            # Obtaining the member '__getitem__' of a type (line 368)
            getitem___154172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 26), kwargs_154171, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 368)
            subscript_call_result_154173 = invoke(stypy.reporting.localization.Localization(__file__, 368, 26), getitem___154172, str_154170)
            
            # Applying the binary operator '-' (line 368)
            result_sub_154174 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 22), '-', int_154169, subscript_call_result_154173)
            
            # Applying the binary operator '-=' (line 368)
            result_isub_154175 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 12), '-=', right_154168, result_sub_154174)
            # Assigning a type to the variable 'right' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'right', result_isub_154175)
            

            if more_types_in_union_154167:
                # SSA join for if statement (line 367)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 369)
        # Getting the type of 'top' (line 369)
        top_154176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'top')
        # Getting the type of 'None' (line 369)
        None_154177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 22), 'None')
        
        (may_be_154178, more_types_in_union_154179) = may_not_be_none(top_154176, None_154177)

        if may_be_154178:

            if more_types_in_union_154179:
                # Runtime conditional SSA (line 369)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'top' (line 370)
            top_154180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'top')
            int_154181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'int')
            
            # Obtaining the type of the subscript
            str_154182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 31), 'str', 'top')
            # Getting the type of 'kwargs' (line 370)
            kwargs_154183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'kwargs')
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___154184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), kwargs_154183, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_154185 = invoke(stypy.reporting.localization.Localization(__file__, 370, 24), getitem___154184, str_154182)
            
            # Applying the binary operator '-' (line 370)
            result_sub_154186 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 20), '-', int_154181, subscript_call_result_154185)
            
            # Applying the binary operator '-=' (line 370)
            result_isub_154187 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 12), '-=', top_154180, result_sub_154186)
            # Assigning a type to the variable 'top' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'top', result_isub_154187)
            

            if more_types_in_union_154179:
                # SSA join for if statement (line 369)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to auto_adjust_subplotpars(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'fig' (line 375)
        fig_154189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 41), 'fig', False)
        # Getting the type of 'renderer' (line 375)
        renderer_154190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 46), 'renderer', False)
        # Processing the call keyword arguments (line 375)
        
        # Obtaining an instance of the builtin type 'tuple' (line 376)
        tuple_154191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 376)
        # Adding element type (line 376)
        # Getting the type of 'max_nrows' (line 376)
        max_nrows_154192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 54), 'max_nrows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 54), tuple_154191, max_nrows_154192)
        # Adding element type (line 376)
        # Getting the type of 'max_ncols' (line 376)
        max_ncols_154193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 65), 'max_ncols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 54), tuple_154191, max_ncols_154193)
        
        keyword_154194 = tuple_154191
        # Getting the type of 'num1num2_list' (line 377)
        num1num2_list_154195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 55), 'num1num2_list', False)
        keyword_154196 = num1num2_list_154195
        # Getting the type of 'subplot_list' (line 378)
        subplot_list_154197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 54), 'subplot_list', False)
        keyword_154198 = subplot_list_154197
        # Getting the type of 'ax_bbox_list' (line 379)
        ax_bbox_list_154199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 54), 'ax_bbox_list', False)
        keyword_154200 = ax_bbox_list_154199
        # Getting the type of 'pad' (line 380)
        pad_154201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 45), 'pad', False)
        keyword_154202 = pad_154201
        # Getting the type of 'h_pad' (line 380)
        h_pad_154203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 56), 'h_pad', False)
        keyword_154204 = h_pad_154203
        # Getting the type of 'w_pad' (line 380)
        w_pad_154205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 69), 'w_pad', False)
        keyword_154206 = w_pad_154205
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_154207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        # Getting the type of 'left' (line 381)
        left_154208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 47), 'left', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 47), tuple_154207, left_154208)
        # Adding element type (line 381)
        # Getting the type of 'bottom' (line 381)
        bottom_154209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 53), 'bottom', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 47), tuple_154207, bottom_154209)
        # Adding element type (line 381)
        # Getting the type of 'right' (line 381)
        right_154210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 61), 'right', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 47), tuple_154207, right_154210)
        # Adding element type (line 381)
        # Getting the type of 'top' (line 381)
        top_154211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 68), 'top', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 47), tuple_154207, top_154211)
        
        keyword_154212 = tuple_154207
        kwargs_154213 = {'nrows_ncols': keyword_154194, 'h_pad': keyword_154204, 'w_pad': keyword_154206, 'subplot_list': keyword_154198, 'pad': keyword_154202, 'rect': keyword_154212, 'num1num2_list': keyword_154196, 'ax_bbox_list': keyword_154200}
        # Getting the type of 'auto_adjust_subplotpars' (line 375)
        auto_adjust_subplotpars_154188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 17), 'auto_adjust_subplotpars', False)
        # Calling auto_adjust_subplotpars(args, kwargs) (line 375)
        auto_adjust_subplotpars_call_result_154214 = invoke(stypy.reporting.localization.Localization(__file__, 375, 17), auto_adjust_subplotpars_154188, *[fig_154189, renderer_154190], **kwargs_154213)
        
        # Assigning a type to the variable 'kwargs' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'kwargs', auto_adjust_subplotpars_call_result_154214)

        if more_types_in_union_154123:
            # SSA join for if statement (line 353)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'kwargs' (line 383)
    kwargs_154215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'kwargs')
    # Assigning a type to the variable 'stypy_return_type' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'stypy_return_type', kwargs_154215)
    
    # ################# End of 'get_tight_layout_figure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_tight_layout_figure' in the type store
    # Getting the type of 'stypy_return_type' (line 262)
    stypy_return_type_154216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_154216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_tight_layout_figure'
    return stypy_return_type_154216

# Assigning a type to the variable 'get_tight_layout_figure' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'get_tight_layout_figure', get_tight_layout_figure)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

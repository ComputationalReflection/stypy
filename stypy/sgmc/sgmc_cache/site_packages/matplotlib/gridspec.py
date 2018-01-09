
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: :mod:`~matplotlib.gridspec` is a module which specifies the location
3: of the subplot in the figure.
4: 
5:     ``GridSpec``
6:         specifies the geometry of the grid that a subplot will be
7:         placed. The number of rows and number of columns of the grid
8:         need to be set. Optionally, the subplot layout parameters
9:         (e.g., left, right, etc.) can be tuned.
10: 
11:     ``SubplotSpec``
12:         specifies the location of the subplot in the given *GridSpec*.
13: 
14: 
15: '''
16: 
17: from __future__ import (absolute_import, division, print_function,
18:                         unicode_literals)
19: 
20: import six
21: from six.moves import zip
22: 
23: import copy
24: import warnings
25: 
26: import matplotlib
27: from matplotlib import rcParams
28: import matplotlib.transforms as mtransforms
29: 
30: import numpy as np
31: 
32: class GridSpecBase(object):
33:     '''
34:     A base class of GridSpec that specifies the geometry of the grid
35:     that a subplot will be placed.
36:     '''
37: 
38:     def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
39:         '''
40:         The number of rows and number of columns of the grid need to
41:         be set. Optionally, the ratio of heights and widths of rows and
42:         columns can be specified.
43:         '''
44:         self._nrows, self._ncols = nrows, ncols
45:         self.set_height_ratios(height_ratios)
46:         self.set_width_ratios(width_ratios)
47: 
48:     def get_geometry(self):
49:         'get the geometry of the grid, e.g., 2,3'
50:         return self._nrows, self._ncols
51: 
52:     def get_subplot_params(self, fig=None):
53:         pass
54: 
55:     def new_subplotspec(self, loc, rowspan=1, colspan=1):
56:         '''
57:         create and return a SuplotSpec instance.
58:         '''
59:         loc1, loc2 = loc
60:         subplotspec = self[loc1:loc1+rowspan, loc2:loc2+colspan]
61:         return subplotspec
62: 
63:     def set_width_ratios(self, width_ratios):
64:         if width_ratios is not None and len(width_ratios) != self._ncols:
65:             raise ValueError('Expected the given number of width ratios to '
66:                              'match the number of columns of the grid')
67:         self._col_width_ratios = width_ratios
68: 
69:     def get_width_ratios(self):
70:         return self._col_width_ratios
71: 
72:     def set_height_ratios(self, height_ratios):
73:         if height_ratios is not None and len(height_ratios) != self._nrows:
74:             raise ValueError('Expected the given number of height ratios to '
75:                              'match the number of rows of the grid')
76:         self._row_height_ratios = height_ratios
77: 
78:     def get_height_ratios(self):
79:         return self._row_height_ratios
80: 
81:     def get_grid_positions(self, fig):
82:         '''
83:         return lists of bottom and top position of rows, left and
84:         right positions of columns.
85:         '''
86:         nrows, ncols = self.get_geometry()
87: 
88:         subplot_params = self.get_subplot_params(fig)
89:         left = subplot_params.left
90:         right = subplot_params.right
91:         bottom = subplot_params.bottom
92:         top = subplot_params.top
93:         wspace = subplot_params.wspace
94:         hspace = subplot_params.hspace
95:         totWidth = right - left
96:         totHeight = top - bottom
97: 
98:         # calculate accumulated heights of columns
99:         cellH = totHeight / (nrows + hspace*(nrows-1))
100:         sepH = hspace * cellH
101:         if self._row_height_ratios is not None:
102:             netHeight = cellH * nrows
103:             tr = float(sum(self._row_height_ratios))
104:             cellHeights = [netHeight * r / tr for r in self._row_height_ratios]
105:         else:
106:             cellHeights = [cellH] * nrows
107:         sepHeights = [0] + ([sepH] * (nrows-1))
108:         cellHs = np.cumsum(np.column_stack([sepHeights, cellHeights]).flat)
109: 
110:         # calculate accumulated widths of rows
111:         cellW = totWidth/(ncols + wspace*(ncols-1))
112:         sepW = wspace*cellW
113:         if self._col_width_ratios is not None:
114:             netWidth = cellW * ncols
115:             tr = float(sum(self._col_width_ratios))
116:             cellWidths = [netWidth*r/tr for r in self._col_width_ratios]
117:         else:
118:             cellWidths = [cellW] * ncols
119:         sepWidths = [0] + ([sepW] * (ncols-1))
120:         cellWs = np.cumsum(np.column_stack([sepWidths, cellWidths]).flat)
121: 
122:         figTops = [top - cellHs[2*rowNum] for rowNum in range(nrows)]
123:         figBottoms = [top - cellHs[2*rowNum+1] for rowNum in range(nrows)]
124:         figLefts = [left + cellWs[2*colNum] for colNum in range(ncols)]
125:         figRights = [left + cellWs[2*colNum+1] for colNum in range(ncols)]
126: 
127:         return figBottoms, figTops, figLefts, figRights
128: 
129:     def __getitem__(self, key):
130:         '''
131:         create and return a SuplotSpec instance.
132:         '''
133:         nrows, ncols = self.get_geometry()
134:         total = nrows*ncols
135: 
136:         if isinstance(key, tuple):
137:             try:
138:                 k1, k2 = key
139:             except ValueError:
140:                 raise ValueError("unrecognized subplot spec")
141: 
142:             if isinstance(k1, slice):
143:                 row1, row2, _ = k1.indices(nrows)
144:             else:
145:                 if k1 < 0:
146:                     k1 += nrows
147:                 if k1 >= nrows or k1 < 0 :
148:                     raise IndexError("index out of range")
149:                 row1, row2 = k1, k1+1
150: 
151:             if isinstance(k2, slice):
152:                 col1, col2, _ = k2.indices(ncols)
153:             else:
154:                 if k2 < 0:
155:                     k2 += ncols
156:                 if k2 >= ncols or k2 < 0 :
157:                     raise IndexError("index out of range")
158:                 col1, col2 = k2, k2+1
159: 
160:             num1 = row1*ncols + col1
161:             num2 = (row2-1)*ncols + (col2-1)
162: 
163:         # single key
164:         else:
165:             if isinstance(key, slice):
166:                 num1, num2, _ = key.indices(total)
167:                 num2 -= 1
168:             else:
169:                 if key < 0:
170:                     key += total
171:                 if key >= total or key < 0 :
172:                     raise IndexError("index out of range")
173:                 num1, num2 = key, None
174: 
175:         return SubplotSpec(self, num1, num2)
176: 
177: 
178: class GridSpec(GridSpecBase):
179:     '''
180:     A class that specifies the geometry of the grid that a subplot
181:     will be placed. The location of grid is determined by similar way
182:     as the SubplotParams.
183:     '''
184: 
185:     def __init__(self, nrows, ncols,
186:                  left=None, bottom=None, right=None, top=None,
187:                  wspace=None, hspace=None,
188:                  width_ratios=None, height_ratios=None):
189:         '''
190:         The number of rows and number of columns of the
191:         grid need to be set. Optionally, the subplot layout parameters
192:         (e.g., left, right, etc.) can be tuned.
193:         '''
194:         self.left = left
195:         self.bottom = bottom
196:         self.right = right
197:         self.top = top
198:         self.wspace = wspace
199:         self.hspace = hspace
200: 
201:         GridSpecBase.__init__(self, nrows, ncols,
202:                               width_ratios=width_ratios,
203:                               height_ratios=height_ratios)
204: 
205:     _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]
206: 
207:     def update(self, **kwargs):
208:         '''
209:         Update the current values.  If any kwarg is None, default to
210:         the current value, if set, otherwise to rc.
211:         '''
212: 
213:         for k, v in six.iteritems(kwargs):
214:             if k in self._AllowedKeys:
215:                 setattr(self, k, v)
216:             else:
217:                 raise AttributeError("%s is unknown keyword" % (k,))
218: 
219:         from matplotlib import _pylab_helpers
220:         from matplotlib.axes import SubplotBase
221:         for figmanager in six.itervalues(_pylab_helpers.Gcf.figs):
222:             for ax in figmanager.canvas.figure.axes:
223:                 # copied from Figure.subplots_adjust
224:                 if not isinstance(ax, SubplotBase):
225:                     # Check if sharing a subplots axis
226:                     if isinstance(ax._sharex, SubplotBase):
227:                         if ax._sharex.get_subplotspec().get_gridspec() == self:
228:                             ax._sharex.update_params()
229:                             ax.set_position(ax._sharex.figbox)
230:                     elif isinstance(ax._sharey, SubplotBase):
231:                         if ax._sharey.get_subplotspec().get_gridspec() == self:
232:                             ax._sharey.update_params()
233:                             ax.set_position(ax._sharey.figbox)
234:                 else:
235:                     ss = ax.get_subplotspec().get_topmost_subplotspec()
236:                     if ss.get_gridspec() == self:
237:                         ax.update_params()
238:                         ax.set_position(ax.figbox)
239: 
240:     def get_subplot_params(self, fig=None):
241:         '''
242:         return a dictionary of subplot layout parameters. The default
243:         parameters are from rcParams unless a figure attribute is set.
244:         '''
245:         from matplotlib.figure import SubplotParams
246:         if fig is None:
247:             kw = {k: rcParams["figure.subplot."+k] for k in self._AllowedKeys}
248:             subplotpars = SubplotParams(**kw)
249:         else:
250:             subplotpars = copy.copy(fig.subplotpars)
251: 
252:         update_kw = {k: getattr(self, k) for k in self._AllowedKeys}
253:         subplotpars.update(**update_kw)
254: 
255:         return subplotpars
256: 
257:     def locally_modified_subplot_params(self):
258:         return [k for k in self._AllowedKeys if getattr(self, k)]
259: 
260:     def tight_layout(self, fig, renderer=None,
261:                      pad=1.08, h_pad=None, w_pad=None, rect=None):
262:         '''
263:         Adjust subplot parameters to give specified padding.
264: 
265:         Parameters
266:         ----------
267: 
268:         pad : float
269:             Padding between the figure edge and the edges of subplots, as a
270:             fraction of the font-size.
271:         h_pad, w_pad : float, optional
272:             Padding (height/width) between edges of adjacent subplots.
273:             Defaults to ``pad_inches``.
274:         rect : tuple of 4 floats, optional
275:             (left, bottom, right, top) rectangle in normalized figure
276:             coordinates that the whole subplots area (including labels) will
277:             fit into.  Default is (0, 0, 1, 1).
278:         '''
279: 
280:         from .tight_layout import (
281:             get_renderer, get_subplotspec_list, get_tight_layout_figure)
282: 
283:         subplotspec_list = get_subplotspec_list(fig.axes, grid_spec=self)
284:         if None in subplotspec_list:
285:             warnings.warn("This figure includes Axes that are not compatible "
286:                           "with tight_layout, so results might be incorrect.")
287: 
288:         if renderer is None:
289:             renderer = get_renderer(fig)
290: 
291:         kwargs = get_tight_layout_figure(
292:             fig, fig.axes, subplotspec_list, renderer,
293:             pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
294:         self.update(**kwargs)
295: 
296: 
297: class GridSpecFromSubplotSpec(GridSpecBase):
298:     '''
299:     GridSpec whose subplot layout parameters are inherited from the
300:     location specified by a given SubplotSpec.
301:     '''
302:     def __init__(self, nrows, ncols,
303:                  subplot_spec,
304:                  wspace=None, hspace=None,
305:                  height_ratios=None, width_ratios=None):
306:         '''
307:         The number of rows and number of columns of the grid need to
308:         be set. An instance of SubplotSpec is also needed to be set
309:         from which the layout parameters will be inherited. The wspace
310:         and hspace of the layout can be optionally specified or the
311:         default values (from the figure or rcParams) will be used.
312:         '''
313:         self._wspace = wspace
314:         self._hspace = hspace
315:         self._subplot_spec = subplot_spec
316: 
317:         GridSpecBase.__init__(self, nrows, ncols,
318:                               width_ratios=width_ratios,
319:                               height_ratios=height_ratios)
320: 
321: 
322:     def get_subplot_params(self, fig=None):
323:         '''Return a dictionary of subplot layout parameters.
324:         '''
325: 
326:         if fig is None:
327:             hspace = rcParams["figure.subplot.hspace"]
328:             wspace = rcParams["figure.subplot.wspace"]
329:         else:
330:             hspace = fig.subplotpars.hspace
331:             wspace = fig.subplotpars.wspace
332: 
333:         if self._hspace is not None:
334:             hspace = self._hspace
335: 
336:         if self._wspace is not None:
337:             wspace = self._wspace
338: 
339:         figbox = self._subplot_spec.get_position(fig, return_all=False)
340:         left, bottom, right, top = figbox.extents
341: 
342:         from matplotlib.figure import SubplotParams
343:         sp = SubplotParams(left=left,
344:                            right=right,
345:                            bottom=bottom,
346:                            top=top,
347:                            wspace=wspace,
348:                            hspace=hspace)
349: 
350:         return sp
351: 
352: 
353:     def get_topmost_subplotspec(self):
354:         '''Get the topmost SubplotSpec instance associated with the subplot.'''
355:         return self._subplot_spec.get_topmost_subplotspec()
356: 
357: 
358: class SubplotSpec(object):
359:     '''Specifies the location of the subplot in the given `GridSpec`.
360:     '''
361: 
362:     def __init__(self, gridspec, num1, num2=None):
363:         '''
364:         The subplot will occupy the num1-th cell of the given
365:         gridspec.  If num2 is provided, the subplot will span between
366:         num1-th cell and num2-th cell.
367: 
368:         The index starts from 0.
369:         '''
370: 
371:         rows, cols = gridspec.get_geometry()
372:         total = rows * cols
373: 
374:         self._gridspec = gridspec
375:         self.num1 = num1
376:         self.num2 = num2
377: 
378:     def get_gridspec(self):
379:         return self._gridspec
380: 
381: 
382:     def get_geometry(self):
383:         '''Get the subplot geometry (``n_rows, n_cols, row, col``).
384: 
385:         Unlike SuplorParams, indexes are 0-based.
386:         '''
387:         rows, cols = self.get_gridspec().get_geometry()
388:         return rows, cols, self.num1, self.num2
389: 
390: 
391:     def get_position(self, fig, return_all=False):
392:         '''Update the subplot position from ``fig.subplotpars``.
393:         '''
394: 
395:         gridspec = self.get_gridspec()
396:         nrows, ncols = gridspec.get_geometry()
397: 
398:         figBottoms, figTops, figLefts, figRights = \
399:             gridspec.get_grid_positions(fig)
400: 
401:         rowNum, colNum =  divmod(self.num1, ncols)
402:         figBottom = figBottoms[rowNum]
403:         figTop = figTops[rowNum]
404:         figLeft = figLefts[colNum]
405:         figRight = figRights[colNum]
406: 
407:         if self.num2 is not None:
408: 
409:             rowNum2, colNum2 =  divmod(self.num2, ncols)
410:             figBottom2 = figBottoms[rowNum2]
411:             figTop2 = figTops[rowNum2]
412:             figLeft2 = figLefts[colNum2]
413:             figRight2 = figRights[colNum2]
414: 
415:             figBottom = min(figBottom, figBottom2)
416:             figLeft = min(figLeft, figLeft2)
417:             figTop = max(figTop, figTop2)
418:             figRight = max(figRight, figRight2)
419: 
420:         figbox = mtransforms.Bbox.from_extents(figLeft, figBottom,
421:                                                figRight, figTop)
422: 
423:         if return_all:
424:             return figbox, rowNum, colNum, nrows, ncols
425:         else:
426:             return figbox
427: 
428:     def get_topmost_subplotspec(self):
429:         'get the topmost SubplotSpec instance associated with the subplot'
430:         gridspec = self.get_gridspec()
431:         if hasattr(gridspec, "get_topmost_subplotspec"):
432:             return gridspec.get_topmost_subplotspec()
433:         else:
434:             return self
435: 
436:     def __eq__(self, other):
437:         # other may not even have the attributes we are checking.
438:         return ((self._gridspec, self.num1, self.num2)
439:                 == (getattr(other, "_gridspec", object()),
440:                     getattr(other, "num1", object()),
441:                     getattr(other, "num2", object())))
442: 
443:     if six.PY2:
444:         def __ne__(self, other):
445:             return not self == other
446: 
447:     def __hash__(self):
448:         return hash((self._gridspec, self.num1, self.num2))
449: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_60271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'unicode', u'\n:mod:`~matplotlib.gridspec` is a module which specifies the location\nof the subplot in the figure.\n\n    ``GridSpec``\n        specifies the geometry of the grid that a subplot will be\n        placed. The number of rows and number of columns of the grid\n        need to be set. Optionally, the subplot layout parameters\n        (e.g., left, right, etc.) can be tuned.\n\n    ``SubplotSpec``\n        specifies the location of the subplot in the given *GridSpec*.\n\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import six' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_60272 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'six')

if (type(import_60272) is not StypyTypeError):

    if (import_60272 != 'pyd_module'):
        __import__(import_60272)
        sys_modules_60273 = sys.modules[import_60272]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'six', sys_modules_60273.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'six', import_60272)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from six.moves import zip' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_60274 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'six.moves')

if (type(import_60274) is not StypyTypeError):

    if (import_60274 != 'pyd_module'):
        __import__(import_60274)
        sys_modules_60275 = sys.modules[import_60274]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'six.moves', sys_modules_60275.module_type_store, module_type_store, ['zip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_60275, sys_modules_60275.module_type_store, module_type_store)
    else:
        from six.moves import zip

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'six.moves', None, module_type_store, ['zip'], [zip])

else:
    # Assigning a type to the variable 'six.moves' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'six.moves', import_60274)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import copy' statement (line 23)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import warnings' statement (line 24)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import matplotlib' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_60276 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib')

if (type(import_60276) is not StypyTypeError):

    if (import_60276 != 'pyd_module'):
        __import__(import_60276)
        sys_modules_60277 = sys.modules[import_60276]
        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', sys_modules_60277.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', import_60276)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from matplotlib import rcParams' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_60278 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib')

if (type(import_60278) is not StypyTypeError):

    if (import_60278 != 'pyd_module'):
        __import__(import_60278)
        sys_modules_60279 = sys.modules[import_60278]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib', sys_modules_60279.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_60279, sys_modules_60279.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib', import_60278)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import matplotlib.transforms' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_60280 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.transforms')

if (type(import_60280) is not StypyTypeError):

    if (import_60280 != 'pyd_module'):
        __import__(import_60280)
        sys_modules_60281 = sys.modules[import_60280]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'mtransforms', sys_modules_60281.module_type_store, module_type_store)
    else:
        import matplotlib.transforms as mtransforms

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'mtransforms', matplotlib.transforms, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.transforms', import_60280)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import numpy' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_60282 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy')

if (type(import_60282) is not StypyTypeError):

    if (import_60282 != 'pyd_module'):
        __import__(import_60282)
        sys_modules_60283 = sys.modules[import_60282]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'np', sys_modules_60283.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy', import_60282)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'GridSpecBase' class

class GridSpecBase(object, ):
    unicode_60284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'unicode', u'\n    A base class of GridSpec that specifies the geometry of the grid\n    that a subplot will be placed.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 38)
        None_60285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 51), 'None')
        # Getting the type of 'None' (line 38)
        None_60286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 70), 'None')
        defaults = [None_60285, None_60286]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.__init__', ['nrows', 'ncols', 'height_ratios', 'width_ratios'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['nrows', 'ncols', 'height_ratios', 'width_ratios'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_60287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'unicode', u'\n        The number of rows and number of columns of the grid need to\n        be set. Optionally, the ratio of heights and widths of rows and\n        columns can be specified.\n        ')
        
        # Assigning a Tuple to a Tuple (line 44):
        
        # Assigning a Name to a Name (line 44):
        # Getting the type of 'nrows' (line 44)
        nrows_60288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'nrows')
        # Assigning a type to the variable 'tuple_assignment_60217' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_60217', nrows_60288)
        
        # Assigning a Name to a Name (line 44):
        # Getting the type of 'ncols' (line 44)
        ncols_60289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), 'ncols')
        # Assigning a type to the variable 'tuple_assignment_60218' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_60218', ncols_60289)
        
        # Assigning a Name to a Attribute (line 44):
        # Getting the type of 'tuple_assignment_60217' (line 44)
        tuple_assignment_60217_60290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_60217')
        # Getting the type of 'self' (line 44)
        self_60291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self')
        # Setting the type of the member '_nrows' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_60291, '_nrows', tuple_assignment_60217_60290)
        
        # Assigning a Name to a Attribute (line 44):
        # Getting the type of 'tuple_assignment_60218' (line 44)
        tuple_assignment_60218_60292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_60218')
        # Getting the type of 'self' (line 44)
        self_60293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'self')
        # Setting the type of the member '_ncols' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 21), self_60293, '_ncols', tuple_assignment_60218_60292)
        
        # Call to set_height_ratios(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'height_ratios' (line 45)
        height_ratios_60296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'height_ratios', False)
        # Processing the call keyword arguments (line 45)
        kwargs_60297 = {}
        # Getting the type of 'self' (line 45)
        self_60294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'set_height_ratios' of a type (line 45)
        set_height_ratios_60295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_60294, 'set_height_ratios')
        # Calling set_height_ratios(args, kwargs) (line 45)
        set_height_ratios_call_result_60298 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), set_height_ratios_60295, *[height_ratios_60296], **kwargs_60297)
        
        
        # Call to set_width_ratios(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'width_ratios' (line 46)
        width_ratios_60301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'width_ratios', False)
        # Processing the call keyword arguments (line 46)
        kwargs_60302 = {}
        # Getting the type of 'self' (line 46)
        self_60299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'set_width_ratios' of a type (line 46)
        set_width_ratios_60300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_60299, 'set_width_ratios')
        # Calling set_width_ratios(args, kwargs) (line 46)
        set_width_ratios_call_result_60303 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), set_width_ratios_60300, *[width_ratios_60301], **kwargs_60302)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_geometry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_geometry'
        module_type_store = module_type_store.open_function_context('get_geometry', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.get_geometry')
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_param_names_list', [])
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.get_geometry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.get_geometry', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_geometry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_geometry(...)' code ##################

        unicode_60304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'unicode', u'get the geometry of the grid, e.g., 2,3')
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_60305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_60306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'self')
        # Obtaining the member '_nrows' of a type (line 50)
        _nrows_60307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), self_60306, '_nrows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 15), tuple_60305, _nrows_60307)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_60308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'self')
        # Obtaining the member '_ncols' of a type (line 50)
        _ncols_60309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 28), self_60308, '_ncols')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 15), tuple_60305, _ncols_60309)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', tuple_60305)
        
        # ################# End of 'get_geometry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_geometry' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_60310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_geometry'
        return stypy_return_type_60310


    @norecursion
    def get_subplot_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 52)
        None_60311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 37), 'None')
        defaults = [None_60311]
        # Create a new context for function 'get_subplot_params'
        module_type_store = module_type_store.open_function_context('get_subplot_params', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.get_subplot_params')
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.get_subplot_params.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.get_subplot_params', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_subplot_params', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_subplot_params(...)' code ##################

        pass
        
        # ################# End of 'get_subplot_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_subplot_params' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_60312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60312)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_subplot_params'
        return stypy_return_type_60312


    @norecursion
    def new_subplotspec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_60313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'int')
        int_60314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 54), 'int')
        defaults = [int_60313, int_60314]
        # Create a new context for function 'new_subplotspec'
        module_type_store = module_type_store.open_function_context('new_subplotspec', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.new_subplotspec')
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_param_names_list', ['loc', 'rowspan', 'colspan'])
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.new_subplotspec.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.new_subplotspec', ['loc', 'rowspan', 'colspan'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_subplotspec', localization, ['loc', 'rowspan', 'colspan'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_subplotspec(...)' code ##################

        unicode_60315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'unicode', u'\n        create and return a SuplotSpec instance.\n        ')
        
        # Assigning a Name to a Tuple (line 59):
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_60316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
        # Getting the type of 'loc' (line 59)
        loc_60317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'loc')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___60318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), loc_60317, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_60319 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), getitem___60318, int_60316)
        
        # Assigning a type to the variable 'tuple_var_assignment_60219' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_60219', subscript_call_result_60319)
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_60320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
        # Getting the type of 'loc' (line 59)
        loc_60321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'loc')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___60322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), loc_60321, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_60323 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), getitem___60322, int_60320)
        
        # Assigning a type to the variable 'tuple_var_assignment_60220' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_60220', subscript_call_result_60323)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_60219' (line 59)
        tuple_var_assignment_60219_60324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_60219')
        # Assigning a type to the variable 'loc1' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'loc1', tuple_var_assignment_60219_60324)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_60220' (line 59)
        tuple_var_assignment_60220_60325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_60220')
        # Assigning a type to the variable 'loc2' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'loc2', tuple_var_assignment_60220_60325)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        # Getting the type of 'loc1' (line 60)
        loc1_60326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'loc1')
        # Getting the type of 'loc1' (line 60)
        loc1_60327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'loc1')
        # Getting the type of 'rowspan' (line 60)
        rowspan_60328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'rowspan')
        # Applying the binary operator '+' (line 60)
        result_add_60329 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 32), '+', loc1_60327, rowspan_60328)
        
        slice_60330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 60, 22), loc1_60326, result_add_60329, None)
        # Getting the type of 'loc2' (line 60)
        loc2_60331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 46), 'loc2')
        # Getting the type of 'loc2' (line 60)
        loc2_60332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'loc2')
        # Getting the type of 'colspan' (line 60)
        colspan_60333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'colspan')
        # Applying the binary operator '+' (line 60)
        result_add_60334 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 51), '+', loc2_60332, colspan_60333)
        
        slice_60335 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 60, 22), loc2_60331, result_add_60334, None)
        # Getting the type of 'self' (line 60)
        self_60336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'self')
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___60337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), self_60336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_60338 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), getitem___60337, (slice_60330, slice_60335))
        
        # Assigning a type to the variable 'subplotspec' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'subplotspec', subscript_call_result_60338)
        # Getting the type of 'subplotspec' (line 61)
        subplotspec_60339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'subplotspec')
        # Assigning a type to the variable 'stypy_return_type' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', subplotspec_60339)
        
        # ################# End of 'new_subplotspec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_subplotspec' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_60340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60340)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_subplotspec'
        return stypy_return_type_60340


    @norecursion
    def set_width_ratios(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_width_ratios'
        module_type_store = module_type_store.open_function_context('set_width_ratios', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.set_width_ratios')
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_param_names_list', ['width_ratios'])
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.set_width_ratios.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.set_width_ratios', ['width_ratios'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_width_ratios', localization, ['width_ratios'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_width_ratios(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'width_ratios' (line 64)
        width_ratios_60341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'width_ratios')
        # Getting the type of 'None' (line 64)
        None_60342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'None')
        # Applying the binary operator 'isnot' (line 64)
        result_is_not_60343 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), 'isnot', width_ratios_60341, None_60342)
        
        
        
        # Call to len(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'width_ratios' (line 64)
        width_ratios_60345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'width_ratios', False)
        # Processing the call keyword arguments (line 64)
        kwargs_60346 = {}
        # Getting the type of 'len' (line 64)
        len_60344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'len', False)
        # Calling len(args, kwargs) (line 64)
        len_call_result_60347 = invoke(stypy.reporting.localization.Localization(__file__, 64, 40), len_60344, *[width_ratios_60345], **kwargs_60346)
        
        # Getting the type of 'self' (line 64)
        self_60348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 61), 'self')
        # Obtaining the member '_ncols' of a type (line 64)
        _ncols_60349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 61), self_60348, '_ncols')
        # Applying the binary operator '!=' (line 64)
        result_ne_60350 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 40), '!=', len_call_result_60347, _ncols_60349)
        
        # Applying the binary operator 'and' (line 64)
        result_and_keyword_60351 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), 'and', result_is_not_60343, result_ne_60350)
        
        # Testing the type of an if condition (line 64)
        if_condition_60352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_and_keyword_60351)
        # Assigning a type to the variable 'if_condition_60352' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_60352', if_condition_60352)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 65)
        # Processing the call arguments (line 65)
        unicode_60354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'unicode', u'Expected the given number of width ratios to match the number of columns of the grid')
        # Processing the call keyword arguments (line 65)
        kwargs_60355 = {}
        # Getting the type of 'ValueError' (line 65)
        ValueError_60353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 65)
        ValueError_call_result_60356 = invoke(stypy.reporting.localization.Localization(__file__, 65, 18), ValueError_60353, *[unicode_60354], **kwargs_60355)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 65, 12), ValueError_call_result_60356, 'raise parameter', BaseException)
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 67):
        
        # Assigning a Name to a Attribute (line 67):
        # Getting the type of 'width_ratios' (line 67)
        width_ratios_60357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'width_ratios')
        # Getting the type of 'self' (line 67)
        self_60358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member '_col_width_ratios' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_60358, '_col_width_ratios', width_ratios_60357)
        
        # ################# End of 'set_width_ratios(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_width_ratios' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_60359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_width_ratios'
        return stypy_return_type_60359


    @norecursion
    def get_width_ratios(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_width_ratios'
        module_type_store = module_type_store.open_function_context('get_width_ratios', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.get_width_ratios')
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_param_names_list', [])
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.get_width_ratios.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.get_width_ratios', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_width_ratios', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_width_ratios(...)' code ##################

        # Getting the type of 'self' (line 70)
        self_60360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'self')
        # Obtaining the member '_col_width_ratios' of a type (line 70)
        _col_width_ratios_60361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), self_60360, '_col_width_ratios')
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', _col_width_ratios_60361)
        
        # ################# End of 'get_width_ratios(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_width_ratios' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_60362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_width_ratios'
        return stypy_return_type_60362


    @norecursion
    def set_height_ratios(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_height_ratios'
        module_type_store = module_type_store.open_function_context('set_height_ratios', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.set_height_ratios')
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_param_names_list', ['height_ratios'])
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.set_height_ratios.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.set_height_ratios', ['height_ratios'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_height_ratios', localization, ['height_ratios'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_height_ratios(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'height_ratios' (line 73)
        height_ratios_60363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'height_ratios')
        # Getting the type of 'None' (line 73)
        None_60364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'None')
        # Applying the binary operator 'isnot' (line 73)
        result_is_not_60365 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), 'isnot', height_ratios_60363, None_60364)
        
        
        
        # Call to len(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'height_ratios' (line 73)
        height_ratios_60367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'height_ratios', False)
        # Processing the call keyword arguments (line 73)
        kwargs_60368 = {}
        # Getting the type of 'len' (line 73)
        len_60366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'len', False)
        # Calling len(args, kwargs) (line 73)
        len_call_result_60369 = invoke(stypy.reporting.localization.Localization(__file__, 73, 41), len_60366, *[height_ratios_60367], **kwargs_60368)
        
        # Getting the type of 'self' (line 73)
        self_60370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 63), 'self')
        # Obtaining the member '_nrows' of a type (line 73)
        _nrows_60371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 63), self_60370, '_nrows')
        # Applying the binary operator '!=' (line 73)
        result_ne_60372 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 41), '!=', len_call_result_60369, _nrows_60371)
        
        # Applying the binary operator 'and' (line 73)
        result_and_keyword_60373 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), 'and', result_is_not_60365, result_ne_60372)
        
        # Testing the type of an if condition (line 73)
        if_condition_60374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_and_keyword_60373)
        # Assigning a type to the variable 'if_condition_60374' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_60374', if_condition_60374)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 74)
        # Processing the call arguments (line 74)
        unicode_60376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'unicode', u'Expected the given number of height ratios to match the number of rows of the grid')
        # Processing the call keyword arguments (line 74)
        kwargs_60377 = {}
        # Getting the type of 'ValueError' (line 74)
        ValueError_60375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 74)
        ValueError_call_result_60378 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), ValueError_60375, *[unicode_60376], **kwargs_60377)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 74, 12), ValueError_call_result_60378, 'raise parameter', BaseException)
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 76):
        
        # Assigning a Name to a Attribute (line 76):
        # Getting the type of 'height_ratios' (line 76)
        height_ratios_60379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 'height_ratios')
        # Getting the type of 'self' (line 76)
        self_60380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member '_row_height_ratios' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_60380, '_row_height_ratios', height_ratios_60379)
        
        # ################# End of 'set_height_ratios(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_height_ratios' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_60381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_height_ratios'
        return stypy_return_type_60381


    @norecursion
    def get_height_ratios(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_height_ratios'
        module_type_store = module_type_store.open_function_context('get_height_ratios', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.get_height_ratios')
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_param_names_list', [])
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.get_height_ratios.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.get_height_ratios', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_height_ratios', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_height_ratios(...)' code ##################

        # Getting the type of 'self' (line 79)
        self_60382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'self')
        # Obtaining the member '_row_height_ratios' of a type (line 79)
        _row_height_ratios_60383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), self_60382, '_row_height_ratios')
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', _row_height_ratios_60383)
        
        # ################# End of 'get_height_ratios(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_height_ratios' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_60384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60384)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_height_ratios'
        return stypy_return_type_60384


    @norecursion
    def get_grid_positions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_grid_positions'
        module_type_store = module_type_store.open_function_context('get_grid_positions', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.get_grid_positions')
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.get_grid_positions.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.get_grid_positions', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_grid_positions', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_grid_positions(...)' code ##################

        unicode_60385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'unicode', u'\n        return lists of bottom and top position of rows, left and\n        right positions of columns.\n        ')
        
        # Assigning a Call to a Tuple (line 86):
        
        # Assigning a Call to a Name:
        
        # Call to get_geometry(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_60388 = {}
        # Getting the type of 'self' (line 86)
        self_60386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'self', False)
        # Obtaining the member 'get_geometry' of a type (line 86)
        get_geometry_60387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), self_60386, 'get_geometry')
        # Calling get_geometry(args, kwargs) (line 86)
        get_geometry_call_result_60389 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), get_geometry_60387, *[], **kwargs_60388)
        
        # Assigning a type to the variable 'call_assignment_60221' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60221', get_geometry_call_result_60389)
        
        # Assigning a Call to a Name (line 86):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_60392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60393 = {}
        # Getting the type of 'call_assignment_60221' (line 86)
        call_assignment_60221_60390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60221', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___60391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), call_assignment_60221_60390, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_60394 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60391, *[int_60392], **kwargs_60393)
        
        # Assigning a type to the variable 'call_assignment_60222' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60222', getitem___call_result_60394)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'call_assignment_60222' (line 86)
        call_assignment_60222_60395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60222')
        # Assigning a type to the variable 'nrows' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'nrows', call_assignment_60222_60395)
        
        # Assigning a Call to a Name (line 86):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_60398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60399 = {}
        # Getting the type of 'call_assignment_60221' (line 86)
        call_assignment_60221_60396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60221', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___60397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), call_assignment_60221_60396, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_60400 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60397, *[int_60398], **kwargs_60399)
        
        # Assigning a type to the variable 'call_assignment_60223' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60223', getitem___call_result_60400)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'call_assignment_60223' (line 86)
        call_assignment_60223_60401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'call_assignment_60223')
        # Assigning a type to the variable 'ncols' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'ncols', call_assignment_60223_60401)
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to get_subplot_params(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'fig' (line 88)
        fig_60404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'fig', False)
        # Processing the call keyword arguments (line 88)
        kwargs_60405 = {}
        # Getting the type of 'self' (line 88)
        self_60402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'self', False)
        # Obtaining the member 'get_subplot_params' of a type (line 88)
        get_subplot_params_60403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), self_60402, 'get_subplot_params')
        # Calling get_subplot_params(args, kwargs) (line 88)
        get_subplot_params_call_result_60406 = invoke(stypy.reporting.localization.Localization(__file__, 88, 25), get_subplot_params_60403, *[fig_60404], **kwargs_60405)
        
        # Assigning a type to the variable 'subplot_params' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'subplot_params', get_subplot_params_call_result_60406)
        
        # Assigning a Attribute to a Name (line 89):
        
        # Assigning a Attribute to a Name (line 89):
        # Getting the type of 'subplot_params' (line 89)
        subplot_params_60407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'subplot_params')
        # Obtaining the member 'left' of a type (line 89)
        left_60408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), subplot_params_60407, 'left')
        # Assigning a type to the variable 'left' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'left', left_60408)
        
        # Assigning a Attribute to a Name (line 90):
        
        # Assigning a Attribute to a Name (line 90):
        # Getting the type of 'subplot_params' (line 90)
        subplot_params_60409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'subplot_params')
        # Obtaining the member 'right' of a type (line 90)
        right_60410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 16), subplot_params_60409, 'right')
        # Assigning a type to the variable 'right' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'right', right_60410)
        
        # Assigning a Attribute to a Name (line 91):
        
        # Assigning a Attribute to a Name (line 91):
        # Getting the type of 'subplot_params' (line 91)
        subplot_params_60411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'subplot_params')
        # Obtaining the member 'bottom' of a type (line 91)
        bottom_60412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 17), subplot_params_60411, 'bottom')
        # Assigning a type to the variable 'bottom' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'bottom', bottom_60412)
        
        # Assigning a Attribute to a Name (line 92):
        
        # Assigning a Attribute to a Name (line 92):
        # Getting the type of 'subplot_params' (line 92)
        subplot_params_60413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'subplot_params')
        # Obtaining the member 'top' of a type (line 92)
        top_60414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 14), subplot_params_60413, 'top')
        # Assigning a type to the variable 'top' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'top', top_60414)
        
        # Assigning a Attribute to a Name (line 93):
        
        # Assigning a Attribute to a Name (line 93):
        # Getting the type of 'subplot_params' (line 93)
        subplot_params_60415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'subplot_params')
        # Obtaining the member 'wspace' of a type (line 93)
        wspace_60416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), subplot_params_60415, 'wspace')
        # Assigning a type to the variable 'wspace' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'wspace', wspace_60416)
        
        # Assigning a Attribute to a Name (line 94):
        
        # Assigning a Attribute to a Name (line 94):
        # Getting the type of 'subplot_params' (line 94)
        subplot_params_60417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'subplot_params')
        # Obtaining the member 'hspace' of a type (line 94)
        hspace_60418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), subplot_params_60417, 'hspace')
        # Assigning a type to the variable 'hspace' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'hspace', hspace_60418)
        
        # Assigning a BinOp to a Name (line 95):
        
        # Assigning a BinOp to a Name (line 95):
        # Getting the type of 'right' (line 95)
        right_60419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'right')
        # Getting the type of 'left' (line 95)
        left_60420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'left')
        # Applying the binary operator '-' (line 95)
        result_sub_60421 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 19), '-', right_60419, left_60420)
        
        # Assigning a type to the variable 'totWidth' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'totWidth', result_sub_60421)
        
        # Assigning a BinOp to a Name (line 96):
        
        # Assigning a BinOp to a Name (line 96):
        # Getting the type of 'top' (line 96)
        top_60422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'top')
        # Getting the type of 'bottom' (line 96)
        bottom_60423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'bottom')
        # Applying the binary operator '-' (line 96)
        result_sub_60424 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 20), '-', top_60422, bottom_60423)
        
        # Assigning a type to the variable 'totHeight' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'totHeight', result_sub_60424)
        
        # Assigning a BinOp to a Name (line 99):
        
        # Assigning a BinOp to a Name (line 99):
        # Getting the type of 'totHeight' (line 99)
        totHeight_60425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'totHeight')
        # Getting the type of 'nrows' (line 99)
        nrows_60426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'nrows')
        # Getting the type of 'hspace' (line 99)
        hspace_60427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'hspace')
        # Getting the type of 'nrows' (line 99)
        nrows_60428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'nrows')
        int_60429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 51), 'int')
        # Applying the binary operator '-' (line 99)
        result_sub_60430 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 45), '-', nrows_60428, int_60429)
        
        # Applying the binary operator '*' (line 99)
        result_mul_60431 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 37), '*', hspace_60427, result_sub_60430)
        
        # Applying the binary operator '+' (line 99)
        result_add_60432 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 29), '+', nrows_60426, result_mul_60431)
        
        # Applying the binary operator 'div' (line 99)
        result_div_60433 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 16), 'div', totHeight_60425, result_add_60432)
        
        # Assigning a type to the variable 'cellH' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'cellH', result_div_60433)
        
        # Assigning a BinOp to a Name (line 100):
        
        # Assigning a BinOp to a Name (line 100):
        # Getting the type of 'hspace' (line 100)
        hspace_60434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'hspace')
        # Getting the type of 'cellH' (line 100)
        cellH_60435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'cellH')
        # Applying the binary operator '*' (line 100)
        result_mul_60436 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '*', hspace_60434, cellH_60435)
        
        # Assigning a type to the variable 'sepH' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'sepH', result_mul_60436)
        
        
        # Getting the type of 'self' (line 101)
        self_60437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'self')
        # Obtaining the member '_row_height_ratios' of a type (line 101)
        _row_height_ratios_60438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 11), self_60437, '_row_height_ratios')
        # Getting the type of 'None' (line 101)
        None_60439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'None')
        # Applying the binary operator 'isnot' (line 101)
        result_is_not_60440 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'isnot', _row_height_ratios_60438, None_60439)
        
        # Testing the type of an if condition (line 101)
        if_condition_60441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_is_not_60440)
        # Assigning a type to the variable 'if_condition_60441' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_60441', if_condition_60441)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 102):
        
        # Assigning a BinOp to a Name (line 102):
        # Getting the type of 'cellH' (line 102)
        cellH_60442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'cellH')
        # Getting the type of 'nrows' (line 102)
        nrows_60443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'nrows')
        # Applying the binary operator '*' (line 102)
        result_mul_60444 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 24), '*', cellH_60442, nrows_60443)
        
        # Assigning a type to the variable 'netHeight' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'netHeight', result_mul_60444)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to float(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to sum(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_60447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'self', False)
        # Obtaining the member '_row_height_ratios' of a type (line 103)
        _row_height_ratios_60448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), self_60447, '_row_height_ratios')
        # Processing the call keyword arguments (line 103)
        kwargs_60449 = {}
        # Getting the type of 'sum' (line 103)
        sum_60446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'sum', False)
        # Calling sum(args, kwargs) (line 103)
        sum_call_result_60450 = invoke(stypy.reporting.localization.Localization(__file__, 103, 23), sum_60446, *[_row_height_ratios_60448], **kwargs_60449)
        
        # Processing the call keyword arguments (line 103)
        kwargs_60451 = {}
        # Getting the type of 'float' (line 103)
        float_60445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'float', False)
        # Calling float(args, kwargs) (line 103)
        float_call_result_60452 = invoke(stypy.reporting.localization.Localization(__file__, 103, 17), float_60445, *[sum_call_result_60450], **kwargs_60451)
        
        # Assigning a type to the variable 'tr' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'tr', float_call_result_60452)
        
        # Assigning a ListComp to a Name (line 104):
        
        # Assigning a ListComp to a Name (line 104):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 104)
        self_60458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 55), 'self')
        # Obtaining the member '_row_height_ratios' of a type (line 104)
        _row_height_ratios_60459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 55), self_60458, '_row_height_ratios')
        comprehension_60460 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 27), _row_height_ratios_60459)
        # Assigning a type to the variable 'r' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'r', comprehension_60460)
        # Getting the type of 'netHeight' (line 104)
        netHeight_60453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'netHeight')
        # Getting the type of 'r' (line 104)
        r_60454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'r')
        # Applying the binary operator '*' (line 104)
        result_mul_60455 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 27), '*', netHeight_60453, r_60454)
        
        # Getting the type of 'tr' (line 104)
        tr_60456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 43), 'tr')
        # Applying the binary operator 'div' (line 104)
        result_div_60457 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 41), 'div', result_mul_60455, tr_60456)
        
        list_60461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 27), list_60461, result_div_60457)
        # Assigning a type to the variable 'cellHeights' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'cellHeights', list_60461)
        # SSA branch for the else part of an if statement (line 101)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 106):
        
        # Assigning a BinOp to a Name (line 106):
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_60462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        # Getting the type of 'cellH' (line 106)
        cellH_60463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'cellH')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 26), list_60462, cellH_60463)
        
        # Getting the type of 'nrows' (line 106)
        nrows_60464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'nrows')
        # Applying the binary operator '*' (line 106)
        result_mul_60465 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 26), '*', list_60462, nrows_60464)
        
        # Assigning a type to the variable 'cellHeights' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'cellHeights', result_mul_60465)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 107):
        
        # Assigning a BinOp to a Name (line 107):
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_60466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        int_60467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_60466, int_60467)
        
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_60468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        # Getting the type of 'sepH' (line 107)
        sepH_60469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'sepH')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 28), list_60468, sepH_60469)
        
        # Getting the type of 'nrows' (line 107)
        nrows_60470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), 'nrows')
        int_60471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 44), 'int')
        # Applying the binary operator '-' (line 107)
        result_sub_60472 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 38), '-', nrows_60470, int_60471)
        
        # Applying the binary operator '*' (line 107)
        result_mul_60473 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 28), '*', list_60468, result_sub_60472)
        
        # Applying the binary operator '+' (line 107)
        result_add_60474 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 21), '+', list_60466, result_mul_60473)
        
        # Assigning a type to the variable 'sepHeights' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'sepHeights', result_add_60474)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to cumsum(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to column_stack(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_60479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'sepHeights' (line 108)
        sepHeights_60480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'sepHeights', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 43), list_60479, sepHeights_60480)
        # Adding element type (line 108)
        # Getting the type of 'cellHeights' (line 108)
        cellHeights_60481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 56), 'cellHeights', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 43), list_60479, cellHeights_60481)
        
        # Processing the call keyword arguments (line 108)
        kwargs_60482 = {}
        # Getting the type of 'np' (line 108)
        np_60477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 108)
        column_stack_60478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), np_60477, 'column_stack')
        # Calling column_stack(args, kwargs) (line 108)
        column_stack_call_result_60483 = invoke(stypy.reporting.localization.Localization(__file__, 108, 27), column_stack_60478, *[list_60479], **kwargs_60482)
        
        # Obtaining the member 'flat' of a type (line 108)
        flat_60484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), column_stack_call_result_60483, 'flat')
        # Processing the call keyword arguments (line 108)
        kwargs_60485 = {}
        # Getting the type of 'np' (line 108)
        np_60475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'np', False)
        # Obtaining the member 'cumsum' of a type (line 108)
        cumsum_60476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), np_60475, 'cumsum')
        # Calling cumsum(args, kwargs) (line 108)
        cumsum_call_result_60486 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), cumsum_60476, *[flat_60484], **kwargs_60485)
        
        # Assigning a type to the variable 'cellHs' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'cellHs', cumsum_call_result_60486)
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        # Getting the type of 'totWidth' (line 111)
        totWidth_60487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'totWidth')
        # Getting the type of 'ncols' (line 111)
        ncols_60488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'ncols')
        # Getting the type of 'wspace' (line 111)
        wspace_60489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'wspace')
        # Getting the type of 'ncols' (line 111)
        ncols_60490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'ncols')
        int_60491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 48), 'int')
        # Applying the binary operator '-' (line 111)
        result_sub_60492 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 42), '-', ncols_60490, int_60491)
        
        # Applying the binary operator '*' (line 111)
        result_mul_60493 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 34), '*', wspace_60489, result_sub_60492)
        
        # Applying the binary operator '+' (line 111)
        result_add_60494 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 26), '+', ncols_60488, result_mul_60493)
        
        # Applying the binary operator 'div' (line 111)
        result_div_60495 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), 'div', totWidth_60487, result_add_60494)
        
        # Assigning a type to the variable 'cellW' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'cellW', result_div_60495)
        
        # Assigning a BinOp to a Name (line 112):
        
        # Assigning a BinOp to a Name (line 112):
        # Getting the type of 'wspace' (line 112)
        wspace_60496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'wspace')
        # Getting the type of 'cellW' (line 112)
        cellW_60497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'cellW')
        # Applying the binary operator '*' (line 112)
        result_mul_60498 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '*', wspace_60496, cellW_60497)
        
        # Assigning a type to the variable 'sepW' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'sepW', result_mul_60498)
        
        
        # Getting the type of 'self' (line 113)
        self_60499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'self')
        # Obtaining the member '_col_width_ratios' of a type (line 113)
        _col_width_ratios_60500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), self_60499, '_col_width_ratios')
        # Getting the type of 'None' (line 113)
        None_60501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'None')
        # Applying the binary operator 'isnot' (line 113)
        result_is_not_60502 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), 'isnot', _col_width_ratios_60500, None_60501)
        
        # Testing the type of an if condition (line 113)
        if_condition_60503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_is_not_60502)
        # Assigning a type to the variable 'if_condition_60503' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_60503', if_condition_60503)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 114):
        
        # Assigning a BinOp to a Name (line 114):
        # Getting the type of 'cellW' (line 114)
        cellW_60504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'cellW')
        # Getting the type of 'ncols' (line 114)
        ncols_60505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'ncols')
        # Applying the binary operator '*' (line 114)
        result_mul_60506 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 23), '*', cellW_60504, ncols_60505)
        
        # Assigning a type to the variable 'netWidth' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'netWidth', result_mul_60506)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to float(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to sum(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_60509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'self', False)
        # Obtaining the member '_col_width_ratios' of a type (line 115)
        _col_width_ratios_60510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), self_60509, '_col_width_ratios')
        # Processing the call keyword arguments (line 115)
        kwargs_60511 = {}
        # Getting the type of 'sum' (line 115)
        sum_60508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'sum', False)
        # Calling sum(args, kwargs) (line 115)
        sum_call_result_60512 = invoke(stypy.reporting.localization.Localization(__file__, 115, 23), sum_60508, *[_col_width_ratios_60510], **kwargs_60511)
        
        # Processing the call keyword arguments (line 115)
        kwargs_60513 = {}
        # Getting the type of 'float' (line 115)
        float_60507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'float', False)
        # Calling float(args, kwargs) (line 115)
        float_call_result_60514 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), float_60507, *[sum_call_result_60512], **kwargs_60513)
        
        # Assigning a type to the variable 'tr' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tr', float_call_result_60514)
        
        # Assigning a ListComp to a Name (line 116):
        
        # Assigning a ListComp to a Name (line 116):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 116)
        self_60520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 49), 'self')
        # Obtaining the member '_col_width_ratios' of a type (line 116)
        _col_width_ratios_60521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 49), self_60520, '_col_width_ratios')
        comprehension_60522 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 26), _col_width_ratios_60521)
        # Assigning a type to the variable 'r' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'r', comprehension_60522)
        # Getting the type of 'netWidth' (line 116)
        netWidth_60515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'netWidth')
        # Getting the type of 'r' (line 116)
        r_60516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'r')
        # Applying the binary operator '*' (line 116)
        result_mul_60517 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 26), '*', netWidth_60515, r_60516)
        
        # Getting the type of 'tr' (line 116)
        tr_60518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'tr')
        # Applying the binary operator 'div' (line 116)
        result_div_60519 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 36), 'div', result_mul_60517, tr_60518)
        
        list_60523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 26), list_60523, result_div_60519)
        # Assigning a type to the variable 'cellWidths' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'cellWidths', list_60523)
        # SSA branch for the else part of an if statement (line 113)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_60524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        # Getting the type of 'cellW' (line 118)
        cellW_60525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'cellW')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 25), list_60524, cellW_60525)
        
        # Getting the type of 'ncols' (line 118)
        ncols_60526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'ncols')
        # Applying the binary operator '*' (line 118)
        result_mul_60527 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 25), '*', list_60524, ncols_60526)
        
        # Assigning a type to the variable 'cellWidths' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'cellWidths', result_mul_60527)
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 119):
        
        # Assigning a BinOp to a Name (line 119):
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_60528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_60529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_60528, int_60529)
        
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_60530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        # Getting the type of 'sepW' (line 119)
        sepW_60531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'sepW')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 27), list_60530, sepW_60531)
        
        # Getting the type of 'ncols' (line 119)
        ncols_60532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'ncols')
        int_60533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 43), 'int')
        # Applying the binary operator '-' (line 119)
        result_sub_60534 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 37), '-', ncols_60532, int_60533)
        
        # Applying the binary operator '*' (line 119)
        result_mul_60535 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 27), '*', list_60530, result_sub_60534)
        
        # Applying the binary operator '+' (line 119)
        result_add_60536 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), '+', list_60528, result_mul_60535)
        
        # Assigning a type to the variable 'sepWidths' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'sepWidths', result_add_60536)
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to cumsum(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to column_stack(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_60541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'sepWidths' (line 120)
        sepWidths_60542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'sepWidths', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 43), list_60541, sepWidths_60542)
        # Adding element type (line 120)
        # Getting the type of 'cellWidths' (line 120)
        cellWidths_60543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'cellWidths', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 43), list_60541, cellWidths_60543)
        
        # Processing the call keyword arguments (line 120)
        kwargs_60544 = {}
        # Getting the type of 'np' (line 120)
        np_60539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 120)
        column_stack_60540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), np_60539, 'column_stack')
        # Calling column_stack(args, kwargs) (line 120)
        column_stack_call_result_60545 = invoke(stypy.reporting.localization.Localization(__file__, 120, 27), column_stack_60540, *[list_60541], **kwargs_60544)
        
        # Obtaining the member 'flat' of a type (line 120)
        flat_60546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), column_stack_call_result_60545, 'flat')
        # Processing the call keyword arguments (line 120)
        kwargs_60547 = {}
        # Getting the type of 'np' (line 120)
        np_60537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'np', False)
        # Obtaining the member 'cumsum' of a type (line 120)
        cumsum_60538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), np_60537, 'cumsum')
        # Calling cumsum(args, kwargs) (line 120)
        cumsum_call_result_60548 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), cumsum_60538, *[flat_60546], **kwargs_60547)
        
        # Assigning a type to the variable 'cellWs' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'cellWs', cumsum_call_result_60548)
        
        # Assigning a ListComp to a Name (line 122):
        
        # Assigning a ListComp to a Name (line 122):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'nrows' (line 122)
        nrows_60558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 62), 'nrows', False)
        # Processing the call keyword arguments (line 122)
        kwargs_60559 = {}
        # Getting the type of 'range' (line 122)
        range_60557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'range', False)
        # Calling range(args, kwargs) (line 122)
        range_call_result_60560 = invoke(stypy.reporting.localization.Localization(__file__, 122, 56), range_60557, *[nrows_60558], **kwargs_60559)
        
        comprehension_60561 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 19), range_call_result_60560)
        # Assigning a type to the variable 'rowNum' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'rowNum', comprehension_60561)
        # Getting the type of 'top' (line 122)
        top_60549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'top')
        
        # Obtaining the type of the subscript
        int_60550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 32), 'int')
        # Getting the type of 'rowNum' (line 122)
        rowNum_60551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'rowNum')
        # Applying the binary operator '*' (line 122)
        result_mul_60552 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 32), '*', int_60550, rowNum_60551)
        
        # Getting the type of 'cellHs' (line 122)
        cellHs_60553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'cellHs')
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___60554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 25), cellHs_60553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_60555 = invoke(stypy.reporting.localization.Localization(__file__, 122, 25), getitem___60554, result_mul_60552)
        
        # Applying the binary operator '-' (line 122)
        result_sub_60556 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 19), '-', top_60549, subscript_call_result_60555)
        
        list_60562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 19), list_60562, result_sub_60556)
        # Assigning a type to the variable 'figTops' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'figTops', list_60562)
        
        # Assigning a ListComp to a Name (line 123):
        
        # Assigning a ListComp to a Name (line 123):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'nrows' (line 123)
        nrows_60574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 67), 'nrows', False)
        # Processing the call keyword arguments (line 123)
        kwargs_60575 = {}
        # Getting the type of 'range' (line 123)
        range_60573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 61), 'range', False)
        # Calling range(args, kwargs) (line 123)
        range_call_result_60576 = invoke(stypy.reporting.localization.Localization(__file__, 123, 61), range_60573, *[nrows_60574], **kwargs_60575)
        
        comprehension_60577 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 22), range_call_result_60576)
        # Assigning a type to the variable 'rowNum' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'rowNum', comprehension_60577)
        # Getting the type of 'top' (line 123)
        top_60563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'top')
        
        # Obtaining the type of the subscript
        int_60564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 35), 'int')
        # Getting the type of 'rowNum' (line 123)
        rowNum_60565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'rowNum')
        # Applying the binary operator '*' (line 123)
        result_mul_60566 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '*', int_60564, rowNum_60565)
        
        int_60567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
        # Applying the binary operator '+' (line 123)
        result_add_60568 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 35), '+', result_mul_60566, int_60567)
        
        # Getting the type of 'cellHs' (line 123)
        cellHs_60569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'cellHs')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___60570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), cellHs_60569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_60571 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), getitem___60570, result_add_60568)
        
        # Applying the binary operator '-' (line 123)
        result_sub_60572 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 22), '-', top_60563, subscript_call_result_60571)
        
        list_60578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 22), list_60578, result_sub_60572)
        # Assigning a type to the variable 'figBottoms' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'figBottoms', list_60578)
        
        # Assigning a ListComp to a Name (line 124):
        
        # Assigning a ListComp to a Name (line 124):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'ncols' (line 124)
        ncols_60588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 64), 'ncols', False)
        # Processing the call keyword arguments (line 124)
        kwargs_60589 = {}
        # Getting the type of 'range' (line 124)
        range_60587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 58), 'range', False)
        # Calling range(args, kwargs) (line 124)
        range_call_result_60590 = invoke(stypy.reporting.localization.Localization(__file__, 124, 58), range_60587, *[ncols_60588], **kwargs_60589)
        
        comprehension_60591 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), range_call_result_60590)
        # Assigning a type to the variable 'colNum' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'colNum', comprehension_60591)
        # Getting the type of 'left' (line 124)
        left_60579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'left')
        
        # Obtaining the type of the subscript
        int_60580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 34), 'int')
        # Getting the type of 'colNum' (line 124)
        colNum_60581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'colNum')
        # Applying the binary operator '*' (line 124)
        result_mul_60582 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 34), '*', int_60580, colNum_60581)
        
        # Getting the type of 'cellWs' (line 124)
        cellWs_60583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'cellWs')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___60584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 27), cellWs_60583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_60585 = invoke(stypy.reporting.localization.Localization(__file__, 124, 27), getitem___60584, result_mul_60582)
        
        # Applying the binary operator '+' (line 124)
        result_add_60586 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '+', left_60579, subscript_call_result_60585)
        
        list_60592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), list_60592, result_add_60586)
        # Assigning a type to the variable 'figLefts' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'figLefts', list_60592)
        
        # Assigning a ListComp to a Name (line 125):
        
        # Assigning a ListComp to a Name (line 125):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'ncols' (line 125)
        ncols_60604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 67), 'ncols', False)
        # Processing the call keyword arguments (line 125)
        kwargs_60605 = {}
        # Getting the type of 'range' (line 125)
        range_60603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 61), 'range', False)
        # Calling range(args, kwargs) (line 125)
        range_call_result_60606 = invoke(stypy.reporting.localization.Localization(__file__, 125, 61), range_60603, *[ncols_60604], **kwargs_60605)
        
        comprehension_60607 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 21), range_call_result_60606)
        # Assigning a type to the variable 'colNum' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'colNum', comprehension_60607)
        # Getting the type of 'left' (line 125)
        left_60593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'left')
        
        # Obtaining the type of the subscript
        int_60594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 35), 'int')
        # Getting the type of 'colNum' (line 125)
        colNum_60595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'colNum')
        # Applying the binary operator '*' (line 125)
        result_mul_60596 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 35), '*', int_60594, colNum_60595)
        
        int_60597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 44), 'int')
        # Applying the binary operator '+' (line 125)
        result_add_60598 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 35), '+', result_mul_60596, int_60597)
        
        # Getting the type of 'cellWs' (line 125)
        cellWs_60599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'cellWs')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___60600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 28), cellWs_60599, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_60601 = invoke(stypy.reporting.localization.Localization(__file__, 125, 28), getitem___60600, result_add_60598)
        
        # Applying the binary operator '+' (line 125)
        result_add_60602 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 21), '+', left_60593, subscript_call_result_60601)
        
        list_60608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 21), list_60608, result_add_60602)
        # Assigning a type to the variable 'figRights' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'figRights', list_60608)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_60609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'figBottoms' (line 127)
        figBottoms_60610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'figBottoms')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), tuple_60609, figBottoms_60610)
        # Adding element type (line 127)
        # Getting the type of 'figTops' (line 127)
        figTops_60611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'figTops')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), tuple_60609, figTops_60611)
        # Adding element type (line 127)
        # Getting the type of 'figLefts' (line 127)
        figLefts_60612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'figLefts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), tuple_60609, figLefts_60612)
        # Adding element type (line 127)
        # Getting the type of 'figRights' (line 127)
        figRights_60613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'figRights')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 15), tuple_60609, figRights_60613)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', tuple_60609)
        
        # ################# End of 'get_grid_positions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_grid_positions' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_60614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_grid_positions'
        return stypy_return_type_60614


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_function_name', 'GridSpecBase.__getitem__')
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecBase.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecBase.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        unicode_60615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'unicode', u'\n        create and return a SuplotSpec instance.\n        ')
        
        # Assigning a Call to a Tuple (line 133):
        
        # Assigning a Call to a Name:
        
        # Call to get_geometry(...): (line 133)
        # Processing the call keyword arguments (line 133)
        kwargs_60618 = {}
        # Getting the type of 'self' (line 133)
        self_60616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self', False)
        # Obtaining the member 'get_geometry' of a type (line 133)
        get_geometry_60617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), self_60616, 'get_geometry')
        # Calling get_geometry(args, kwargs) (line 133)
        get_geometry_call_result_60619 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), get_geometry_60617, *[], **kwargs_60618)
        
        # Assigning a type to the variable 'call_assignment_60224' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60224', get_geometry_call_result_60619)
        
        # Assigning a Call to a Name (line 133):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_60622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60623 = {}
        # Getting the type of 'call_assignment_60224' (line 133)
        call_assignment_60224_60620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60224', False)
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___60621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), call_assignment_60224_60620, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_60624 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60621, *[int_60622], **kwargs_60623)
        
        # Assigning a type to the variable 'call_assignment_60225' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60225', getitem___call_result_60624)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'call_assignment_60225' (line 133)
        call_assignment_60225_60625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60225')
        # Assigning a type to the variable 'nrows' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'nrows', call_assignment_60225_60625)
        
        # Assigning a Call to a Name (line 133):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_60628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        # Processing the call keyword arguments
        kwargs_60629 = {}
        # Getting the type of 'call_assignment_60224' (line 133)
        call_assignment_60224_60626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60224', False)
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___60627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), call_assignment_60224_60626, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_60630 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60627, *[int_60628], **kwargs_60629)
        
        # Assigning a type to the variable 'call_assignment_60226' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60226', getitem___call_result_60630)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'call_assignment_60226' (line 133)
        call_assignment_60226_60631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'call_assignment_60226')
        # Assigning a type to the variable 'ncols' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'ncols', call_assignment_60226_60631)
        
        # Assigning a BinOp to a Name (line 134):
        
        # Assigning a BinOp to a Name (line 134):
        # Getting the type of 'nrows' (line 134)
        nrows_60632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'nrows')
        # Getting the type of 'ncols' (line 134)
        ncols_60633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'ncols')
        # Applying the binary operator '*' (line 134)
        result_mul_60634 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '*', nrows_60632, ncols_60633)
        
        # Assigning a type to the variable 'total' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'total', result_mul_60634)
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        # Getting the type of 'tuple' (line 136)
        tuple_60635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'tuple')
        # Getting the type of 'key' (line 136)
        key_60636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'key')
        
        (may_be_60637, more_types_in_union_60638) = may_be_subtype(tuple_60635, key_60636)

        if may_be_60637:

            if more_types_in_union_60638:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'key' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'key', remove_not_subtype_from_union(key_60636, tuple))
            
            
            # SSA begins for try-except statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Name to a Tuple (line 138):
            
            # Assigning a Subscript to a Name (line 138):
            
            # Obtaining the type of the subscript
            int_60639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'int')
            # Getting the type of 'key' (line 138)
            key_60640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'key')
            # Obtaining the member '__getitem__' of a type (line 138)
            getitem___60641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), key_60640, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 138)
            subscript_call_result_60642 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), getitem___60641, int_60639)
            
            # Assigning a type to the variable 'tuple_var_assignment_60227' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'tuple_var_assignment_60227', subscript_call_result_60642)
            
            # Assigning a Subscript to a Name (line 138):
            
            # Obtaining the type of the subscript
            int_60643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'int')
            # Getting the type of 'key' (line 138)
            key_60644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'key')
            # Obtaining the member '__getitem__' of a type (line 138)
            getitem___60645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), key_60644, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 138)
            subscript_call_result_60646 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), getitem___60645, int_60643)
            
            # Assigning a type to the variable 'tuple_var_assignment_60228' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'tuple_var_assignment_60228', subscript_call_result_60646)
            
            # Assigning a Name to a Name (line 138):
            # Getting the type of 'tuple_var_assignment_60227' (line 138)
            tuple_var_assignment_60227_60647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'tuple_var_assignment_60227')
            # Assigning a type to the variable 'k1' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'k1', tuple_var_assignment_60227_60647)
            
            # Assigning a Name to a Name (line 138):
            # Getting the type of 'tuple_var_assignment_60228' (line 138)
            tuple_var_assignment_60228_60648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'tuple_var_assignment_60228')
            # Assigning a type to the variable 'k2' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'k2', tuple_var_assignment_60228_60648)
            # SSA branch for the except part of a try statement (line 137)
            # SSA branch for the except 'ValueError' branch of a try statement (line 137)
            module_type_store.open_ssa_branch('except')
            
            # Call to ValueError(...): (line 140)
            # Processing the call arguments (line 140)
            unicode_60650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'unicode', u'unrecognized subplot spec')
            # Processing the call keyword arguments (line 140)
            kwargs_60651 = {}
            # Getting the type of 'ValueError' (line 140)
            ValueError_60649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 140)
            ValueError_call_result_60652 = invoke(stypy.reporting.localization.Localization(__file__, 140, 22), ValueError_60649, *[unicode_60650], **kwargs_60651)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 140, 16), ValueError_call_result_60652, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 137)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Type idiom detected: calculating its left and rigth part (line 142)
            # Getting the type of 'slice' (line 142)
            slice_60653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'slice')
            # Getting the type of 'k1' (line 142)
            k1_60654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'k1')
            
            (may_be_60655, more_types_in_union_60656) = may_be_subtype(slice_60653, k1_60654)

            if may_be_60655:

                if more_types_in_union_60656:
                    # Runtime conditional SSA (line 142)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'k1' (line 142)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'k1', remove_not_subtype_from_union(k1_60654, slice))
                
                # Assigning a Call to a Tuple (line 143):
                
                # Assigning a Call to a Name:
                
                # Call to indices(...): (line 143)
                # Processing the call arguments (line 143)
                # Getting the type of 'nrows' (line 143)
                nrows_60659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 43), 'nrows', False)
                # Processing the call keyword arguments (line 143)
                kwargs_60660 = {}
                # Getting the type of 'k1' (line 143)
                k1_60657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 32), 'k1', False)
                # Obtaining the member 'indices' of a type (line 143)
                indices_60658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 32), k1_60657, 'indices')
                # Calling indices(args, kwargs) (line 143)
                indices_call_result_60661 = invoke(stypy.reporting.localization.Localization(__file__, 143, 32), indices_60658, *[nrows_60659], **kwargs_60660)
                
                # Assigning a type to the variable 'call_assignment_60229' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60229', indices_call_result_60661)
                
                # Assigning a Call to a Name (line 143):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60665 = {}
                # Getting the type of 'call_assignment_60229' (line 143)
                call_assignment_60229_60662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60229', False)
                # Obtaining the member '__getitem__' of a type (line 143)
                getitem___60663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), call_assignment_60229_60662, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60666 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60663, *[int_60664], **kwargs_60665)
                
                # Assigning a type to the variable 'call_assignment_60230' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60230', getitem___call_result_60666)
                
                # Assigning a Name to a Name (line 143):
                # Getting the type of 'call_assignment_60230' (line 143)
                call_assignment_60230_60667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60230')
                # Assigning a type to the variable 'row1' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'row1', call_assignment_60230_60667)
                
                # Assigning a Call to a Name (line 143):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60671 = {}
                # Getting the type of 'call_assignment_60229' (line 143)
                call_assignment_60229_60668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60229', False)
                # Obtaining the member '__getitem__' of a type (line 143)
                getitem___60669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), call_assignment_60229_60668, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60672 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60669, *[int_60670], **kwargs_60671)
                
                # Assigning a type to the variable 'call_assignment_60231' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60231', getitem___call_result_60672)
                
                # Assigning a Name to a Name (line 143):
                # Getting the type of 'call_assignment_60231' (line 143)
                call_assignment_60231_60673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60231')
                # Assigning a type to the variable 'row2' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'row2', call_assignment_60231_60673)
                
                # Assigning a Call to a Name (line 143):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60677 = {}
                # Getting the type of 'call_assignment_60229' (line 143)
                call_assignment_60229_60674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60229', False)
                # Obtaining the member '__getitem__' of a type (line 143)
                getitem___60675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), call_assignment_60229_60674, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60678 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60675, *[int_60676], **kwargs_60677)
                
                # Assigning a type to the variable 'call_assignment_60232' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60232', getitem___call_result_60678)
                
                # Assigning a Name to a Name (line 143):
                # Getting the type of 'call_assignment_60232' (line 143)
                call_assignment_60232_60679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'call_assignment_60232')
                # Assigning a type to the variable '_' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), '_', call_assignment_60232_60679)

                if more_types_in_union_60656:
                    # Runtime conditional SSA for else branch (line 142)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_60655) or more_types_in_union_60656):
                # Assigning a type to the variable 'k1' (line 142)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'k1', remove_subtype_from_union(k1_60654, slice))
                
                
                # Getting the type of 'k1' (line 145)
                k1_60680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'k1')
                int_60681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'int')
                # Applying the binary operator '<' (line 145)
                result_lt_60682 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 19), '<', k1_60680, int_60681)
                
                # Testing the type of an if condition (line 145)
                if_condition_60683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 16), result_lt_60682)
                # Assigning a type to the variable 'if_condition_60683' (line 145)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'if_condition_60683', if_condition_60683)
                # SSA begins for if statement (line 145)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'k1' (line 146)
                k1_60684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'k1')
                # Getting the type of 'nrows' (line 146)
                nrows_60685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'nrows')
                # Applying the binary operator '+=' (line 146)
                result_iadd_60686 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 20), '+=', k1_60684, nrows_60685)
                # Assigning a type to the variable 'k1' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'k1', result_iadd_60686)
                
                # SSA join for if statement (line 145)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'k1' (line 147)
                k1_60687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'k1')
                # Getting the type of 'nrows' (line 147)
                nrows_60688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'nrows')
                # Applying the binary operator '>=' (line 147)
                result_ge_60689 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), '>=', k1_60687, nrows_60688)
                
                
                # Getting the type of 'k1' (line 147)
                k1_60690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'k1')
                int_60691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 39), 'int')
                # Applying the binary operator '<' (line 147)
                result_lt_60692 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 34), '<', k1_60690, int_60691)
                
                # Applying the binary operator 'or' (line 147)
                result_or_keyword_60693 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), 'or', result_ge_60689, result_lt_60692)
                
                # Testing the type of an if condition (line 147)
                if_condition_60694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 16), result_or_keyword_60693)
                # Assigning a type to the variable 'if_condition_60694' (line 147)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'if_condition_60694', if_condition_60694)
                # SSA begins for if statement (line 147)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to IndexError(...): (line 148)
                # Processing the call arguments (line 148)
                unicode_60696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'unicode', u'index out of range')
                # Processing the call keyword arguments (line 148)
                kwargs_60697 = {}
                # Getting the type of 'IndexError' (line 148)
                IndexError_60695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'IndexError', False)
                # Calling IndexError(args, kwargs) (line 148)
                IndexError_call_result_60698 = invoke(stypy.reporting.localization.Localization(__file__, 148, 26), IndexError_60695, *[unicode_60696], **kwargs_60697)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 20), IndexError_call_result_60698, 'raise parameter', BaseException)
                # SSA join for if statement (line 147)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Tuple to a Tuple (line 149):
                
                # Assigning a Name to a Name (line 149):
                # Getting the type of 'k1' (line 149)
                k1_60699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'k1')
                # Assigning a type to the variable 'tuple_assignment_60233' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'tuple_assignment_60233', k1_60699)
                
                # Assigning a BinOp to a Name (line 149):
                # Getting the type of 'k1' (line 149)
                k1_60700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'k1')
                int_60701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 36), 'int')
                # Applying the binary operator '+' (line 149)
                result_add_60702 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 33), '+', k1_60700, int_60701)
                
                # Assigning a type to the variable 'tuple_assignment_60234' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'tuple_assignment_60234', result_add_60702)
                
                # Assigning a Name to a Name (line 149):
                # Getting the type of 'tuple_assignment_60233' (line 149)
                tuple_assignment_60233_60703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'tuple_assignment_60233')
                # Assigning a type to the variable 'row1' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'row1', tuple_assignment_60233_60703)
                
                # Assigning a Name to a Name (line 149):
                # Getting the type of 'tuple_assignment_60234' (line 149)
                tuple_assignment_60234_60704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'tuple_assignment_60234')
                # Assigning a type to the variable 'row2' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'row2', tuple_assignment_60234_60704)

                if (may_be_60655 and more_types_in_union_60656):
                    # SSA join for if statement (line 142)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 151)
            # Getting the type of 'slice' (line 151)
            slice_60705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'slice')
            # Getting the type of 'k2' (line 151)
            k2_60706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'k2')
            
            (may_be_60707, more_types_in_union_60708) = may_be_subtype(slice_60705, k2_60706)

            if may_be_60707:

                if more_types_in_union_60708:
                    # Runtime conditional SSA (line 151)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'k2' (line 151)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'k2', remove_not_subtype_from_union(k2_60706, slice))
                
                # Assigning a Call to a Tuple (line 152):
                
                # Assigning a Call to a Name:
                
                # Call to indices(...): (line 152)
                # Processing the call arguments (line 152)
                # Getting the type of 'ncols' (line 152)
                ncols_60711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'ncols', False)
                # Processing the call keyword arguments (line 152)
                kwargs_60712 = {}
                # Getting the type of 'k2' (line 152)
                k2_60709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'k2', False)
                # Obtaining the member 'indices' of a type (line 152)
                indices_60710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 32), k2_60709, 'indices')
                # Calling indices(args, kwargs) (line 152)
                indices_call_result_60713 = invoke(stypy.reporting.localization.Localization(__file__, 152, 32), indices_60710, *[ncols_60711], **kwargs_60712)
                
                # Assigning a type to the variable 'call_assignment_60235' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60235', indices_call_result_60713)
                
                # Assigning a Call to a Name (line 152):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60717 = {}
                # Getting the type of 'call_assignment_60235' (line 152)
                call_assignment_60235_60714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60235', False)
                # Obtaining the member '__getitem__' of a type (line 152)
                getitem___60715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), call_assignment_60235_60714, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60718 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60715, *[int_60716], **kwargs_60717)
                
                # Assigning a type to the variable 'call_assignment_60236' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60236', getitem___call_result_60718)
                
                # Assigning a Name to a Name (line 152):
                # Getting the type of 'call_assignment_60236' (line 152)
                call_assignment_60236_60719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60236')
                # Assigning a type to the variable 'col1' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'col1', call_assignment_60236_60719)
                
                # Assigning a Call to a Name (line 152):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60723 = {}
                # Getting the type of 'call_assignment_60235' (line 152)
                call_assignment_60235_60720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60235', False)
                # Obtaining the member '__getitem__' of a type (line 152)
                getitem___60721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), call_assignment_60235_60720, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60724 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60721, *[int_60722], **kwargs_60723)
                
                # Assigning a type to the variable 'call_assignment_60237' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60237', getitem___call_result_60724)
                
                # Assigning a Name to a Name (line 152):
                # Getting the type of 'call_assignment_60237' (line 152)
                call_assignment_60237_60725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60237')
                # Assigning a type to the variable 'col2' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'col2', call_assignment_60237_60725)
                
                # Assigning a Call to a Name (line 152):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60729 = {}
                # Getting the type of 'call_assignment_60235' (line 152)
                call_assignment_60235_60726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60235', False)
                # Obtaining the member '__getitem__' of a type (line 152)
                getitem___60727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), call_assignment_60235_60726, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60730 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60727, *[int_60728], **kwargs_60729)
                
                # Assigning a type to the variable 'call_assignment_60238' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60238', getitem___call_result_60730)
                
                # Assigning a Name to a Name (line 152):
                # Getting the type of 'call_assignment_60238' (line 152)
                call_assignment_60238_60731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'call_assignment_60238')
                # Assigning a type to the variable '_' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), '_', call_assignment_60238_60731)

                if more_types_in_union_60708:
                    # Runtime conditional SSA for else branch (line 151)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_60707) or more_types_in_union_60708):
                # Assigning a type to the variable 'k2' (line 151)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'k2', remove_subtype_from_union(k2_60706, slice))
                
                
                # Getting the type of 'k2' (line 154)
                k2_60732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'k2')
                int_60733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'int')
                # Applying the binary operator '<' (line 154)
                result_lt_60734 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 19), '<', k2_60732, int_60733)
                
                # Testing the type of an if condition (line 154)
                if_condition_60735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 16), result_lt_60734)
                # Assigning a type to the variable 'if_condition_60735' (line 154)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'if_condition_60735', if_condition_60735)
                # SSA begins for if statement (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'k2' (line 155)
                k2_60736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'k2')
                # Getting the type of 'ncols' (line 155)
                ncols_60737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'ncols')
                # Applying the binary operator '+=' (line 155)
                result_iadd_60738 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 20), '+=', k2_60736, ncols_60737)
                # Assigning a type to the variable 'k2' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'k2', result_iadd_60738)
                
                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'k2' (line 156)
                k2_60739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'k2')
                # Getting the type of 'ncols' (line 156)
                ncols_60740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'ncols')
                # Applying the binary operator '>=' (line 156)
                result_ge_60741 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 19), '>=', k2_60739, ncols_60740)
                
                
                # Getting the type of 'k2' (line 156)
                k2_60742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'k2')
                int_60743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 39), 'int')
                # Applying the binary operator '<' (line 156)
                result_lt_60744 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 34), '<', k2_60742, int_60743)
                
                # Applying the binary operator 'or' (line 156)
                result_or_keyword_60745 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 19), 'or', result_ge_60741, result_lt_60744)
                
                # Testing the type of an if condition (line 156)
                if_condition_60746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 16), result_or_keyword_60745)
                # Assigning a type to the variable 'if_condition_60746' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'if_condition_60746', if_condition_60746)
                # SSA begins for if statement (line 156)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to IndexError(...): (line 157)
                # Processing the call arguments (line 157)
                unicode_60748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'unicode', u'index out of range')
                # Processing the call keyword arguments (line 157)
                kwargs_60749 = {}
                # Getting the type of 'IndexError' (line 157)
                IndexError_60747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'IndexError', False)
                # Calling IndexError(args, kwargs) (line 157)
                IndexError_call_result_60750 = invoke(stypy.reporting.localization.Localization(__file__, 157, 26), IndexError_60747, *[unicode_60748], **kwargs_60749)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 20), IndexError_call_result_60750, 'raise parameter', BaseException)
                # SSA join for if statement (line 156)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Tuple to a Tuple (line 158):
                
                # Assigning a Name to a Name (line 158):
                # Getting the type of 'k2' (line 158)
                k2_60751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'k2')
                # Assigning a type to the variable 'tuple_assignment_60239' (line 158)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_assignment_60239', k2_60751)
                
                # Assigning a BinOp to a Name (line 158):
                # Getting the type of 'k2' (line 158)
                k2_60752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'k2')
                int_60753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'int')
                # Applying the binary operator '+' (line 158)
                result_add_60754 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 33), '+', k2_60752, int_60753)
                
                # Assigning a type to the variable 'tuple_assignment_60240' (line 158)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_assignment_60240', result_add_60754)
                
                # Assigning a Name to a Name (line 158):
                # Getting the type of 'tuple_assignment_60239' (line 158)
                tuple_assignment_60239_60755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_assignment_60239')
                # Assigning a type to the variable 'col1' (line 158)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'col1', tuple_assignment_60239_60755)
                
                # Assigning a Name to a Name (line 158):
                # Getting the type of 'tuple_assignment_60240' (line 158)
                tuple_assignment_60240_60756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_assignment_60240')
                # Assigning a type to the variable 'col2' (line 158)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'col2', tuple_assignment_60240_60756)

                if (may_be_60707 and more_types_in_union_60708):
                    # SSA join for if statement (line 151)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a BinOp to a Name (line 160):
            
            # Assigning a BinOp to a Name (line 160):
            # Getting the type of 'row1' (line 160)
            row1_60757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'row1')
            # Getting the type of 'ncols' (line 160)
            ncols_60758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'ncols')
            # Applying the binary operator '*' (line 160)
            result_mul_60759 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 19), '*', row1_60757, ncols_60758)
            
            # Getting the type of 'col1' (line 160)
            col1_60760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'col1')
            # Applying the binary operator '+' (line 160)
            result_add_60761 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 19), '+', result_mul_60759, col1_60760)
            
            # Assigning a type to the variable 'num1' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'num1', result_add_60761)
            
            # Assigning a BinOp to a Name (line 161):
            
            # Assigning a BinOp to a Name (line 161):
            # Getting the type of 'row2' (line 161)
            row2_60762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'row2')
            int_60763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'int')
            # Applying the binary operator '-' (line 161)
            result_sub_60764 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 20), '-', row2_60762, int_60763)
            
            # Getting the type of 'ncols' (line 161)
            ncols_60765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'ncols')
            # Applying the binary operator '*' (line 161)
            result_mul_60766 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 19), '*', result_sub_60764, ncols_60765)
            
            # Getting the type of 'col2' (line 161)
            col2_60767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'col2')
            int_60768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 42), 'int')
            # Applying the binary operator '-' (line 161)
            result_sub_60769 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 37), '-', col2_60767, int_60768)
            
            # Applying the binary operator '+' (line 161)
            result_add_60770 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 19), '+', result_mul_60766, result_sub_60769)
            
            # Assigning a type to the variable 'num2' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'num2', result_add_60770)

            if more_types_in_union_60638:
                # Runtime conditional SSA for else branch (line 136)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_60637) or more_types_in_union_60638):
            # Assigning a type to the variable 'key' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'key', remove_subtype_from_union(key_60636, tuple))
            
            # Type idiom detected: calculating its left and rigth part (line 165)
            # Getting the type of 'slice' (line 165)
            slice_60771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'slice')
            # Getting the type of 'key' (line 165)
            key_60772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'key')
            
            (may_be_60773, more_types_in_union_60774) = may_be_subtype(slice_60771, key_60772)

            if may_be_60773:

                if more_types_in_union_60774:
                    # Runtime conditional SSA (line 165)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'key' (line 165)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'key', remove_not_subtype_from_union(key_60772, slice))
                
                # Assigning a Call to a Tuple (line 166):
                
                # Assigning a Call to a Name:
                
                # Call to indices(...): (line 166)
                # Processing the call arguments (line 166)
                # Getting the type of 'total' (line 166)
                total_60777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 44), 'total', False)
                # Processing the call keyword arguments (line 166)
                kwargs_60778 = {}
                # Getting the type of 'key' (line 166)
                key_60775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'key', False)
                # Obtaining the member 'indices' of a type (line 166)
                indices_60776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), key_60775, 'indices')
                # Calling indices(args, kwargs) (line 166)
                indices_call_result_60779 = invoke(stypy.reporting.localization.Localization(__file__, 166, 32), indices_60776, *[total_60777], **kwargs_60778)
                
                # Assigning a type to the variable 'call_assignment_60241' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60241', indices_call_result_60779)
                
                # Assigning a Call to a Name (line 166):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60783 = {}
                # Getting the type of 'call_assignment_60241' (line 166)
                call_assignment_60241_60780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60241', False)
                # Obtaining the member '__getitem__' of a type (line 166)
                getitem___60781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), call_assignment_60241_60780, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60784 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60781, *[int_60782], **kwargs_60783)
                
                # Assigning a type to the variable 'call_assignment_60242' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60242', getitem___call_result_60784)
                
                # Assigning a Name to a Name (line 166):
                # Getting the type of 'call_assignment_60242' (line 166)
                call_assignment_60242_60785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60242')
                # Assigning a type to the variable 'num1' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'num1', call_assignment_60242_60785)
                
                # Assigning a Call to a Name (line 166):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60789 = {}
                # Getting the type of 'call_assignment_60241' (line 166)
                call_assignment_60241_60786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60241', False)
                # Obtaining the member '__getitem__' of a type (line 166)
                getitem___60787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), call_assignment_60241_60786, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60790 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60787, *[int_60788], **kwargs_60789)
                
                # Assigning a type to the variable 'call_assignment_60243' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60243', getitem___call_result_60790)
                
                # Assigning a Name to a Name (line 166):
                # Getting the type of 'call_assignment_60243' (line 166)
                call_assignment_60243_60791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60243')
                # Assigning a type to the variable 'num2' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'num2', call_assignment_60243_60791)
                
                # Assigning a Call to a Name (line 166):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_60794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 16), 'int')
                # Processing the call keyword arguments
                kwargs_60795 = {}
                # Getting the type of 'call_assignment_60241' (line 166)
                call_assignment_60241_60792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60241', False)
                # Obtaining the member '__getitem__' of a type (line 166)
                getitem___60793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), call_assignment_60241_60792, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_60796 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___60793, *[int_60794], **kwargs_60795)
                
                # Assigning a type to the variable 'call_assignment_60244' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60244', getitem___call_result_60796)
                
                # Assigning a Name to a Name (line 166):
                # Getting the type of 'call_assignment_60244' (line 166)
                call_assignment_60244_60797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'call_assignment_60244')
                # Assigning a type to the variable '_' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), '_', call_assignment_60244_60797)
                
                # Getting the type of 'num2' (line 167)
                num2_60798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'num2')
                int_60799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'int')
                # Applying the binary operator '-=' (line 167)
                result_isub_60800 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 16), '-=', num2_60798, int_60799)
                # Assigning a type to the variable 'num2' (line 167)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'num2', result_isub_60800)
                

                if more_types_in_union_60774:
                    # Runtime conditional SSA for else branch (line 165)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_60773) or more_types_in_union_60774):
                # Assigning a type to the variable 'key' (line 165)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'key', remove_subtype_from_union(key_60772, slice))
                
                
                # Getting the type of 'key' (line 169)
                key_60801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'key')
                int_60802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 25), 'int')
                # Applying the binary operator '<' (line 169)
                result_lt_60803 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 19), '<', key_60801, int_60802)
                
                # Testing the type of an if condition (line 169)
                if_condition_60804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 16), result_lt_60803)
                # Assigning a type to the variable 'if_condition_60804' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'if_condition_60804', if_condition_60804)
                # SSA begins for if statement (line 169)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'key' (line 170)
                key_60805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'key')
                # Getting the type of 'total' (line 170)
                total_60806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'total')
                # Applying the binary operator '+=' (line 170)
                result_iadd_60807 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 20), '+=', key_60805, total_60806)
                # Assigning a type to the variable 'key' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'key', result_iadd_60807)
                
                # SSA join for if statement (line 169)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                # Evaluating a boolean operation
                
                # Getting the type of 'key' (line 171)
                key_60808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'key')
                # Getting the type of 'total' (line 171)
                total_60809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'total')
                # Applying the binary operator '>=' (line 171)
                result_ge_60810 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 19), '>=', key_60808, total_60809)
                
                
                # Getting the type of 'key' (line 171)
                key_60811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'key')
                int_60812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 41), 'int')
                # Applying the binary operator '<' (line 171)
                result_lt_60813 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 35), '<', key_60811, int_60812)
                
                # Applying the binary operator 'or' (line 171)
                result_or_keyword_60814 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 19), 'or', result_ge_60810, result_lt_60813)
                
                # Testing the type of an if condition (line 171)
                if_condition_60815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 16), result_or_keyword_60814)
                # Assigning a type to the variable 'if_condition_60815' (line 171)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'if_condition_60815', if_condition_60815)
                # SSA begins for if statement (line 171)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to IndexError(...): (line 172)
                # Processing the call arguments (line 172)
                unicode_60817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 37), 'unicode', u'index out of range')
                # Processing the call keyword arguments (line 172)
                kwargs_60818 = {}
                # Getting the type of 'IndexError' (line 172)
                IndexError_60816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 26), 'IndexError', False)
                # Calling IndexError(args, kwargs) (line 172)
                IndexError_call_result_60819 = invoke(stypy.reporting.localization.Localization(__file__, 172, 26), IndexError_60816, *[unicode_60817], **kwargs_60818)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 172, 20), IndexError_call_result_60819, 'raise parameter', BaseException)
                # SSA join for if statement (line 171)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Tuple to a Tuple (line 173):
                
                # Assigning a Name to a Name (line 173):
                # Getting the type of 'key' (line 173)
                key_60820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'key')
                # Assigning a type to the variable 'tuple_assignment_60245' (line 173)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'tuple_assignment_60245', key_60820)
                
                # Assigning a Name to a Name (line 173):
                # Getting the type of 'None' (line 173)
                None_60821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), 'None')
                # Assigning a type to the variable 'tuple_assignment_60246' (line 173)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'tuple_assignment_60246', None_60821)
                
                # Assigning a Name to a Name (line 173):
                # Getting the type of 'tuple_assignment_60245' (line 173)
                tuple_assignment_60245_60822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'tuple_assignment_60245')
                # Assigning a type to the variable 'num1' (line 173)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'num1', tuple_assignment_60245_60822)
                
                # Assigning a Name to a Name (line 173):
                # Getting the type of 'tuple_assignment_60246' (line 173)
                tuple_assignment_60246_60823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'tuple_assignment_60246')
                # Assigning a type to the variable 'num2' (line 173)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'num2', tuple_assignment_60246_60823)

                if (may_be_60773 and more_types_in_union_60774):
                    # SSA join for if statement (line 165)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_60637 and more_types_in_union_60638):
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to SubplotSpec(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_60825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'self', False)
        # Getting the type of 'num1' (line 175)
        num1_60826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 33), 'num1', False)
        # Getting the type of 'num2' (line 175)
        num2_60827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 39), 'num2', False)
        # Processing the call keyword arguments (line 175)
        kwargs_60828 = {}
        # Getting the type of 'SubplotSpec' (line 175)
        SubplotSpec_60824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'SubplotSpec', False)
        # Calling SubplotSpec(args, kwargs) (line 175)
        SubplotSpec_call_result_60829 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), SubplotSpec_60824, *[self_60825, num1_60826, num2_60827], **kwargs_60828)
        
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', SubplotSpec_call_result_60829)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_60830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60830)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_60830


# Assigning a type to the variable 'GridSpecBase' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'GridSpecBase', GridSpecBase)
# Declaration of the 'GridSpec' class
# Getting the type of 'GridSpecBase' (line 178)
GridSpecBase_60831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'GridSpecBase')

class GridSpec(GridSpecBase_60831, ):
    unicode_60832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'unicode', u'\n    A class that specifies the geometry of the grid that a subplot\n    will be placed. The location of grid is determined by similar way\n    as the SubplotParams.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 186)
        None_60833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'None')
        # Getting the type of 'None' (line 186)
        None_60834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 35), 'None')
        # Getting the type of 'None' (line 186)
        None_60835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 47), 'None')
        # Getting the type of 'None' (line 186)
        None_60836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 57), 'None')
        # Getting the type of 'None' (line 187)
        None_60837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'None')
        # Getting the type of 'None' (line 187)
        None_60838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 'None')
        # Getting the type of 'None' (line 188)
        None_60839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 30), 'None')
        # Getting the type of 'None' (line 188)
        None_60840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 50), 'None')
        defaults = [None_60833, None_60834, None_60835, None_60836, None_60837, None_60838, None_60839, None_60840]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpec.__init__', ['nrows', 'ncols', 'left', 'bottom', 'right', 'top', 'wspace', 'hspace', 'width_ratios', 'height_ratios'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['nrows', 'ncols', 'left', 'bottom', 'right', 'top', 'wspace', 'hspace', 'width_ratios', 'height_ratios'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_60841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'unicode', u'\n        The number of rows and number of columns of the\n        grid need to be set. Optionally, the subplot layout parameters\n        (e.g., left, right, etc.) can be tuned.\n        ')
        
        # Assigning a Name to a Attribute (line 194):
        
        # Assigning a Name to a Attribute (line 194):
        # Getting the type of 'left' (line 194)
        left_60842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'left')
        # Getting the type of 'self' (line 194)
        self_60843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self')
        # Setting the type of the member 'left' of a type (line 194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_60843, 'left', left_60842)
        
        # Assigning a Name to a Attribute (line 195):
        
        # Assigning a Name to a Attribute (line 195):
        # Getting the type of 'bottom' (line 195)
        bottom_60844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'bottom')
        # Getting the type of 'self' (line 195)
        self_60845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self')
        # Setting the type of the member 'bottom' of a type (line 195)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_60845, 'bottom', bottom_60844)
        
        # Assigning a Name to a Attribute (line 196):
        
        # Assigning a Name to a Attribute (line 196):
        # Getting the type of 'right' (line 196)
        right_60846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'right')
        # Getting the type of 'self' (line 196)
        self_60847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member 'right' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_60847, 'right', right_60846)
        
        # Assigning a Name to a Attribute (line 197):
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'top' (line 197)
        top_60848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'top')
        # Getting the type of 'self' (line 197)
        self_60849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self')
        # Setting the type of the member 'top' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_60849, 'top', top_60848)
        
        # Assigning a Name to a Attribute (line 198):
        
        # Assigning a Name to a Attribute (line 198):
        # Getting the type of 'wspace' (line 198)
        wspace_60850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'wspace')
        # Getting the type of 'self' (line 198)
        self_60851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member 'wspace' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_60851, 'wspace', wspace_60850)
        
        # Assigning a Name to a Attribute (line 199):
        
        # Assigning a Name to a Attribute (line 199):
        # Getting the type of 'hspace' (line 199)
        hspace_60852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'hspace')
        # Getting the type of 'self' (line 199)
        self_60853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member 'hspace' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_60853, 'hspace', hspace_60852)
        
        # Call to __init__(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_60856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'self', False)
        # Getting the type of 'nrows' (line 201)
        nrows_60857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'nrows', False)
        # Getting the type of 'ncols' (line 201)
        ncols_60858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 43), 'ncols', False)
        # Processing the call keyword arguments (line 201)
        # Getting the type of 'width_ratios' (line 202)
        width_ratios_60859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'width_ratios', False)
        keyword_60860 = width_ratios_60859
        # Getting the type of 'height_ratios' (line 203)
        height_ratios_60861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'height_ratios', False)
        keyword_60862 = height_ratios_60861
        kwargs_60863 = {'height_ratios': keyword_60862, 'width_ratios': keyword_60860}
        # Getting the type of 'GridSpecBase' (line 201)
        GridSpecBase_60854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'GridSpecBase', False)
        # Obtaining the member '__init__' of a type (line 201)
        init___60855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), GridSpecBase_60854, '__init__')
        # Calling __init__(args, kwargs) (line 201)
        init___call_result_60864 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), init___60855, *[self_60856, nrows_60857, ncols_60858], **kwargs_60863)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()

    
    # Assigning a List to a Name (line 205):

    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpec.update.__dict__.__setitem__('stypy_localization', localization)
        GridSpec.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpec.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpec.update.__dict__.__setitem__('stypy_function_name', 'GridSpec.update')
        GridSpec.update.__dict__.__setitem__('stypy_param_names_list', [])
        GridSpec.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpec.update.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        GridSpec.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpec.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpec.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpec.update.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpec.update', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        unicode_60865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, (-1)), 'unicode', u'\n        Update the current values.  If any kwarg is None, default to\n        the current value, if set, otherwise to rc.\n        ')
        
        
        # Call to iteritems(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'kwargs' (line 213)
        kwargs_60868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 34), 'kwargs', False)
        # Processing the call keyword arguments (line 213)
        kwargs_60869 = {}
        # Getting the type of 'six' (line 213)
        six_60866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 213)
        iteritems_60867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), six_60866, 'iteritems')
        # Calling iteritems(args, kwargs) (line 213)
        iteritems_call_result_60870 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), iteritems_60867, *[kwargs_60868], **kwargs_60869)
        
        # Testing the type of a for loop iterable (line 213)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 213, 8), iteritems_call_result_60870)
        # Getting the type of the for loop variable (line 213)
        for_loop_var_60871 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 213, 8), iteritems_call_result_60870)
        # Assigning a type to the variable 'k' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 8), for_loop_var_60871))
        # Assigning a type to the variable 'v' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 8), for_loop_var_60871))
        # SSA begins for a for statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'k' (line 214)
        k_60872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'k')
        # Getting the type of 'self' (line 214)
        self_60873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'self')
        # Obtaining the member '_AllowedKeys' of a type (line 214)
        _AllowedKeys_60874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), self_60873, '_AllowedKeys')
        # Applying the binary operator 'in' (line 214)
        result_contains_60875 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 15), 'in', k_60872, _AllowedKeys_60874)
        
        # Testing the type of an if condition (line 214)
        if_condition_60876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 12), result_contains_60875)
        # Assigning a type to the variable 'if_condition_60876' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'if_condition_60876', if_condition_60876)
        # SSA begins for if statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'self' (line 215)
        self_60878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'self', False)
        # Getting the type of 'k' (line 215)
        k_60879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 30), 'k', False)
        # Getting the type of 'v' (line 215)
        v_60880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'v', False)
        # Processing the call keyword arguments (line 215)
        kwargs_60881 = {}
        # Getting the type of 'setattr' (line 215)
        setattr_60877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 215)
        setattr_call_result_60882 = invoke(stypy.reporting.localization.Localization(__file__, 215, 16), setattr_60877, *[self_60878, k_60879, v_60880], **kwargs_60881)
        
        # SSA branch for the else part of an if statement (line 214)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 217)
        # Processing the call arguments (line 217)
        unicode_60884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'unicode', u'%s is unknown keyword')
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_60885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 64), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        # Getting the type of 'k' (line 217)
        k_60886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 64), tuple_60885, k_60886)
        
        # Applying the binary operator '%' (line 217)
        result_mod_60887 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 37), '%', unicode_60884, tuple_60885)
        
        # Processing the call keyword arguments (line 217)
        kwargs_60888 = {}
        # Getting the type of 'AttributeError' (line 217)
        AttributeError_60883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 217)
        AttributeError_call_result_60889 = invoke(stypy.reporting.localization.Localization(__file__, 217, 22), AttributeError_60883, *[result_mod_60887], **kwargs_60888)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 217, 16), AttributeError_call_result_60889, 'raise parameter', BaseException)
        # SSA join for if statement (line 214)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 219, 8))
        
        # 'from matplotlib import _pylab_helpers' statement (line 219)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_60890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 219, 8), 'matplotlib')

        if (type(import_60890) is not StypyTypeError):

            if (import_60890 != 'pyd_module'):
                __import__(import_60890)
                sys_modules_60891 = sys.modules[import_60890]
                import_from_module(stypy.reporting.localization.Localization(__file__, 219, 8), 'matplotlib', sys_modules_60891.module_type_store, module_type_store, ['_pylab_helpers'])
                nest_module(stypy.reporting.localization.Localization(__file__, 219, 8), __file__, sys_modules_60891, sys_modules_60891.module_type_store, module_type_store)
            else:
                from matplotlib import _pylab_helpers

                import_from_module(stypy.reporting.localization.Localization(__file__, 219, 8), 'matplotlib', None, module_type_store, ['_pylab_helpers'], [_pylab_helpers])

        else:
            # Assigning a type to the variable 'matplotlib' (line 219)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'matplotlib', import_60890)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 220, 8))
        
        # 'from matplotlib.axes import SubplotBase' statement (line 220)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_60892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 220, 8), 'matplotlib.axes')

        if (type(import_60892) is not StypyTypeError):

            if (import_60892 != 'pyd_module'):
                __import__(import_60892)
                sys_modules_60893 = sys.modules[import_60892]
                import_from_module(stypy.reporting.localization.Localization(__file__, 220, 8), 'matplotlib.axes', sys_modules_60893.module_type_store, module_type_store, ['SubplotBase'])
                nest_module(stypy.reporting.localization.Localization(__file__, 220, 8), __file__, sys_modules_60893, sys_modules_60893.module_type_store, module_type_store)
            else:
                from matplotlib.axes import SubplotBase

                import_from_module(stypy.reporting.localization.Localization(__file__, 220, 8), 'matplotlib.axes', None, module_type_store, ['SubplotBase'], [SubplotBase])

        else:
            # Assigning a type to the variable 'matplotlib.axes' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'matplotlib.axes', import_60892)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        
        # Call to itervalues(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of '_pylab_helpers' (line 221)
        _pylab_helpers_60896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 41), '_pylab_helpers', False)
        # Obtaining the member 'Gcf' of a type (line 221)
        Gcf_60897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 41), _pylab_helpers_60896, 'Gcf')
        # Obtaining the member 'figs' of a type (line 221)
        figs_60898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 41), Gcf_60897, 'figs')
        # Processing the call keyword arguments (line 221)
        kwargs_60899 = {}
        # Getting the type of 'six' (line 221)
        six_60894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 26), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 221)
        itervalues_60895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 26), six_60894, 'itervalues')
        # Calling itervalues(args, kwargs) (line 221)
        itervalues_call_result_60900 = invoke(stypy.reporting.localization.Localization(__file__, 221, 26), itervalues_60895, *[figs_60898], **kwargs_60899)
        
        # Testing the type of a for loop iterable (line 221)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 221, 8), itervalues_call_result_60900)
        # Getting the type of the for loop variable (line 221)
        for_loop_var_60901 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 221, 8), itervalues_call_result_60900)
        # Assigning a type to the variable 'figmanager' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'figmanager', for_loop_var_60901)
        # SSA begins for a for statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'figmanager' (line 222)
        figmanager_60902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'figmanager')
        # Obtaining the member 'canvas' of a type (line 222)
        canvas_60903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), figmanager_60902, 'canvas')
        # Obtaining the member 'figure' of a type (line 222)
        figure_60904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), canvas_60903, 'figure')
        # Obtaining the member 'axes' of a type (line 222)
        axes_60905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), figure_60904, 'axes')
        # Testing the type of a for loop iterable (line 222)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 12), axes_60905)
        # Getting the type of the for loop variable (line 222)
        for_loop_var_60906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 12), axes_60905)
        # Assigning a type to the variable 'ax' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'ax', for_loop_var_60906)
        # SSA begins for a for statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to isinstance(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'ax' (line 224)
        ax_60908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 34), 'ax', False)
        # Getting the type of 'SubplotBase' (line 224)
        SubplotBase_60909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'SubplotBase', False)
        # Processing the call keyword arguments (line 224)
        kwargs_60910 = {}
        # Getting the type of 'isinstance' (line 224)
        isinstance_60907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 224)
        isinstance_call_result_60911 = invoke(stypy.reporting.localization.Localization(__file__, 224, 23), isinstance_60907, *[ax_60908, SubplotBase_60909], **kwargs_60910)
        
        # Applying the 'not' unary operator (line 224)
        result_not__60912 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), 'not', isinstance_call_result_60911)
        
        # Testing the type of an if condition (line 224)
        if_condition_60913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 16), result_not__60912)
        # Assigning a type to the variable 'if_condition_60913' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'if_condition_60913', if_condition_60913)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to isinstance(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'ax' (line 226)
        ax_60915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'ax', False)
        # Obtaining the member '_sharex' of a type (line 226)
        _sharex_60916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 34), ax_60915, '_sharex')
        # Getting the type of 'SubplotBase' (line 226)
        SubplotBase_60917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 46), 'SubplotBase', False)
        # Processing the call keyword arguments (line 226)
        kwargs_60918 = {}
        # Getting the type of 'isinstance' (line 226)
        isinstance_60914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 226)
        isinstance_call_result_60919 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), isinstance_60914, *[_sharex_60916, SubplotBase_60917], **kwargs_60918)
        
        # Testing the type of an if condition (line 226)
        if_condition_60920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 20), isinstance_call_result_60919)
        # Assigning a type to the variable 'if_condition_60920' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'if_condition_60920', if_condition_60920)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to get_gridspec(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_60927 = {}
        
        # Call to get_subplotspec(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_60924 = {}
        # Getting the type of 'ax' (line 227)
        ax_60921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'ax', False)
        # Obtaining the member '_sharex' of a type (line 227)
        _sharex_60922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), ax_60921, '_sharex')
        # Obtaining the member 'get_subplotspec' of a type (line 227)
        get_subplotspec_60923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), _sharex_60922, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 227)
        get_subplotspec_call_result_60925 = invoke(stypy.reporting.localization.Localization(__file__, 227, 27), get_subplotspec_60923, *[], **kwargs_60924)
        
        # Obtaining the member 'get_gridspec' of a type (line 227)
        get_gridspec_60926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), get_subplotspec_call_result_60925, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 227)
        get_gridspec_call_result_60928 = invoke(stypy.reporting.localization.Localization(__file__, 227, 27), get_gridspec_60926, *[], **kwargs_60927)
        
        # Getting the type of 'self' (line 227)
        self_60929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 74), 'self')
        # Applying the binary operator '==' (line 227)
        result_eq_60930 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 27), '==', get_gridspec_call_result_60928, self_60929)
        
        # Testing the type of an if condition (line 227)
        if_condition_60931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 24), result_eq_60930)
        # Assigning a type to the variable 'if_condition_60931' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 24), 'if_condition_60931', if_condition_60931)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update_params(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_60935 = {}
        # Getting the type of 'ax' (line 228)
        ax_60932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'ax', False)
        # Obtaining the member '_sharex' of a type (line 228)
        _sharex_60933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 28), ax_60932, '_sharex')
        # Obtaining the member 'update_params' of a type (line 228)
        update_params_60934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 28), _sharex_60933, 'update_params')
        # Calling update_params(args, kwargs) (line 228)
        update_params_call_result_60936 = invoke(stypy.reporting.localization.Localization(__file__, 228, 28), update_params_60934, *[], **kwargs_60935)
        
        
        # Call to set_position(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'ax' (line 229)
        ax_60939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 44), 'ax', False)
        # Obtaining the member '_sharex' of a type (line 229)
        _sharex_60940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 44), ax_60939, '_sharex')
        # Obtaining the member 'figbox' of a type (line 229)
        figbox_60941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 44), _sharex_60940, 'figbox')
        # Processing the call keyword arguments (line 229)
        kwargs_60942 = {}
        # Getting the type of 'ax' (line 229)
        ax_60937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 28), 'ax', False)
        # Obtaining the member 'set_position' of a type (line 229)
        set_position_60938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 28), ax_60937, 'set_position')
        # Calling set_position(args, kwargs) (line 229)
        set_position_call_result_60943 = invoke(stypy.reporting.localization.Localization(__file__, 229, 28), set_position_60938, *[figbox_60941], **kwargs_60942)
        
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 226)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'ax' (line 230)
        ax_60945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 36), 'ax', False)
        # Obtaining the member '_sharey' of a type (line 230)
        _sharey_60946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 36), ax_60945, '_sharey')
        # Getting the type of 'SubplotBase' (line 230)
        SubplotBase_60947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'SubplotBase', False)
        # Processing the call keyword arguments (line 230)
        kwargs_60948 = {}
        # Getting the type of 'isinstance' (line 230)
        isinstance_60944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 230)
        isinstance_call_result_60949 = invoke(stypy.reporting.localization.Localization(__file__, 230, 25), isinstance_60944, *[_sharey_60946, SubplotBase_60947], **kwargs_60948)
        
        # Testing the type of an if condition (line 230)
        if_condition_60950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 25), isinstance_call_result_60949)
        # Assigning a type to the variable 'if_condition_60950' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'if_condition_60950', if_condition_60950)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to get_gridspec(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_60957 = {}
        
        # Call to get_subplotspec(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_60954 = {}
        # Getting the type of 'ax' (line 231)
        ax_60951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'ax', False)
        # Obtaining the member '_sharey' of a type (line 231)
        _sharey_60952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 27), ax_60951, '_sharey')
        # Obtaining the member 'get_subplotspec' of a type (line 231)
        get_subplotspec_60953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 27), _sharey_60952, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 231)
        get_subplotspec_call_result_60955 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), get_subplotspec_60953, *[], **kwargs_60954)
        
        # Obtaining the member 'get_gridspec' of a type (line 231)
        get_gridspec_60956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 27), get_subplotspec_call_result_60955, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 231)
        get_gridspec_call_result_60958 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), get_gridspec_60956, *[], **kwargs_60957)
        
        # Getting the type of 'self' (line 231)
        self_60959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 74), 'self')
        # Applying the binary operator '==' (line 231)
        result_eq_60960 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 27), '==', get_gridspec_call_result_60958, self_60959)
        
        # Testing the type of an if condition (line 231)
        if_condition_60961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 24), result_eq_60960)
        # Assigning a type to the variable 'if_condition_60961' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'if_condition_60961', if_condition_60961)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update_params(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_60965 = {}
        # Getting the type of 'ax' (line 232)
        ax_60962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'ax', False)
        # Obtaining the member '_sharey' of a type (line 232)
        _sharey_60963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), ax_60962, '_sharey')
        # Obtaining the member 'update_params' of a type (line 232)
        update_params_60964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), _sharey_60963, 'update_params')
        # Calling update_params(args, kwargs) (line 232)
        update_params_call_result_60966 = invoke(stypy.reporting.localization.Localization(__file__, 232, 28), update_params_60964, *[], **kwargs_60965)
        
        
        # Call to set_position(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'ax' (line 233)
        ax_60969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 44), 'ax', False)
        # Obtaining the member '_sharey' of a type (line 233)
        _sharey_60970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 44), ax_60969, '_sharey')
        # Obtaining the member 'figbox' of a type (line 233)
        figbox_60971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 44), _sharey_60970, 'figbox')
        # Processing the call keyword arguments (line 233)
        kwargs_60972 = {}
        # Getting the type of 'ax' (line 233)
        ax_60967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'ax', False)
        # Obtaining the member 'set_position' of a type (line 233)
        set_position_60968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 28), ax_60967, 'set_position')
        # Calling set_position(args, kwargs) (line 233)
        set_position_call_result_60973 = invoke(stypy.reporting.localization.Localization(__file__, 233, 28), set_position_60968, *[figbox_60971], **kwargs_60972)
        
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 224)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to get_topmost_subplotspec(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_60979 = {}
        
        # Call to get_subplotspec(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_60976 = {}
        # Getting the type of 'ax' (line 235)
        ax_60974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'ax', False)
        # Obtaining the member 'get_subplotspec' of a type (line 235)
        get_subplotspec_60975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 25), ax_60974, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 235)
        get_subplotspec_call_result_60977 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), get_subplotspec_60975, *[], **kwargs_60976)
        
        # Obtaining the member 'get_topmost_subplotspec' of a type (line 235)
        get_topmost_subplotspec_60978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 25), get_subplotspec_call_result_60977, 'get_topmost_subplotspec')
        # Calling get_topmost_subplotspec(args, kwargs) (line 235)
        get_topmost_subplotspec_call_result_60980 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), get_topmost_subplotspec_60978, *[], **kwargs_60979)
        
        # Assigning a type to the variable 'ss' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'ss', get_topmost_subplotspec_call_result_60980)
        
        
        
        # Call to get_gridspec(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_60983 = {}
        # Getting the type of 'ss' (line 236)
        ss_60981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'ss', False)
        # Obtaining the member 'get_gridspec' of a type (line 236)
        get_gridspec_60982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 23), ss_60981, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 236)
        get_gridspec_call_result_60984 = invoke(stypy.reporting.localization.Localization(__file__, 236, 23), get_gridspec_60982, *[], **kwargs_60983)
        
        # Getting the type of 'self' (line 236)
        self_60985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 44), 'self')
        # Applying the binary operator '==' (line 236)
        result_eq_60986 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 23), '==', get_gridspec_call_result_60984, self_60985)
        
        # Testing the type of an if condition (line 236)
        if_condition_60987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 20), result_eq_60986)
        # Assigning a type to the variable 'if_condition_60987' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'if_condition_60987', if_condition_60987)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update_params(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_60990 = {}
        # Getting the type of 'ax' (line 237)
        ax_60988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'ax', False)
        # Obtaining the member 'update_params' of a type (line 237)
        update_params_60989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 24), ax_60988, 'update_params')
        # Calling update_params(args, kwargs) (line 237)
        update_params_call_result_60991 = invoke(stypy.reporting.localization.Localization(__file__, 237, 24), update_params_60989, *[], **kwargs_60990)
        
        
        # Call to set_position(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'ax' (line 238)
        ax_60994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 40), 'ax', False)
        # Obtaining the member 'figbox' of a type (line 238)
        figbox_60995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 40), ax_60994, 'figbox')
        # Processing the call keyword arguments (line 238)
        kwargs_60996 = {}
        # Getting the type of 'ax' (line 238)
        ax_60992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'ax', False)
        # Obtaining the member 'set_position' of a type (line 238)
        set_position_60993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), ax_60992, 'set_position')
        # Calling set_position(args, kwargs) (line 238)
        set_position_call_result_60997 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), set_position_60993, *[figbox_60995], **kwargs_60996)
        
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_60998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_60998


    @norecursion
    def get_subplot_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 240)
        None_60999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 37), 'None')
        defaults = [None_60999]
        # Create a new context for function 'get_subplot_params'
        module_type_store = module_type_store.open_function_context('get_subplot_params', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_localization', localization)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_function_name', 'GridSpec.get_subplot_params')
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpec.get_subplot_params.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpec.get_subplot_params', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_subplot_params', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_subplot_params(...)' code ##################

        unicode_61000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'unicode', u'\n        return a dictionary of subplot layout parameters. The default\n        parameters are from rcParams unless a figure attribute is set.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 245, 8))
        
        # 'from matplotlib.figure import SubplotParams' statement (line 245)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_61001 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.figure')

        if (type(import_61001) is not StypyTypeError):

            if (import_61001 != 'pyd_module'):
                __import__(import_61001)
                sys_modules_61002 = sys.modules[import_61001]
                import_from_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.figure', sys_modules_61002.module_type_store, module_type_store, ['SubplotParams'])
                nest_module(stypy.reporting.localization.Localization(__file__, 245, 8), __file__, sys_modules_61002, sys_modules_61002.module_type_store, module_type_store)
            else:
                from matplotlib.figure import SubplotParams

                import_from_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.figure', None, module_type_store, ['SubplotParams'], [SubplotParams])

        else:
            # Assigning a type to the variable 'matplotlib.figure' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.figure', import_61001)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Type idiom detected: calculating its left and rigth part (line 246)
        # Getting the type of 'fig' (line 246)
        fig_61003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'fig')
        # Getting the type of 'None' (line 246)
        None_61004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'None')
        
        (may_be_61005, more_types_in_union_61006) = may_be_none(fig_61003, None_61004)

        if may_be_61005:

            if more_types_in_union_61006:
                # Runtime conditional SSA (line 246)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a DictComp to a Name (line 247):
            
            # Assigning a DictComp to a Name (line 247):
            # Calculating dict comprehension
            module_type_store = module_type_store.open_function_context('dict comprehension expression', 247, 18, True)
            # Calculating comprehension expression
            # Getting the type of 'self' (line 247)
            self_61014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 60), 'self')
            # Obtaining the member '_AllowedKeys' of a type (line 247)
            _AllowedKeys_61015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 60), self_61014, '_AllowedKeys')
            comprehension_61016 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 18), _AllowedKeys_61015)
            # Assigning a type to the variable 'k' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'k', comprehension_61016)
            # Getting the type of 'k' (line 247)
            k_61007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'k')
            
            # Obtaining the type of the subscript
            unicode_61008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 30), 'unicode', u'figure.subplot.')
            # Getting the type of 'k' (line 247)
            k_61009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 48), 'k')
            # Applying the binary operator '+' (line 247)
            result_add_61010 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 30), '+', unicode_61008, k_61009)
            
            # Getting the type of 'rcParams' (line 247)
            rcParams_61011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 247)
            getitem___61012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 21), rcParams_61011, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 247)
            subscript_call_result_61013 = invoke(stypy.reporting.localization.Localization(__file__, 247, 21), getitem___61012, result_add_61010)
            
            dict_61017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 18), 'dict')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 18), dict_61017, (k_61007, subscript_call_result_61013))
            # Assigning a type to the variable 'kw' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'kw', dict_61017)
            
            # Assigning a Call to a Name (line 248):
            
            # Assigning a Call to a Name (line 248):
            
            # Call to SubplotParams(...): (line 248)
            # Processing the call keyword arguments (line 248)
            # Getting the type of 'kw' (line 248)
            kw_61019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'kw', False)
            kwargs_61020 = {'kw_61019': kw_61019}
            # Getting the type of 'SubplotParams' (line 248)
            SubplotParams_61018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 26), 'SubplotParams', False)
            # Calling SubplotParams(args, kwargs) (line 248)
            SubplotParams_call_result_61021 = invoke(stypy.reporting.localization.Localization(__file__, 248, 26), SubplotParams_61018, *[], **kwargs_61020)
            
            # Assigning a type to the variable 'subplotpars' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'subplotpars', SubplotParams_call_result_61021)

            if more_types_in_union_61006:
                # Runtime conditional SSA for else branch (line 246)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_61005) or more_types_in_union_61006):
            
            # Assigning a Call to a Name (line 250):
            
            # Assigning a Call to a Name (line 250):
            
            # Call to copy(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'fig' (line 250)
            fig_61024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 36), 'fig', False)
            # Obtaining the member 'subplotpars' of a type (line 250)
            subplotpars_61025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 36), fig_61024, 'subplotpars')
            # Processing the call keyword arguments (line 250)
            kwargs_61026 = {}
            # Getting the type of 'copy' (line 250)
            copy_61022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 26), 'copy', False)
            # Obtaining the member 'copy' of a type (line 250)
            copy_61023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 26), copy_61022, 'copy')
            # Calling copy(args, kwargs) (line 250)
            copy_call_result_61027 = invoke(stypy.reporting.localization.Localization(__file__, 250, 26), copy_61023, *[subplotpars_61025], **kwargs_61026)
            
            # Assigning a type to the variable 'subplotpars' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'subplotpars', copy_call_result_61027)

            if (may_be_61005 and more_types_in_union_61006):
                # SSA join for if statement (line 246)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a DictComp to a Name (line 252):
        
        # Assigning a DictComp to a Name (line 252):
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 252, 21, True)
        # Calculating comprehension expression
        # Getting the type of 'self' (line 252)
        self_61034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 50), 'self')
        # Obtaining the member '_AllowedKeys' of a type (line 252)
        _AllowedKeys_61035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 50), self_61034, '_AllowedKeys')
        comprehension_61036 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 21), _AllowedKeys_61035)
        # Assigning a type to the variable 'k' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'k', comprehension_61036)
        # Getting the type of 'k' (line 252)
        k_61028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'k')
        
        # Call to getattr(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'self' (line 252)
        self_61030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 32), 'self', False)
        # Getting the type of 'k' (line 252)
        k_61031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'k', False)
        # Processing the call keyword arguments (line 252)
        kwargs_61032 = {}
        # Getting the type of 'getattr' (line 252)
        getattr_61029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 252)
        getattr_call_result_61033 = invoke(stypy.reporting.localization.Localization(__file__, 252, 24), getattr_61029, *[self_61030, k_61031], **kwargs_61032)
        
        dict_61037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 21), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 21), dict_61037, (k_61028, getattr_call_result_61033))
        # Assigning a type to the variable 'update_kw' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'update_kw', dict_61037)
        
        # Call to update(...): (line 253)
        # Processing the call keyword arguments (line 253)
        # Getting the type of 'update_kw' (line 253)
        update_kw_61040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 29), 'update_kw', False)
        kwargs_61041 = {'update_kw_61040': update_kw_61040}
        # Getting the type of 'subplotpars' (line 253)
        subplotpars_61038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'subplotpars', False)
        # Obtaining the member 'update' of a type (line 253)
        update_61039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), subplotpars_61038, 'update')
        # Calling update(args, kwargs) (line 253)
        update_call_result_61042 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), update_61039, *[], **kwargs_61041)
        
        # Getting the type of 'subplotpars' (line 255)
        subplotpars_61043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'subplotpars')
        # Assigning a type to the variable 'stypy_return_type' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type', subplotpars_61043)
        
        # ################# End of 'get_subplot_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_subplot_params' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_61044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61044)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_subplot_params'
        return stypy_return_type_61044


    @norecursion
    def locally_modified_subplot_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'locally_modified_subplot_params'
        module_type_store = module_type_store.open_function_context('locally_modified_subplot_params', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_localization', localization)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_function_name', 'GridSpec.locally_modified_subplot_params')
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_param_names_list', [])
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpec.locally_modified_subplot_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpec.locally_modified_subplot_params', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'locally_modified_subplot_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'locally_modified_subplot_params(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 258)
        self_61051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'self')
        # Obtaining the member '_AllowedKeys' of a type (line 258)
        _AllowedKeys_61052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 27), self_61051, '_AllowedKeys')
        comprehension_61053 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 16), _AllowedKeys_61052)
        # Assigning a type to the variable 'k' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'k', comprehension_61053)
        
        # Call to getattr(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'self' (line 258)
        self_61047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 56), 'self', False)
        # Getting the type of 'k' (line 258)
        k_61048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 62), 'k', False)
        # Processing the call keyword arguments (line 258)
        kwargs_61049 = {}
        # Getting the type of 'getattr' (line 258)
        getattr_61046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 48), 'getattr', False)
        # Calling getattr(args, kwargs) (line 258)
        getattr_call_result_61050 = invoke(stypy.reporting.localization.Localization(__file__, 258, 48), getattr_61046, *[self_61047, k_61048], **kwargs_61049)
        
        # Getting the type of 'k' (line 258)
        k_61045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'k')
        list_61054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 16), list_61054, k_61045)
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', list_61054)
        
        # ################# End of 'locally_modified_subplot_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'locally_modified_subplot_params' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_61055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'locally_modified_subplot_params'
        return stypy_return_type_61055


    @norecursion
    def tight_layout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 260)
        None_61056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 41), 'None')
        float_61057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 25), 'float')
        # Getting the type of 'None' (line 261)
        None_61058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'None')
        # Getting the type of 'None' (line 261)
        None_61059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 49), 'None')
        # Getting the type of 'None' (line 261)
        None_61060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 60), 'None')
        defaults = [None_61056, float_61057, None_61058, None_61059, None_61060]
        # Create a new context for function 'tight_layout'
        module_type_store = module_type_store.open_function_context('tight_layout', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpec.tight_layout.__dict__.__setitem__('stypy_localization', localization)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_function_name', 'GridSpec.tight_layout')
        GridSpec.tight_layout.__dict__.__setitem__('stypy_param_names_list', ['fig', 'renderer', 'pad', 'h_pad', 'w_pad', 'rect'])
        GridSpec.tight_layout.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpec.tight_layout.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpec.tight_layout', ['fig', 'renderer', 'pad', 'h_pad', 'w_pad', 'rect'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tight_layout', localization, ['fig', 'renderer', 'pad', 'h_pad', 'w_pad', 'rect'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tight_layout(...)' code ##################

        unicode_61061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, (-1)), 'unicode', u'\n        Adjust subplot parameters to give specified padding.\n\n        Parameters\n        ----------\n\n        pad : float\n            Padding between the figure edge and the edges of subplots, as a\n            fraction of the font-size.\n        h_pad, w_pad : float, optional\n            Padding (height/width) between edges of adjacent subplots.\n            Defaults to ``pad_inches``.\n        rect : tuple of 4 floats, optional\n            (left, bottom, right, top) rectangle in normalized figure\n            coordinates that the whole subplots area (including labels) will\n            fit into.  Default is (0, 0, 1, 1).\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 280, 8))
        
        # 'from matplotlib.tight_layout import get_renderer, get_subplotspec_list, get_tight_layout_figure' statement (line 280)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_61062 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 280, 8), 'matplotlib.tight_layout')

        if (type(import_61062) is not StypyTypeError):

            if (import_61062 != 'pyd_module'):
                __import__(import_61062)
                sys_modules_61063 = sys.modules[import_61062]
                import_from_module(stypy.reporting.localization.Localization(__file__, 280, 8), 'matplotlib.tight_layout', sys_modules_61063.module_type_store, module_type_store, ['get_renderer', 'get_subplotspec_list', 'get_tight_layout_figure'])
                nest_module(stypy.reporting.localization.Localization(__file__, 280, 8), __file__, sys_modules_61063, sys_modules_61063.module_type_store, module_type_store)
            else:
                from matplotlib.tight_layout import get_renderer, get_subplotspec_list, get_tight_layout_figure

                import_from_module(stypy.reporting.localization.Localization(__file__, 280, 8), 'matplotlib.tight_layout', None, module_type_store, ['get_renderer', 'get_subplotspec_list', 'get_tight_layout_figure'], [get_renderer, get_subplotspec_list, get_tight_layout_figure])

        else:
            # Assigning a type to the variable 'matplotlib.tight_layout' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'matplotlib.tight_layout', import_61062)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to get_subplotspec_list(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'fig' (line 283)
        fig_61065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 48), 'fig', False)
        # Obtaining the member 'axes' of a type (line 283)
        axes_61066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 48), fig_61065, 'axes')
        # Processing the call keyword arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_61067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 68), 'self', False)
        keyword_61068 = self_61067
        kwargs_61069 = {'grid_spec': keyword_61068}
        # Getting the type of 'get_subplotspec_list' (line 283)
        get_subplotspec_list_61064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'get_subplotspec_list', False)
        # Calling get_subplotspec_list(args, kwargs) (line 283)
        get_subplotspec_list_call_result_61070 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), get_subplotspec_list_61064, *[axes_61066], **kwargs_61069)
        
        # Assigning a type to the variable 'subplotspec_list' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'subplotspec_list', get_subplotspec_list_call_result_61070)
        
        
        # Getting the type of 'None' (line 284)
        None_61071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'None')
        # Getting the type of 'subplotspec_list' (line 284)
        subplotspec_list_61072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'subplotspec_list')
        # Applying the binary operator 'in' (line 284)
        result_contains_61073 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'in', None_61071, subplotspec_list_61072)
        
        # Testing the type of an if condition (line 284)
        if_condition_61074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_contains_61073)
        # Assigning a type to the variable 'if_condition_61074' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_61074', if_condition_61074)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 285)
        # Processing the call arguments (line 285)
        unicode_61077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 26), 'unicode', u'This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.')
        # Processing the call keyword arguments (line 285)
        kwargs_61078 = {}
        # Getting the type of 'warnings' (line 285)
        warnings_61075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 285)
        warn_61076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), warnings_61075, 'warn')
        # Calling warn(args, kwargs) (line 285)
        warn_call_result_61079 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), warn_61076, *[unicode_61077], **kwargs_61078)
        
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 288)
        # Getting the type of 'renderer' (line 288)
        renderer_61080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'renderer')
        # Getting the type of 'None' (line 288)
        None_61081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'None')
        
        (may_be_61082, more_types_in_union_61083) = may_be_none(renderer_61080, None_61081)

        if may_be_61082:

            if more_types_in_union_61083:
                # Runtime conditional SSA (line 288)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 289):
            
            # Assigning a Call to a Name (line 289):
            
            # Call to get_renderer(...): (line 289)
            # Processing the call arguments (line 289)
            # Getting the type of 'fig' (line 289)
            fig_61085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 36), 'fig', False)
            # Processing the call keyword arguments (line 289)
            kwargs_61086 = {}
            # Getting the type of 'get_renderer' (line 289)
            get_renderer_61084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'get_renderer', False)
            # Calling get_renderer(args, kwargs) (line 289)
            get_renderer_call_result_61087 = invoke(stypy.reporting.localization.Localization(__file__, 289, 23), get_renderer_61084, *[fig_61085], **kwargs_61086)
            
            # Assigning a type to the variable 'renderer' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'renderer', get_renderer_call_result_61087)

            if more_types_in_union_61083:
                # SSA join for if statement (line 288)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to get_tight_layout_figure(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'fig' (line 292)
        fig_61089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'fig', False)
        # Getting the type of 'fig' (line 292)
        fig_61090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'fig', False)
        # Obtaining the member 'axes' of a type (line 292)
        axes_61091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 17), fig_61090, 'axes')
        # Getting the type of 'subplotspec_list' (line 292)
        subplotspec_list_61092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 27), 'subplotspec_list', False)
        # Getting the type of 'renderer' (line 292)
        renderer_61093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 45), 'renderer', False)
        # Processing the call keyword arguments (line 291)
        # Getting the type of 'pad' (line 293)
        pad_61094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'pad', False)
        keyword_61095 = pad_61094
        # Getting the type of 'h_pad' (line 293)
        h_pad_61096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 'h_pad', False)
        keyword_61097 = h_pad_61096
        # Getting the type of 'w_pad' (line 293)
        w_pad_61098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 40), 'w_pad', False)
        keyword_61099 = w_pad_61098
        # Getting the type of 'rect' (line 293)
        rect_61100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 52), 'rect', False)
        keyword_61101 = rect_61100
        kwargs_61102 = {'w_pad': keyword_61099, 'h_pad': keyword_61097, 'pad': keyword_61095, 'rect': keyword_61101}
        # Getting the type of 'get_tight_layout_figure' (line 291)
        get_tight_layout_figure_61088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'get_tight_layout_figure', False)
        # Calling get_tight_layout_figure(args, kwargs) (line 291)
        get_tight_layout_figure_call_result_61103 = invoke(stypy.reporting.localization.Localization(__file__, 291, 17), get_tight_layout_figure_61088, *[fig_61089, axes_61091, subplotspec_list_61092, renderer_61093], **kwargs_61102)
        
        # Assigning a type to the variable 'kwargs' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'kwargs', get_tight_layout_figure_call_result_61103)
        
        # Call to update(...): (line 294)
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'kwargs' (line 294)
        kwargs_61106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'kwargs', False)
        kwargs_61107 = {'kwargs_61106': kwargs_61106}
        # Getting the type of 'self' (line 294)
        self_61104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self', False)
        # Obtaining the member 'update' of a type (line 294)
        update_61105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_61104, 'update')
        # Calling update(args, kwargs) (line 294)
        update_call_result_61108 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), update_61105, *[], **kwargs_61107)
        
        
        # ################# End of 'tight_layout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tight_layout' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_61109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tight_layout'
        return stypy_return_type_61109


# Assigning a type to the variable 'GridSpec' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'GridSpec', GridSpec)

# Assigning a List to a Name (line 205):

# Obtaining an instance of the builtin type 'list' (line 205)
list_61110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 205)
# Adding element type (line 205)
unicode_61111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 20), 'unicode', u'left')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_61110, unicode_61111)
# Adding element type (line 205)
unicode_61112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 28), 'unicode', u'bottom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_61110, unicode_61112)
# Adding element type (line 205)
unicode_61113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 38), 'unicode', u'right')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_61110, unicode_61113)
# Adding element type (line 205)
unicode_61114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 47), 'unicode', u'top')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_61110, unicode_61114)
# Adding element type (line 205)
unicode_61115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 54), 'unicode', u'wspace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_61110, unicode_61115)
# Adding element type (line 205)
unicode_61116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 64), 'unicode', u'hspace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 19), list_61110, unicode_61116)

# Getting the type of 'GridSpec'
GridSpec_61117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GridSpec')
# Setting the type of the member '_AllowedKeys' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GridSpec_61117, '_AllowedKeys', list_61110)
# Declaration of the 'GridSpecFromSubplotSpec' class
# Getting the type of 'GridSpecBase' (line 297)
GridSpecBase_61118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'GridSpecBase')

class GridSpecFromSubplotSpec(GridSpecBase_61118, ):
    unicode_61119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'unicode', u'\n    GridSpec whose subplot layout parameters are inherited from the\n    location specified by a given SubplotSpec.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 304)
        None_61120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'None')
        # Getting the type of 'None' (line 304)
        None_61121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 37), 'None')
        # Getting the type of 'None' (line 305)
        None_61122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 31), 'None')
        # Getting the type of 'None' (line 305)
        None_61123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 50), 'None')
        defaults = [None_61120, None_61121, None_61122, None_61123]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecFromSubplotSpec.__init__', ['nrows', 'ncols', 'subplot_spec', 'wspace', 'hspace', 'height_ratios', 'width_ratios'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['nrows', 'ncols', 'subplot_spec', 'wspace', 'hspace', 'height_ratios', 'width_ratios'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_61124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, (-1)), 'unicode', u'\n        The number of rows and number of columns of the grid need to\n        be set. An instance of SubplotSpec is also needed to be set\n        from which the layout parameters will be inherited. The wspace\n        and hspace of the layout can be optionally specified or the\n        default values (from the figure or rcParams) will be used.\n        ')
        
        # Assigning a Name to a Attribute (line 313):
        
        # Assigning a Name to a Attribute (line 313):
        # Getting the type of 'wspace' (line 313)
        wspace_61125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'wspace')
        # Getting the type of 'self' (line 313)
        self_61126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self')
        # Setting the type of the member '_wspace' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_61126, '_wspace', wspace_61125)
        
        # Assigning a Name to a Attribute (line 314):
        
        # Assigning a Name to a Attribute (line 314):
        # Getting the type of 'hspace' (line 314)
        hspace_61127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 'hspace')
        # Getting the type of 'self' (line 314)
        self_61128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self')
        # Setting the type of the member '_hspace' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_61128, '_hspace', hspace_61127)
        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of 'subplot_spec' (line 315)
        subplot_spec_61129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 29), 'subplot_spec')
        # Getting the type of 'self' (line 315)
        self_61130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member '_subplot_spec' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_61130, '_subplot_spec', subplot_spec_61129)
        
        # Call to __init__(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_61133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 30), 'self', False)
        # Getting the type of 'nrows' (line 317)
        nrows_61134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 36), 'nrows', False)
        # Getting the type of 'ncols' (line 317)
        ncols_61135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 43), 'ncols', False)
        # Processing the call keyword arguments (line 317)
        # Getting the type of 'width_ratios' (line 318)
        width_ratios_61136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 43), 'width_ratios', False)
        keyword_61137 = width_ratios_61136
        # Getting the type of 'height_ratios' (line 319)
        height_ratios_61138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 44), 'height_ratios', False)
        keyword_61139 = height_ratios_61138
        kwargs_61140 = {'height_ratios': keyword_61139, 'width_ratios': keyword_61137}
        # Getting the type of 'GridSpecBase' (line 317)
        GridSpecBase_61131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'GridSpecBase', False)
        # Obtaining the member '__init__' of a type (line 317)
        init___61132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), GridSpecBase_61131, '__init__')
        # Calling __init__(args, kwargs) (line 317)
        init___call_result_61141 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), init___61132, *[self_61133, nrows_61134, ncols_61135], **kwargs_61140)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_subplot_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 322)
        None_61142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 37), 'None')
        defaults = [None_61142]
        # Create a new context for function 'get_subplot_params'
        module_type_store = module_type_store.open_function_context('get_subplot_params', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_localization', localization)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_function_name', 'GridSpecFromSubplotSpec.get_subplot_params')
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecFromSubplotSpec.get_subplot_params.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecFromSubplotSpec.get_subplot_params', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_subplot_params', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_subplot_params(...)' code ##################

        unicode_61143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, (-1)), 'unicode', u'Return a dictionary of subplot layout parameters.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 326)
        # Getting the type of 'fig' (line 326)
        fig_61144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'fig')
        # Getting the type of 'None' (line 326)
        None_61145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 18), 'None')
        
        (may_be_61146, more_types_in_union_61147) = may_be_none(fig_61144, None_61145)

        if may_be_61146:

            if more_types_in_union_61147:
                # Runtime conditional SSA (line 326)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 327):
            
            # Assigning a Subscript to a Name (line 327):
            
            # Obtaining the type of the subscript
            unicode_61148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 30), 'unicode', u'figure.subplot.hspace')
            # Getting the type of 'rcParams' (line 327)
            rcParams_61149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___61150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 21), rcParams_61149, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 327)
            subscript_call_result_61151 = invoke(stypy.reporting.localization.Localization(__file__, 327, 21), getitem___61150, unicode_61148)
            
            # Assigning a type to the variable 'hspace' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'hspace', subscript_call_result_61151)
            
            # Assigning a Subscript to a Name (line 328):
            
            # Assigning a Subscript to a Name (line 328):
            
            # Obtaining the type of the subscript
            unicode_61152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 30), 'unicode', u'figure.subplot.wspace')
            # Getting the type of 'rcParams' (line 328)
            rcParams_61153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 328)
            getitem___61154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), rcParams_61153, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 328)
            subscript_call_result_61155 = invoke(stypy.reporting.localization.Localization(__file__, 328, 21), getitem___61154, unicode_61152)
            
            # Assigning a type to the variable 'wspace' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'wspace', subscript_call_result_61155)

            if more_types_in_union_61147:
                # Runtime conditional SSA for else branch (line 326)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_61146) or more_types_in_union_61147):
            
            # Assigning a Attribute to a Name (line 330):
            
            # Assigning a Attribute to a Name (line 330):
            # Getting the type of 'fig' (line 330)
            fig_61156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 21), 'fig')
            # Obtaining the member 'subplotpars' of a type (line 330)
            subplotpars_61157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 21), fig_61156, 'subplotpars')
            # Obtaining the member 'hspace' of a type (line 330)
            hspace_61158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 21), subplotpars_61157, 'hspace')
            # Assigning a type to the variable 'hspace' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'hspace', hspace_61158)
            
            # Assigning a Attribute to a Name (line 331):
            
            # Assigning a Attribute to a Name (line 331):
            # Getting the type of 'fig' (line 331)
            fig_61159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'fig')
            # Obtaining the member 'subplotpars' of a type (line 331)
            subplotpars_61160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), fig_61159, 'subplotpars')
            # Obtaining the member 'wspace' of a type (line 331)
            wspace_61161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), subplotpars_61160, 'wspace')
            # Assigning a type to the variable 'wspace' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'wspace', wspace_61161)

            if (may_be_61146 and more_types_in_union_61147):
                # SSA join for if statement (line 326)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 333)
        self_61162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'self')
        # Obtaining the member '_hspace' of a type (line 333)
        _hspace_61163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 11), self_61162, '_hspace')
        # Getting the type of 'None' (line 333)
        None_61164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'None')
        # Applying the binary operator 'isnot' (line 333)
        result_is_not_61165 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 11), 'isnot', _hspace_61163, None_61164)
        
        # Testing the type of an if condition (line 333)
        if_condition_61166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), result_is_not_61165)
        # Assigning a type to the variable 'if_condition_61166' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_61166', if_condition_61166)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 334):
        
        # Assigning a Attribute to a Name (line 334):
        # Getting the type of 'self' (line 334)
        self_61167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'self')
        # Obtaining the member '_hspace' of a type (line 334)
        _hspace_61168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 21), self_61167, '_hspace')
        # Assigning a type to the variable 'hspace' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'hspace', _hspace_61168)
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 336)
        self_61169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'self')
        # Obtaining the member '_wspace' of a type (line 336)
        _wspace_61170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), self_61169, '_wspace')
        # Getting the type of 'None' (line 336)
        None_61171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 31), 'None')
        # Applying the binary operator 'isnot' (line 336)
        result_is_not_61172 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), 'isnot', _wspace_61170, None_61171)
        
        # Testing the type of an if condition (line 336)
        if_condition_61173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_is_not_61172)
        # Assigning a type to the variable 'if_condition_61173' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_61173', if_condition_61173)
        # SSA begins for if statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 337):
        
        # Assigning a Attribute to a Name (line 337):
        # Getting the type of 'self' (line 337)
        self_61174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'self')
        # Obtaining the member '_wspace' of a type (line 337)
        _wspace_61175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 21), self_61174, '_wspace')
        # Assigning a type to the variable 'wspace' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'wspace', _wspace_61175)
        # SSA join for if statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 339):
        
        # Assigning a Call to a Name (line 339):
        
        # Call to get_position(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'fig' (line 339)
        fig_61179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 49), 'fig', False)
        # Processing the call keyword arguments (line 339)
        # Getting the type of 'False' (line 339)
        False_61180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 65), 'False', False)
        keyword_61181 = False_61180
        kwargs_61182 = {'return_all': keyword_61181}
        # Getting the type of 'self' (line 339)
        self_61176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'self', False)
        # Obtaining the member '_subplot_spec' of a type (line 339)
        _subplot_spec_61177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 17), self_61176, '_subplot_spec')
        # Obtaining the member 'get_position' of a type (line 339)
        get_position_61178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 17), _subplot_spec_61177, 'get_position')
        # Calling get_position(args, kwargs) (line 339)
        get_position_call_result_61183 = invoke(stypy.reporting.localization.Localization(__file__, 339, 17), get_position_61178, *[fig_61179], **kwargs_61182)
        
        # Assigning a type to the variable 'figbox' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'figbox', get_position_call_result_61183)
        
        # Assigning a Attribute to a Tuple (line 340):
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_61184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        # Getting the type of 'figbox' (line 340)
        figbox_61185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'figbox')
        # Obtaining the member 'extents' of a type (line 340)
        extents_61186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), figbox_61185, 'extents')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___61187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), extents_61186, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_61188 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___61187, int_61184)
        
        # Assigning a type to the variable 'tuple_var_assignment_60247' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60247', subscript_call_result_61188)
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_61189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        # Getting the type of 'figbox' (line 340)
        figbox_61190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'figbox')
        # Obtaining the member 'extents' of a type (line 340)
        extents_61191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), figbox_61190, 'extents')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___61192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), extents_61191, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_61193 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___61192, int_61189)
        
        # Assigning a type to the variable 'tuple_var_assignment_60248' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60248', subscript_call_result_61193)
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_61194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        # Getting the type of 'figbox' (line 340)
        figbox_61195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'figbox')
        # Obtaining the member 'extents' of a type (line 340)
        extents_61196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), figbox_61195, 'extents')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___61197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), extents_61196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_61198 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___61197, int_61194)
        
        # Assigning a type to the variable 'tuple_var_assignment_60249' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60249', subscript_call_result_61198)
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_61199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        # Getting the type of 'figbox' (line 340)
        figbox_61200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'figbox')
        # Obtaining the member 'extents' of a type (line 340)
        extents_61201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), figbox_61200, 'extents')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___61202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), extents_61201, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_61203 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___61202, int_61199)
        
        # Assigning a type to the variable 'tuple_var_assignment_60250' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60250', subscript_call_result_61203)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_60247' (line 340)
        tuple_var_assignment_60247_61204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60247')
        # Assigning a type to the variable 'left' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'left', tuple_var_assignment_60247_61204)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_60248' (line 340)
        tuple_var_assignment_60248_61205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60248')
        # Assigning a type to the variable 'bottom' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 14), 'bottom', tuple_var_assignment_60248_61205)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_60249' (line 340)
        tuple_var_assignment_60249_61206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60249')
        # Assigning a type to the variable 'right' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), 'right', tuple_var_assignment_60249_61206)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_60250' (line 340)
        tuple_var_assignment_60250_61207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_60250')
        # Assigning a type to the variable 'top' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 29), 'top', tuple_var_assignment_60250_61207)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 342, 8))
        
        # 'from matplotlib.figure import SubplotParams' statement (line 342)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_61208 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 342, 8), 'matplotlib.figure')

        if (type(import_61208) is not StypyTypeError):

            if (import_61208 != 'pyd_module'):
                __import__(import_61208)
                sys_modules_61209 = sys.modules[import_61208]
                import_from_module(stypy.reporting.localization.Localization(__file__, 342, 8), 'matplotlib.figure', sys_modules_61209.module_type_store, module_type_store, ['SubplotParams'])
                nest_module(stypy.reporting.localization.Localization(__file__, 342, 8), __file__, sys_modules_61209, sys_modules_61209.module_type_store, module_type_store)
            else:
                from matplotlib.figure import SubplotParams

                import_from_module(stypy.reporting.localization.Localization(__file__, 342, 8), 'matplotlib.figure', None, module_type_store, ['SubplotParams'], [SubplotParams])

        else:
            # Assigning a type to the variable 'matplotlib.figure' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'matplotlib.figure', import_61208)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to SubplotParams(...): (line 343)
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'left' (line 343)
        left_61211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'left', False)
        keyword_61212 = left_61211
        # Getting the type of 'right' (line 344)
        right_61213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), 'right', False)
        keyword_61214 = right_61213
        # Getting the type of 'bottom' (line 345)
        bottom_61215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 34), 'bottom', False)
        keyword_61216 = bottom_61215
        # Getting the type of 'top' (line 346)
        top_61217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 31), 'top', False)
        keyword_61218 = top_61217
        # Getting the type of 'wspace' (line 347)
        wspace_61219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 34), 'wspace', False)
        keyword_61220 = wspace_61219
        # Getting the type of 'hspace' (line 348)
        hspace_61221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), 'hspace', False)
        keyword_61222 = hspace_61221
        kwargs_61223 = {'right': keyword_61214, 'bottom': keyword_61216, 'top': keyword_61218, 'wspace': keyword_61220, 'hspace': keyword_61222, 'left': keyword_61212}
        # Getting the type of 'SubplotParams' (line 343)
        SubplotParams_61210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 13), 'SubplotParams', False)
        # Calling SubplotParams(args, kwargs) (line 343)
        SubplotParams_call_result_61224 = invoke(stypy.reporting.localization.Localization(__file__, 343, 13), SubplotParams_61210, *[], **kwargs_61223)
        
        # Assigning a type to the variable 'sp' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'sp', SubplotParams_call_result_61224)
        # Getting the type of 'sp' (line 350)
        sp_61225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 15), 'sp')
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', sp_61225)
        
        # ################# End of 'get_subplot_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_subplot_params' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_61226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_subplot_params'
        return stypy_return_type_61226


    @norecursion
    def get_topmost_subplotspec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_topmost_subplotspec'
        module_type_store = module_type_store.open_function_context('get_topmost_subplotspec', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_localization', localization)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_type_store', module_type_store)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_function_name', 'GridSpecFromSubplotSpec.get_topmost_subplotspec')
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_param_names_list', [])
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_varargs_param_name', None)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_call_defaults', defaults)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_call_varargs', varargs)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GridSpecFromSubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GridSpecFromSubplotSpec.get_topmost_subplotspec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_topmost_subplotspec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_topmost_subplotspec(...)' code ##################

        unicode_61227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'unicode', u'Get the topmost SubplotSpec instance associated with the subplot.')
        
        # Call to get_topmost_subplotspec(...): (line 355)
        # Processing the call keyword arguments (line 355)
        kwargs_61231 = {}
        # Getting the type of 'self' (line 355)
        self_61228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), 'self', False)
        # Obtaining the member '_subplot_spec' of a type (line 355)
        _subplot_spec_61229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), self_61228, '_subplot_spec')
        # Obtaining the member 'get_topmost_subplotspec' of a type (line 355)
        get_topmost_subplotspec_61230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), _subplot_spec_61229, 'get_topmost_subplotspec')
        # Calling get_topmost_subplotspec(args, kwargs) (line 355)
        get_topmost_subplotspec_call_result_61232 = invoke(stypy.reporting.localization.Localization(__file__, 355, 15), get_topmost_subplotspec_61230, *[], **kwargs_61231)
        
        # Assigning a type to the variable 'stypy_return_type' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'stypy_return_type', get_topmost_subplotspec_call_result_61232)
        
        # ################# End of 'get_topmost_subplotspec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_topmost_subplotspec' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_61233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_topmost_subplotspec'
        return stypy_return_type_61233


# Assigning a type to the variable 'GridSpecFromSubplotSpec' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'GridSpecFromSubplotSpec', GridSpecFromSubplotSpec)
# Declaration of the 'SubplotSpec' class

class SubplotSpec(object, ):
    unicode_61234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, (-1)), 'unicode', u'Specifies the location of the subplot in the given `GridSpec`.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 362)
        None_61235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 44), 'None')
        defaults = [None_61235]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 362, 4, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.__init__', ['gridspec', 'num1', 'num2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['gridspec', 'num1', 'num2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_61236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, (-1)), 'unicode', u'\n        The subplot will occupy the num1-th cell of the given\n        gridspec.  If num2 is provided, the subplot will span between\n        num1-th cell and num2-th cell.\n\n        The index starts from 0.\n        ')
        
        # Assigning a Call to a Tuple (line 371):
        
        # Assigning a Call to a Name:
        
        # Call to get_geometry(...): (line 371)
        # Processing the call keyword arguments (line 371)
        kwargs_61239 = {}
        # Getting the type of 'gridspec' (line 371)
        gridspec_61237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'gridspec', False)
        # Obtaining the member 'get_geometry' of a type (line 371)
        get_geometry_61238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 21), gridspec_61237, 'get_geometry')
        # Calling get_geometry(args, kwargs) (line 371)
        get_geometry_call_result_61240 = invoke(stypy.reporting.localization.Localization(__file__, 371, 21), get_geometry_61238, *[], **kwargs_61239)
        
        # Assigning a type to the variable 'call_assignment_60251' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60251', get_geometry_call_result_61240)
        
        # Assigning a Call to a Name (line 371):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61244 = {}
        # Getting the type of 'call_assignment_60251' (line 371)
        call_assignment_60251_61241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60251', False)
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___61242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), call_assignment_60251_61241, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61245 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61242, *[int_61243], **kwargs_61244)
        
        # Assigning a type to the variable 'call_assignment_60252' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60252', getitem___call_result_61245)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'call_assignment_60252' (line 371)
        call_assignment_60252_61246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60252')
        # Assigning a type to the variable 'rows' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'rows', call_assignment_60252_61246)
        
        # Assigning a Call to a Name (line 371):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61250 = {}
        # Getting the type of 'call_assignment_60251' (line 371)
        call_assignment_60251_61247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60251', False)
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___61248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), call_assignment_60251_61247, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61251 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61248, *[int_61249], **kwargs_61250)
        
        # Assigning a type to the variable 'call_assignment_60253' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60253', getitem___call_result_61251)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'call_assignment_60253' (line 371)
        call_assignment_60253_61252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'call_assignment_60253')
        # Assigning a type to the variable 'cols' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 14), 'cols', call_assignment_60253_61252)
        
        # Assigning a BinOp to a Name (line 372):
        
        # Assigning a BinOp to a Name (line 372):
        # Getting the type of 'rows' (line 372)
        rows_61253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'rows')
        # Getting the type of 'cols' (line 372)
        cols_61254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'cols')
        # Applying the binary operator '*' (line 372)
        result_mul_61255 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 16), '*', rows_61253, cols_61254)
        
        # Assigning a type to the variable 'total' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'total', result_mul_61255)
        
        # Assigning a Name to a Attribute (line 374):
        
        # Assigning a Name to a Attribute (line 374):
        # Getting the type of 'gridspec' (line 374)
        gridspec_61256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 25), 'gridspec')
        # Getting the type of 'self' (line 374)
        self_61257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'self')
        # Setting the type of the member '_gridspec' of a type (line 374)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), self_61257, '_gridspec', gridspec_61256)
        
        # Assigning a Name to a Attribute (line 375):
        
        # Assigning a Name to a Attribute (line 375):
        # Getting the type of 'num1' (line 375)
        num1_61258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 20), 'num1')
        # Getting the type of 'self' (line 375)
        self_61259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'self')
        # Setting the type of the member 'num1' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), self_61259, 'num1', num1_61258)
        
        # Assigning a Name to a Attribute (line 376):
        
        # Assigning a Name to a Attribute (line 376):
        # Getting the type of 'num2' (line 376)
        num2_61260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 20), 'num2')
        # Getting the type of 'self' (line 376)
        self_61261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'self')
        # Setting the type of the member 'num2' of a type (line 376)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), self_61261, 'num2', num2_61260)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_gridspec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_gridspec'
        module_type_store = module_type_store.open_function_context('get_gridspec', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_localization', localization)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_function_name', 'SubplotSpec.get_gridspec')
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotSpec.get_gridspec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.get_gridspec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_gridspec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_gridspec(...)' code ##################

        # Getting the type of 'self' (line 379)
        self_61262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'self')
        # Obtaining the member '_gridspec' of a type (line 379)
        _gridspec_61263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), self_61262, '_gridspec')
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', _gridspec_61263)
        
        # ################# End of 'get_gridspec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_gridspec' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_61264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61264)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_gridspec'
        return stypy_return_type_61264


    @norecursion
    def get_geometry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_geometry'
        module_type_store = module_type_store.open_function_context('get_geometry', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_localization', localization)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_function_name', 'SubplotSpec.get_geometry')
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotSpec.get_geometry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.get_geometry', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_geometry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_geometry(...)' code ##################

        unicode_61265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, (-1)), 'unicode', u'Get the subplot geometry (``n_rows, n_cols, row, col``).\n\n        Unlike SuplorParams, indexes are 0-based.\n        ')
        
        # Assigning a Call to a Tuple (line 387):
        
        # Assigning a Call to a Name:
        
        # Call to get_geometry(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_61271 = {}
        
        # Call to get_gridspec(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_61268 = {}
        # Getting the type of 'self' (line 387)
        self_61266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 21), 'self', False)
        # Obtaining the member 'get_gridspec' of a type (line 387)
        get_gridspec_61267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 21), self_61266, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 387)
        get_gridspec_call_result_61269 = invoke(stypy.reporting.localization.Localization(__file__, 387, 21), get_gridspec_61267, *[], **kwargs_61268)
        
        # Obtaining the member 'get_geometry' of a type (line 387)
        get_geometry_61270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 21), get_gridspec_call_result_61269, 'get_geometry')
        # Calling get_geometry(args, kwargs) (line 387)
        get_geometry_call_result_61272 = invoke(stypy.reporting.localization.Localization(__file__, 387, 21), get_geometry_61270, *[], **kwargs_61271)
        
        # Assigning a type to the variable 'call_assignment_60254' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60254', get_geometry_call_result_61272)
        
        # Assigning a Call to a Name (line 387):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61276 = {}
        # Getting the type of 'call_assignment_60254' (line 387)
        call_assignment_60254_61273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60254', False)
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___61274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), call_assignment_60254_61273, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61277 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61274, *[int_61275], **kwargs_61276)
        
        # Assigning a type to the variable 'call_assignment_60255' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60255', getitem___call_result_61277)
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'call_assignment_60255' (line 387)
        call_assignment_60255_61278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60255')
        # Assigning a type to the variable 'rows' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'rows', call_assignment_60255_61278)
        
        # Assigning a Call to a Name (line 387):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61282 = {}
        # Getting the type of 'call_assignment_60254' (line 387)
        call_assignment_60254_61279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60254', False)
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___61280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), call_assignment_60254_61279, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61283 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61280, *[int_61281], **kwargs_61282)
        
        # Assigning a type to the variable 'call_assignment_60256' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60256', getitem___call_result_61283)
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'call_assignment_60256' (line 387)
        call_assignment_60256_61284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'call_assignment_60256')
        # Assigning a type to the variable 'cols' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 14), 'cols', call_assignment_60256_61284)
        
        # Obtaining an instance of the builtin type 'tuple' (line 388)
        tuple_61285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 388)
        # Adding element type (line 388)
        # Getting the type of 'rows' (line 388)
        rows_61286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'rows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 15), tuple_61285, rows_61286)
        # Adding element type (line 388)
        # Getting the type of 'cols' (line 388)
        cols_61287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 21), 'cols')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 15), tuple_61285, cols_61287)
        # Adding element type (line 388)
        # Getting the type of 'self' (line 388)
        self_61288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 27), 'self')
        # Obtaining the member 'num1' of a type (line 388)
        num1_61289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 27), self_61288, 'num1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 15), tuple_61285, num1_61289)
        # Adding element type (line 388)
        # Getting the type of 'self' (line 388)
        self_61290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 38), 'self')
        # Obtaining the member 'num2' of a type (line 388)
        num2_61291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 38), self_61290, 'num2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 15), tuple_61285, num2_61291)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', tuple_61285)
        
        # ################# End of 'get_geometry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_geometry' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_61292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_geometry'
        return stypy_return_type_61292


    @norecursion
    def get_position(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 391)
        False_61293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 43), 'False')
        defaults = [False_61293]
        # Create a new context for function 'get_position'
        module_type_store = module_type_store.open_function_context('get_position', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotSpec.get_position.__dict__.__setitem__('stypy_localization', localization)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_function_name', 'SubplotSpec.get_position')
        SubplotSpec.get_position.__dict__.__setitem__('stypy_param_names_list', ['fig', 'return_all'])
        SubplotSpec.get_position.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotSpec.get_position.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.get_position', ['fig', 'return_all'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_position', localization, ['fig', 'return_all'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_position(...)' code ##################

        unicode_61294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, (-1)), 'unicode', u'Update the subplot position from ``fig.subplotpars``.\n        ')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to get_gridspec(...): (line 395)
        # Processing the call keyword arguments (line 395)
        kwargs_61297 = {}
        # Getting the type of 'self' (line 395)
        self_61295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'self', False)
        # Obtaining the member 'get_gridspec' of a type (line 395)
        get_gridspec_61296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), self_61295, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 395)
        get_gridspec_call_result_61298 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), get_gridspec_61296, *[], **kwargs_61297)
        
        # Assigning a type to the variable 'gridspec' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'gridspec', get_gridspec_call_result_61298)
        
        # Assigning a Call to a Tuple (line 396):
        
        # Assigning a Call to a Name:
        
        # Call to get_geometry(...): (line 396)
        # Processing the call keyword arguments (line 396)
        kwargs_61301 = {}
        # Getting the type of 'gridspec' (line 396)
        gridspec_61299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'gridspec', False)
        # Obtaining the member 'get_geometry' of a type (line 396)
        get_geometry_61300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 23), gridspec_61299, 'get_geometry')
        # Calling get_geometry(args, kwargs) (line 396)
        get_geometry_call_result_61302 = invoke(stypy.reporting.localization.Localization(__file__, 396, 23), get_geometry_61300, *[], **kwargs_61301)
        
        # Assigning a type to the variable 'call_assignment_60257' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60257', get_geometry_call_result_61302)
        
        # Assigning a Call to a Name (line 396):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61306 = {}
        # Getting the type of 'call_assignment_60257' (line 396)
        call_assignment_60257_61303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60257', False)
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___61304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), call_assignment_60257_61303, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61307 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61304, *[int_61305], **kwargs_61306)
        
        # Assigning a type to the variable 'call_assignment_60258' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60258', getitem___call_result_61307)
        
        # Assigning a Name to a Name (line 396):
        # Getting the type of 'call_assignment_60258' (line 396)
        call_assignment_60258_61308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60258')
        # Assigning a type to the variable 'nrows' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'nrows', call_assignment_60258_61308)
        
        # Assigning a Call to a Name (line 396):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61312 = {}
        # Getting the type of 'call_assignment_60257' (line 396)
        call_assignment_60257_61309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60257', False)
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___61310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), call_assignment_60257_61309, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61313 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61310, *[int_61311], **kwargs_61312)
        
        # Assigning a type to the variable 'call_assignment_60259' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60259', getitem___call_result_61313)
        
        # Assigning a Name to a Name (line 396):
        # Getting the type of 'call_assignment_60259' (line 396)
        call_assignment_60259_61314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'call_assignment_60259')
        # Assigning a type to the variable 'ncols' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 15), 'ncols', call_assignment_60259_61314)
        
        # Assigning a Call to a Tuple (line 398):
        
        # Assigning a Call to a Name:
        
        # Call to get_grid_positions(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'fig' (line 399)
        fig_61317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 40), 'fig', False)
        # Processing the call keyword arguments (line 399)
        kwargs_61318 = {}
        # Getting the type of 'gridspec' (line 399)
        gridspec_61315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'gridspec', False)
        # Obtaining the member 'get_grid_positions' of a type (line 399)
        get_grid_positions_61316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), gridspec_61315, 'get_grid_positions')
        # Calling get_grid_positions(args, kwargs) (line 399)
        get_grid_positions_call_result_61319 = invoke(stypy.reporting.localization.Localization(__file__, 399, 12), get_grid_positions_61316, *[fig_61317], **kwargs_61318)
        
        # Assigning a type to the variable 'call_assignment_60260' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60260', get_grid_positions_call_result_61319)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61323 = {}
        # Getting the type of 'call_assignment_60260' (line 398)
        call_assignment_60260_61320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60260', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___61321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_60260_61320, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61324 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61321, *[int_61322], **kwargs_61323)
        
        # Assigning a type to the variable 'call_assignment_60261' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60261', getitem___call_result_61324)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'call_assignment_60261' (line 398)
        call_assignment_60261_61325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60261')
        # Assigning a type to the variable 'figBottoms' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'figBottoms', call_assignment_60261_61325)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61329 = {}
        # Getting the type of 'call_assignment_60260' (line 398)
        call_assignment_60260_61326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60260', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___61327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_60260_61326, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61330 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61327, *[int_61328], **kwargs_61329)
        
        # Assigning a type to the variable 'call_assignment_60262' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60262', getitem___call_result_61330)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'call_assignment_60262' (line 398)
        call_assignment_60262_61331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60262')
        # Assigning a type to the variable 'figTops' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'figTops', call_assignment_60262_61331)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61335 = {}
        # Getting the type of 'call_assignment_60260' (line 398)
        call_assignment_60260_61332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60260', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___61333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_60260_61332, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61336 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61333, *[int_61334], **kwargs_61335)
        
        # Assigning a type to the variable 'call_assignment_60263' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60263', getitem___call_result_61336)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'call_assignment_60263' (line 398)
        call_assignment_60263_61337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60263')
        # Assigning a type to the variable 'figLefts' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'figLefts', call_assignment_60263_61337)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61341 = {}
        # Getting the type of 'call_assignment_60260' (line 398)
        call_assignment_60260_61338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60260', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___61339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_60260_61338, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61342 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61339, *[int_61340], **kwargs_61341)
        
        # Assigning a type to the variable 'call_assignment_60264' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60264', getitem___call_result_61342)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'call_assignment_60264' (line 398)
        call_assignment_60264_61343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_60264')
        # Assigning a type to the variable 'figRights' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 39), 'figRights', call_assignment_60264_61343)
        
        # Assigning a Call to a Tuple (line 401):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'self' (line 401)
        self_61345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'self', False)
        # Obtaining the member 'num1' of a type (line 401)
        num1_61346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 33), self_61345, 'num1')
        # Getting the type of 'ncols' (line 401)
        ncols_61347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 44), 'ncols', False)
        # Processing the call keyword arguments (line 401)
        kwargs_61348 = {}
        # Getting the type of 'divmod' (line 401)
        divmod_61344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 26), 'divmod', False)
        # Calling divmod(args, kwargs) (line 401)
        divmod_call_result_61349 = invoke(stypy.reporting.localization.Localization(__file__, 401, 26), divmod_61344, *[num1_61346, ncols_61347], **kwargs_61348)
        
        # Assigning a type to the variable 'call_assignment_60265' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60265', divmod_call_result_61349)
        
        # Assigning a Call to a Name (line 401):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61353 = {}
        # Getting the type of 'call_assignment_60265' (line 401)
        call_assignment_60265_61350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60265', False)
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___61351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), call_assignment_60265_61350, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61354 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61351, *[int_61352], **kwargs_61353)
        
        # Assigning a type to the variable 'call_assignment_60266' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60266', getitem___call_result_61354)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'call_assignment_60266' (line 401)
        call_assignment_60266_61355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60266')
        # Assigning a type to the variable 'rowNum' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'rowNum', call_assignment_60266_61355)
        
        # Assigning a Call to a Name (line 401):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61359 = {}
        # Getting the type of 'call_assignment_60265' (line 401)
        call_assignment_60265_61356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60265', False)
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___61357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), call_assignment_60265_61356, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61360 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61357, *[int_61358], **kwargs_61359)
        
        # Assigning a type to the variable 'call_assignment_60267' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60267', getitem___call_result_61360)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'call_assignment_60267' (line 401)
        call_assignment_60267_61361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'call_assignment_60267')
        # Assigning a type to the variable 'colNum' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'colNum', call_assignment_60267_61361)
        
        # Assigning a Subscript to a Name (line 402):
        
        # Assigning a Subscript to a Name (line 402):
        
        # Obtaining the type of the subscript
        # Getting the type of 'rowNum' (line 402)
        rowNum_61362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 31), 'rowNum')
        # Getting the type of 'figBottoms' (line 402)
        figBottoms_61363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'figBottoms')
        # Obtaining the member '__getitem__' of a type (line 402)
        getitem___61364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), figBottoms_61363, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 402)
        subscript_call_result_61365 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), getitem___61364, rowNum_61362)
        
        # Assigning a type to the variable 'figBottom' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'figBottom', subscript_call_result_61365)
        
        # Assigning a Subscript to a Name (line 403):
        
        # Assigning a Subscript to a Name (line 403):
        
        # Obtaining the type of the subscript
        # Getting the type of 'rowNum' (line 403)
        rowNum_61366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 25), 'rowNum')
        # Getting the type of 'figTops' (line 403)
        figTops_61367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'figTops')
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___61368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 17), figTops_61367, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 403)
        subscript_call_result_61369 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), getitem___61368, rowNum_61366)
        
        # Assigning a type to the variable 'figTop' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'figTop', subscript_call_result_61369)
        
        # Assigning a Subscript to a Name (line 404):
        
        # Assigning a Subscript to a Name (line 404):
        
        # Obtaining the type of the subscript
        # Getting the type of 'colNum' (line 404)
        colNum_61370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 27), 'colNum')
        # Getting the type of 'figLefts' (line 404)
        figLefts_61371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 18), 'figLefts')
        # Obtaining the member '__getitem__' of a type (line 404)
        getitem___61372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 18), figLefts_61371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 404)
        subscript_call_result_61373 = invoke(stypy.reporting.localization.Localization(__file__, 404, 18), getitem___61372, colNum_61370)
        
        # Assigning a type to the variable 'figLeft' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'figLeft', subscript_call_result_61373)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        # Getting the type of 'colNum' (line 405)
        colNum_61374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'colNum')
        # Getting the type of 'figRights' (line 405)
        figRights_61375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 19), 'figRights')
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___61376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 19), figRights_61375, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_61377 = invoke(stypy.reporting.localization.Localization(__file__, 405, 19), getitem___61376, colNum_61374)
        
        # Assigning a type to the variable 'figRight' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'figRight', subscript_call_result_61377)
        
        
        # Getting the type of 'self' (line 407)
        self_61378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'self')
        # Obtaining the member 'num2' of a type (line 407)
        num2_61379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 11), self_61378, 'num2')
        # Getting the type of 'None' (line 407)
        None_61380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'None')
        # Applying the binary operator 'isnot' (line 407)
        result_is_not_61381 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 11), 'isnot', num2_61379, None_61380)
        
        # Testing the type of an if condition (line 407)
        if_condition_61382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 8), result_is_not_61381)
        # Assigning a type to the variable 'if_condition_61382' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'if_condition_61382', if_condition_61382)
        # SSA begins for if statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 409):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'self' (line 409)
        self_61384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'self', False)
        # Obtaining the member 'num2' of a type (line 409)
        num2_61385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 39), self_61384, 'num2')
        # Getting the type of 'ncols' (line 409)
        ncols_61386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 50), 'ncols', False)
        # Processing the call keyword arguments (line 409)
        kwargs_61387 = {}
        # Getting the type of 'divmod' (line 409)
        divmod_61383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'divmod', False)
        # Calling divmod(args, kwargs) (line 409)
        divmod_call_result_61388 = invoke(stypy.reporting.localization.Localization(__file__, 409, 32), divmod_61383, *[num2_61385, ncols_61386], **kwargs_61387)
        
        # Assigning a type to the variable 'call_assignment_60268' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60268', divmod_call_result_61388)
        
        # Assigning a Call to a Name (line 409):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 12), 'int')
        # Processing the call keyword arguments
        kwargs_61392 = {}
        # Getting the type of 'call_assignment_60268' (line 409)
        call_assignment_60268_61389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60268', False)
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___61390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), call_assignment_60268_61389, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61393 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61390, *[int_61391], **kwargs_61392)
        
        # Assigning a type to the variable 'call_assignment_60269' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60269', getitem___call_result_61393)
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'call_assignment_60269' (line 409)
        call_assignment_60269_61394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60269')
        # Assigning a type to the variable 'rowNum2' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'rowNum2', call_assignment_60269_61394)
        
        # Assigning a Call to a Name (line 409):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 12), 'int')
        # Processing the call keyword arguments
        kwargs_61398 = {}
        # Getting the type of 'call_assignment_60268' (line 409)
        call_assignment_60268_61395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60268', False)
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___61396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), call_assignment_60268_61395, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61399 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61396, *[int_61397], **kwargs_61398)
        
        # Assigning a type to the variable 'call_assignment_60270' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60270', getitem___call_result_61399)
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'call_assignment_60270' (line 409)
        call_assignment_60270_61400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'call_assignment_60270')
        # Assigning a type to the variable 'colNum2' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'colNum2', call_assignment_60270_61400)
        
        # Assigning a Subscript to a Name (line 410):
        
        # Assigning a Subscript to a Name (line 410):
        
        # Obtaining the type of the subscript
        # Getting the type of 'rowNum2' (line 410)
        rowNum2_61401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'rowNum2')
        # Getting the type of 'figBottoms' (line 410)
        figBottoms_61402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'figBottoms')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___61403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 25), figBottoms_61402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_61404 = invoke(stypy.reporting.localization.Localization(__file__, 410, 25), getitem___61403, rowNum2_61401)
        
        # Assigning a type to the variable 'figBottom2' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'figBottom2', subscript_call_result_61404)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        # Getting the type of 'rowNum2' (line 411)
        rowNum2_61405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 30), 'rowNum2')
        # Getting the type of 'figTops' (line 411)
        figTops_61406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'figTops')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___61407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 22), figTops_61406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_61408 = invoke(stypy.reporting.localization.Localization(__file__, 411, 22), getitem___61407, rowNum2_61405)
        
        # Assigning a type to the variable 'figTop2' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'figTop2', subscript_call_result_61408)
        
        # Assigning a Subscript to a Name (line 412):
        
        # Assigning a Subscript to a Name (line 412):
        
        # Obtaining the type of the subscript
        # Getting the type of 'colNum2' (line 412)
        colNum2_61409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'colNum2')
        # Getting the type of 'figLefts' (line 412)
        figLefts_61410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), 'figLefts')
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___61411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 23), figLefts_61410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_61412 = invoke(stypy.reporting.localization.Localization(__file__, 412, 23), getitem___61411, colNum2_61409)
        
        # Assigning a type to the variable 'figLeft2' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'figLeft2', subscript_call_result_61412)
        
        # Assigning a Subscript to a Name (line 413):
        
        # Assigning a Subscript to a Name (line 413):
        
        # Obtaining the type of the subscript
        # Getting the type of 'colNum2' (line 413)
        colNum2_61413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'colNum2')
        # Getting the type of 'figRights' (line 413)
        figRights_61414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 24), 'figRights')
        # Obtaining the member '__getitem__' of a type (line 413)
        getitem___61415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 24), figRights_61414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 413)
        subscript_call_result_61416 = invoke(stypy.reporting.localization.Localization(__file__, 413, 24), getitem___61415, colNum2_61413)
        
        # Assigning a type to the variable 'figRight2' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'figRight2', subscript_call_result_61416)
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to min(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'figBottom' (line 415)
        figBottom_61418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 28), 'figBottom', False)
        # Getting the type of 'figBottom2' (line 415)
        figBottom2_61419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 39), 'figBottom2', False)
        # Processing the call keyword arguments (line 415)
        kwargs_61420 = {}
        # Getting the type of 'min' (line 415)
        min_61417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 24), 'min', False)
        # Calling min(args, kwargs) (line 415)
        min_call_result_61421 = invoke(stypy.reporting.localization.Localization(__file__, 415, 24), min_61417, *[figBottom_61418, figBottom2_61419], **kwargs_61420)
        
        # Assigning a type to the variable 'figBottom' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'figBottom', min_call_result_61421)
        
        # Assigning a Call to a Name (line 416):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to min(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'figLeft' (line 416)
        figLeft_61423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 26), 'figLeft', False)
        # Getting the type of 'figLeft2' (line 416)
        figLeft2_61424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 35), 'figLeft2', False)
        # Processing the call keyword arguments (line 416)
        kwargs_61425 = {}
        # Getting the type of 'min' (line 416)
        min_61422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'min', False)
        # Calling min(args, kwargs) (line 416)
        min_call_result_61426 = invoke(stypy.reporting.localization.Localization(__file__, 416, 22), min_61422, *[figLeft_61423, figLeft2_61424], **kwargs_61425)
        
        # Assigning a type to the variable 'figLeft' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'figLeft', min_call_result_61426)
        
        # Assigning a Call to a Name (line 417):
        
        # Assigning a Call to a Name (line 417):
        
        # Call to max(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'figTop' (line 417)
        figTop_61428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'figTop', False)
        # Getting the type of 'figTop2' (line 417)
        figTop2_61429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 33), 'figTop2', False)
        # Processing the call keyword arguments (line 417)
        kwargs_61430 = {}
        # Getting the type of 'max' (line 417)
        max_61427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), 'max', False)
        # Calling max(args, kwargs) (line 417)
        max_call_result_61431 = invoke(stypy.reporting.localization.Localization(__file__, 417, 21), max_61427, *[figTop_61428, figTop2_61429], **kwargs_61430)
        
        # Assigning a type to the variable 'figTop' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'figTop', max_call_result_61431)
        
        # Assigning a Call to a Name (line 418):
        
        # Assigning a Call to a Name (line 418):
        
        # Call to max(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'figRight' (line 418)
        figRight_61433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 27), 'figRight', False)
        # Getting the type of 'figRight2' (line 418)
        figRight2_61434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 37), 'figRight2', False)
        # Processing the call keyword arguments (line 418)
        kwargs_61435 = {}
        # Getting the type of 'max' (line 418)
        max_61432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'max', False)
        # Calling max(args, kwargs) (line 418)
        max_call_result_61436 = invoke(stypy.reporting.localization.Localization(__file__, 418, 23), max_61432, *[figRight_61433, figRight2_61434], **kwargs_61435)
        
        # Assigning a type to the variable 'figRight' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'figRight', max_call_result_61436)
        # SSA join for if statement (line 407)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 420):
        
        # Assigning a Call to a Name (line 420):
        
        # Call to from_extents(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'figLeft' (line 420)
        figLeft_61440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 47), 'figLeft', False)
        # Getting the type of 'figBottom' (line 420)
        figBottom_61441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 56), 'figBottom', False)
        # Getting the type of 'figRight' (line 421)
        figRight_61442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 47), 'figRight', False)
        # Getting the type of 'figTop' (line 421)
        figTop_61443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 57), 'figTop', False)
        # Processing the call keyword arguments (line 420)
        kwargs_61444 = {}
        # Getting the type of 'mtransforms' (line 420)
        mtransforms_61437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 'mtransforms', False)
        # Obtaining the member 'Bbox' of a type (line 420)
        Bbox_61438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 17), mtransforms_61437, 'Bbox')
        # Obtaining the member 'from_extents' of a type (line 420)
        from_extents_61439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 17), Bbox_61438, 'from_extents')
        # Calling from_extents(args, kwargs) (line 420)
        from_extents_call_result_61445 = invoke(stypy.reporting.localization.Localization(__file__, 420, 17), from_extents_61439, *[figLeft_61440, figBottom_61441, figRight_61442, figTop_61443], **kwargs_61444)
        
        # Assigning a type to the variable 'figbox' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'figbox', from_extents_call_result_61445)
        
        # Getting the type of 'return_all' (line 423)
        return_all_61446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'return_all')
        # Testing the type of an if condition (line 423)
        if_condition_61447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), return_all_61446)
        # Assigning a type to the variable 'if_condition_61447' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_61447', if_condition_61447)
        # SSA begins for if statement (line 423)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 424)
        tuple_61448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 424)
        # Adding element type (line 424)
        # Getting the type of 'figbox' (line 424)
        figbox_61449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'figbox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 19), tuple_61448, figbox_61449)
        # Adding element type (line 424)
        # Getting the type of 'rowNum' (line 424)
        rowNum_61450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'rowNum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 19), tuple_61448, rowNum_61450)
        # Adding element type (line 424)
        # Getting the type of 'colNum' (line 424)
        colNum_61451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 35), 'colNum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 19), tuple_61448, colNum_61451)
        # Adding element type (line 424)
        # Getting the type of 'nrows' (line 424)
        nrows_61452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 43), 'nrows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 19), tuple_61448, nrows_61452)
        # Adding element type (line 424)
        # Getting the type of 'ncols' (line 424)
        ncols_61453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 50), 'ncols')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 19), tuple_61448, ncols_61453)
        
        # Assigning a type to the variable 'stypy_return_type' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'stypy_return_type', tuple_61448)
        # SSA branch for the else part of an if statement (line 423)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'figbox' (line 426)
        figbox_61454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 19), 'figbox')
        # Assigning a type to the variable 'stypy_return_type' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'stypy_return_type', figbox_61454)
        # SSA join for if statement (line 423)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_position(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_position' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_61455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61455)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_position'
        return stypy_return_type_61455


    @norecursion
    def get_topmost_subplotspec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_topmost_subplotspec'
        module_type_store = module_type_store.open_function_context('get_topmost_subplotspec', 428, 4, False)
        # Assigning a type to the variable 'self' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_localization', localization)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_function_name', 'SubplotSpec.get_topmost_subplotspec')
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotSpec.get_topmost_subplotspec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.get_topmost_subplotspec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_topmost_subplotspec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_topmost_subplotspec(...)' code ##################

        unicode_61456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 8), 'unicode', u'get the topmost SubplotSpec instance associated with the subplot')
        
        # Assigning a Call to a Name (line 430):
        
        # Assigning a Call to a Name (line 430):
        
        # Call to get_gridspec(...): (line 430)
        # Processing the call keyword arguments (line 430)
        kwargs_61459 = {}
        # Getting the type of 'self' (line 430)
        self_61457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'self', False)
        # Obtaining the member 'get_gridspec' of a type (line 430)
        get_gridspec_61458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 19), self_61457, 'get_gridspec')
        # Calling get_gridspec(args, kwargs) (line 430)
        get_gridspec_call_result_61460 = invoke(stypy.reporting.localization.Localization(__file__, 430, 19), get_gridspec_61458, *[], **kwargs_61459)
        
        # Assigning a type to the variable 'gridspec' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'gridspec', get_gridspec_call_result_61460)
        
        # Type idiom detected: calculating its left and rigth part (line 431)
        unicode_61461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 29), 'unicode', u'get_topmost_subplotspec')
        # Getting the type of 'gridspec' (line 431)
        gridspec_61462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'gridspec')
        
        (may_be_61463, more_types_in_union_61464) = may_provide_member(unicode_61461, gridspec_61462)

        if may_be_61463:

            if more_types_in_union_61464:
                # Runtime conditional SSA (line 431)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'gridspec' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'gridspec', remove_not_member_provider_from_union(gridspec_61462, u'get_topmost_subplotspec'))
            
            # Call to get_topmost_subplotspec(...): (line 432)
            # Processing the call keyword arguments (line 432)
            kwargs_61467 = {}
            # Getting the type of 'gridspec' (line 432)
            gridspec_61465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 'gridspec', False)
            # Obtaining the member 'get_topmost_subplotspec' of a type (line 432)
            get_topmost_subplotspec_61466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 19), gridspec_61465, 'get_topmost_subplotspec')
            # Calling get_topmost_subplotspec(args, kwargs) (line 432)
            get_topmost_subplotspec_call_result_61468 = invoke(stypy.reporting.localization.Localization(__file__, 432, 19), get_topmost_subplotspec_61466, *[], **kwargs_61467)
            
            # Assigning a type to the variable 'stypy_return_type' (line 432)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'stypy_return_type', get_topmost_subplotspec_call_result_61468)

            if more_types_in_union_61464:
                # Runtime conditional SSA for else branch (line 431)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_61463) or more_types_in_union_61464):
            # Assigning a type to the variable 'gridspec' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'gridspec', remove_member_provider_from_union(gridspec_61462, u'get_topmost_subplotspec'))
            # Getting the type of 'self' (line 434)
            self_61469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'stypy_return_type', self_61469)

            if (may_be_61463 and more_types_in_union_61464):
                # SSA join for if statement (line 431)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_topmost_subplotspec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_topmost_subplotspec' in the type store
        # Getting the type of 'stypy_return_type' (line 428)
        stypy_return_type_61470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_topmost_subplotspec'
        return stypy_return_type_61470


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'SubplotSpec.stypy__eq__')
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotSpec.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 438)
        tuple_61471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 438)
        # Adding element type (line 438)
        # Getting the type of 'self' (line 438)
        self_61472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 17), 'self')
        # Obtaining the member '_gridspec' of a type (line 438)
        _gridspec_61473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 17), self_61472, '_gridspec')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 17), tuple_61471, _gridspec_61473)
        # Adding element type (line 438)
        # Getting the type of 'self' (line 438)
        self_61474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'self')
        # Obtaining the member 'num1' of a type (line 438)
        num1_61475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 33), self_61474, 'num1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 17), tuple_61471, num1_61475)
        # Adding element type (line 438)
        # Getting the type of 'self' (line 438)
        self_61476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 44), 'self')
        # Obtaining the member 'num2' of a type (line 438)
        num2_61477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 44), self_61476, 'num2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 17), tuple_61471, num2_61477)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 439)
        tuple_61478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 439)
        # Adding element type (line 439)
        
        # Call to getattr(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'other' (line 439)
        other_61480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'other', False)
        unicode_61481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 35), 'unicode', u'_gridspec')
        
        # Call to object(...): (line 439)
        # Processing the call keyword arguments (line 439)
        kwargs_61483 = {}
        # Getting the type of 'object' (line 439)
        object_61482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 48), 'object', False)
        # Calling object(args, kwargs) (line 439)
        object_call_result_61484 = invoke(stypy.reporting.localization.Localization(__file__, 439, 48), object_61482, *[], **kwargs_61483)
        
        # Processing the call keyword arguments (line 439)
        kwargs_61485 = {}
        # Getting the type of 'getattr' (line 439)
        getattr_61479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 439)
        getattr_call_result_61486 = invoke(stypy.reporting.localization.Localization(__file__, 439, 20), getattr_61479, *[other_61480, unicode_61481, object_call_result_61484], **kwargs_61485)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 20), tuple_61478, getattr_call_result_61486)
        # Adding element type (line 439)
        
        # Call to getattr(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'other' (line 440)
        other_61488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 28), 'other', False)
        unicode_61489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 35), 'unicode', u'num1')
        
        # Call to object(...): (line 440)
        # Processing the call keyword arguments (line 440)
        kwargs_61491 = {}
        # Getting the type of 'object' (line 440)
        object_61490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 43), 'object', False)
        # Calling object(args, kwargs) (line 440)
        object_call_result_61492 = invoke(stypy.reporting.localization.Localization(__file__, 440, 43), object_61490, *[], **kwargs_61491)
        
        # Processing the call keyword arguments (line 440)
        kwargs_61493 = {}
        # Getting the type of 'getattr' (line 440)
        getattr_61487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 440)
        getattr_call_result_61494 = invoke(stypy.reporting.localization.Localization(__file__, 440, 20), getattr_61487, *[other_61488, unicode_61489, object_call_result_61492], **kwargs_61493)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 20), tuple_61478, getattr_call_result_61494)
        # Adding element type (line 439)
        
        # Call to getattr(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'other' (line 441)
        other_61496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 28), 'other', False)
        unicode_61497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 35), 'unicode', u'num2')
        
        # Call to object(...): (line 441)
        # Processing the call keyword arguments (line 441)
        kwargs_61499 = {}
        # Getting the type of 'object' (line 441)
        object_61498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'object', False)
        # Calling object(args, kwargs) (line 441)
        object_call_result_61500 = invoke(stypy.reporting.localization.Localization(__file__, 441, 43), object_61498, *[], **kwargs_61499)
        
        # Processing the call keyword arguments (line 441)
        kwargs_61501 = {}
        # Getting the type of 'getattr' (line 441)
        getattr_61495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 441)
        getattr_call_result_61502 = invoke(stypy.reporting.localization.Localization(__file__, 441, 20), getattr_61495, *[other_61496, unicode_61497, object_call_result_61500], **kwargs_61501)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 20), tuple_61478, getattr_call_result_61502)
        
        # Applying the binary operator '==' (line 438)
        result_eq_61503 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 16), '==', tuple_61471, tuple_61478)
        
        # Assigning a type to the variable 'stypy_return_type' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', result_eq_61503)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_61504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_61504


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_function_name', 'SubplotSpec.stypy__hash__')
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotSpec.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotSpec.stypy__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        
        # Call to hash(...): (line 448)
        # Processing the call arguments (line 448)
        
        # Obtaining an instance of the builtin type 'tuple' (line 448)
        tuple_61506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 448)
        # Adding element type (line 448)
        # Getting the type of 'self' (line 448)
        self_61507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 21), 'self', False)
        # Obtaining the member '_gridspec' of a type (line 448)
        _gridspec_61508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 21), self_61507, '_gridspec')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 21), tuple_61506, _gridspec_61508)
        # Adding element type (line 448)
        # Getting the type of 'self' (line 448)
        self_61509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 37), 'self', False)
        # Obtaining the member 'num1' of a type (line 448)
        num1_61510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 37), self_61509, 'num1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 21), tuple_61506, num1_61510)
        # Adding element type (line 448)
        # Getting the type of 'self' (line 448)
        self_61511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 48), 'self', False)
        # Obtaining the member 'num2' of a type (line 448)
        num2_61512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 48), self_61511, 'num2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 21), tuple_61506, num2_61512)
        
        # Processing the call keyword arguments (line 448)
        kwargs_61513 = {}
        # Getting the type of 'hash' (line 448)
        hash_61505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 448)
        hash_call_result_61514 = invoke(stypy.reporting.localization.Localization(__file__, 448, 15), hash_61505, *[tuple_61506], **kwargs_61513)
        
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'stypy_return_type', hash_call_result_61514)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_61515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_61515


# Assigning a type to the variable 'SubplotSpec' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'SubplotSpec', SubplotSpec)

# Getting the type of 'six' (line 443)
six_61516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'six')
# Obtaining the member 'PY2' of a type (line 443)
PY2_61517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 7), six_61516, 'PY2')
# Testing the type of an if condition (line 443)
if_condition_61518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), PY2_61517)
# Assigning a type to the variable 'if_condition_61518' (line 443)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_61518', if_condition_61518)
# SSA begins for if statement (line 443)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def __ne__(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__ne__'
    module_type_store = module_type_store.open_function_context('__ne__', 444, 8, False)
    
    # Passed parameters checking function
    __ne__.stypy_localization = localization
    __ne__.stypy_type_of_self = None
    __ne__.stypy_type_store = module_type_store
    __ne__.stypy_function_name = '__ne__'
    __ne__.stypy_param_names_list = ['self', 'other']
    __ne__.stypy_varargs_param_name = None
    __ne__.stypy_kwargs_param_name = None
    __ne__.stypy_call_defaults = defaults
    __ne__.stypy_call_varargs = varargs
    __ne__.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__ne__', ['self', 'other'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__ne__', localization, ['self', 'other'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__ne__(...)' code ##################

    
    
    # Getting the type of 'self' (line 445)
    self_61519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 23), 'self')
    # Getting the type of 'other' (line 445)
    other_61520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 31), 'other')
    # Applying the binary operator '==' (line 445)
    result_eq_61521 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 23), '==', self_61519, other_61520)
    
    # Applying the 'not' unary operator (line 445)
    result_not__61522 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 19), 'not', result_eq_61521)
    
    # Assigning a type to the variable 'stypy_return_type' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'stypy_return_type', result_not__61522)
    
    # ################# End of '__ne__(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__ne__' in the type store
    # Getting the type of 'stypy_return_type' (line 444)
    stypy_return_type_61523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_61523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__ne__'
    return stypy_return_type_61523

# Assigning a type to the variable '__ne__' (line 444)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), '__ne__', __ne__)
# SSA join for if statement (line 443)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

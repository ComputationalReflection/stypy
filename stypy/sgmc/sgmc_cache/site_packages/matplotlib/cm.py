
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module provides a large set of colormaps, functions for
3: registering new colormaps and for getting a colormap by name,
4: and a mixin class for adding color mapping functionality.
5: 
6: '''
7: from __future__ import (absolute_import, division, print_function,
8:                         unicode_literals)
9: 
10: import six
11: 
12: import os
13: import numpy as np
14: from numpy import ma
15: import matplotlib as mpl
16: import matplotlib.colors as colors
17: import matplotlib.cbook as cbook
18: from matplotlib._cm import datad, _deprecation_datad
19: from matplotlib._cm import cubehelix
20: from matplotlib._cm_listed import cmaps as cmaps_listed
21: 
22: cmap_d = _deprecation_datad()
23: 
24: # reverse all the colormaps.
25: # reversed colormaps have '_r' appended to the name.
26: 
27: 
28: def _reverser(f):
29:     def freversed(x):
30:         return f(1 - x)
31:     return freversed
32: 
33: 
34: def revcmap(data):
35:     '''Can only handle specification *data* in dictionary format.'''
36:     data_r = {}
37:     for key, val in six.iteritems(data):
38:         if callable(val):
39:             valnew = _reverser(val)
40:             # This doesn't work: lambda x: val(1-x)
41:             # The same "val" (the first one) is used
42:             # each time, so the colors are identical
43:             # and the result is shades of gray.
44:         else:
45:             # Flip x and exchange the y values facing x = 0 and x = 1.
46:             valnew = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(val)]
47:         data_r[key] = valnew
48:     return data_r
49: 
50: 
51: def _reverse_cmap_spec(spec):
52:     '''Reverses cmap specification *spec*, can handle both dict and tuple
53:     type specs.'''
54: 
55:     if 'listed' in spec:
56:         return {'listed': spec['listed'][::-1]}
57: 
58:     if 'red' in spec:
59:         return revcmap(spec)
60:     else:
61:         revspec = list(reversed(spec))
62:         if len(revspec[0]) == 2:    # e.g., (1, (1.0, 0.0, 1.0))
63:             revspec = [(1.0 - a, b) for a, b in revspec]
64:         return revspec
65: 
66: 
67: def _generate_cmap(name, lutsize):
68:     '''Generates the requested cmap from its *name*.  The lut size is
69:     *lutsize*.'''
70: 
71:     # Use superclass method to avoid deprecation warnings during initial load.
72:     spec = dict.__getitem__(datad, name)
73: 
74:     # Generate the colormap object.
75:     if 'red' in spec:
76:         return colors.LinearSegmentedColormap(name, spec, lutsize)
77:     elif 'listed' in spec:
78:         return colors.ListedColormap(spec['listed'], name)
79:     else:
80:         return colors.LinearSegmentedColormap.from_list(name, spec, lutsize)
81: 
82: LUTSIZE = mpl.rcParams['image.lut']
83: 
84: # Generate the reversed specifications (all at once, to avoid
85: # modify-when-iterating).
86: datad.update({cmapname + '_r': _reverse_cmap_spec(spec)
87:               for cmapname, spec in six.iteritems(datad)})
88: 
89: # Precache the cmaps with ``lutsize = LUTSIZE``.
90: # Also add the reversed ones added in the section above:
91: for cmapname in datad:
92:     cmap_d[cmapname] = _generate_cmap(cmapname, LUTSIZE)
93: 
94: cmap_d.update(cmaps_listed)
95: 
96: locals().update(cmap_d)
97: 
98: 
99: # Continue with definitions ...
100: 
101: 
102: def register_cmap(name=None, cmap=None, data=None, lut=None):
103:     '''
104:     Add a colormap to the set recognized by :func:`get_cmap`.
105: 
106:     It can be used in two ways::
107: 
108:         register_cmap(name='swirly', cmap=swirly_cmap)
109: 
110:         register_cmap(name='choppy', data=choppydata, lut=128)
111: 
112:     In the first case, *cmap* must be a :class:`matplotlib.colors.Colormap`
113:     instance.  The *name* is optional; if absent, the name will
114:     be the :attr:`~matplotlib.colors.Colormap.name` attribute of the *cmap*.
115: 
116:     In the second case, the three arguments are passed to
117:     the :class:`~matplotlib.colors.LinearSegmentedColormap` initializer,
118:     and the resulting colormap is registered.
119: 
120:     '''
121:     if name is None:
122:         try:
123:             name = cmap.name
124:         except AttributeError:
125:             raise ValueError("Arguments must include a name or a Colormap")
126: 
127:     if not isinstance(name, six.string_types):
128:         raise ValueError("Colormap name must be a string")
129: 
130:     if isinstance(cmap, colors.Colormap):
131:         cmap_d[name] = cmap
132:         return
133: 
134:     # For the remainder, let exceptions propagate.
135:     if lut is None:
136:         lut = mpl.rcParams['image.lut']
137:     cmap = colors.LinearSegmentedColormap(name, data, lut)
138:     cmap_d[name] = cmap
139: 
140: 
141: def get_cmap(name=None, lut=None):
142:     '''
143:     Get a colormap instance, defaulting to rc values if *name* is None.
144: 
145:     Colormaps added with :func:`register_cmap` take precedence over
146:     built-in colormaps.
147: 
148:     If *name* is a :class:`matplotlib.colors.Colormap` instance, it will be
149:     returned.
150: 
151:     If *lut* is not None it must be an integer giving the number of
152:     entries desired in the lookup table, and *name* must be a standard
153:     mpl colormap name.
154:     '''
155:     if name is None:
156:         name = mpl.rcParams['image.cmap']
157: 
158:     if isinstance(name, colors.Colormap):
159:         return name
160: 
161:     if name in cmap_d:
162:         if lut is None:
163:             return cmap_d[name]
164:         else:
165:             return cmap_d[name]._resample(lut)
166:     else:
167:         raise ValueError(
168:             "Colormap %s is not recognized. Possible values are: %s"
169:             % (name, ', '.join(sorted(cmap_d))))
170: 
171: 
172: class ScalarMappable(object):
173:     '''
174:     This is a mixin class to support scalar data to RGBA mapping.
175:     The ScalarMappable makes use of data normalization before returning
176:     RGBA colors from the given colormap.
177: 
178:     '''
179:     def __init__(self, norm=None, cmap=None):
180:         r'''
181: 
182:         Parameters
183:         ----------
184:         norm : :class:`matplotlib.colors.Normalize` instance
185:             The normalizing object which scales data, typically into the
186:             interval ``[0, 1]``.
187:             If *None*, *norm* defaults to a *colors.Normalize* object which
188:             initializes its scaling based on the first data processed.
189:         cmap : str or :class:`~matplotlib.colors.Colormap` instance
190:             The colormap used to map normalized data values to RGBA colors.
191:         '''
192: 
193:         self.callbacksSM = cbook.CallbackRegistry()
194: 
195:         if cmap is None:
196:             cmap = get_cmap()
197:         if norm is None:
198:             norm = colors.Normalize()
199: 
200:         self._A = None
201:         #: The Normalization instance of this ScalarMappable.
202:         self.norm = norm
203:         #: The Colormap instance of this ScalarMappable.
204:         self.cmap = get_cmap(cmap)
205:         #: The last colorbar associated with this ScalarMappable. May be None.
206:         self.colorbar = None
207:         self.update_dict = {'array': False}
208: 
209:     def to_rgba(self, x, alpha=None, bytes=False, norm=True):
210:         '''
211:         Return a normalized rgba array corresponding to *x*.
212: 
213:         In the normal case, *x* is a 1-D or 2-D sequence of scalars, and
214:         the corresponding ndarray of rgba values will be returned,
215:         based on the norm and colormap set for this ScalarMappable.
216: 
217:         There is one special case, for handling images that are already
218:         rgb or rgba, such as might have been read from an image file.
219:         If *x* is an ndarray with 3 dimensions,
220:         and the last dimension is either 3 or 4, then it will be
221:         treated as an rgb or rgba array, and no mapping will be done.
222:         The array can be uint8, or it can be floating point with
223:         values in the 0-1 range; otherwise a ValueError will be raised.
224:         If it is a masked array, the mask will be ignored.
225:         If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
226:         will be used to fill in the transparency.  If the last dimension
227:         is 4, the *alpha* kwarg is ignored; it does not
228:         replace the pre-existing alpha.  A ValueError will be raised
229:         if the third dimension is other than 3 or 4.
230: 
231:         In either case, if *bytes* is *False* (default), the rgba
232:         array will be floats in the 0-1 range; if it is *True*,
233:         the returned rgba array will be uint8 in the 0 to 255 range.
234: 
235:         If norm is False, no normalization of the input data is
236:         performed, and it is assumed to be in the range (0-1).
237: 
238:         '''
239:         # First check for special case, image input:
240:         try:
241:             if x.ndim == 3:
242:                 if x.shape[2] == 3:
243:                     if alpha is None:
244:                         alpha = 1
245:                     if x.dtype == np.uint8:
246:                         alpha = np.uint8(alpha * 255)
247:                     m, n = x.shape[:2]
248:                     xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
249:                     xx[:, :, :3] = x
250:                     xx[:, :, 3] = alpha
251:                 elif x.shape[2] == 4:
252:                     xx = x
253:                 else:
254:                     raise ValueError("third dimension must be 3 or 4")
255:                 if xx.dtype.kind == 'f':
256:                     if norm and xx.max() > 1 or xx.min() < 0:
257:                         raise ValueError("Floating point image RGB values "
258:                                          "must be in the 0..1 range.")
259:                     if bytes:
260:                         xx = (xx * 255).astype(np.uint8)
261:                 elif xx.dtype == np.uint8:
262:                     if not bytes:
263:                         xx = xx.astype(float) / 255
264:                 else:
265:                     raise ValueError("Image RGB array must be uint8 or "
266:                                      "floating point; found %s" % xx.dtype)
267:                 return xx
268:         except AttributeError:
269:             # e.g., x is not an ndarray; so try mapping it
270:             pass
271: 
272:         # This is the normal case, mapping a scalar array:
273:         x = ma.asarray(x)
274:         if norm:
275:             x = self.norm(x)
276:         rgba = self.cmap(x, alpha=alpha, bytes=bytes)
277:         return rgba
278: 
279:     def set_array(self, A):
280:         'Set the image array from numpy array *A*'
281:         self._A = A
282:         self.update_dict['array'] = True
283: 
284:     def get_array(self):
285:         'Return the array'
286:         return self._A
287: 
288:     def get_cmap(self):
289:         'return the colormap'
290:         return self.cmap
291: 
292:     def get_clim(self):
293:         'return the min, max of the color limits for image scaling'
294:         return self.norm.vmin, self.norm.vmax
295: 
296:     def set_clim(self, vmin=None, vmax=None):
297:         '''
298:         set the norm limits for image scaling; if *vmin* is a length2
299:         sequence, interpret it as ``(vmin, vmax)`` which is used to
300:         support setp
301: 
302:         ACCEPTS: a length 2 sequence of floats
303:         '''
304:         if vmax is None:
305:             try:
306:                 vmin, vmax = vmin
307:             except (TypeError, ValueError):
308:                 pass
309:         if vmin is not None:
310:             self.norm.vmin = vmin
311:         if vmax is not None:
312:             self.norm.vmax = vmax
313:         self.changed()
314: 
315:     def set_cmap(self, cmap):
316:         '''
317:         set the colormap for luminance data
318: 
319:         ACCEPTS: a colormap or registered colormap name
320:         '''
321:         cmap = get_cmap(cmap)
322:         self.cmap = cmap
323:         self.changed()
324: 
325:     def set_norm(self, norm):
326:         'set the normalization instance'
327:         if norm is None:
328:             norm = colors.Normalize()
329:         self.norm = norm
330:         self.changed()
331: 
332:     def autoscale(self):
333:         '''
334:         Autoscale the scalar limits on the norm instance using the
335:         current array
336:         '''
337:         if self._A is None:
338:             raise TypeError('You must first set_array for mappable')
339:         self.norm.autoscale(self._A)
340:         self.changed()
341: 
342:     def autoscale_None(self):
343:         '''
344:         Autoscale the scalar limits on the norm instance using the
345:         current array, changing only limits that are None
346:         '''
347:         if self._A is None:
348:             raise TypeError('You must first set_array for mappable')
349:         self.norm.autoscale_None(self._A)
350:         self.changed()
351: 
352:     def add_checker(self, checker):
353:         '''
354:         Add an entry to a dictionary of boolean flags
355:         that are set to True when the mappable is changed.
356:         '''
357:         self.update_dict[checker] = False
358: 
359:     def check_update(self, checker):
360:         '''
361:         If mappable has changed since the last check,
362:         return True; else return False
363:         '''
364:         if self.update_dict[checker]:
365:             self.update_dict[checker] = False
366:             return True
367:         return False
368: 
369:     def changed(self):
370:         '''
371:         Call this whenever the mappable is changed to notify all the
372:         callbackSM listeners to the 'changed' signal
373:         '''
374:         self.callbacksSM.process('changed', self)
375: 
376:         for key in self.update_dict:
377:             self.update_dict[key] = True
378:         self.stale = True
379: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_25514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'unicode', u'\nThis module provides a large set of colormaps, functions for\nregistering new colormaps and for getting a colormap by name,\nand a mixin class for adding color mapping functionality.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import six' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25515 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six')

if (type(import_25515) is not StypyTypeError):

    if (import_25515 != 'pyd_module'):
        __import__(import_25515)
        sys_modules_25516 = sys.modules[import_25515]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', sys_modules_25516.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', import_25515)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import os' statement (line 12)
import os

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25517 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_25517) is not StypyTypeError):

    if (import_25517 != 'pyd_module'):
        __import__(import_25517)
        sys_modules_25518 = sys.modules[import_25517]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_25518.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_25517)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy import ma' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25519 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_25519) is not StypyTypeError):

    if (import_25519 != 'pyd_module'):
        __import__(import_25519)
        sys_modules_25520 = sys.modules[import_25519]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', sys_modules_25520.module_type_store, module_type_store, ['ma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_25520, sys_modules_25520.module_type_store, module_type_store)
    else:
        from numpy import ma

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', None, module_type_store, ['ma'], [ma])

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_25519)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import matplotlib' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25521 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib')

if (type(import_25521) is not StypyTypeError):

    if (import_25521 != 'pyd_module'):
        __import__(import_25521)
        sys_modules_25522 = sys.modules[import_25521]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'mpl', sys_modules_25522.module_type_store, module_type_store)
    else:
        import matplotlib as mpl

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'mpl', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', import_25521)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import matplotlib.colors' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25523 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.colors')

if (type(import_25523) is not StypyTypeError):

    if (import_25523 != 'pyd_module'):
        __import__(import_25523)
        sys_modules_25524 = sys.modules[import_25523]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'colors', sys_modules_25524.module_type_store, module_type_store)
    else:
        import matplotlib.colors as colors

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'colors', matplotlib.colors, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.colors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.colors', import_25523)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import matplotlib.cbook' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25525 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.cbook')

if (type(import_25525) is not StypyTypeError):

    if (import_25525 != 'pyd_module'):
        __import__(import_25525)
        sys_modules_25526 = sys.modules[import_25525]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'cbook', sys_modules_25526.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.cbook', import_25525)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from matplotlib._cm import datad, _deprecation_datad' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25527 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib._cm')

if (type(import_25527) is not StypyTypeError):

    if (import_25527 != 'pyd_module'):
        __import__(import_25527)
        sys_modules_25528 = sys.modules[import_25527]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib._cm', sys_modules_25528.module_type_store, module_type_store, ['datad', '_deprecation_datad'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_25528, sys_modules_25528.module_type_store, module_type_store)
    else:
        from matplotlib._cm import datad, _deprecation_datad

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib._cm', None, module_type_store, ['datad', '_deprecation_datad'], [datad, _deprecation_datad])

else:
    # Assigning a type to the variable 'matplotlib._cm' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib._cm', import_25527)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib._cm import cubehelix' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25529 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib._cm')

if (type(import_25529) is not StypyTypeError):

    if (import_25529 != 'pyd_module'):
        __import__(import_25529)
        sys_modules_25530 = sys.modules[import_25529]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib._cm', sys_modules_25530.module_type_store, module_type_store, ['cubehelix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_25530, sys_modules_25530.module_type_store, module_type_store)
    else:
        from matplotlib._cm import cubehelix

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib._cm', None, module_type_store, ['cubehelix'], [cubehelix])

else:
    # Assigning a type to the variable 'matplotlib._cm' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib._cm', import_25529)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib._cm_listed import cmaps_listed' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25531 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib._cm_listed')

if (type(import_25531) is not StypyTypeError):

    if (import_25531 != 'pyd_module'):
        __import__(import_25531)
        sys_modules_25532 = sys.modules[import_25531]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib._cm_listed', sys_modules_25532.module_type_store, module_type_store, ['cmaps'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_25532, sys_modules_25532.module_type_store, module_type_store)
    else:
        from matplotlib._cm_listed import cmaps as cmaps_listed

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib._cm_listed', None, module_type_store, ['cmaps'], [cmaps_listed])

else:
    # Assigning a type to the variable 'matplotlib._cm_listed' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib._cm_listed', import_25531)

# Adding an alias
module_type_store.add_alias('cmaps_listed', 'cmaps')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Assigning a Call to a Name (line 22):

# Assigning a Call to a Name (line 22):

# Call to _deprecation_datad(...): (line 22)
# Processing the call keyword arguments (line 22)
kwargs_25534 = {}
# Getting the type of '_deprecation_datad' (line 22)
_deprecation_datad_25533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), '_deprecation_datad', False)
# Calling _deprecation_datad(args, kwargs) (line 22)
_deprecation_datad_call_result_25535 = invoke(stypy.reporting.localization.Localization(__file__, 22, 9), _deprecation_datad_25533, *[], **kwargs_25534)

# Assigning a type to the variable 'cmap_d' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'cmap_d', _deprecation_datad_call_result_25535)

@norecursion
def _reverser(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_reverser'
    module_type_store = module_type_store.open_function_context('_reverser', 28, 0, False)
    
    # Passed parameters checking function
    _reverser.stypy_localization = localization
    _reverser.stypy_type_of_self = None
    _reverser.stypy_type_store = module_type_store
    _reverser.stypy_function_name = '_reverser'
    _reverser.stypy_param_names_list = ['f']
    _reverser.stypy_varargs_param_name = None
    _reverser.stypy_kwargs_param_name = None
    _reverser.stypy_call_defaults = defaults
    _reverser.stypy_call_varargs = varargs
    _reverser.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_reverser', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_reverser', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_reverser(...)' code ##################


    @norecursion
    def freversed(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'freversed'
        module_type_store = module_type_store.open_function_context('freversed', 29, 4, False)
        
        # Passed parameters checking function
        freversed.stypy_localization = localization
        freversed.stypy_type_of_self = None
        freversed.stypy_type_store = module_type_store
        freversed.stypy_function_name = 'freversed'
        freversed.stypy_param_names_list = ['x']
        freversed.stypy_varargs_param_name = None
        freversed.stypy_kwargs_param_name = None
        freversed.stypy_call_defaults = defaults
        freversed.stypy_call_varargs = varargs
        freversed.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'freversed', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'freversed', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'freversed(...)' code ##################

        
        # Call to f(...): (line 30)
        # Processing the call arguments (line 30)
        int_25537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
        # Getting the type of 'x' (line 30)
        x_25538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'x', False)
        # Applying the binary operator '-' (line 30)
        result_sub_25539 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 17), '-', int_25537, x_25538)
        
        # Processing the call keyword arguments (line 30)
        kwargs_25540 = {}
        # Getting the type of 'f' (line 30)
        f_25536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'f', False)
        # Calling f(args, kwargs) (line 30)
        f_call_result_25541 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), f_25536, *[result_sub_25539], **kwargs_25540)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', f_call_result_25541)
        
        # ################# End of 'freversed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'freversed' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_25542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'freversed'
        return stypy_return_type_25542

    # Assigning a type to the variable 'freversed' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'freversed', freversed)
    # Getting the type of 'freversed' (line 31)
    freversed_25543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'freversed')
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', freversed_25543)
    
    # ################# End of '_reverser(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_reverser' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_25544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25544)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_reverser'
    return stypy_return_type_25544

# Assigning a type to the variable '_reverser' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '_reverser', _reverser)

@norecursion
def revcmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'revcmap'
    module_type_store = module_type_store.open_function_context('revcmap', 34, 0, False)
    
    # Passed parameters checking function
    revcmap.stypy_localization = localization
    revcmap.stypy_type_of_self = None
    revcmap.stypy_type_store = module_type_store
    revcmap.stypy_function_name = 'revcmap'
    revcmap.stypy_param_names_list = ['data']
    revcmap.stypy_varargs_param_name = None
    revcmap.stypy_kwargs_param_name = None
    revcmap.stypy_call_defaults = defaults
    revcmap.stypy_call_varargs = varargs
    revcmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'revcmap', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'revcmap', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'revcmap(...)' code ##################

    unicode_25545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'unicode', u'Can only handle specification *data* in dictionary format.')
    
    # Assigning a Dict to a Name (line 36):
    
    # Assigning a Dict to a Name (line 36):
    
    # Obtaining an instance of the builtin type 'dict' (line 36)
    dict_25546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 36)
    
    # Assigning a type to the variable 'data_r' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'data_r', dict_25546)
    
    
    # Call to iteritems(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'data' (line 37)
    data_25549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'data', False)
    # Processing the call keyword arguments (line 37)
    kwargs_25550 = {}
    # Getting the type of 'six' (line 37)
    six_25547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'six', False)
    # Obtaining the member 'iteritems' of a type (line 37)
    iteritems_25548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), six_25547, 'iteritems')
    # Calling iteritems(args, kwargs) (line 37)
    iteritems_call_result_25551 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), iteritems_25548, *[data_25549], **kwargs_25550)
    
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 4), iteritems_call_result_25551)
    # Getting the type of the for loop variable (line 37)
    for_loop_var_25552 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 4), iteritems_call_result_25551)
    # Assigning a type to the variable 'key' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), for_loop_var_25552))
    # Assigning a type to the variable 'val' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), for_loop_var_25552))
    # SSA begins for a for statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to callable(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'val' (line 38)
    val_25554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'val', False)
    # Processing the call keyword arguments (line 38)
    kwargs_25555 = {}
    # Getting the type of 'callable' (line 38)
    callable_25553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'callable', False)
    # Calling callable(args, kwargs) (line 38)
    callable_call_result_25556 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), callable_25553, *[val_25554], **kwargs_25555)
    
    # Testing the type of an if condition (line 38)
    if_condition_25557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), callable_call_result_25556)
    # Assigning a type to the variable 'if_condition_25557' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_25557', if_condition_25557)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to _reverser(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'val' (line 39)
    val_25559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'val', False)
    # Processing the call keyword arguments (line 39)
    kwargs_25560 = {}
    # Getting the type of '_reverser' (line 39)
    _reverser_25558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), '_reverser', False)
    # Calling _reverser(args, kwargs) (line 39)
    _reverser_call_result_25561 = invoke(stypy.reporting.localization.Localization(__file__, 39, 21), _reverser_25558, *[val_25559], **kwargs_25560)
    
    # Assigning a type to the variable 'valnew' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'valnew', _reverser_call_result_25561)
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 46):
    
    # Assigning a ListComp to a Name (line 46):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to reversed(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'val' (line 46)
    val_25569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 66), 'val', False)
    # Processing the call keyword arguments (line 46)
    kwargs_25570 = {}
    # Getting the type of 'reversed' (line 46)
    reversed_25568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 57), 'reversed', False)
    # Calling reversed(args, kwargs) (line 46)
    reversed_call_result_25571 = invoke(stypy.reporting.localization.Localization(__file__, 46, 57), reversed_25568, *[val_25569], **kwargs_25570)
    
    comprehension_25572 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), reversed_call_result_25571)
    # Assigning a type to the variable 'x' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), comprehension_25572))
    # Assigning a type to the variable 'y0' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'y0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), comprehension_25572))
    # Assigning a type to the variable 'y1' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'y1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), comprehension_25572))
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_25562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    float_25563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'float')
    # Getting the type of 'x' (line 46)
    x_25564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'x')
    # Applying the binary operator '-' (line 46)
    result_sub_25565 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 23), '-', float_25563, x_25564)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 23), tuple_25562, result_sub_25565)
    # Adding element type (line 46)
    # Getting the type of 'y1' (line 46)
    y1_25566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'y1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 23), tuple_25562, y1_25566)
    # Adding element type (line 46)
    # Getting the type of 'y0' (line 46)
    y0_25567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'y0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 23), tuple_25562, y0_25567)
    
    list_25573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_25573, tuple_25562)
    # Assigning a type to the variable 'valnew' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'valnew', list_25573)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 47):
    
    # Assigning a Name to a Subscript (line 47):
    # Getting the type of 'valnew' (line 47)
    valnew_25574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'valnew')
    # Getting the type of 'data_r' (line 47)
    data_r_25575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'data_r')
    # Getting the type of 'key' (line 47)
    key_25576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'key')
    # Storing an element on a container (line 47)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), data_r_25575, (key_25576, valnew_25574))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'data_r' (line 48)
    data_r_25577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'data_r')
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', data_r_25577)
    
    # ################# End of 'revcmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'revcmap' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_25578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25578)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'revcmap'
    return stypy_return_type_25578

# Assigning a type to the variable 'revcmap' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'revcmap', revcmap)

@norecursion
def _reverse_cmap_spec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_reverse_cmap_spec'
    module_type_store = module_type_store.open_function_context('_reverse_cmap_spec', 51, 0, False)
    
    # Passed parameters checking function
    _reverse_cmap_spec.stypy_localization = localization
    _reverse_cmap_spec.stypy_type_of_self = None
    _reverse_cmap_spec.stypy_type_store = module_type_store
    _reverse_cmap_spec.stypy_function_name = '_reverse_cmap_spec'
    _reverse_cmap_spec.stypy_param_names_list = ['spec']
    _reverse_cmap_spec.stypy_varargs_param_name = None
    _reverse_cmap_spec.stypy_kwargs_param_name = None
    _reverse_cmap_spec.stypy_call_defaults = defaults
    _reverse_cmap_spec.stypy_call_varargs = varargs
    _reverse_cmap_spec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_reverse_cmap_spec', ['spec'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_reverse_cmap_spec', localization, ['spec'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_reverse_cmap_spec(...)' code ##################

    unicode_25579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'unicode', u'Reverses cmap specification *spec*, can handle both dict and tuple\n    type specs.')
    
    
    unicode_25580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 7), 'unicode', u'listed')
    # Getting the type of 'spec' (line 55)
    spec_25581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'spec')
    # Applying the binary operator 'in' (line 55)
    result_contains_25582 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), 'in', unicode_25580, spec_25581)
    
    # Testing the type of an if condition (line 55)
    if_condition_25583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), result_contains_25582)
    # Assigning a type to the variable 'if_condition_25583' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_25583', if_condition_25583)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'dict' (line 56)
    dict_25584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 56)
    # Adding element type (key, value) (line 56)
    unicode_25585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'unicode', u'listed')
    
    # Obtaining the type of the subscript
    int_25586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 43), 'int')
    slice_25587 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 26), None, None, int_25586)
    
    # Obtaining the type of the subscript
    unicode_25588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'unicode', u'listed')
    # Getting the type of 'spec' (line 56)
    spec_25589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'spec')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___25590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 26), spec_25589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_25591 = invoke(stypy.reporting.localization.Localization(__file__, 56, 26), getitem___25590, unicode_25588)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___25592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 26), subscript_call_result_25591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_25593 = invoke(stypy.reporting.localization.Localization(__file__, 56, 26), getitem___25592, slice_25587)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 15), dict_25584, (unicode_25585, subscript_call_result_25593))
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', dict_25584)
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    unicode_25594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 7), 'unicode', u'red')
    # Getting the type of 'spec' (line 58)
    spec_25595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'spec')
    # Applying the binary operator 'in' (line 58)
    result_contains_25596 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), 'in', unicode_25594, spec_25595)
    
    # Testing the type of an if condition (line 58)
    if_condition_25597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_contains_25596)
    # Assigning a type to the variable 'if_condition_25597' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_25597', if_condition_25597)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to revcmap(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'spec' (line 59)
    spec_25599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'spec', False)
    # Processing the call keyword arguments (line 59)
    kwargs_25600 = {}
    # Getting the type of 'revcmap' (line 59)
    revcmap_25598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'revcmap', False)
    # Calling revcmap(args, kwargs) (line 59)
    revcmap_call_result_25601 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), revcmap_25598, *[spec_25599], **kwargs_25600)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', revcmap_call_result_25601)
    # SSA branch for the else part of an if statement (line 58)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to list(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to reversed(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'spec' (line 61)
    spec_25604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'spec', False)
    # Processing the call keyword arguments (line 61)
    kwargs_25605 = {}
    # Getting the type of 'reversed' (line 61)
    reversed_25603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'reversed', False)
    # Calling reversed(args, kwargs) (line 61)
    reversed_call_result_25606 = invoke(stypy.reporting.localization.Localization(__file__, 61, 23), reversed_25603, *[spec_25604], **kwargs_25605)
    
    # Processing the call keyword arguments (line 61)
    kwargs_25607 = {}
    # Getting the type of 'list' (line 61)
    list_25602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'list', False)
    # Calling list(args, kwargs) (line 61)
    list_call_result_25608 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), list_25602, *[reversed_call_result_25606], **kwargs_25607)
    
    # Assigning a type to the variable 'revspec' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'revspec', list_call_result_25608)
    
    
    
    # Call to len(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Obtaining the type of the subscript
    int_25610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'int')
    # Getting the type of 'revspec' (line 62)
    revspec_25611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'revspec', False)
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___25612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), revspec_25611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_25613 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), getitem___25612, int_25610)
    
    # Processing the call keyword arguments (line 62)
    kwargs_25614 = {}
    # Getting the type of 'len' (line 62)
    len_25609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'len', False)
    # Calling len(args, kwargs) (line 62)
    len_call_result_25615 = invoke(stypy.reporting.localization.Localization(__file__, 62, 11), len_25609, *[subscript_call_result_25613], **kwargs_25614)
    
    int_25616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'int')
    # Applying the binary operator '==' (line 62)
    result_eq_25617 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '==', len_call_result_25615, int_25616)
    
    # Testing the type of an if condition (line 62)
    if_condition_25618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_eq_25617)
    # Assigning a type to the variable 'if_condition_25618' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_25618', if_condition_25618)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 63):
    
    # Assigning a ListComp to a Name (line 63):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'revspec' (line 63)
    revspec_25624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'revspec')
    comprehension_25625 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 23), revspec_25624)
    # Assigning a type to the variable 'a' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 23), comprehension_25625))
    # Assigning a type to the variable 'b' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 23), comprehension_25625))
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_25619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    float_25620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'float')
    # Getting the type of 'a' (line 63)
    a_25621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'a')
    # Applying the binary operator '-' (line 63)
    result_sub_25622 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 24), '-', float_25620, a_25621)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), tuple_25619, result_sub_25622)
    # Adding element type (line 63)
    # Getting the type of 'b' (line 63)
    b_25623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), tuple_25619, b_25623)
    
    list_25626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 23), list_25626, tuple_25619)
    # Assigning a type to the variable 'revspec' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'revspec', list_25626)
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'revspec' (line 64)
    revspec_25627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'revspec')
    # Assigning a type to the variable 'stypy_return_type' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', revspec_25627)
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_reverse_cmap_spec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_reverse_cmap_spec' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_25628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25628)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_reverse_cmap_spec'
    return stypy_return_type_25628

# Assigning a type to the variable '_reverse_cmap_spec' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_reverse_cmap_spec', _reverse_cmap_spec)

@norecursion
def _generate_cmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_generate_cmap'
    module_type_store = module_type_store.open_function_context('_generate_cmap', 67, 0, False)
    
    # Passed parameters checking function
    _generate_cmap.stypy_localization = localization
    _generate_cmap.stypy_type_of_self = None
    _generate_cmap.stypy_type_store = module_type_store
    _generate_cmap.stypy_function_name = '_generate_cmap'
    _generate_cmap.stypy_param_names_list = ['name', 'lutsize']
    _generate_cmap.stypy_varargs_param_name = None
    _generate_cmap.stypy_kwargs_param_name = None
    _generate_cmap.stypy_call_defaults = defaults
    _generate_cmap.stypy_call_varargs = varargs
    _generate_cmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_generate_cmap', ['name', 'lutsize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_generate_cmap', localization, ['name', 'lutsize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_generate_cmap(...)' code ##################

    unicode_25629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'unicode', u'Generates the requested cmap from its *name*.  The lut size is\n    *lutsize*.')
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to __getitem__(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'datad' (line 72)
    datad_25632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'datad', False)
    # Getting the type of 'name' (line 72)
    name_25633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'name', False)
    # Processing the call keyword arguments (line 72)
    kwargs_25634 = {}
    # Getting the type of 'dict' (line 72)
    dict_25630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'dict', False)
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___25631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), dict_25630, '__getitem__')
    # Calling __getitem__(args, kwargs) (line 72)
    getitem___call_result_25635 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), getitem___25631, *[datad_25632, name_25633], **kwargs_25634)
    
    # Assigning a type to the variable 'spec' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'spec', getitem___call_result_25635)
    
    
    unicode_25636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 7), 'unicode', u'red')
    # Getting the type of 'spec' (line 75)
    spec_25637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'spec')
    # Applying the binary operator 'in' (line 75)
    result_contains_25638 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'in', unicode_25636, spec_25637)
    
    # Testing the type of an if condition (line 75)
    if_condition_25639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_contains_25638)
    # Assigning a type to the variable 'if_condition_25639' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_25639', if_condition_25639)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinearSegmentedColormap(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'name' (line 76)
    name_25642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'name', False)
    # Getting the type of 'spec' (line 76)
    spec_25643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 52), 'spec', False)
    # Getting the type of 'lutsize' (line 76)
    lutsize_25644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 58), 'lutsize', False)
    # Processing the call keyword arguments (line 76)
    kwargs_25645 = {}
    # Getting the type of 'colors' (line 76)
    colors_25640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'colors', False)
    # Obtaining the member 'LinearSegmentedColormap' of a type (line 76)
    LinearSegmentedColormap_25641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 15), colors_25640, 'LinearSegmentedColormap')
    # Calling LinearSegmentedColormap(args, kwargs) (line 76)
    LinearSegmentedColormap_call_result_25646 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), LinearSegmentedColormap_25641, *[name_25642, spec_25643, lutsize_25644], **kwargs_25645)
    
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', LinearSegmentedColormap_call_result_25646)
    # SSA branch for the else part of an if statement (line 75)
    module_type_store.open_ssa_branch('else')
    
    
    unicode_25647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'unicode', u'listed')
    # Getting the type of 'spec' (line 77)
    spec_25648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'spec')
    # Applying the binary operator 'in' (line 77)
    result_contains_25649 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), 'in', unicode_25647, spec_25648)
    
    # Testing the type of an if condition (line 77)
    if_condition_25650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 9), result_contains_25649)
    # Assigning a type to the variable 'if_condition_25650' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'if_condition_25650', if_condition_25650)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ListedColormap(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining the type of the subscript
    unicode_25653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'unicode', u'listed')
    # Getting the type of 'spec' (line 78)
    spec_25654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___25655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 37), spec_25654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_25656 = invoke(stypy.reporting.localization.Localization(__file__, 78, 37), getitem___25655, unicode_25653)
    
    # Getting the type of 'name' (line 78)
    name_25657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 53), 'name', False)
    # Processing the call keyword arguments (line 78)
    kwargs_25658 = {}
    # Getting the type of 'colors' (line 78)
    colors_25651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'colors', False)
    # Obtaining the member 'ListedColormap' of a type (line 78)
    ListedColormap_25652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), colors_25651, 'ListedColormap')
    # Calling ListedColormap(args, kwargs) (line 78)
    ListedColormap_call_result_25659 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), ListedColormap_25652, *[subscript_call_result_25656, name_25657], **kwargs_25658)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', ListedColormap_call_result_25659)
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    # Call to from_list(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'name' (line 80)
    name_25663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 56), 'name', False)
    # Getting the type of 'spec' (line 80)
    spec_25664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 62), 'spec', False)
    # Getting the type of 'lutsize' (line 80)
    lutsize_25665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 68), 'lutsize', False)
    # Processing the call keyword arguments (line 80)
    kwargs_25666 = {}
    # Getting the type of 'colors' (line 80)
    colors_25660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'colors', False)
    # Obtaining the member 'LinearSegmentedColormap' of a type (line 80)
    LinearSegmentedColormap_25661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), colors_25660, 'LinearSegmentedColormap')
    # Obtaining the member 'from_list' of a type (line 80)
    from_list_25662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), LinearSegmentedColormap_25661, 'from_list')
    # Calling from_list(args, kwargs) (line 80)
    from_list_call_result_25667 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), from_list_25662, *[name_25663, spec_25664, lutsize_25665], **kwargs_25666)
    
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', from_list_call_result_25667)
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_generate_cmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_generate_cmap' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_25668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25668)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_generate_cmap'
    return stypy_return_type_25668

# Assigning a type to the variable '_generate_cmap' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), '_generate_cmap', _generate_cmap)

# Assigning a Subscript to a Name (line 82):

# Assigning a Subscript to a Name (line 82):

# Obtaining the type of the subscript
unicode_25669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'unicode', u'image.lut')
# Getting the type of 'mpl' (line 82)
mpl_25670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 10), 'mpl')
# Obtaining the member 'rcParams' of a type (line 82)
rcParams_25671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 10), mpl_25670, 'rcParams')
# Obtaining the member '__getitem__' of a type (line 82)
getitem___25672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 10), rcParams_25671, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 82)
subscript_call_result_25673 = invoke(stypy.reporting.localization.Localization(__file__, 82, 10), getitem___25672, unicode_25669)

# Assigning a type to the variable 'LUTSIZE' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'LUTSIZE', subscript_call_result_25673)

# Call to update(...): (line 86)
# Processing the call arguments (line 86)
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 86, 14, True)
# Calculating comprehension expression

# Call to iteritems(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of 'datad' (line 87)
datad_25685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 50), 'datad', False)
# Processing the call keyword arguments (line 87)
kwargs_25686 = {}
# Getting the type of 'six' (line 87)
six_25683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'six', False)
# Obtaining the member 'iteritems' of a type (line 87)
iteritems_25684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 36), six_25683, 'iteritems')
# Calling iteritems(args, kwargs) (line 87)
iteritems_call_result_25687 = invoke(stypy.reporting.localization.Localization(__file__, 87, 36), iteritems_25684, *[datad_25685], **kwargs_25686)

comprehension_25688 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), iteritems_call_result_25687)
# Assigning a type to the variable 'cmapname' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'cmapname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), comprehension_25688))
# Assigning a type to the variable 'spec' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), comprehension_25688))
# Getting the type of 'cmapname' (line 86)
cmapname_25676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'cmapname', False)
unicode_25677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'unicode', u'_r')
# Applying the binary operator '+' (line 86)
result_add_25678 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 14), '+', cmapname_25676, unicode_25677)


# Call to _reverse_cmap_spec(...): (line 86)
# Processing the call arguments (line 86)
# Getting the type of 'spec' (line 86)
spec_25680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 50), 'spec', False)
# Processing the call keyword arguments (line 86)
kwargs_25681 = {}
# Getting the type of '_reverse_cmap_spec' (line 86)
_reverse_cmap_spec_25679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), '_reverse_cmap_spec', False)
# Calling _reverse_cmap_spec(args, kwargs) (line 86)
_reverse_cmap_spec_call_result_25682 = invoke(stypy.reporting.localization.Localization(__file__, 86, 31), _reverse_cmap_spec_25679, *[spec_25680], **kwargs_25681)

dict_25689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), dict_25689, (result_add_25678, _reverse_cmap_spec_call_result_25682))
# Processing the call keyword arguments (line 86)
kwargs_25690 = {}
# Getting the type of 'datad' (line 86)
datad_25674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'datad', False)
# Obtaining the member 'update' of a type (line 86)
update_25675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 0), datad_25674, 'update')
# Calling update(args, kwargs) (line 86)
update_call_result_25691 = invoke(stypy.reporting.localization.Localization(__file__, 86, 0), update_25675, *[dict_25689], **kwargs_25690)


# Getting the type of 'datad' (line 91)
datad_25692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'datad')
# Testing the type of a for loop iterable (line 91)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 0), datad_25692)
# Getting the type of the for loop variable (line 91)
for_loop_var_25693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 0), datad_25692)
# Assigning a type to the variable 'cmapname' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'cmapname', for_loop_var_25693)
# SSA begins for a for statement (line 91)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Call to a Subscript (line 92):

# Assigning a Call to a Subscript (line 92):

# Call to _generate_cmap(...): (line 92)
# Processing the call arguments (line 92)
# Getting the type of 'cmapname' (line 92)
cmapname_25695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'cmapname', False)
# Getting the type of 'LUTSIZE' (line 92)
LUTSIZE_25696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 48), 'LUTSIZE', False)
# Processing the call keyword arguments (line 92)
kwargs_25697 = {}
# Getting the type of '_generate_cmap' (line 92)
_generate_cmap_25694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), '_generate_cmap', False)
# Calling _generate_cmap(args, kwargs) (line 92)
_generate_cmap_call_result_25698 = invoke(stypy.reporting.localization.Localization(__file__, 92, 23), _generate_cmap_25694, *[cmapname_25695, LUTSIZE_25696], **kwargs_25697)

# Getting the type of 'cmap_d' (line 92)
cmap_d_25699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'cmap_d')
# Getting the type of 'cmapname' (line 92)
cmapname_25700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'cmapname')
# Storing an element on a container (line 92)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), cmap_d_25699, (cmapname_25700, _generate_cmap_call_result_25698))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Call to update(...): (line 94)
# Processing the call arguments (line 94)
# Getting the type of 'cmaps_listed' (line 94)
cmaps_listed_25703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'cmaps_listed', False)
# Processing the call keyword arguments (line 94)
kwargs_25704 = {}
# Getting the type of 'cmap_d' (line 94)
cmap_d_25701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'cmap_d', False)
# Obtaining the member 'update' of a type (line 94)
update_25702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 0), cmap_d_25701, 'update')
# Calling update(args, kwargs) (line 94)
update_call_result_25705 = invoke(stypy.reporting.localization.Localization(__file__, 94, 0), update_25702, *[cmaps_listed_25703], **kwargs_25704)


# Call to update(...): (line 96)
# Processing the call arguments (line 96)
# Getting the type of 'cmap_d' (line 96)
cmap_d_25710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'cmap_d', False)
# Processing the call keyword arguments (line 96)
kwargs_25711 = {}

# Call to locals(...): (line 96)
# Processing the call keyword arguments (line 96)
kwargs_25707 = {}
# Getting the type of 'locals' (line 96)
locals_25706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'locals', False)
# Calling locals(args, kwargs) (line 96)
locals_call_result_25708 = invoke(stypy.reporting.localization.Localization(__file__, 96, 0), locals_25706, *[], **kwargs_25707)

# Obtaining the member 'update' of a type (line 96)
update_25709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 0), locals_call_result_25708, 'update')
# Calling update(args, kwargs) (line 96)
update_call_result_25712 = invoke(stypy.reporting.localization.Localization(__file__, 96, 0), update_25709, *[cmap_d_25710], **kwargs_25711)


@norecursion
def register_cmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 102)
    None_25713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'None')
    # Getting the type of 'None' (line 102)
    None_25714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'None')
    # Getting the type of 'None' (line 102)
    None_25715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 45), 'None')
    # Getting the type of 'None' (line 102)
    None_25716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 55), 'None')
    defaults = [None_25713, None_25714, None_25715, None_25716]
    # Create a new context for function 'register_cmap'
    module_type_store = module_type_store.open_function_context('register_cmap', 102, 0, False)
    
    # Passed parameters checking function
    register_cmap.stypy_localization = localization
    register_cmap.stypy_type_of_self = None
    register_cmap.stypy_type_store = module_type_store
    register_cmap.stypy_function_name = 'register_cmap'
    register_cmap.stypy_param_names_list = ['name', 'cmap', 'data', 'lut']
    register_cmap.stypy_varargs_param_name = None
    register_cmap.stypy_kwargs_param_name = None
    register_cmap.stypy_call_defaults = defaults
    register_cmap.stypy_call_varargs = varargs
    register_cmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'register_cmap', ['name', 'cmap', 'data', 'lut'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'register_cmap', localization, ['name', 'cmap', 'data', 'lut'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'register_cmap(...)' code ##################

    unicode_25717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'unicode', u"\n    Add a colormap to the set recognized by :func:`get_cmap`.\n\n    It can be used in two ways::\n\n        register_cmap(name='swirly', cmap=swirly_cmap)\n\n        register_cmap(name='choppy', data=choppydata, lut=128)\n\n    In the first case, *cmap* must be a :class:`matplotlib.colors.Colormap`\n    instance.  The *name* is optional; if absent, the name will\n    be the :attr:`~matplotlib.colors.Colormap.name` attribute of the *cmap*.\n\n    In the second case, the three arguments are passed to\n    the :class:`~matplotlib.colors.LinearSegmentedColormap` initializer,\n    and the resulting colormap is registered.\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 121)
    # Getting the type of 'name' (line 121)
    name_25718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'name')
    # Getting the type of 'None' (line 121)
    None_25719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'None')
    
    (may_be_25720, more_types_in_union_25721) = may_be_none(name_25718, None_25719)

    if may_be_25720:

        if more_types_in_union_25721:
            # Runtime conditional SSA (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # SSA begins for try-except statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 123):
        
        # Assigning a Attribute to a Name (line 123):
        # Getting the type of 'cmap' (line 123)
        cmap_25722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'cmap')
        # Obtaining the member 'name' of a type (line 123)
        name_25723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 19), cmap_25722, 'name')
        # Assigning a type to the variable 'name' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'name', name_25723)
        # SSA branch for the except part of a try statement (line 122)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 122)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 125)
        # Processing the call arguments (line 125)
        unicode_25725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'unicode', u'Arguments must include a name or a Colormap')
        # Processing the call keyword arguments (line 125)
        kwargs_25726 = {}
        # Getting the type of 'ValueError' (line 125)
        ValueError_25724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 125)
        ValueError_call_result_25727 = invoke(stypy.reporting.localization.Localization(__file__, 125, 18), ValueError_25724, *[unicode_25725], **kwargs_25726)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 125, 12), ValueError_call_result_25727, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_25721:
            # SSA join for if statement (line 121)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to isinstance(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'name' (line 127)
    name_25729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'name', False)
    # Getting the type of 'six' (line 127)
    six_25730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'six', False)
    # Obtaining the member 'string_types' of a type (line 127)
    string_types_25731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 28), six_25730, 'string_types')
    # Processing the call keyword arguments (line 127)
    kwargs_25732 = {}
    # Getting the type of 'isinstance' (line 127)
    isinstance_25728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 127)
    isinstance_call_result_25733 = invoke(stypy.reporting.localization.Localization(__file__, 127, 11), isinstance_25728, *[name_25729, string_types_25731], **kwargs_25732)
    
    # Applying the 'not' unary operator (line 127)
    result_not__25734 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), 'not', isinstance_call_result_25733)
    
    # Testing the type of an if condition (line 127)
    if_condition_25735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 4), result_not__25734)
    # Assigning a type to the variable 'if_condition_25735' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'if_condition_25735', if_condition_25735)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 128)
    # Processing the call arguments (line 128)
    unicode_25737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 25), 'unicode', u'Colormap name must be a string')
    # Processing the call keyword arguments (line 128)
    kwargs_25738 = {}
    # Getting the type of 'ValueError' (line 128)
    ValueError_25736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 128)
    ValueError_call_result_25739 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), ValueError_25736, *[unicode_25737], **kwargs_25738)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 8), ValueError_call_result_25739, 'raise parameter', BaseException)
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'cmap' (line 130)
    cmap_25741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'cmap', False)
    # Getting the type of 'colors' (line 130)
    colors_25742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'colors', False)
    # Obtaining the member 'Colormap' of a type (line 130)
    Colormap_25743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), colors_25742, 'Colormap')
    # Processing the call keyword arguments (line 130)
    kwargs_25744 = {}
    # Getting the type of 'isinstance' (line 130)
    isinstance_25740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 130)
    isinstance_call_result_25745 = invoke(stypy.reporting.localization.Localization(__file__, 130, 7), isinstance_25740, *[cmap_25741, Colormap_25743], **kwargs_25744)
    
    # Testing the type of an if condition (line 130)
    if_condition_25746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), isinstance_call_result_25745)
    # Assigning a type to the variable 'if_condition_25746' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_25746', if_condition_25746)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 131):
    
    # Assigning a Name to a Subscript (line 131):
    # Getting the type of 'cmap' (line 131)
    cmap_25747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'cmap')
    # Getting the type of 'cmap_d' (line 131)
    cmap_d_25748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'cmap_d')
    # Getting the type of 'name' (line 131)
    name_25749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'name')
    # Storing an element on a container (line 131)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 8), cmap_d_25748, (name_25749, cmap_25747))
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 135)
    # Getting the type of 'lut' (line 135)
    lut_25750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 7), 'lut')
    # Getting the type of 'None' (line 135)
    None_25751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'None')
    
    (may_be_25752, more_types_in_union_25753) = may_be_none(lut_25750, None_25751)

    if may_be_25752:

        if more_types_in_union_25753:
            # Runtime conditional SSA (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 136):
        
        # Assigning a Subscript to a Name (line 136):
        
        # Obtaining the type of the subscript
        unicode_25754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 27), 'unicode', u'image.lut')
        # Getting the type of 'mpl' (line 136)
        mpl_25755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 14), 'mpl')
        # Obtaining the member 'rcParams' of a type (line 136)
        rcParams_25756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 14), mpl_25755, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___25757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 14), rcParams_25756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_25758 = invoke(stypy.reporting.localization.Localization(__file__, 136, 14), getitem___25757, unicode_25754)
        
        # Assigning a type to the variable 'lut' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'lut', subscript_call_result_25758)

        if more_types_in_union_25753:
            # SSA join for if statement (line 135)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to LinearSegmentedColormap(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'name' (line 137)
    name_25761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'name', False)
    # Getting the type of 'data' (line 137)
    data_25762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 48), 'data', False)
    # Getting the type of 'lut' (line 137)
    lut_25763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 'lut', False)
    # Processing the call keyword arguments (line 137)
    kwargs_25764 = {}
    # Getting the type of 'colors' (line 137)
    colors_25759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'colors', False)
    # Obtaining the member 'LinearSegmentedColormap' of a type (line 137)
    LinearSegmentedColormap_25760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), colors_25759, 'LinearSegmentedColormap')
    # Calling LinearSegmentedColormap(args, kwargs) (line 137)
    LinearSegmentedColormap_call_result_25765 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), LinearSegmentedColormap_25760, *[name_25761, data_25762, lut_25763], **kwargs_25764)
    
    # Assigning a type to the variable 'cmap' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'cmap', LinearSegmentedColormap_call_result_25765)
    
    # Assigning a Name to a Subscript (line 138):
    
    # Assigning a Name to a Subscript (line 138):
    # Getting the type of 'cmap' (line 138)
    cmap_25766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'cmap')
    # Getting the type of 'cmap_d' (line 138)
    cmap_d_25767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'cmap_d')
    # Getting the type of 'name' (line 138)
    name_25768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'name')
    # Storing an element on a container (line 138)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 4), cmap_d_25767, (name_25768, cmap_25766))
    
    # ################# End of 'register_cmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'register_cmap' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_25769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25769)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'register_cmap'
    return stypy_return_type_25769

# Assigning a type to the variable 'register_cmap' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'register_cmap', register_cmap)

@norecursion
def get_cmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 141)
    None_25770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'None')
    # Getting the type of 'None' (line 141)
    None_25771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'None')
    defaults = [None_25770, None_25771]
    # Create a new context for function 'get_cmap'
    module_type_store = module_type_store.open_function_context('get_cmap', 141, 0, False)
    
    # Passed parameters checking function
    get_cmap.stypy_localization = localization
    get_cmap.stypy_type_of_self = None
    get_cmap.stypy_type_store = module_type_store
    get_cmap.stypy_function_name = 'get_cmap'
    get_cmap.stypy_param_names_list = ['name', 'lut']
    get_cmap.stypy_varargs_param_name = None
    get_cmap.stypy_kwargs_param_name = None
    get_cmap.stypy_call_defaults = defaults
    get_cmap.stypy_call_varargs = varargs
    get_cmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_cmap', ['name', 'lut'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_cmap', localization, ['name', 'lut'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_cmap(...)' code ##################

    unicode_25772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'unicode', u'\n    Get a colormap instance, defaulting to rc values if *name* is None.\n\n    Colormaps added with :func:`register_cmap` take precedence over\n    built-in colormaps.\n\n    If *name* is a :class:`matplotlib.colors.Colormap` instance, it will be\n    returned.\n\n    If *lut* is not None it must be an integer giving the number of\n    entries desired in the lookup table, and *name* must be a standard\n    mpl colormap name.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 155)
    # Getting the type of 'name' (line 155)
    name_25773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 7), 'name')
    # Getting the type of 'None' (line 155)
    None_25774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'None')
    
    (may_be_25775, more_types_in_union_25776) = may_be_none(name_25773, None_25774)

    if may_be_25775:

        if more_types_in_union_25776:
            # Runtime conditional SSA (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 156):
        
        # Assigning a Subscript to a Name (line 156):
        
        # Obtaining the type of the subscript
        unicode_25777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'unicode', u'image.cmap')
        # Getting the type of 'mpl' (line 156)
        mpl_25778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'mpl')
        # Obtaining the member 'rcParams' of a type (line 156)
        rcParams_25779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), mpl_25778, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___25780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), rcParams_25779, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_25781 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), getitem___25780, unicode_25777)
        
        # Assigning a type to the variable 'name' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'name', subscript_call_result_25781)

        if more_types_in_union_25776:
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isinstance(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'name' (line 158)
    name_25783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'name', False)
    # Getting the type of 'colors' (line 158)
    colors_25784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'colors', False)
    # Obtaining the member 'Colormap' of a type (line 158)
    Colormap_25785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 24), colors_25784, 'Colormap')
    # Processing the call keyword arguments (line 158)
    kwargs_25786 = {}
    # Getting the type of 'isinstance' (line 158)
    isinstance_25782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 158)
    isinstance_call_result_25787 = invoke(stypy.reporting.localization.Localization(__file__, 158, 7), isinstance_25782, *[name_25783, Colormap_25785], **kwargs_25786)
    
    # Testing the type of an if condition (line 158)
    if_condition_25788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 4), isinstance_call_result_25787)
    # Assigning a type to the variable 'if_condition_25788' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'if_condition_25788', if_condition_25788)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'name' (line 159)
    name_25789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', name_25789)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'name' (line 161)
    name_25790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 7), 'name')
    # Getting the type of 'cmap_d' (line 161)
    cmap_d_25791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'cmap_d')
    # Applying the binary operator 'in' (line 161)
    result_contains_25792 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 7), 'in', name_25790, cmap_d_25791)
    
    # Testing the type of an if condition (line 161)
    if_condition_25793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 4), result_contains_25792)
    # Assigning a type to the variable 'if_condition_25793' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'if_condition_25793', if_condition_25793)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 162)
    # Getting the type of 'lut' (line 162)
    lut_25794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'lut')
    # Getting the type of 'None' (line 162)
    None_25795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'None')
    
    (may_be_25796, more_types_in_union_25797) = may_be_none(lut_25794, None_25795)

    if may_be_25796:

        if more_types_in_union_25797:
            # Runtime conditional SSA (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 163)
        name_25798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'name')
        # Getting the type of 'cmap_d' (line 163)
        cmap_d_25799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'cmap_d')
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___25800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), cmap_d_25799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_25801 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), getitem___25800, name_25798)
        
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'stypy_return_type', subscript_call_result_25801)

        if more_types_in_union_25797:
            # Runtime conditional SSA for else branch (line 162)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_25796) or more_types_in_union_25797):
        
        # Call to _resample(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'lut' (line 165)
        lut_25807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 42), 'lut', False)
        # Processing the call keyword arguments (line 165)
        kwargs_25808 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 165)
        name_25802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'name', False)
        # Getting the type of 'cmap_d' (line 165)
        cmap_d_25803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'cmap_d', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___25804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), cmap_d_25803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_25805 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), getitem___25804, name_25802)
        
        # Obtaining the member '_resample' of a type (line 165)
        _resample_25806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), subscript_call_result_25805, '_resample')
        # Calling _resample(args, kwargs) (line 165)
        _resample_call_result_25809 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), _resample_25806, *[lut_25807], **kwargs_25808)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'stypy_return_type', _resample_call_result_25809)

        if (may_be_25796 and more_types_in_union_25797):
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 161)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 167)
    # Processing the call arguments (line 167)
    unicode_25811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'unicode', u'Colormap %s is not recognized. Possible values are: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 169)
    tuple_25812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 169)
    # Adding element type (line 169)
    # Getting the type of 'name' (line 169)
    name_25813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 15), tuple_25812, name_25813)
    # Adding element type (line 169)
    
    # Call to join(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Call to sorted(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'cmap_d' (line 169)
    cmap_d_25817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), 'cmap_d', False)
    # Processing the call keyword arguments (line 169)
    kwargs_25818 = {}
    # Getting the type of 'sorted' (line 169)
    sorted_25816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'sorted', False)
    # Calling sorted(args, kwargs) (line 169)
    sorted_call_result_25819 = invoke(stypy.reporting.localization.Localization(__file__, 169, 31), sorted_25816, *[cmap_d_25817], **kwargs_25818)
    
    # Processing the call keyword arguments (line 169)
    kwargs_25820 = {}
    unicode_25814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'unicode', u', ')
    # Obtaining the member 'join' of a type (line 169)
    join_25815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), unicode_25814, 'join')
    # Calling join(args, kwargs) (line 169)
    join_call_result_25821 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), join_25815, *[sorted_call_result_25819], **kwargs_25820)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 15), tuple_25812, join_call_result_25821)
    
    # Applying the binary operator '%' (line 168)
    result_mod_25822 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 12), '%', unicode_25811, tuple_25812)
    
    # Processing the call keyword arguments (line 167)
    kwargs_25823 = {}
    # Getting the type of 'ValueError' (line 167)
    ValueError_25810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 167)
    ValueError_call_result_25824 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), ValueError_25810, *[result_mod_25822], **kwargs_25823)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 167, 8), ValueError_call_result_25824, 'raise parameter', BaseException)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_cmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_cmap' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_25825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25825)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_cmap'
    return stypy_return_type_25825

# Assigning a type to the variable 'get_cmap' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'get_cmap', get_cmap)
# Declaration of the 'ScalarMappable' class

class ScalarMappable(object, ):
    unicode_25826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, (-1)), 'unicode', u'\n    This is a mixin class to support scalar data to RGBA mapping.\n    The ScalarMappable makes use of data normalization before returning\n    RGBA colors from the given colormap.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 179)
        None_25827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'None')
        # Getting the type of 'None' (line 179)
        None_25828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 39), 'None')
        defaults = [None_25827, None_25828]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.__init__', ['norm', 'cmap'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['norm', 'cmap'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_25829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'unicode', u'\n\n        Parameters\n        ----------\n        norm : :class:`matplotlib.colors.Normalize` instance\n            The normalizing object which scales data, typically into the\n            interval ``[0, 1]``.\n            If *None*, *norm* defaults to a *colors.Normalize* object which\n            initializes its scaling based on the first data processed.\n        cmap : str or :class:`~matplotlib.colors.Colormap` instance\n            The colormap used to map normalized data values to RGBA colors.\n        ')
        
        # Assigning a Call to a Attribute (line 193):
        
        # Assigning a Call to a Attribute (line 193):
        
        # Call to CallbackRegistry(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_25832 = {}
        # Getting the type of 'cbook' (line 193)
        cbook_25830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'cbook', False)
        # Obtaining the member 'CallbackRegistry' of a type (line 193)
        CallbackRegistry_25831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 27), cbook_25830, 'CallbackRegistry')
        # Calling CallbackRegistry(args, kwargs) (line 193)
        CallbackRegistry_call_result_25833 = invoke(stypy.reporting.localization.Localization(__file__, 193, 27), CallbackRegistry_25831, *[], **kwargs_25832)
        
        # Getting the type of 'self' (line 193)
        self_25834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member 'callbacksSM' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_25834, 'callbacksSM', CallbackRegistry_call_result_25833)
        
        # Type idiom detected: calculating its left and rigth part (line 195)
        # Getting the type of 'cmap' (line 195)
        cmap_25835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'cmap')
        # Getting the type of 'None' (line 195)
        None_25836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'None')
        
        (may_be_25837, more_types_in_union_25838) = may_be_none(cmap_25835, None_25836)

        if may_be_25837:

            if more_types_in_union_25838:
                # Runtime conditional SSA (line 195)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 196):
            
            # Assigning a Call to a Name (line 196):
            
            # Call to get_cmap(...): (line 196)
            # Processing the call keyword arguments (line 196)
            kwargs_25840 = {}
            # Getting the type of 'get_cmap' (line 196)
            get_cmap_25839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'get_cmap', False)
            # Calling get_cmap(args, kwargs) (line 196)
            get_cmap_call_result_25841 = invoke(stypy.reporting.localization.Localization(__file__, 196, 19), get_cmap_25839, *[], **kwargs_25840)
            
            # Assigning a type to the variable 'cmap' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'cmap', get_cmap_call_result_25841)

            if more_types_in_union_25838:
                # SSA join for if statement (line 195)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 197)
        # Getting the type of 'norm' (line 197)
        norm_25842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'norm')
        # Getting the type of 'None' (line 197)
        None_25843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'None')
        
        (may_be_25844, more_types_in_union_25845) = may_be_none(norm_25842, None_25843)

        if may_be_25844:

            if more_types_in_union_25845:
                # Runtime conditional SSA (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 198):
            
            # Assigning a Call to a Name (line 198):
            
            # Call to Normalize(...): (line 198)
            # Processing the call keyword arguments (line 198)
            kwargs_25848 = {}
            # Getting the type of 'colors' (line 198)
            colors_25846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'colors', False)
            # Obtaining the member 'Normalize' of a type (line 198)
            Normalize_25847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), colors_25846, 'Normalize')
            # Calling Normalize(args, kwargs) (line 198)
            Normalize_call_result_25849 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), Normalize_25847, *[], **kwargs_25848)
            
            # Assigning a type to the variable 'norm' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'norm', Normalize_call_result_25849)

            if more_types_in_union_25845:
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 200):
        
        # Assigning a Name to a Attribute (line 200):
        # Getting the type of 'None' (line 200)
        None_25850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'None')
        # Getting the type of 'self' (line 200)
        self_25851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member '_A' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_25851, '_A', None_25850)
        
        # Assigning a Name to a Attribute (line 202):
        
        # Assigning a Name to a Attribute (line 202):
        # Getting the type of 'norm' (line 202)
        norm_25852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'norm')
        # Getting the type of 'self' (line 202)
        self_25853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member 'norm' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_25853, 'norm', norm_25852)
        
        # Assigning a Call to a Attribute (line 204):
        
        # Assigning a Call to a Attribute (line 204):
        
        # Call to get_cmap(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'cmap' (line 204)
        cmap_25855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'cmap', False)
        # Processing the call keyword arguments (line 204)
        kwargs_25856 = {}
        # Getting the type of 'get_cmap' (line 204)
        get_cmap_25854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'get_cmap', False)
        # Calling get_cmap(args, kwargs) (line 204)
        get_cmap_call_result_25857 = invoke(stypy.reporting.localization.Localization(__file__, 204, 20), get_cmap_25854, *[cmap_25855], **kwargs_25856)
        
        # Getting the type of 'self' (line 204)
        self_25858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'cmap' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_25858, 'cmap', get_cmap_call_result_25857)
        
        # Assigning a Name to a Attribute (line 206):
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'None' (line 206)
        None_25859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'None')
        # Getting the type of 'self' (line 206)
        self_25860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'colorbar' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_25860, 'colorbar', None_25859)
        
        # Assigning a Dict to a Attribute (line 207):
        
        # Assigning a Dict to a Attribute (line 207):
        
        # Obtaining an instance of the builtin type 'dict' (line 207)
        dict_25861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 207)
        # Adding element type (key, value) (line 207)
        unicode_25862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'unicode', u'array')
        # Getting the type of 'False' (line 207)
        False_25863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 37), 'False')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 27), dict_25861, (unicode_25862, False_25863))
        
        # Getting the type of 'self' (line 207)
        self_25864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'update_dict' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_25864, 'update_dict', dict_25861)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def to_rgba(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 209)
        None_25865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 31), 'None')
        # Getting the type of 'False' (line 209)
        False_25866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'False')
        # Getting the type of 'True' (line 209)
        True_25867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 55), 'True')
        defaults = [None_25865, False_25866, True_25867]
        # Create a new context for function 'to_rgba'
        module_type_store = module_type_store.open_function_context('to_rgba', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.to_rgba')
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_param_names_list', ['x', 'alpha', 'bytes', 'norm'])
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.to_rgba.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.to_rgba', ['x', 'alpha', 'bytes', 'norm'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'to_rgba', localization, ['x', 'alpha', 'bytes', 'norm'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'to_rgba(...)' code ##################

        unicode_25868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'unicode', u'\n        Return a normalized rgba array corresponding to *x*.\n\n        In the normal case, *x* is a 1-D or 2-D sequence of scalars, and\n        the corresponding ndarray of rgba values will be returned,\n        based on the norm and colormap set for this ScalarMappable.\n\n        There is one special case, for handling images that are already\n        rgb or rgba, such as might have been read from an image file.\n        If *x* is an ndarray with 3 dimensions,\n        and the last dimension is either 3 or 4, then it will be\n        treated as an rgb or rgba array, and no mapping will be done.\n        The array can be uint8, or it can be floating point with\n        values in the 0-1 range; otherwise a ValueError will be raised.\n        If it is a masked array, the mask will be ignored.\n        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)\n        will be used to fill in the transparency.  If the last dimension\n        is 4, the *alpha* kwarg is ignored; it does not\n        replace the pre-existing alpha.  A ValueError will be raised\n        if the third dimension is other than 3 or 4.\n\n        In either case, if *bytes* is *False* (default), the rgba\n        array will be floats in the 0-1 range; if it is *True*,\n        the returned rgba array will be uint8 in the 0 to 255 range.\n\n        If norm is False, no normalization of the input data is\n        performed, and it is assumed to be in the range (0-1).\n\n        ')
        
        
        # SSA begins for try-except statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Getting the type of 'x' (line 241)
        x_25869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'x')
        # Obtaining the member 'ndim' of a type (line 241)
        ndim_25870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), x_25869, 'ndim')
        int_25871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 25), 'int')
        # Applying the binary operator '==' (line 241)
        result_eq_25872 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '==', ndim_25870, int_25871)
        
        # Testing the type of an if condition (line 241)
        if_condition_25873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), result_eq_25872)
        # Assigning a type to the variable 'if_condition_25873' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'if_condition_25873', if_condition_25873)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_25874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 27), 'int')
        # Getting the type of 'x' (line 242)
        x_25875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'x')
        # Obtaining the member 'shape' of a type (line 242)
        shape_25876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 19), x_25875, 'shape')
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___25877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 19), shape_25876, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_25878 = invoke(stypy.reporting.localization.Localization(__file__, 242, 19), getitem___25877, int_25874)
        
        int_25879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 33), 'int')
        # Applying the binary operator '==' (line 242)
        result_eq_25880 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 19), '==', subscript_call_result_25878, int_25879)
        
        # Testing the type of an if condition (line 242)
        if_condition_25881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 16), result_eq_25880)
        # Assigning a type to the variable 'if_condition_25881' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'if_condition_25881', if_condition_25881)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 243)
        # Getting the type of 'alpha' (line 243)
        alpha_25882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 23), 'alpha')
        # Getting the type of 'None' (line 243)
        None_25883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 32), 'None')
        
        (may_be_25884, more_types_in_union_25885) = may_be_none(alpha_25882, None_25883)

        if may_be_25884:

            if more_types_in_union_25885:
                # Runtime conditional SSA (line 243)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 244):
            
            # Assigning a Num to a Name (line 244):
            int_25886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 32), 'int')
            # Assigning a type to the variable 'alpha' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'alpha', int_25886)

            if more_types_in_union_25885:
                # SSA join for if statement (line 243)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'x' (line 245)
        x_25887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 'x')
        # Obtaining the member 'dtype' of a type (line 245)
        dtype_25888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 23), x_25887, 'dtype')
        # Getting the type of 'np' (line 245)
        np_25889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'np')
        # Obtaining the member 'uint8' of a type (line 245)
        uint8_25890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 34), np_25889, 'uint8')
        # Applying the binary operator '==' (line 245)
        result_eq_25891 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 23), '==', dtype_25888, uint8_25890)
        
        # Testing the type of an if condition (line 245)
        if_condition_25892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 20), result_eq_25891)
        # Assigning a type to the variable 'if_condition_25892' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'if_condition_25892', if_condition_25892)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to uint8(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'alpha' (line 246)
        alpha_25895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 41), 'alpha', False)
        int_25896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 49), 'int')
        # Applying the binary operator '*' (line 246)
        result_mul_25897 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 41), '*', alpha_25895, int_25896)
        
        # Processing the call keyword arguments (line 246)
        kwargs_25898 = {}
        # Getting the type of 'np' (line 246)
        np_25893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 32), 'np', False)
        # Obtaining the member 'uint8' of a type (line 246)
        uint8_25894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 32), np_25893, 'uint8')
        # Calling uint8(args, kwargs) (line 246)
        uint8_call_result_25899 = invoke(stypy.reporting.localization.Localization(__file__, 246, 32), uint8_25894, *[result_mul_25897], **kwargs_25898)
        
        # Assigning a type to the variable 'alpha' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'alpha', uint8_call_result_25899)
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 247):
        
        # Assigning a Subscript to a Name (line 247):
        
        # Obtaining the type of the subscript
        int_25900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'int')
        
        # Obtaining the type of the subscript
        int_25901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 36), 'int')
        slice_25902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 247, 27), None, int_25901, None)
        # Getting the type of 'x' (line 247)
        x_25903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 27), 'x')
        # Obtaining the member 'shape' of a type (line 247)
        shape_25904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 27), x_25903, 'shape')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 27), shape_25904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25906 = invoke(stypy.reporting.localization.Localization(__file__, 247, 27), getitem___25905, slice_25902)
        
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), subscript_call_result_25906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25908 = invoke(stypy.reporting.localization.Localization(__file__, 247, 20), getitem___25907, int_25900)
        
        # Assigning a type to the variable 'tuple_var_assignment_25510' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'tuple_var_assignment_25510', subscript_call_result_25908)
        
        # Assigning a Subscript to a Name (line 247):
        
        # Obtaining the type of the subscript
        int_25909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'int')
        
        # Obtaining the type of the subscript
        int_25910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 36), 'int')
        slice_25911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 247, 27), None, int_25910, None)
        # Getting the type of 'x' (line 247)
        x_25912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 27), 'x')
        # Obtaining the member 'shape' of a type (line 247)
        shape_25913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 27), x_25912, 'shape')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 27), shape_25913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25915 = invoke(stypy.reporting.localization.Localization(__file__, 247, 27), getitem___25914, slice_25911)
        
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), subscript_call_result_25915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25917 = invoke(stypy.reporting.localization.Localization(__file__, 247, 20), getitem___25916, int_25909)
        
        # Assigning a type to the variable 'tuple_var_assignment_25511' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'tuple_var_assignment_25511', subscript_call_result_25917)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'tuple_var_assignment_25510' (line 247)
        tuple_var_assignment_25510_25918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'tuple_var_assignment_25510')
        # Assigning a type to the variable 'm' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'm', tuple_var_assignment_25510_25918)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'tuple_var_assignment_25511' (line 247)
        tuple_var_assignment_25511_25919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'tuple_var_assignment_25511')
        # Assigning a type to the variable 'n' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'n', tuple_var_assignment_25511_25919)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to empty(...): (line 248)
        # Processing the call keyword arguments (line 248)
        
        # Obtaining an instance of the builtin type 'tuple' (line 248)
        tuple_25922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 248)
        # Adding element type (line 248)
        # Getting the type of 'm' (line 248)
        m_25923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 41), tuple_25922, m_25923)
        # Adding element type (line 248)
        # Getting the type of 'n' (line 248)
        n_25924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 44), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 41), tuple_25922, n_25924)
        # Adding element type (line 248)
        int_25925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 41), tuple_25922, int_25925)
        
        keyword_25926 = tuple_25922
        # Getting the type of 'x' (line 248)
        x_25927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 57), 'x', False)
        # Obtaining the member 'dtype' of a type (line 248)
        dtype_25928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 57), x_25927, 'dtype')
        keyword_25929 = dtype_25928
        kwargs_25930 = {'dtype': keyword_25929, 'shape': keyword_25926}
        # Getting the type of 'np' (line 248)
        np_25920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'np', False)
        # Obtaining the member 'empty' of a type (line 248)
        empty_25921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), np_25920, 'empty')
        # Calling empty(args, kwargs) (line 248)
        empty_call_result_25931 = invoke(stypy.reporting.localization.Localization(__file__, 248, 25), empty_25921, *[], **kwargs_25930)
        
        # Assigning a type to the variable 'xx' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'xx', empty_call_result_25931)
        
        # Assigning a Name to a Subscript (line 249):
        
        # Assigning a Name to a Subscript (line 249):
        # Getting the type of 'x' (line 249)
        x_25932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 35), 'x')
        # Getting the type of 'xx' (line 249)
        xx_25933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'xx')
        slice_25934 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 20), None, None, None)
        slice_25935 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 20), None, None, None)
        int_25936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 30), 'int')
        slice_25937 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 20), None, int_25936, None)
        # Storing an element on a container (line 249)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), xx_25933, ((slice_25934, slice_25935, slice_25937), x_25932))
        
        # Assigning a Name to a Subscript (line 250):
        
        # Assigning a Name to a Subscript (line 250):
        # Getting the type of 'alpha' (line 250)
        alpha_25938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 34), 'alpha')
        # Getting the type of 'xx' (line 250)
        xx_25939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'xx')
        slice_25940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 250, 20), None, None, None)
        slice_25941 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 250, 20), None, None, None)
        int_25942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 29), 'int')
        # Storing an element on a container (line 250)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), xx_25939, ((slice_25940, slice_25941, int_25942), alpha_25938))
        # SSA branch for the else part of an if statement (line 242)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_25943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'int')
        # Getting the type of 'x' (line 251)
        x_25944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'x')
        # Obtaining the member 'shape' of a type (line 251)
        shape_25945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 21), x_25944, 'shape')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___25946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 21), shape_25945, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_25947 = invoke(stypy.reporting.localization.Localization(__file__, 251, 21), getitem___25946, int_25943)
        
        int_25948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 35), 'int')
        # Applying the binary operator '==' (line 251)
        result_eq_25949 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 21), '==', subscript_call_result_25947, int_25948)
        
        # Testing the type of an if condition (line 251)
        if_condition_25950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 21), result_eq_25949)
        # Assigning a type to the variable 'if_condition_25950' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'if_condition_25950', if_condition_25950)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 252):
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'x' (line 252)
        x_25951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 25), 'x')
        # Assigning a type to the variable 'xx' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 20), 'xx', x_25951)
        # SSA branch for the else part of an if statement (line 251)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 254)
        # Processing the call arguments (line 254)
        unicode_25953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 37), 'unicode', u'third dimension must be 3 or 4')
        # Processing the call keyword arguments (line 254)
        kwargs_25954 = {}
        # Getting the type of 'ValueError' (line 254)
        ValueError_25952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 254)
        ValueError_call_result_25955 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), ValueError_25952, *[unicode_25953], **kwargs_25954)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 254, 20), ValueError_call_result_25955, 'raise parameter', BaseException)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'xx' (line 255)
        xx_25956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'xx')
        # Obtaining the member 'dtype' of a type (line 255)
        dtype_25957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 19), xx_25956, 'dtype')
        # Obtaining the member 'kind' of a type (line 255)
        kind_25958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 19), dtype_25957, 'kind')
        unicode_25959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 36), 'unicode', u'f')
        # Applying the binary operator '==' (line 255)
        result_eq_25960 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 19), '==', kind_25958, unicode_25959)
        
        # Testing the type of an if condition (line 255)
        if_condition_25961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 16), result_eq_25960)
        # Assigning a type to the variable 'if_condition_25961' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'if_condition_25961', if_condition_25961)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'norm' (line 256)
        norm_25962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'norm')
        
        
        # Call to max(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_25965 = {}
        # Getting the type of 'xx' (line 256)
        xx_25963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'xx', False)
        # Obtaining the member 'max' of a type (line 256)
        max_25964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 32), xx_25963, 'max')
        # Calling max(args, kwargs) (line 256)
        max_call_result_25966 = invoke(stypy.reporting.localization.Localization(__file__, 256, 32), max_25964, *[], **kwargs_25965)
        
        int_25967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 43), 'int')
        # Applying the binary operator '>' (line 256)
        result_gt_25968 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 32), '>', max_call_result_25966, int_25967)
        
        # Applying the binary operator 'and' (line 256)
        result_and_keyword_25969 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 23), 'and', norm_25962, result_gt_25968)
        
        
        
        # Call to min(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_25972 = {}
        # Getting the type of 'xx' (line 256)
        xx_25970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 48), 'xx', False)
        # Obtaining the member 'min' of a type (line 256)
        min_25971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 48), xx_25970, 'min')
        # Calling min(args, kwargs) (line 256)
        min_call_result_25973 = invoke(stypy.reporting.localization.Localization(__file__, 256, 48), min_25971, *[], **kwargs_25972)
        
        int_25974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 59), 'int')
        # Applying the binary operator '<' (line 256)
        result_lt_25975 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 48), '<', min_call_result_25973, int_25974)
        
        # Applying the binary operator 'or' (line 256)
        result_or_keyword_25976 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 23), 'or', result_and_keyword_25969, result_lt_25975)
        
        # Testing the type of an if condition (line 256)
        if_condition_25977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 20), result_or_keyword_25976)
        # Assigning a type to the variable 'if_condition_25977' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'if_condition_25977', if_condition_25977)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 257)
        # Processing the call arguments (line 257)
        unicode_25979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 41), 'unicode', u'Floating point image RGB values must be in the 0..1 range.')
        # Processing the call keyword arguments (line 257)
        kwargs_25980 = {}
        # Getting the type of 'ValueError' (line 257)
        ValueError_25978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 257)
        ValueError_call_result_25981 = invoke(stypy.reporting.localization.Localization(__file__, 257, 30), ValueError_25978, *[unicode_25979], **kwargs_25980)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 257, 24), ValueError_call_result_25981, 'raise parameter', BaseException)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'bytes' (line 259)
        bytes_25982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'bytes')
        # Testing the type of an if condition (line 259)
        if_condition_25983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 20), bytes_25982)
        # Assigning a type to the variable 'if_condition_25983' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'if_condition_25983', if_condition_25983)
        # SSA begins for if statement (line 259)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to astype(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'np' (line 260)
        np_25988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 47), 'np', False)
        # Obtaining the member 'uint8' of a type (line 260)
        uint8_25989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 47), np_25988, 'uint8')
        # Processing the call keyword arguments (line 260)
        kwargs_25990 = {}
        # Getting the type of 'xx' (line 260)
        xx_25984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'xx', False)
        int_25985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 35), 'int')
        # Applying the binary operator '*' (line 260)
        result_mul_25986 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 30), '*', xx_25984, int_25985)
        
        # Obtaining the member 'astype' of a type (line 260)
        astype_25987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 30), result_mul_25986, 'astype')
        # Calling astype(args, kwargs) (line 260)
        astype_call_result_25991 = invoke(stypy.reporting.localization.Localization(__file__, 260, 30), astype_25987, *[uint8_25989], **kwargs_25990)
        
        # Assigning a type to the variable 'xx' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'xx', astype_call_result_25991)
        # SSA join for if statement (line 259)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 255)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'xx' (line 261)
        xx_25992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 21), 'xx')
        # Obtaining the member 'dtype' of a type (line 261)
        dtype_25993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 21), xx_25992, 'dtype')
        # Getting the type of 'np' (line 261)
        np_25994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'np')
        # Obtaining the member 'uint8' of a type (line 261)
        uint8_25995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 33), np_25994, 'uint8')
        # Applying the binary operator '==' (line 261)
        result_eq_25996 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 21), '==', dtype_25993, uint8_25995)
        
        # Testing the type of an if condition (line 261)
        if_condition_25997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 21), result_eq_25996)
        # Assigning a type to the variable 'if_condition_25997' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 21), 'if_condition_25997', if_condition_25997)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'bytes' (line 262)
        bytes_25998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 27), 'bytes')
        # Applying the 'not' unary operator (line 262)
        result_not__25999 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 23), 'not', bytes_25998)
        
        # Testing the type of an if condition (line 262)
        if_condition_26000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 20), result_not__25999)
        # Assigning a type to the variable 'if_condition_26000' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'if_condition_26000', if_condition_26000)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 263):
        
        # Assigning a BinOp to a Name (line 263):
        
        # Call to astype(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'float' (line 263)
        float_26003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 39), 'float', False)
        # Processing the call keyword arguments (line 263)
        kwargs_26004 = {}
        # Getting the type of 'xx' (line 263)
        xx_26001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'xx', False)
        # Obtaining the member 'astype' of a type (line 263)
        astype_26002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 29), xx_26001, 'astype')
        # Calling astype(args, kwargs) (line 263)
        astype_call_result_26005 = invoke(stypy.reporting.localization.Localization(__file__, 263, 29), astype_26002, *[float_26003], **kwargs_26004)
        
        int_26006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 48), 'int')
        # Applying the binary operator 'div' (line 263)
        result_div_26007 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 29), 'div', astype_call_result_26005, int_26006)
        
        # Assigning a type to the variable 'xx' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'xx', result_div_26007)
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 261)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 265)
        # Processing the call arguments (line 265)
        unicode_26009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 37), 'unicode', u'Image RGB array must be uint8 or floating point; found %s')
        # Getting the type of 'xx' (line 266)
        xx_26010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 66), 'xx', False)
        # Obtaining the member 'dtype' of a type (line 266)
        dtype_26011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 66), xx_26010, 'dtype')
        # Applying the binary operator '%' (line 265)
        result_mod_26012 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 37), '%', unicode_26009, dtype_26011)
        
        # Processing the call keyword arguments (line 265)
        kwargs_26013 = {}
        # Getting the type of 'ValueError' (line 265)
        ValueError_26008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 265)
        ValueError_call_result_26014 = invoke(stypy.reporting.localization.Localization(__file__, 265, 26), ValueError_26008, *[result_mod_26012], **kwargs_26013)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 265, 20), ValueError_call_result_26014, 'raise parameter', BaseException)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'xx' (line 267)
        xx_26015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'xx')
        # Assigning a type to the variable 'stypy_return_type' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'stypy_return_type', xx_26015)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 240)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 240)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to asarray(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'x' (line 273)
        x_26018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'x', False)
        # Processing the call keyword arguments (line 273)
        kwargs_26019 = {}
        # Getting the type of 'ma' (line 273)
        ma_26016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'ma', False)
        # Obtaining the member 'asarray' of a type (line 273)
        asarray_26017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), ma_26016, 'asarray')
        # Calling asarray(args, kwargs) (line 273)
        asarray_call_result_26020 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), asarray_26017, *[x_26018], **kwargs_26019)
        
        # Assigning a type to the variable 'x' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'x', asarray_call_result_26020)
        
        # Getting the type of 'norm' (line 274)
        norm_26021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'norm')
        # Testing the type of an if condition (line 274)
        if_condition_26022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 8), norm_26021)
        # Assigning a type to the variable 'if_condition_26022' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'if_condition_26022', if_condition_26022)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to norm(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'x' (line 275)
        x_26025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'x', False)
        # Processing the call keyword arguments (line 275)
        kwargs_26026 = {}
        # Getting the type of 'self' (line 275)
        self_26023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'self', False)
        # Obtaining the member 'norm' of a type (line 275)
        norm_26024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), self_26023, 'norm')
        # Calling norm(args, kwargs) (line 275)
        norm_call_result_26027 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), norm_26024, *[x_26025], **kwargs_26026)
        
        # Assigning a type to the variable 'x' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'x', norm_call_result_26027)
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to cmap(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'x' (line 276)
        x_26030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 25), 'x', False)
        # Processing the call keyword arguments (line 276)
        # Getting the type of 'alpha' (line 276)
        alpha_26031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'alpha', False)
        keyword_26032 = alpha_26031
        # Getting the type of 'bytes' (line 276)
        bytes_26033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 47), 'bytes', False)
        keyword_26034 = bytes_26033
        kwargs_26035 = {'alpha': keyword_26032, 'bytes': keyword_26034}
        # Getting the type of 'self' (line 276)
        self_26028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'self', False)
        # Obtaining the member 'cmap' of a type (line 276)
        cmap_26029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 15), self_26028, 'cmap')
        # Calling cmap(args, kwargs) (line 276)
        cmap_call_result_26036 = invoke(stypy.reporting.localization.Localization(__file__, 276, 15), cmap_26029, *[x_26030], **kwargs_26035)
        
        # Assigning a type to the variable 'rgba' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'rgba', cmap_call_result_26036)
        # Getting the type of 'rgba' (line 277)
        rgba_26037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'rgba')
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stypy_return_type', rgba_26037)
        
        # ################# End of 'to_rgba(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'to_rgba' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_26038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'to_rgba'
        return stypy_return_type_26038


    @norecursion
    def set_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_array'
        module_type_store = module_type_store.open_function_context('set_array', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.set_array.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.set_array')
        ScalarMappable.set_array.__dict__.__setitem__('stypy_param_names_list', ['A'])
        ScalarMappable.set_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.set_array.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.set_array', ['A'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_array', localization, ['A'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_array(...)' code ##################

        unicode_26039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'unicode', u'Set the image array from numpy array *A*')
        
        # Assigning a Name to a Attribute (line 281):
        
        # Assigning a Name to a Attribute (line 281):
        # Getting the type of 'A' (line 281)
        A_26040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'A')
        # Getting the type of 'self' (line 281)
        self_26041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'self')
        # Setting the type of the member '_A' of a type (line 281)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), self_26041, '_A', A_26040)
        
        # Assigning a Name to a Subscript (line 282):
        
        # Assigning a Name to a Subscript (line 282):
        # Getting the type of 'True' (line 282)
        True_26042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'True')
        # Getting the type of 'self' (line 282)
        self_26043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self')
        # Obtaining the member 'update_dict' of a type (line 282)
        update_dict_26044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_26043, 'update_dict')
        unicode_26045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 25), 'unicode', u'array')
        # Storing an element on a container (line 282)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 8), update_dict_26044, (unicode_26045, True_26042))
        
        # ################# End of 'set_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_array' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_26046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26046)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_array'
        return stypy_return_type_26046


    @norecursion
    def get_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_array'
        module_type_store = module_type_store.open_function_context('get_array', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.get_array.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.get_array')
        ScalarMappable.get_array.__dict__.__setitem__('stypy_param_names_list', [])
        ScalarMappable.get_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.get_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.get_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_array(...)' code ##################

        unicode_26047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'unicode', u'Return the array')
        # Getting the type of 'self' (line 286)
        self_26048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'self')
        # Obtaining the member '_A' of a type (line 286)
        _A_26049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), self_26048, '_A')
        # Assigning a type to the variable 'stypy_return_type' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type', _A_26049)
        
        # ################# End of 'get_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_array' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_26050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_array'
        return stypy_return_type_26050


    @norecursion
    def get_cmap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_cmap'
        module_type_store = module_type_store.open_function_context('get_cmap', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.get_cmap')
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_param_names_list', [])
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.get_cmap.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.get_cmap', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_cmap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_cmap(...)' code ##################

        unicode_26051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 8), 'unicode', u'return the colormap')
        # Getting the type of 'self' (line 290)
        self_26052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'self')
        # Obtaining the member 'cmap' of a type (line 290)
        cmap_26053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), self_26052, 'cmap')
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', cmap_26053)
        
        # ################# End of 'get_cmap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_cmap' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_26054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_cmap'
        return stypy_return_type_26054


    @norecursion
    def get_clim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_clim'
        module_type_store = module_type_store.open_function_context('get_clim', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.get_clim')
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_param_names_list', [])
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.get_clim.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.get_clim', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_clim', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_clim(...)' code ##################

        unicode_26055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 8), 'unicode', u'return the min, max of the color limits for image scaling')
        
        # Obtaining an instance of the builtin type 'tuple' (line 294)
        tuple_26056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 294)
        # Adding element type (line 294)
        # Getting the type of 'self' (line 294)
        self_26057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'self')
        # Obtaining the member 'norm' of a type (line 294)
        norm_26058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), self_26057, 'norm')
        # Obtaining the member 'vmin' of a type (line 294)
        vmin_26059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), norm_26058, 'vmin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 15), tuple_26056, vmin_26059)
        # Adding element type (line 294)
        # Getting the type of 'self' (line 294)
        self_26060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'self')
        # Obtaining the member 'norm' of a type (line 294)
        norm_26061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 31), self_26060, 'norm')
        # Obtaining the member 'vmax' of a type (line 294)
        vmax_26062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 31), norm_26061, 'vmax')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 15), tuple_26056, vmax_26062)
        
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', tuple_26056)
        
        # ################# End of 'get_clim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_clim' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_26063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_clim'
        return stypy_return_type_26063


    @norecursion
    def set_clim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 296)
        None_26064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 28), 'None')
        # Getting the type of 'None' (line 296)
        None_26065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'None')
        defaults = [None_26064, None_26065]
        # Create a new context for function 'set_clim'
        module_type_store = module_type_store.open_function_context('set_clim', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.set_clim')
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_param_names_list', ['vmin', 'vmax'])
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.set_clim.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.set_clim', ['vmin', 'vmax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_clim', localization, ['vmin', 'vmax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_clim(...)' code ##################

        unicode_26066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'unicode', u'\n        set the norm limits for image scaling; if *vmin* is a length2\n        sequence, interpret it as ``(vmin, vmax)`` which is used to\n        support setp\n\n        ACCEPTS: a length 2 sequence of floats\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 304)
        # Getting the type of 'vmax' (line 304)
        vmax_26067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'vmax')
        # Getting the type of 'None' (line 304)
        None_26068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'None')
        
        (may_be_26069, more_types_in_union_26070) = may_be_none(vmax_26067, None_26068)

        if may_be_26069:

            if more_types_in_union_26070:
                # Runtime conditional SSA (line 304)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Name to a Tuple (line 306):
            
            # Assigning a Subscript to a Name (line 306):
            
            # Obtaining the type of the subscript
            int_26071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 16), 'int')
            # Getting the type of 'vmin' (line 306)
            vmin_26072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'vmin')
            # Obtaining the member '__getitem__' of a type (line 306)
            getitem___26073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 16), vmin_26072, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 306)
            subscript_call_result_26074 = invoke(stypy.reporting.localization.Localization(__file__, 306, 16), getitem___26073, int_26071)
            
            # Assigning a type to the variable 'tuple_var_assignment_25512' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'tuple_var_assignment_25512', subscript_call_result_26074)
            
            # Assigning a Subscript to a Name (line 306):
            
            # Obtaining the type of the subscript
            int_26075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 16), 'int')
            # Getting the type of 'vmin' (line 306)
            vmin_26076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'vmin')
            # Obtaining the member '__getitem__' of a type (line 306)
            getitem___26077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 16), vmin_26076, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 306)
            subscript_call_result_26078 = invoke(stypy.reporting.localization.Localization(__file__, 306, 16), getitem___26077, int_26075)
            
            # Assigning a type to the variable 'tuple_var_assignment_25513' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'tuple_var_assignment_25513', subscript_call_result_26078)
            
            # Assigning a Name to a Name (line 306):
            # Getting the type of 'tuple_var_assignment_25512' (line 306)
            tuple_var_assignment_25512_26079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'tuple_var_assignment_25512')
            # Assigning a type to the variable 'vmin' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'vmin', tuple_var_assignment_25512_26079)
            
            # Assigning a Name to a Name (line 306):
            # Getting the type of 'tuple_var_assignment_25513' (line 306)
            tuple_var_assignment_25513_26080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'tuple_var_assignment_25513')
            # Assigning a type to the variable 'vmax' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'vmax', tuple_var_assignment_25513_26080)
            # SSA branch for the except part of a try statement (line 305)
            # SSA branch for the except 'Tuple' branch of a try statement (line 305)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 305)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_26070:
                # SSA join for if statement (line 304)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 309)
        # Getting the type of 'vmin' (line 309)
        vmin_26081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'vmin')
        # Getting the type of 'None' (line 309)
        None_26082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'None')
        
        (may_be_26083, more_types_in_union_26084) = may_not_be_none(vmin_26081, None_26082)

        if may_be_26083:

            if more_types_in_union_26084:
                # Runtime conditional SSA (line 309)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 310):
            
            # Assigning a Name to a Attribute (line 310):
            # Getting the type of 'vmin' (line 310)
            vmin_26085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'vmin')
            # Getting the type of 'self' (line 310)
            self_26086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'self')
            # Obtaining the member 'norm' of a type (line 310)
            norm_26087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), self_26086, 'norm')
            # Setting the type of the member 'vmin' of a type (line 310)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), norm_26087, 'vmin', vmin_26085)

            if more_types_in_union_26084:
                # SSA join for if statement (line 309)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 311)
        # Getting the type of 'vmax' (line 311)
        vmax_26088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'vmax')
        # Getting the type of 'None' (line 311)
        None_26089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'None')
        
        (may_be_26090, more_types_in_union_26091) = may_not_be_none(vmax_26088, None_26089)

        if may_be_26090:

            if more_types_in_union_26091:
                # Runtime conditional SSA (line 311)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 312):
            
            # Assigning a Name to a Attribute (line 312):
            # Getting the type of 'vmax' (line 312)
            vmax_26092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'vmax')
            # Getting the type of 'self' (line 312)
            self_26093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'self')
            # Obtaining the member 'norm' of a type (line 312)
            norm_26094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), self_26093, 'norm')
            # Setting the type of the member 'vmax' of a type (line 312)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), norm_26094, 'vmax', vmax_26092)

            if more_types_in_union_26091:
                # SSA join for if statement (line 311)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to changed(...): (line 313)
        # Processing the call keyword arguments (line 313)
        kwargs_26097 = {}
        # Getting the type of 'self' (line 313)
        self_26095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self', False)
        # Obtaining the member 'changed' of a type (line 313)
        changed_26096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_26095, 'changed')
        # Calling changed(args, kwargs) (line 313)
        changed_call_result_26098 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), changed_26096, *[], **kwargs_26097)
        
        
        # ################# End of 'set_clim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clim' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_26099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clim'
        return stypy_return_type_26099


    @norecursion
    def set_cmap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cmap'
        module_type_store = module_type_store.open_function_context('set_cmap', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.set_cmap')
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_param_names_list', ['cmap'])
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.set_cmap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.set_cmap', ['cmap'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_cmap', localization, ['cmap'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_cmap(...)' code ##################

        unicode_26100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, (-1)), 'unicode', u'\n        set the colormap for luminance data\n\n        ACCEPTS: a colormap or registered colormap name\n        ')
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to get_cmap(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'cmap' (line 321)
        cmap_26102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'cmap', False)
        # Processing the call keyword arguments (line 321)
        kwargs_26103 = {}
        # Getting the type of 'get_cmap' (line 321)
        get_cmap_26101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 15), 'get_cmap', False)
        # Calling get_cmap(args, kwargs) (line 321)
        get_cmap_call_result_26104 = invoke(stypy.reporting.localization.Localization(__file__, 321, 15), get_cmap_26101, *[cmap_26102], **kwargs_26103)
        
        # Assigning a type to the variable 'cmap' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'cmap', get_cmap_call_result_26104)
        
        # Assigning a Name to a Attribute (line 322):
        
        # Assigning a Name to a Attribute (line 322):
        # Getting the type of 'cmap' (line 322)
        cmap_26105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'cmap')
        # Getting the type of 'self' (line 322)
        self_26106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self')
        # Setting the type of the member 'cmap' of a type (line 322)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_26106, 'cmap', cmap_26105)
        
        # Call to changed(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_26109 = {}
        # Getting the type of 'self' (line 323)
        self_26107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self', False)
        # Obtaining the member 'changed' of a type (line 323)
        changed_26108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_26107, 'changed')
        # Calling changed(args, kwargs) (line 323)
        changed_call_result_26110 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), changed_26108, *[], **kwargs_26109)
        
        
        # ################# End of 'set_cmap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cmap' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_26111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cmap'
        return stypy_return_type_26111


    @norecursion
    def set_norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_norm'
        module_type_store = module_type_store.open_function_context('set_norm', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.set_norm')
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_param_names_list', ['norm'])
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.set_norm.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.set_norm', ['norm'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_norm', localization, ['norm'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_norm(...)' code ##################

        unicode_26112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 8), 'unicode', u'set the normalization instance')
        
        # Type idiom detected: calculating its left and rigth part (line 327)
        # Getting the type of 'norm' (line 327)
        norm_26113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'norm')
        # Getting the type of 'None' (line 327)
        None_26114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'None')
        
        (may_be_26115, more_types_in_union_26116) = may_be_none(norm_26113, None_26114)

        if may_be_26115:

            if more_types_in_union_26116:
                # Runtime conditional SSA (line 327)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 328):
            
            # Assigning a Call to a Name (line 328):
            
            # Call to Normalize(...): (line 328)
            # Processing the call keyword arguments (line 328)
            kwargs_26119 = {}
            # Getting the type of 'colors' (line 328)
            colors_26117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'colors', False)
            # Obtaining the member 'Normalize' of a type (line 328)
            Normalize_26118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 19), colors_26117, 'Normalize')
            # Calling Normalize(args, kwargs) (line 328)
            Normalize_call_result_26120 = invoke(stypy.reporting.localization.Localization(__file__, 328, 19), Normalize_26118, *[], **kwargs_26119)
            
            # Assigning a type to the variable 'norm' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'norm', Normalize_call_result_26120)

            if more_types_in_union_26116:
                # SSA join for if statement (line 327)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 329):
        
        # Assigning a Name to a Attribute (line 329):
        # Getting the type of 'norm' (line 329)
        norm_26121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'norm')
        # Getting the type of 'self' (line 329)
        self_26122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self')
        # Setting the type of the member 'norm' of a type (line 329)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_26122, 'norm', norm_26121)
        
        # Call to changed(...): (line 330)
        # Processing the call keyword arguments (line 330)
        kwargs_26125 = {}
        # Getting the type of 'self' (line 330)
        self_26123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self', False)
        # Obtaining the member 'changed' of a type (line 330)
        changed_26124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_26123, 'changed')
        # Calling changed(args, kwargs) (line 330)
        changed_call_result_26126 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), changed_26124, *[], **kwargs_26125)
        
        
        # ################# End of 'set_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_26127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_norm'
        return stypy_return_type_26127


    @norecursion
    def autoscale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'autoscale'
        module_type_store = module_type_store.open_function_context('autoscale', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.autoscale')
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_param_names_list', [])
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.autoscale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.autoscale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'autoscale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'autoscale(...)' code ##################

        unicode_26128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, (-1)), 'unicode', u'\n        Autoscale the scalar limits on the norm instance using the\n        current array\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 337)
        # Getting the type of 'self' (line 337)
        self_26129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 11), 'self')
        # Obtaining the member '_A' of a type (line 337)
        _A_26130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 11), self_26129, '_A')
        # Getting the type of 'None' (line 337)
        None_26131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'None')
        
        (may_be_26132, more_types_in_union_26133) = may_be_none(_A_26130, None_26131)

        if may_be_26132:

            if more_types_in_union_26133:
                # Runtime conditional SSA (line 337)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 338)
            # Processing the call arguments (line 338)
            unicode_26135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 28), 'unicode', u'You must first set_array for mappable')
            # Processing the call keyword arguments (line 338)
            kwargs_26136 = {}
            # Getting the type of 'TypeError' (line 338)
            TypeError_26134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 338)
            TypeError_call_result_26137 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), TypeError_26134, *[unicode_26135], **kwargs_26136)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 338, 12), TypeError_call_result_26137, 'raise parameter', BaseException)

            if more_types_in_union_26133:
                # SSA join for if statement (line 337)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to autoscale(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'self' (line 339)
        self_26141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 28), 'self', False)
        # Obtaining the member '_A' of a type (line 339)
        _A_26142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 28), self_26141, '_A')
        # Processing the call keyword arguments (line 339)
        kwargs_26143 = {}
        # Getting the type of 'self' (line 339)
        self_26138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self', False)
        # Obtaining the member 'norm' of a type (line 339)
        norm_26139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_26138, 'norm')
        # Obtaining the member 'autoscale' of a type (line 339)
        autoscale_26140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), norm_26139, 'autoscale')
        # Calling autoscale(args, kwargs) (line 339)
        autoscale_call_result_26144 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), autoscale_26140, *[_A_26142], **kwargs_26143)
        
        
        # Call to changed(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_26147 = {}
        # Getting the type of 'self' (line 340)
        self_26145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self', False)
        # Obtaining the member 'changed' of a type (line 340)
        changed_26146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_26145, 'changed')
        # Calling changed(args, kwargs) (line 340)
        changed_call_result_26148 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), changed_26146, *[], **kwargs_26147)
        
        
        # ################# End of 'autoscale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'autoscale' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_26149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'autoscale'
        return stypy_return_type_26149


    @norecursion
    def autoscale_None(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'autoscale_None'
        module_type_store = module_type_store.open_function_context('autoscale_None', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.autoscale_None')
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_param_names_list', [])
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.autoscale_None.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.autoscale_None', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'autoscale_None', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'autoscale_None(...)' code ##################

        unicode_26150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, (-1)), 'unicode', u'\n        Autoscale the scalar limits on the norm instance using the\n        current array, changing only limits that are None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 347)
        # Getting the type of 'self' (line 347)
        self_26151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'self')
        # Obtaining the member '_A' of a type (line 347)
        _A_26152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 11), self_26151, '_A')
        # Getting the type of 'None' (line 347)
        None_26153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'None')
        
        (may_be_26154, more_types_in_union_26155) = may_be_none(_A_26152, None_26153)

        if may_be_26154:

            if more_types_in_union_26155:
                # Runtime conditional SSA (line 347)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 348)
            # Processing the call arguments (line 348)
            unicode_26157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 28), 'unicode', u'You must first set_array for mappable')
            # Processing the call keyword arguments (line 348)
            kwargs_26158 = {}
            # Getting the type of 'TypeError' (line 348)
            TypeError_26156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 348)
            TypeError_call_result_26159 = invoke(stypy.reporting.localization.Localization(__file__, 348, 18), TypeError_26156, *[unicode_26157], **kwargs_26158)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 348, 12), TypeError_call_result_26159, 'raise parameter', BaseException)

            if more_types_in_union_26155:
                # SSA join for if statement (line 347)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to autoscale_None(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'self' (line 349)
        self_26163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 33), 'self', False)
        # Obtaining the member '_A' of a type (line 349)
        _A_26164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 33), self_26163, '_A')
        # Processing the call keyword arguments (line 349)
        kwargs_26165 = {}
        # Getting the type of 'self' (line 349)
        self_26160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self', False)
        # Obtaining the member 'norm' of a type (line 349)
        norm_26161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_26160, 'norm')
        # Obtaining the member 'autoscale_None' of a type (line 349)
        autoscale_None_26162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), norm_26161, 'autoscale_None')
        # Calling autoscale_None(args, kwargs) (line 349)
        autoscale_None_call_result_26166 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), autoscale_None_26162, *[_A_26164], **kwargs_26165)
        
        
        # Call to changed(...): (line 350)
        # Processing the call keyword arguments (line 350)
        kwargs_26169 = {}
        # Getting the type of 'self' (line 350)
        self_26167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member 'changed' of a type (line 350)
        changed_26168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_26167, 'changed')
        # Calling changed(args, kwargs) (line 350)
        changed_call_result_26170 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), changed_26168, *[], **kwargs_26169)
        
        
        # ################# End of 'autoscale_None(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'autoscale_None' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_26171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'autoscale_None'
        return stypy_return_type_26171


    @norecursion
    def add_checker(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_checker'
        module_type_store = module_type_store.open_function_context('add_checker', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.add_checker')
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_param_names_list', ['checker'])
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.add_checker.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.add_checker', ['checker'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_checker', localization, ['checker'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_checker(...)' code ##################

        unicode_26172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, (-1)), 'unicode', u'\n        Add an entry to a dictionary of boolean flags\n        that are set to True when the mappable is changed.\n        ')
        
        # Assigning a Name to a Subscript (line 357):
        
        # Assigning a Name to a Subscript (line 357):
        # Getting the type of 'False' (line 357)
        False_26173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 36), 'False')
        # Getting the type of 'self' (line 357)
        self_26174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'self')
        # Obtaining the member 'update_dict' of a type (line 357)
        update_dict_26175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), self_26174, 'update_dict')
        # Getting the type of 'checker' (line 357)
        checker_26176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 25), 'checker')
        # Storing an element on a container (line 357)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 8), update_dict_26175, (checker_26176, False_26173))
        
        # ################# End of 'add_checker(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_checker' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_26177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_checker'
        return stypy_return_type_26177


    @norecursion
    def check_update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_update'
        module_type_store = module_type_store.open_function_context('check_update', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.check_update.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.check_update')
        ScalarMappable.check_update.__dict__.__setitem__('stypy_param_names_list', ['checker'])
        ScalarMappable.check_update.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.check_update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.check_update', ['checker'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_update', localization, ['checker'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_update(...)' code ##################

        unicode_26178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, (-1)), 'unicode', u'\n        If mappable has changed since the last check,\n        return True; else return False\n        ')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'checker' (line 364)
        checker_26179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 28), 'checker')
        # Getting the type of 'self' (line 364)
        self_26180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'self')
        # Obtaining the member 'update_dict' of a type (line 364)
        update_dict_26181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 11), self_26180, 'update_dict')
        # Obtaining the member '__getitem__' of a type (line 364)
        getitem___26182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 11), update_dict_26181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 364)
        subscript_call_result_26183 = invoke(stypy.reporting.localization.Localization(__file__, 364, 11), getitem___26182, checker_26179)
        
        # Testing the type of an if condition (line 364)
        if_condition_26184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 8), subscript_call_result_26183)
        # Assigning a type to the variable 'if_condition_26184' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'if_condition_26184', if_condition_26184)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 365):
        
        # Assigning a Name to a Subscript (line 365):
        # Getting the type of 'False' (line 365)
        False_26185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 40), 'False')
        # Getting the type of 'self' (line 365)
        self_26186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'self')
        # Obtaining the member 'update_dict' of a type (line 365)
        update_dict_26187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 12), self_26186, 'update_dict')
        # Getting the type of 'checker' (line 365)
        checker_26188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 29), 'checker')
        # Storing an element on a container (line 365)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 12), update_dict_26187, (checker_26188, False_26185))
        # Getting the type of 'True' (line 366)
        True_26189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'stypy_return_type', True_26189)
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 367)
        False_26190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'stypy_return_type', False_26190)
        
        # ################# End of 'check_update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_update' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_26191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_update'
        return stypy_return_type_26191


    @norecursion
    def changed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'changed'
        module_type_store = module_type_store.open_function_context('changed', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ScalarMappable.changed.__dict__.__setitem__('stypy_localization', localization)
        ScalarMappable.changed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ScalarMappable.changed.__dict__.__setitem__('stypy_type_store', module_type_store)
        ScalarMappable.changed.__dict__.__setitem__('stypy_function_name', 'ScalarMappable.changed')
        ScalarMappable.changed.__dict__.__setitem__('stypy_param_names_list', [])
        ScalarMappable.changed.__dict__.__setitem__('stypy_varargs_param_name', None)
        ScalarMappable.changed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ScalarMappable.changed.__dict__.__setitem__('stypy_call_defaults', defaults)
        ScalarMappable.changed.__dict__.__setitem__('stypy_call_varargs', varargs)
        ScalarMappable.changed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ScalarMappable.changed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ScalarMappable.changed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'changed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'changed(...)' code ##################

        unicode_26192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, (-1)), 'unicode', u"\n        Call this whenever the mappable is changed to notify all the\n        callbackSM listeners to the 'changed' signal\n        ")
        
        # Call to process(...): (line 374)
        # Processing the call arguments (line 374)
        unicode_26196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 33), 'unicode', u'changed')
        # Getting the type of 'self' (line 374)
        self_26197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 44), 'self', False)
        # Processing the call keyword arguments (line 374)
        kwargs_26198 = {}
        # Getting the type of 'self' (line 374)
        self_26193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'self', False)
        # Obtaining the member 'callbacksSM' of a type (line 374)
        callbacksSM_26194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), self_26193, 'callbacksSM')
        # Obtaining the member 'process' of a type (line 374)
        process_26195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), callbacksSM_26194, 'process')
        # Calling process(args, kwargs) (line 374)
        process_call_result_26199 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), process_26195, *[unicode_26196, self_26197], **kwargs_26198)
        
        
        # Getting the type of 'self' (line 376)
        self_26200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'self')
        # Obtaining the member 'update_dict' of a type (line 376)
        update_dict_26201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 19), self_26200, 'update_dict')
        # Testing the type of a for loop iterable (line 376)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 376, 8), update_dict_26201)
        # Getting the type of the for loop variable (line 376)
        for_loop_var_26202 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 376, 8), update_dict_26201)
        # Assigning a type to the variable 'key' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'key', for_loop_var_26202)
        # SSA begins for a for statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 377):
        
        # Assigning a Name to a Subscript (line 377):
        # Getting the type of 'True' (line 377)
        True_26203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'True')
        # Getting the type of 'self' (line 377)
        self_26204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'self')
        # Obtaining the member 'update_dict' of a type (line 377)
        update_dict_26205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 12), self_26204, 'update_dict')
        # Getting the type of 'key' (line 377)
        key_26206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 29), 'key')
        # Storing an element on a container (line 377)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), update_dict_26205, (key_26206, True_26203))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 378):
        
        # Assigning a Name to a Attribute (line 378):
        # Getting the type of 'True' (line 378)
        True_26207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 21), 'True')
        # Getting the type of 'self' (line 378)
        self_26208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 378)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_26208, 'stale', True_26207)
        
        # ################# End of 'changed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'changed' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_26209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'changed'
        return stypy_return_type_26209


# Assigning a type to the variable 'ScalarMappable' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'ScalarMappable', ScalarMappable)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

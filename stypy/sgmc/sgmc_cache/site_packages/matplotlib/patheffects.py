
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Defines classes for path effects. The path effects are supported in
3: :class:`~matplotlib.text.Text`, :class:`~matplotlib.lines.Line2D`
4: and :class:`~matplotlib.patches.Patch`.
5: '''
6: 
7: from __future__ import (absolute_import, division, print_function,
8:                         unicode_literals)
9: 
10: import six
11: 
12: from matplotlib.backend_bases import RendererBase
13: from matplotlib import colors as mcolors
14: from matplotlib import patches as mpatches
15: from matplotlib import transforms as mtransforms
16: 
17: 
18: class AbstractPathEffect(object):
19:     '''
20:     A base class for path effects.
21: 
22:     Subclasses should override the ``draw_path`` method to add effect
23:     functionality.
24: 
25:     '''
26:     def __init__(self, offset=(0., 0.)):
27:         '''
28:         Parameters
29:         ----------
30:         offset : pair of floats
31:             The offset to apply to the path, measured in points.
32:         '''
33:         self._offset = offset
34:         self._offset_trans = mtransforms.Affine2D()
35: 
36:     def _offset_transform(self, renderer, transform):
37:         '''Apply the offset to the given transform.'''
38:         offset_x = renderer.points_to_pixels(self._offset[0])
39:         offset_y = renderer.points_to_pixels(self._offset[1])
40:         return transform + self._offset_trans.clear().translate(offset_x,
41:                                                                 offset_y)
42: 
43:     def _update_gc(self, gc, new_gc_dict):
44:         '''
45:         Update the given GraphicsCollection with the given
46:         dictionary of properties. The keys in the dictionary are used to
47:         identify the appropriate set_ method on the gc.
48: 
49:         '''
50:         new_gc_dict = new_gc_dict.copy()
51: 
52:         dashes = new_gc_dict.pop("dashes", None)
53:         if dashes:
54:             gc.set_dashes(**dashes)
55: 
56:         for k, v in six.iteritems(new_gc_dict):
57:             set_method = getattr(gc, 'set_' + k, None)
58:             if not callable(set_method):
59:                 raise AttributeError('Unknown property {0}'.format(k))
60:             set_method(v)
61:         return gc
62: 
63:     def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
64:         '''
65:         Derived should override this method. The arguments are the same
66:         as :meth:`matplotlib.backend_bases.RendererBase.draw_path`
67:         except the first argument is a renderer.
68: 
69:         '''
70:         # Get the real renderer, not a PathEffectRenderer.
71:         if isinstance(renderer, PathEffectRenderer):
72:             renderer = renderer._renderer
73:         return renderer.draw_path(gc, tpath, affine, rgbFace)
74: 
75: 
76: class PathEffectRenderer(RendererBase):
77:     '''
78:     Implements a Renderer which contains another renderer.
79: 
80:     This proxy then intercepts draw calls, calling the appropriate
81:     :class:`AbstractPathEffect` draw method.
82: 
83:     .. note::
84:         Not all methods have been overridden on this RendererBase subclass.
85:         It may be necessary to add further methods to extend the PathEffects
86:         capabilities further.
87: 
88:     '''
89:     def __init__(self, path_effects, renderer):
90:         '''
91:         Parameters
92:         ----------
93:         path_effects : iterable of :class:`AbstractPathEffect`
94:             The path effects which this renderer represents.
95:         renderer : :class:`matplotlib.backend_bases.RendererBase` instance
96: 
97:         '''
98:         self._path_effects = path_effects
99:         self._renderer = renderer
100: 
101:     def new_gc(self):
102:         return self._renderer.new_gc()
103: 
104:     def copy_with_path_effect(self, path_effects):
105:         return self.__class__(path_effects, self._renderer)
106: 
107:     def draw_path(self, gc, tpath, affine, rgbFace=None):
108:         for path_effect in self._path_effects:
109:             path_effect.draw_path(self._renderer, gc, tpath, affine,
110:                                   rgbFace)
111: 
112:     def draw_markers(self, gc, marker_path, marker_trans, path, *args,
113:                              **kwargs):
114:         # We do a little shimmy so that all markers are drawn for each path
115:         # effect in turn. Essentially, we induce recursion (depth 1) which is
116:         # terminated once we have just a single path effect to work with.
117:         if len(self._path_effects) == 1:
118:             # Call the base path effect function - this uses the unoptimised
119:             # approach of calling "draw_path" multiple times.
120:             return RendererBase.draw_markers(self, gc, marker_path,
121:                                              marker_trans, path, *args,
122:                                              **kwargs)
123: 
124:         for path_effect in self._path_effects:
125:             renderer = self.copy_with_path_effect([path_effect])
126:             # Recursively call this method, only next time we will only have
127:             # one path effect.
128:             renderer.draw_markers(gc, marker_path, marker_trans, path,
129:                                   *args, **kwargs)
130: 
131:     def draw_path_collection(self, gc, master_transform, paths, *args,
132:                              **kwargs):
133:         # We do a little shimmy so that all paths are drawn for each path
134:         # effect in turn. Essentially, we induce recursion (depth 1) which is
135:         # terminated once we have just a single path effect to work with.
136:         if len(self._path_effects) == 1:
137:             # Call the base path effect function - this uses the unoptimised
138:             # approach of calling "draw_path" multiple times.
139:             return RendererBase.draw_path_collection(self, gc,
140:                                                      master_transform, paths,
141:                                                      *args, **kwargs)
142: 
143:         for path_effect in self._path_effects:
144:             renderer = self.copy_with_path_effect([path_effect])
145:             # Recursively call this method, only next time we will only have
146:             # one path effect.
147:             renderer.draw_path_collection(gc, master_transform, paths,
148:                                           *args, **kwargs)
149: 
150:     def points_to_pixels(self, points):
151:         return self._renderer.points_to_pixels(points)
152: 
153:     def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath):
154:         # Implements the naive text drawing as is found in RendererBase.
155:         path, transform = self._get_text_path_transform(x, y, s, prop,
156:                                                         angle, ismath)
157:         color = gc.get_rgb()
158:         gc.set_linewidth(0.0)
159:         self.draw_path(gc, path, transform, rgbFace=color)
160: 
161:     def __getattribute__(self, name):
162:         if name in ['_text2path', 'flipy', 'height', 'width']:
163:             return getattr(self._renderer, name)
164:         else:
165:             return object.__getattribute__(self, name)
166: 
167: 
168: class Normal(AbstractPathEffect):
169:     '''
170:     The "identity" PathEffect.
171: 
172:     The Normal PathEffect's sole purpose is to draw the original artist with
173:     no special path effect.
174:     '''
175:     pass
176: 
177: 
178: class Stroke(AbstractPathEffect):
179:     '''A line based PathEffect which re-draws a stroke.'''
180:     def __init__(self, offset=(0, 0), **kwargs):
181:         '''
182:         The path will be stroked with its gc updated with the given
183:         keyword arguments, i.e., the keyword arguments should be valid
184:         gc parameter values.
185:         '''
186:         super(Stroke, self).__init__(offset)
187:         self._gc = kwargs
188: 
189:     def draw_path(self, renderer, gc, tpath, affine, rgbFace):
190:         '''
191:         draw the path with updated gc.
192:         '''
193:         # Do not modify the input! Use copy instead.
194: 
195:         gc0 = renderer.new_gc()
196:         gc0.copy_properties(gc)
197: 
198:         gc0 = self._update_gc(gc0, self._gc)
199:         trans = self._offset_transform(renderer, affine)
200:         renderer.draw_path(gc0, tpath, trans, rgbFace)
201:         gc0.restore()
202: 
203: 
204: class withStroke(Stroke):
205:     '''
206:     Adds a simple :class:`Stroke` and then draws the
207:     original Artist to avoid needing to call :class:`Normal`.
208: 
209:     '''
210:     def draw_path(self, renderer, gc, tpath, affine, rgbFace):
211:         Stroke.draw_path(self, renderer, gc, tpath, affine, rgbFace)
212:         renderer.draw_path(gc, tpath, affine, rgbFace)
213: 
214: 
215: class SimplePatchShadow(AbstractPathEffect):
216:     '''A simple shadow via a filled patch.'''
217:     def __init__(self, offset=(2, -2),
218:                  shadow_rgbFace=None, alpha=None,
219:                  rho=0.3, **kwargs):
220:         '''
221:         Parameters
222:         ----------
223:         offset : pair of floats
224:             The offset of the shadow in points.
225:         shadow_rgbFace : color
226:             The shadow color.
227:         alpha : float
228:             The alpha transparency of the created shadow patch.
229:             Default is 0.3.
230:             http://matplotlib.1069221.n5.nabble.com/path-effects-question-td27630.html
231:         rho : float
232:             A scale factor to apply to the rgbFace color if `shadow_rgbFace`
233:             is not specified. Default is 0.3.
234:         **kwargs
235:             Extra keywords are stored and passed through to
236:             :meth:`AbstractPathEffect._update_gc`.
237: 
238:         '''
239:         super(SimplePatchShadow, self).__init__(offset)
240: 
241:         if shadow_rgbFace is None:
242:             self._shadow_rgbFace = shadow_rgbFace
243:         else:
244:             self._shadow_rgbFace = mcolors.to_rgba(shadow_rgbFace)
245: 
246:         if alpha is None:
247:             alpha = 0.3
248: 
249:         self._alpha = alpha
250:         self._rho = rho
251: 
252:         #: The dictionary of keywords to update the graphics collection with.
253:         self._gc = kwargs
254: 
255:         #: The offset transform object. The offset isn't calculated yet
256:         #: as we don't know how big the figure will be in pixels.
257:         self._offset_tran = mtransforms.Affine2D()
258: 
259:     def draw_path(self, renderer, gc, tpath, affine, rgbFace):
260:         '''
261:         Overrides the standard draw_path to add the shadow offset and
262:         necessary color changes for the shadow.
263: 
264:         '''
265:         # IMPORTANT: Do not modify the input - we copy everything instead.
266:         affine0 = self._offset_transform(renderer, affine)
267:         gc0 = renderer.new_gc()
268:         gc0.copy_properties(gc)
269: 
270:         if self._shadow_rgbFace is None:
271:             r,g,b = (rgbFace or (1., 1., 1.))[:3]
272:             # Scale the colors by a factor to improve the shadow effect.
273:             shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
274:         else:
275:             shadow_rgbFace = self._shadow_rgbFace
276: 
277:         gc0.set_foreground("none")
278:         gc0.set_alpha(self._alpha)
279:         gc0.set_linewidth(0)
280: 
281:         gc0 = self._update_gc(gc0, self._gc)
282:         renderer.draw_path(gc0, tpath, affine0, shadow_rgbFace)
283:         gc0.restore()
284: 
285: 
286: class withSimplePatchShadow(SimplePatchShadow):
287:     '''
288:     Adds a simple :class:`SimplePatchShadow` and then draws the
289:     original Artist to avoid needing to call :class:`Normal`.
290: 
291:     '''
292:     def draw_path(self, renderer, gc, tpath, affine, rgbFace):
293:         SimplePatchShadow.draw_path(self, renderer, gc, tpath, affine, rgbFace)
294:         renderer.draw_path(gc, tpath, affine, rgbFace)
295: 
296: 
297: class SimpleLineShadow(AbstractPathEffect):
298:     '''A simple shadow via a line.'''
299:     def __init__(self, offset=(2,-2),
300:                  shadow_color='k', alpha=0.3, rho=0.3, **kwargs):
301:         '''
302:         Parameters
303:         ----------
304:         offset : pair of floats
305:             The offset to apply to the path, in points.
306:         shadow_color : color
307:             The shadow color. Default is black.
308:             A value of ``None`` takes the original artist's color
309:             with a scale factor of `rho`.
310:         alpha : float
311:             The alpha transparency of the created shadow patch.
312:             Default is 0.3.
313:         rho : float
314:             A scale factor to apply to the rgbFace color if `shadow_rgbFace`
315:             is ``None``. Default is 0.3.
316:         **kwargs
317:             Extra keywords are stored and passed through to
318:             :meth:`AbstractPathEffect._update_gc`.
319: 
320:         '''
321:         super(SimpleLineShadow, self).__init__(offset)
322:         if shadow_color is None:
323:             self._shadow_color = shadow_color
324:         else:
325:             self._shadow_color = mcolors.to_rgba(shadow_color)
326:         self._alpha = alpha
327:         self._rho = rho
328: 
329:         #: The dictionary of keywords to update the graphics collection with.
330:         self._gc = kwargs
331: 
332:         #: The offset transform object. The offset isn't calculated yet
333:         #: as we don't know how big the figure will be in pixels.
334:         self._offset_tran = mtransforms.Affine2D()
335: 
336:     def draw_path(self, renderer, gc, tpath, affine, rgbFace):
337:         '''
338:         Overrides the standard draw_path to add the shadow offset and
339:         necessary color changes for the shadow.
340: 
341:         '''
342:         # IMPORTANT: Do not modify the input - we copy everything instead.
343:         affine0 = self._offset_transform(renderer, affine)
344:         gc0 = renderer.new_gc()
345:         gc0.copy_properties(gc)
346: 
347:         if self._shadow_color is None:
348:             r,g,b = (gc0.get_foreground() or (1., 1., 1.))[:3]
349:             # Scale the colors by a factor to improve the shadow effect.
350:             shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
351:         else:
352:             shadow_rgbFace = self._shadow_color
353: 
354:         fill_color = None
355: 
356:         gc0.set_foreground(shadow_rgbFace)
357:         gc0.set_alpha(self._alpha)
358: 
359:         gc0 = self._update_gc(gc0, self._gc)
360:         renderer.draw_path(gc0, tpath, affine0, fill_color)
361:         gc0.restore()
362: 
363: 
364: class PathPatchEffect(AbstractPathEffect):
365:     '''
366:     Draws a :class:`~matplotlib.patches.PathPatch` instance whose Path
367:     comes from the original PathEffect artist.
368: 
369:     '''
370:     def __init__(self, offset=(0, 0), **kwargs):
371:         '''
372:         Parameters
373:         ----------
374:         offset : pair of floats
375:             The offset to apply to the path, in points.
376:         **kwargs :
377:             All keyword arguments are passed through to the
378:             :class:`~matplotlib.patches.PathPatch` constructor. The
379:             properties which cannot be overridden are "path", "clip_box"
380:             "transform" and "clip_path".
381:         '''
382:         super(PathPatchEffect, self).__init__(offset=offset)
383:         self.patch = mpatches.PathPatch([], **kwargs)
384: 
385:     def draw_path(self, renderer, gc, tpath, affine, rgbFace):
386:         affine = self._offset_transform(renderer, affine)
387:         self.patch._path = tpath
388:         self.patch.set_transform(affine)
389:         self.patch.set_clip_box(gc.get_clip_rectangle())
390:         clip_path = gc.get_clip_path()
391:         if clip_path:
392:             self.patch.set_clip_path(*clip_path)
393:         self.patch.draw(renderer)
394: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_113755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'unicode', u'\nDefines classes for path effects. The path effects are supported in\n:class:`~matplotlib.text.Text`, :class:`~matplotlib.lines.Line2D`\nand :class:`~matplotlib.patches.Patch`.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import six' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_113756 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six')

if (type(import_113756) is not StypyTypeError):

    if (import_113756 != 'pyd_module'):
        __import__(import_113756)
        sys_modules_113757 = sys.modules[import_113756]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', sys_modules_113757.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', import_113756)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.backend_bases import RendererBase' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_113758 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases')

if (type(import_113758) is not StypyTypeError):

    if (import_113758 != 'pyd_module'):
        __import__(import_113758)
        sys_modules_113759 = sys.modules[import_113758]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases', sys_modules_113759.module_type_store, module_type_store, ['RendererBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_113759, sys_modules_113759.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import RendererBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases', None, module_type_store, ['RendererBase'], [RendererBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backend_bases', import_113758)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib import mcolors' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_113760 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib')

if (type(import_113760) is not StypyTypeError):

    if (import_113760 != 'pyd_module'):
        __import__(import_113760)
        sys_modules_113761 = sys.modules[import_113760]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib', sys_modules_113761.module_type_store, module_type_store, ['colors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_113761, sys_modules_113761.module_type_store, module_type_store)
    else:
        from matplotlib import colors as mcolors

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib', None, module_type_store, ['colors'], [mcolors])

else:
    # Assigning a type to the variable 'matplotlib' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib', import_113760)

# Adding an alias
module_type_store.add_alias('mcolors', 'colors')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import mpatches' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_113762 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_113762) is not StypyTypeError):

    if (import_113762 != 'pyd_module'):
        __import__(import_113762)
        sys_modules_113763 = sys.modules[import_113762]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_113763.module_type_store, module_type_store, ['patches'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_113763, sys_modules_113763.module_type_store, module_type_store)
    else:
        from matplotlib import patches as mpatches

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['patches'], [mpatches])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_113762)

# Adding an alias
module_type_store.add_alias('mpatches', 'patches')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib import mtransforms' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_113764 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib')

if (type(import_113764) is not StypyTypeError):

    if (import_113764 != 'pyd_module'):
        __import__(import_113764)
        sys_modules_113765 = sys.modules[import_113764]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', sys_modules_113765.module_type_store, module_type_store, ['transforms'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_113765, sys_modules_113765.module_type_store, module_type_store)
    else:
        from matplotlib import transforms as mtransforms

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', None, module_type_store, ['transforms'], [mtransforms])

else:
    # Assigning a type to the variable 'matplotlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', import_113764)

# Adding an alias
module_type_store.add_alias('mtransforms', 'transforms')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'AbstractPathEffect' class

class AbstractPathEffect(object, ):
    unicode_113766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'unicode', u'\n    A base class for path effects.\n\n    Subclasses should override the ``draw_path`` method to add effect\n    functionality.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 26)
        tuple_113767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 26)
        # Adding element type (line 26)
        float_113768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 31), tuple_113767, float_113768)
        # Adding element type (line 26)
        float_113769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 31), tuple_113767, float_113769)
        
        defaults = [tuple_113767]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbstractPathEffect.__init__', ['offset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_113770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'unicode', u'\n        Parameters\n        ----------\n        offset : pair of floats\n            The offset to apply to the path, measured in points.\n        ')
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'offset' (line 33)
        offset_113771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'offset')
        # Getting the type of 'self' (line 33)
        self_113772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member '_offset' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_113772, '_offset', offset_113771)
        
        # Assigning a Call to a Attribute (line 34):
        
        # Assigning a Call to a Attribute (line 34):
        
        # Call to Affine2D(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_113775 = {}
        # Getting the type of 'mtransforms' (line 34)
        mtransforms_113773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 34)
        Affine2D_113774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 29), mtransforms_113773, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 34)
        Affine2D_call_result_113776 = invoke(stypy.reporting.localization.Localization(__file__, 34, 29), Affine2D_113774, *[], **kwargs_113775)
        
        # Getting the type of 'self' (line 34)
        self_113777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member '_offset_trans' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_113777, '_offset_trans', Affine2D_call_result_113776)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _offset_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_offset_transform'
        module_type_store = module_type_store.open_function_context('_offset_transform', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_localization', localization)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_function_name', 'AbstractPathEffect._offset_transform')
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'transform'])
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbstractPathEffect._offset_transform.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbstractPathEffect._offset_transform', ['renderer', 'transform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_offset_transform', localization, ['renderer', 'transform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_offset_transform(...)' code ##################

        unicode_113778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'unicode', u'Apply the offset to the given transform.')
        
        # Assigning a Call to a Name (line 38):
        
        # Assigning a Call to a Name (line 38):
        
        # Call to points_to_pixels(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Obtaining the type of the subscript
        int_113781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 58), 'int')
        # Getting the type of 'self' (line 38)
        self_113782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'self', False)
        # Obtaining the member '_offset' of a type (line 38)
        _offset_113783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), self_113782, '_offset')
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___113784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), _offset_113783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_113785 = invoke(stypy.reporting.localization.Localization(__file__, 38, 45), getitem___113784, int_113781)
        
        # Processing the call keyword arguments (line 38)
        kwargs_113786 = {}
        # Getting the type of 'renderer' (line 38)
        renderer_113779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'renderer', False)
        # Obtaining the member 'points_to_pixels' of a type (line 38)
        points_to_pixels_113780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 19), renderer_113779, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 38)
        points_to_pixels_call_result_113787 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), points_to_pixels_113780, *[subscript_call_result_113785], **kwargs_113786)
        
        # Assigning a type to the variable 'offset_x' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'offset_x', points_to_pixels_call_result_113787)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to points_to_pixels(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        int_113790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 58), 'int')
        # Getting the type of 'self' (line 39)
        self_113791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 45), 'self', False)
        # Obtaining the member '_offset' of a type (line 39)
        _offset_113792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 45), self_113791, '_offset')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___113793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 45), _offset_113792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_113794 = invoke(stypy.reporting.localization.Localization(__file__, 39, 45), getitem___113793, int_113790)
        
        # Processing the call keyword arguments (line 39)
        kwargs_113795 = {}
        # Getting the type of 'renderer' (line 39)
        renderer_113788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'renderer', False)
        # Obtaining the member 'points_to_pixels' of a type (line 39)
        points_to_pixels_113789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), renderer_113788, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 39)
        points_to_pixels_call_result_113796 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), points_to_pixels_113789, *[subscript_call_result_113794], **kwargs_113795)
        
        # Assigning a type to the variable 'offset_y' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'offset_y', points_to_pixels_call_result_113796)
        # Getting the type of 'transform' (line 40)
        transform_113797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'transform')
        
        # Call to translate(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'offset_x' (line 40)
        offset_x_113804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 64), 'offset_x', False)
        # Getting the type of 'offset_y' (line 41)
        offset_y_113805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 64), 'offset_y', False)
        # Processing the call keyword arguments (line 40)
        kwargs_113806 = {}
        
        # Call to clear(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_113801 = {}
        # Getting the type of 'self' (line 40)
        self_113798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'self', False)
        # Obtaining the member '_offset_trans' of a type (line 40)
        _offset_trans_113799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), self_113798, '_offset_trans')
        # Obtaining the member 'clear' of a type (line 40)
        clear_113800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), _offset_trans_113799, 'clear')
        # Calling clear(args, kwargs) (line 40)
        clear_call_result_113802 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), clear_113800, *[], **kwargs_113801)
        
        # Obtaining the member 'translate' of a type (line 40)
        translate_113803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), clear_call_result_113802, 'translate')
        # Calling translate(args, kwargs) (line 40)
        translate_call_result_113807 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), translate_113803, *[offset_x_113804, offset_y_113805], **kwargs_113806)
        
        # Applying the binary operator '+' (line 40)
        result_add_113808 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), '+', transform_113797, translate_call_result_113807)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', result_add_113808)
        
        # ################# End of '_offset_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_offset_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_113809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113809)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_offset_transform'
        return stypy_return_type_113809


    @norecursion
    def _update_gc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_gc'
        module_type_store = module_type_store.open_function_context('_update_gc', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_localization', localization)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_function_name', 'AbstractPathEffect._update_gc')
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_param_names_list', ['gc', 'new_gc_dict'])
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbstractPathEffect._update_gc.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbstractPathEffect._update_gc', ['gc', 'new_gc_dict'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_gc', localization, ['gc', 'new_gc_dict'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_gc(...)' code ##################

        unicode_113810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'unicode', u'\n        Update the given GraphicsCollection with the given\n        dictionary of properties. The keys in the dictionary are used to\n        identify the appropriate set_ method on the gc.\n\n        ')
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to copy(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_113813 = {}
        # Getting the type of 'new_gc_dict' (line 50)
        new_gc_dict_113811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'new_gc_dict', False)
        # Obtaining the member 'copy' of a type (line 50)
        copy_113812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 22), new_gc_dict_113811, 'copy')
        # Calling copy(args, kwargs) (line 50)
        copy_call_result_113814 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), copy_113812, *[], **kwargs_113813)
        
        # Assigning a type to the variable 'new_gc_dict' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'new_gc_dict', copy_call_result_113814)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to pop(...): (line 52)
        # Processing the call arguments (line 52)
        unicode_113817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 33), 'unicode', u'dashes')
        # Getting the type of 'None' (line 52)
        None_113818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'None', False)
        # Processing the call keyword arguments (line 52)
        kwargs_113819 = {}
        # Getting the type of 'new_gc_dict' (line 52)
        new_gc_dict_113815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'new_gc_dict', False)
        # Obtaining the member 'pop' of a type (line 52)
        pop_113816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), new_gc_dict_113815, 'pop')
        # Calling pop(args, kwargs) (line 52)
        pop_call_result_113820 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), pop_113816, *[unicode_113817, None_113818], **kwargs_113819)
        
        # Assigning a type to the variable 'dashes' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'dashes', pop_call_result_113820)
        
        # Getting the type of 'dashes' (line 53)
        dashes_113821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'dashes')
        # Testing the type of an if condition (line 53)
        if_condition_113822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), dashes_113821)
        # Assigning a type to the variable 'if_condition_113822' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_113822', if_condition_113822)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_dashes(...): (line 54)
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'dashes' (line 54)
        dashes_113825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'dashes', False)
        kwargs_113826 = {'dashes_113825': dashes_113825}
        # Getting the type of 'gc' (line 54)
        gc_113823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'gc', False)
        # Obtaining the member 'set_dashes' of a type (line 54)
        set_dashes_113824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), gc_113823, 'set_dashes')
        # Calling set_dashes(args, kwargs) (line 54)
        set_dashes_call_result_113827 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), set_dashes_113824, *[], **kwargs_113826)
        
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to iteritems(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'new_gc_dict' (line 56)
        new_gc_dict_113830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'new_gc_dict', False)
        # Processing the call keyword arguments (line 56)
        kwargs_113831 = {}
        # Getting the type of 'six' (line 56)
        six_113828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 56)
        iteritems_113829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 20), six_113828, 'iteritems')
        # Calling iteritems(args, kwargs) (line 56)
        iteritems_call_result_113832 = invoke(stypy.reporting.localization.Localization(__file__, 56, 20), iteritems_113829, *[new_gc_dict_113830], **kwargs_113831)
        
        # Testing the type of a for loop iterable (line 56)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 8), iteritems_call_result_113832)
        # Getting the type of the for loop variable (line 56)
        for_loop_var_113833 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 8), iteritems_call_result_113832)
        # Assigning a type to the variable 'k' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), for_loop_var_113833))
        # Assigning a type to the variable 'v' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), for_loop_var_113833))
        # SSA begins for a for statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to getattr(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'gc' (line 57)
        gc_113835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'gc', False)
        unicode_113836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'unicode', u'set_')
        # Getting the type of 'k' (line 57)
        k_113837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 46), 'k', False)
        # Applying the binary operator '+' (line 57)
        result_add_113838 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 37), '+', unicode_113836, k_113837)
        
        # Getting the type of 'None' (line 57)
        None_113839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 49), 'None', False)
        # Processing the call keyword arguments (line 57)
        kwargs_113840 = {}
        # Getting the type of 'getattr' (line 57)
        getattr_113834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'getattr', False)
        # Calling getattr(args, kwargs) (line 57)
        getattr_call_result_113841 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), getattr_113834, *[gc_113835, result_add_113838, None_113839], **kwargs_113840)
        
        # Assigning a type to the variable 'set_method' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'set_method', getattr_call_result_113841)
        
        
        
        # Call to callable(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'set_method' (line 58)
        set_method_113843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'set_method', False)
        # Processing the call keyword arguments (line 58)
        kwargs_113844 = {}
        # Getting the type of 'callable' (line 58)
        callable_113842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'callable', False)
        # Calling callable(args, kwargs) (line 58)
        callable_call_result_113845 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), callable_113842, *[set_method_113843], **kwargs_113844)
        
        # Applying the 'not' unary operator (line 58)
        result_not__113846 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 15), 'not', callable_call_result_113845)
        
        # Testing the type of an if condition (line 58)
        if_condition_113847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 12), result_not__113846)
        # Assigning a type to the variable 'if_condition_113847' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'if_condition_113847', if_condition_113847)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AttributeError(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to format(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'k' (line 59)
        k_113851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 67), 'k', False)
        # Processing the call keyword arguments (line 59)
        kwargs_113852 = {}
        unicode_113849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'unicode', u'Unknown property {0}')
        # Obtaining the member 'format' of a type (line 59)
        format_113850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 37), unicode_113849, 'format')
        # Calling format(args, kwargs) (line 59)
        format_call_result_113853 = invoke(stypy.reporting.localization.Localization(__file__, 59, 37), format_113850, *[k_113851], **kwargs_113852)
        
        # Processing the call keyword arguments (line 59)
        kwargs_113854 = {}
        # Getting the type of 'AttributeError' (line 59)
        AttributeError_113848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 59)
        AttributeError_call_result_113855 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), AttributeError_113848, *[format_call_result_113853], **kwargs_113854)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 59, 16), AttributeError_call_result_113855, 'raise parameter', BaseException)
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_method(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'v' (line 60)
        v_113857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'v', False)
        # Processing the call keyword arguments (line 60)
        kwargs_113858 = {}
        # Getting the type of 'set_method' (line 60)
        set_method_113856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'set_method', False)
        # Calling set_method(args, kwargs) (line 60)
        set_method_call_result_113859 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), set_method_113856, *[v_113857], **kwargs_113858)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'gc' (line 61)
        gc_113860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'gc')
        # Assigning a type to the variable 'stypy_return_type' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', gc_113860)
        
        # ################# End of '_update_gc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_gc' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_113861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113861)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_gc'
        return stypy_return_type_113861


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 63)
        None_113862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 61), 'None')
        defaults = [None_113862]
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_localization', localization)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_function_name', 'AbstractPathEffect.draw_path')
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AbstractPathEffect.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AbstractPathEffect.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        unicode_113863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'unicode', u'\n        Derived should override this method. The arguments are the same\n        as :meth:`matplotlib.backend_bases.RendererBase.draw_path`\n        except the first argument is a renderer.\n\n        ')
        
        
        # Call to isinstance(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'renderer' (line 71)
        renderer_113865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'renderer', False)
        # Getting the type of 'PathEffectRenderer' (line 71)
        PathEffectRenderer_113866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'PathEffectRenderer', False)
        # Processing the call keyword arguments (line 71)
        kwargs_113867 = {}
        # Getting the type of 'isinstance' (line 71)
        isinstance_113864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 71)
        isinstance_call_result_113868 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), isinstance_113864, *[renderer_113865, PathEffectRenderer_113866], **kwargs_113867)
        
        # Testing the type of an if condition (line 71)
        if_condition_113869 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), isinstance_call_result_113868)
        # Assigning a type to the variable 'if_condition_113869' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_113869', if_condition_113869)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 72):
        
        # Assigning a Attribute to a Name (line 72):
        # Getting the type of 'renderer' (line 72)
        renderer_113870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'renderer')
        # Obtaining the member '_renderer' of a type (line 72)
        _renderer_113871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), renderer_113870, '_renderer')
        # Assigning a type to the variable 'renderer' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'renderer', _renderer_113871)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_path(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'gc' (line 73)
        gc_113874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'gc', False)
        # Getting the type of 'tpath' (line 73)
        tpath_113875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'tpath', False)
        # Getting the type of 'affine' (line 73)
        affine_113876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'affine', False)
        # Getting the type of 'rgbFace' (line 73)
        rgbFace_113877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 53), 'rgbFace', False)
        # Processing the call keyword arguments (line 73)
        kwargs_113878 = {}
        # Getting the type of 'renderer' (line 73)
        renderer_113872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'renderer', False)
        # Obtaining the member 'draw_path' of a type (line 73)
        draw_path_113873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), renderer_113872, 'draw_path')
        # Calling draw_path(args, kwargs) (line 73)
        draw_path_call_result_113879 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), draw_path_113873, *[gc_113874, tpath_113875, affine_113876, rgbFace_113877], **kwargs_113878)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', draw_path_call_result_113879)
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_113880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113880)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_113880


# Assigning a type to the variable 'AbstractPathEffect' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'AbstractPathEffect', AbstractPathEffect)
# Declaration of the 'PathEffectRenderer' class
# Getting the type of 'RendererBase' (line 76)
RendererBase_113881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'RendererBase')

class PathEffectRenderer(RendererBase_113881, ):
    unicode_113882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'unicode', u'\n    Implements a Renderer which contains another renderer.\n\n    This proxy then intercepts draw calls, calling the appropriate\n    :class:`AbstractPathEffect` draw method.\n\n    .. note::\n        Not all methods have been overridden on this RendererBase subclass.\n        It may be necessary to add further methods to extend the PathEffects\n        capabilities further.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.__init__', ['path_effects', 'renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['path_effects', 'renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_113883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'unicode', u'\n        Parameters\n        ----------\n        path_effects : iterable of :class:`AbstractPathEffect`\n            The path effects which this renderer represents.\n        renderer : :class:`matplotlib.backend_bases.RendererBase` instance\n\n        ')
        
        # Assigning a Name to a Attribute (line 98):
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'path_effects' (line 98)
        path_effects_113884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 29), 'path_effects')
        # Getting the type of 'self' (line 98)
        self_113885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member '_path_effects' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_113885, '_path_effects', path_effects_113884)
        
        # Assigning a Name to a Attribute (line 99):
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'renderer' (line 99)
        renderer_113886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'renderer')
        # Getting the type of 'self' (line 99)
        self_113887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_113887, '_renderer', renderer_113886)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def new_gc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_gc'
        module_type_store = module_type_store.open_function_context('new_gc', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.new_gc')
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_param_names_list', [])
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.new_gc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.new_gc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_gc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_gc(...)' code ##################

        
        # Call to new_gc(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_113891 = {}
        # Getting the type of 'self' (line 102)
        self_113888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 102)
        _renderer_113889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), self_113888, '_renderer')
        # Obtaining the member 'new_gc' of a type (line 102)
        new_gc_113890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), _renderer_113889, 'new_gc')
        # Calling new_gc(args, kwargs) (line 102)
        new_gc_call_result_113892 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), new_gc_113890, *[], **kwargs_113891)
        
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', new_gc_call_result_113892)
        
        # ################# End of 'new_gc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_gc' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_113893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_gc'
        return stypy_return_type_113893


    @norecursion
    def copy_with_path_effect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy_with_path_effect'
        module_type_store = module_type_store.open_function_context('copy_with_path_effect', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.copy_with_path_effect')
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_param_names_list', ['path_effects'])
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.copy_with_path_effect.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.copy_with_path_effect', ['path_effects'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_with_path_effect', localization, ['path_effects'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_with_path_effect(...)' code ##################

        
        # Call to __class__(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'path_effects' (line 105)
        path_effects_113896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'path_effects', False)
        # Getting the type of 'self' (line 105)
        self_113897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 44), 'self', False)
        # Obtaining the member '_renderer' of a type (line 105)
        _renderer_113898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 44), self_113897, '_renderer')
        # Processing the call keyword arguments (line 105)
        kwargs_113899 = {}
        # Getting the type of 'self' (line 105)
        self_113894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 105)
        class___113895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), self_113894, '__class__')
        # Calling __class__(args, kwargs) (line 105)
        class___call_result_113900 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), class___113895, *[path_effects_113896, _renderer_113898], **kwargs_113899)
        
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', class___call_result_113900)
        
        # ################# End of 'copy_with_path_effect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_with_path_effect' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_113901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_with_path_effect'
        return stypy_return_type_113901


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 107)
        None_113902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 51), 'None')
        defaults = [None_113902]
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.draw_path')
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_param_names_list', ['gc', 'tpath', 'affine', 'rgbFace'])
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.draw_path', ['gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        
        # Getting the type of 'self' (line 108)
        self_113903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'self')
        # Obtaining the member '_path_effects' of a type (line 108)
        _path_effects_113904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), self_113903, '_path_effects')
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), _path_effects_113904)
        # Getting the type of the for loop variable (line 108)
        for_loop_var_113905 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), _path_effects_113904)
        # Assigning a type to the variable 'path_effect' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'path_effect', for_loop_var_113905)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to draw_path(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_113908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'self', False)
        # Obtaining the member '_renderer' of a type (line 109)
        _renderer_113909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 34), self_113908, '_renderer')
        # Getting the type of 'gc' (line 109)
        gc_113910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 50), 'gc', False)
        # Getting the type of 'tpath' (line 109)
        tpath_113911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 54), 'tpath', False)
        # Getting the type of 'affine' (line 109)
        affine_113912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 61), 'affine', False)
        # Getting the type of 'rgbFace' (line 110)
        rgbFace_113913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'rgbFace', False)
        # Processing the call keyword arguments (line 109)
        kwargs_113914 = {}
        # Getting the type of 'path_effect' (line 109)
        path_effect_113906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'path_effect', False)
        # Obtaining the member 'draw_path' of a type (line 109)
        draw_path_113907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), path_effect_113906, 'draw_path')
        # Calling draw_path(args, kwargs) (line 109)
        draw_path_call_result_113915 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), draw_path_113907, *[_renderer_113909, gc_113910, tpath_113911, affine_113912, rgbFace_113913], **kwargs_113914)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_113916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113916)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_113916


    @norecursion
    def draw_markers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_markers'
        module_type_store = module_type_store.open_function_context('draw_markers', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.draw_markers')
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_param_names_list', ['gc', 'marker_path', 'marker_trans', 'path'])
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.draw_markers.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.draw_markers', ['gc', 'marker_path', 'marker_trans', 'path'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_markers', localization, ['gc', 'marker_path', 'marker_trans', 'path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_markers(...)' code ##################

        
        
        
        # Call to len(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_113918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'self', False)
        # Obtaining the member '_path_effects' of a type (line 117)
        _path_effects_113919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), self_113918, '_path_effects')
        # Processing the call keyword arguments (line 117)
        kwargs_113920 = {}
        # Getting the type of 'len' (line 117)
        len_113917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'len', False)
        # Calling len(args, kwargs) (line 117)
        len_call_result_113921 = invoke(stypy.reporting.localization.Localization(__file__, 117, 11), len_113917, *[_path_effects_113919], **kwargs_113920)
        
        int_113922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'int')
        # Applying the binary operator '==' (line 117)
        result_eq_113923 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 11), '==', len_call_result_113921, int_113922)
        
        # Testing the type of an if condition (line 117)
        if_condition_113924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), result_eq_113923)
        # Assigning a type to the variable 'if_condition_113924' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_113924', if_condition_113924)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw_markers(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_113927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'self', False)
        # Getting the type of 'gc' (line 120)
        gc_113928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'gc', False)
        # Getting the type of 'marker_path' (line 120)
        marker_path_113929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'marker_path', False)
        # Getting the type of 'marker_trans' (line 121)
        marker_trans_113930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'marker_trans', False)
        # Getting the type of 'path' (line 121)
        path_113931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 59), 'path', False)
        # Getting the type of 'args' (line 121)
        args_113932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 66), 'args', False)
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'kwargs' (line 122)
        kwargs_113933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 47), 'kwargs', False)
        kwargs_113934 = {'kwargs_113933': kwargs_113933}
        # Getting the type of 'RendererBase' (line 120)
        RendererBase_113925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'RendererBase', False)
        # Obtaining the member 'draw_markers' of a type (line 120)
        draw_markers_113926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), RendererBase_113925, 'draw_markers')
        # Calling draw_markers(args, kwargs) (line 120)
        draw_markers_call_result_113935 = invoke(stypy.reporting.localization.Localization(__file__, 120, 19), draw_markers_113926, *[self_113927, gc_113928, marker_path_113929, marker_trans_113930, path_113931, args_113932], **kwargs_113934)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'stypy_return_type', draw_markers_call_result_113935)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 124)
        self_113936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'self')
        # Obtaining the member '_path_effects' of a type (line 124)
        _path_effects_113937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 27), self_113936, '_path_effects')
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), _path_effects_113937)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_113938 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), _path_effects_113937)
        # Assigning a type to the variable 'path_effect' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'path_effect', for_loop_var_113938)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to copy_with_path_effect(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_113941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'path_effect' (line 125)
        path_effect_113942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 51), 'path_effect', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 50), list_113941, path_effect_113942)
        
        # Processing the call keyword arguments (line 125)
        kwargs_113943 = {}
        # Getting the type of 'self' (line 125)
        self_113939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'self', False)
        # Obtaining the member 'copy_with_path_effect' of a type (line 125)
        copy_with_path_effect_113940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 23), self_113939, 'copy_with_path_effect')
        # Calling copy_with_path_effect(args, kwargs) (line 125)
        copy_with_path_effect_call_result_113944 = invoke(stypy.reporting.localization.Localization(__file__, 125, 23), copy_with_path_effect_113940, *[list_113941], **kwargs_113943)
        
        # Assigning a type to the variable 'renderer' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'renderer', copy_with_path_effect_call_result_113944)
        
        # Call to draw_markers(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'gc' (line 128)
        gc_113947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'gc', False)
        # Getting the type of 'marker_path' (line 128)
        marker_path_113948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'marker_path', False)
        # Getting the type of 'marker_trans' (line 128)
        marker_trans_113949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'marker_trans', False)
        # Getting the type of 'path' (line 128)
        path_113950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 65), 'path', False)
        # Getting the type of 'args' (line 129)
        args_113951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'args', False)
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'kwargs' (line 129)
        kwargs_113952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 43), 'kwargs', False)
        kwargs_113953 = {'kwargs_113952': kwargs_113952}
        # Getting the type of 'renderer' (line 128)
        renderer_113945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'renderer', False)
        # Obtaining the member 'draw_markers' of a type (line 128)
        draw_markers_113946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), renderer_113945, 'draw_markers')
        # Calling draw_markers(args, kwargs) (line 128)
        draw_markers_call_result_113954 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), draw_markers_113946, *[gc_113947, marker_path_113948, marker_trans_113949, path_113950, args_113951], **kwargs_113953)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_markers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_markers' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_113955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_markers'
        return stypy_return_type_113955


    @norecursion
    def draw_path_collection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path_collection'
        module_type_store = module_type_store.open_function_context('draw_path_collection', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.draw_path_collection')
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_param_names_list', ['gc', 'master_transform', 'paths'])
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.draw_path_collection.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.draw_path_collection', ['gc', 'master_transform', 'paths'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path_collection', localization, ['gc', 'master_transform', 'paths'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path_collection(...)' code ##################

        
        
        
        # Call to len(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'self' (line 136)
        self_113957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'self', False)
        # Obtaining the member '_path_effects' of a type (line 136)
        _path_effects_113958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 15), self_113957, '_path_effects')
        # Processing the call keyword arguments (line 136)
        kwargs_113959 = {}
        # Getting the type of 'len' (line 136)
        len_113956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'len', False)
        # Calling len(args, kwargs) (line 136)
        len_call_result_113960 = invoke(stypy.reporting.localization.Localization(__file__, 136, 11), len_113956, *[_path_effects_113958], **kwargs_113959)
        
        int_113961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 38), 'int')
        # Applying the binary operator '==' (line 136)
        result_eq_113962 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), '==', len_call_result_113960, int_113961)
        
        # Testing the type of an if condition (line 136)
        if_condition_113963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 8), result_eq_113962)
        # Assigning a type to the variable 'if_condition_113963' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'if_condition_113963', if_condition_113963)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw_path_collection(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_113966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 53), 'self', False)
        # Getting the type of 'gc' (line 139)
        gc_113967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 59), 'gc', False)
        # Getting the type of 'master_transform' (line 140)
        master_transform_113968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'master_transform', False)
        # Getting the type of 'paths' (line 140)
        paths_113969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 71), 'paths', False)
        # Getting the type of 'args' (line 141)
        args_113970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 54), 'args', False)
        # Processing the call keyword arguments (line 139)
        # Getting the type of 'kwargs' (line 141)
        kwargs_113971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 62), 'kwargs', False)
        kwargs_113972 = {'kwargs_113971': kwargs_113971}
        # Getting the type of 'RendererBase' (line 139)
        RendererBase_113964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'RendererBase', False)
        # Obtaining the member 'draw_path_collection' of a type (line 139)
        draw_path_collection_113965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 19), RendererBase_113964, 'draw_path_collection')
        # Calling draw_path_collection(args, kwargs) (line 139)
        draw_path_collection_call_result_113973 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), draw_path_collection_113965, *[self_113966, gc_113967, master_transform_113968, paths_113969, args_113970], **kwargs_113972)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'stypy_return_type', draw_path_collection_call_result_113973)
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 143)
        self_113974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'self')
        # Obtaining the member '_path_effects' of a type (line 143)
        _path_effects_113975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 27), self_113974, '_path_effects')
        # Testing the type of a for loop iterable (line 143)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 8), _path_effects_113975)
        # Getting the type of the for loop variable (line 143)
        for_loop_var_113976 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 8), _path_effects_113975)
        # Assigning a type to the variable 'path_effect' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'path_effect', for_loop_var_113976)
        # SSA begins for a for statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to copy_with_path_effect(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_113979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        # Getting the type of 'path_effect' (line 144)
        path_effect_113980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'path_effect', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 50), list_113979, path_effect_113980)
        
        # Processing the call keyword arguments (line 144)
        kwargs_113981 = {}
        # Getting the type of 'self' (line 144)
        self_113977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'self', False)
        # Obtaining the member 'copy_with_path_effect' of a type (line 144)
        copy_with_path_effect_113978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 23), self_113977, 'copy_with_path_effect')
        # Calling copy_with_path_effect(args, kwargs) (line 144)
        copy_with_path_effect_call_result_113982 = invoke(stypy.reporting.localization.Localization(__file__, 144, 23), copy_with_path_effect_113978, *[list_113979], **kwargs_113981)
        
        # Assigning a type to the variable 'renderer' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'renderer', copy_with_path_effect_call_result_113982)
        
        # Call to draw_path_collection(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'gc' (line 147)
        gc_113985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'gc', False)
        # Getting the type of 'master_transform' (line 147)
        master_transform_113986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 46), 'master_transform', False)
        # Getting the type of 'paths' (line 147)
        paths_113987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 64), 'paths', False)
        # Getting the type of 'args' (line 148)
        args_113988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'args', False)
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'kwargs' (line 148)
        kwargs_113989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'kwargs', False)
        kwargs_113990 = {'kwargs_113989': kwargs_113989}
        # Getting the type of 'renderer' (line 147)
        renderer_113983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'renderer', False)
        # Obtaining the member 'draw_path_collection' of a type (line 147)
        draw_path_collection_113984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), renderer_113983, 'draw_path_collection')
        # Calling draw_path_collection(args, kwargs) (line 147)
        draw_path_collection_call_result_113991 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), draw_path_collection_113984, *[gc_113985, master_transform_113986, paths_113987, args_113988], **kwargs_113990)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_path_collection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path_collection' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_113992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path_collection'
        return stypy_return_type_113992


    @norecursion
    def points_to_pixels(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'points_to_pixels'
        module_type_store = module_type_store.open_function_context('points_to_pixels', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.points_to_pixels')
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_param_names_list', ['points'])
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.points_to_pixels.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.points_to_pixels', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'points_to_pixels', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'points_to_pixels(...)' code ##################

        
        # Call to points_to_pixels(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'points' (line 151)
        points_113996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 47), 'points', False)
        # Processing the call keyword arguments (line 151)
        kwargs_113997 = {}
        # Getting the type of 'self' (line 151)
        self_113993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'self', False)
        # Obtaining the member '_renderer' of a type (line 151)
        _renderer_113994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), self_113993, '_renderer')
        # Obtaining the member 'points_to_pixels' of a type (line 151)
        points_to_pixels_113995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), _renderer_113994, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 151)
        points_to_pixels_call_result_113998 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), points_to_pixels_113995, *[points_113996], **kwargs_113997)
        
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', points_to_pixels_call_result_113998)
        
        # ################# End of 'points_to_pixels(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'points_to_pixels' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_113999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'points_to_pixels'
        return stypy_return_type_113999


    @norecursion
    def _draw_text_as_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_draw_text_as_path'
        module_type_store = module_type_store.open_function_context('_draw_text_as_path', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer._draw_text_as_path')
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath'])
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer._draw_text_as_path.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer._draw_text_as_path', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_draw_text_as_path', localization, ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_draw_text_as_path(...)' code ##################

        
        # Assigning a Call to a Tuple (line 155):
        
        # Assigning a Call to a Name:
        
        # Call to _get_text_path_transform(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'x' (line 155)
        x_114002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 56), 'x', False)
        # Getting the type of 'y' (line 155)
        y_114003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 59), 'y', False)
        # Getting the type of 's' (line 155)
        s_114004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 62), 's', False)
        # Getting the type of 'prop' (line 155)
        prop_114005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 65), 'prop', False)
        # Getting the type of 'angle' (line 156)
        angle_114006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 56), 'angle', False)
        # Getting the type of 'ismath' (line 156)
        ismath_114007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 63), 'ismath', False)
        # Processing the call keyword arguments (line 155)
        kwargs_114008 = {}
        # Getting the type of 'self' (line 155)
        self_114000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'self', False)
        # Obtaining the member '_get_text_path_transform' of a type (line 155)
        _get_text_path_transform_114001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 26), self_114000, '_get_text_path_transform')
        # Calling _get_text_path_transform(args, kwargs) (line 155)
        _get_text_path_transform_call_result_114009 = invoke(stypy.reporting.localization.Localization(__file__, 155, 26), _get_text_path_transform_114001, *[x_114002, y_114003, s_114004, prop_114005, angle_114006, ismath_114007], **kwargs_114008)
        
        # Assigning a type to the variable 'call_assignment_113746' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113746', _get_text_path_transform_call_result_114009)
        
        # Assigning a Call to a Name (line 155):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_114012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 8), 'int')
        # Processing the call keyword arguments
        kwargs_114013 = {}
        # Getting the type of 'call_assignment_113746' (line 155)
        call_assignment_113746_114010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113746', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___114011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), call_assignment_113746_114010, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_114014 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___114011, *[int_114012], **kwargs_114013)
        
        # Assigning a type to the variable 'call_assignment_113747' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113747', getitem___call_result_114014)
        
        # Assigning a Name to a Name (line 155):
        # Getting the type of 'call_assignment_113747' (line 155)
        call_assignment_113747_114015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113747')
        # Assigning a type to the variable 'path' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'path', call_assignment_113747_114015)
        
        # Assigning a Call to a Name (line 155):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_114018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 8), 'int')
        # Processing the call keyword arguments
        kwargs_114019 = {}
        # Getting the type of 'call_assignment_113746' (line 155)
        call_assignment_113746_114016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113746', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___114017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), call_assignment_113746_114016, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_114020 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___114017, *[int_114018], **kwargs_114019)
        
        # Assigning a type to the variable 'call_assignment_113748' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113748', getitem___call_result_114020)
        
        # Assigning a Name to a Name (line 155):
        # Getting the type of 'call_assignment_113748' (line 155)
        call_assignment_113748_114021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'call_assignment_113748')
        # Assigning a type to the variable 'transform' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'transform', call_assignment_113748_114021)
        
        # Assigning a Call to a Name (line 157):
        
        # Assigning a Call to a Name (line 157):
        
        # Call to get_rgb(...): (line 157)
        # Processing the call keyword arguments (line 157)
        kwargs_114024 = {}
        # Getting the type of 'gc' (line 157)
        gc_114022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'gc', False)
        # Obtaining the member 'get_rgb' of a type (line 157)
        get_rgb_114023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), gc_114022, 'get_rgb')
        # Calling get_rgb(args, kwargs) (line 157)
        get_rgb_call_result_114025 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), get_rgb_114023, *[], **kwargs_114024)
        
        # Assigning a type to the variable 'color' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'color', get_rgb_call_result_114025)
        
        # Call to set_linewidth(...): (line 158)
        # Processing the call arguments (line 158)
        float_114028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'float')
        # Processing the call keyword arguments (line 158)
        kwargs_114029 = {}
        # Getting the type of 'gc' (line 158)
        gc_114026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'gc', False)
        # Obtaining the member 'set_linewidth' of a type (line 158)
        set_linewidth_114027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), gc_114026, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 158)
        set_linewidth_call_result_114030 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), set_linewidth_114027, *[float_114028], **kwargs_114029)
        
        
        # Call to draw_path(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'gc' (line 159)
        gc_114033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'gc', False)
        # Getting the type of 'path' (line 159)
        path_114034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'path', False)
        # Getting the type of 'transform' (line 159)
        transform_114035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'transform', False)
        # Processing the call keyword arguments (line 159)
        # Getting the type of 'color' (line 159)
        color_114036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 52), 'color', False)
        keyword_114037 = color_114036
        kwargs_114038 = {'rgbFace': keyword_114037}
        # Getting the type of 'self' (line 159)
        self_114031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'draw_path' of a type (line 159)
        draw_path_114032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_114031, 'draw_path')
        # Calling draw_path(args, kwargs) (line 159)
        draw_path_call_result_114039 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), draw_path_114032, *[gc_114033, path_114034, transform_114035], **kwargs_114038)
        
        
        # ################# End of '_draw_text_as_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_draw_text_as_path' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_114040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114040)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_draw_text_as_path'
        return stypy_return_type_114040


    @norecursion
    def __getattribute__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattribute__'
        module_type_store = module_type_store.open_function_context('__getattribute__', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_localization', localization)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_function_name', 'PathEffectRenderer.__getattribute__')
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathEffectRenderer.__getattribute__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathEffectRenderer.__getattribute__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattribute__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattribute__(...)' code ##################

        
        
        # Getting the type of 'name' (line 162)
        name_114041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'name')
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_114042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        unicode_114043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'unicode', u'_text2path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_114042, unicode_114043)
        # Adding element type (line 162)
        unicode_114044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 34), 'unicode', u'flipy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_114042, unicode_114044)
        # Adding element type (line 162)
        unicode_114045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 43), 'unicode', u'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_114042, unicode_114045)
        # Adding element type (line 162)
        unicode_114046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 53), 'unicode', u'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_114042, unicode_114046)
        
        # Applying the binary operator 'in' (line 162)
        result_contains_114047 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), 'in', name_114041, list_114042)
        
        # Testing the type of an if condition (line 162)
        if_condition_114048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_contains_114047)
        # Assigning a type to the variable 'if_condition_114048' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_114048', if_condition_114048)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to getattr(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'self' (line 163)
        self_114050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'self', False)
        # Obtaining the member '_renderer' of a type (line 163)
        _renderer_114051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), self_114050, '_renderer')
        # Getting the type of 'name' (line 163)
        name_114052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), 'name', False)
        # Processing the call keyword arguments (line 163)
        kwargs_114053 = {}
        # Getting the type of 'getattr' (line 163)
        getattr_114049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 163)
        getattr_call_result_114054 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), getattr_114049, *[_renderer_114051, name_114052], **kwargs_114053)
        
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'stypy_return_type', getattr_call_result_114054)
        # SSA branch for the else part of an if statement (line 162)
        module_type_store.open_ssa_branch('else')
        
        # Call to __getattribute__(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'self' (line 165)
        self_114057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 43), 'self', False)
        # Getting the type of 'name' (line 165)
        name_114058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'name', False)
        # Processing the call keyword arguments (line 165)
        kwargs_114059 = {}
        # Getting the type of 'object' (line 165)
        object_114055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'object', False)
        # Obtaining the member '__getattribute__' of a type (line 165)
        getattribute___114056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), object_114055, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 165)
        getattribute___call_result_114060 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), getattribute___114056, *[self_114057, name_114058], **kwargs_114059)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'stypy_return_type', getattribute___call_result_114060)
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattribute__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattribute__' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_114061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattribute__'
        return stypy_return_type_114061


# Assigning a type to the variable 'PathEffectRenderer' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'PathEffectRenderer', PathEffectRenderer)
# Declaration of the 'Normal' class
# Getting the type of 'AbstractPathEffect' (line 168)
AbstractPathEffect_114062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'AbstractPathEffect')

class Normal(AbstractPathEffect_114062, ):
    unicode_114063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'unicode', u'\n    The "identity" PathEffect.\n\n    The Normal PathEffect\'s sole purpose is to draw the original artist with\n    no special path effect.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 0, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Normal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Normal' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'Normal', Normal)
# Declaration of the 'Stroke' class
# Getting the type of 'AbstractPathEffect' (line 178)
AbstractPathEffect_114064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'AbstractPathEffect')

class Stroke(AbstractPathEffect_114064, ):
    unicode_114065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 4), 'unicode', u'A line based PathEffect which re-draws a stroke.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_114066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        int_114067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 31), tuple_114066, int_114067)
        # Adding element type (line 180)
        int_114068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 31), tuple_114066, int_114068)
        
        defaults = [tuple_114066]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Stroke.__init__', ['offset'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_114069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, (-1)), 'unicode', u'\n        The path will be stroked with its gc updated with the given\n        keyword arguments, i.e., the keyword arguments should be valid\n        gc parameter values.\n        ')
        
        # Call to __init__(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'offset' (line 186)
        offset_114076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'offset', False)
        # Processing the call keyword arguments (line 186)
        kwargs_114077 = {}
        
        # Call to super(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'Stroke' (line 186)
        Stroke_114071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), 'Stroke', False)
        # Getting the type of 'self' (line 186)
        self_114072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'self', False)
        # Processing the call keyword arguments (line 186)
        kwargs_114073 = {}
        # Getting the type of 'super' (line 186)
        super_114070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'super', False)
        # Calling super(args, kwargs) (line 186)
        super_call_result_114074 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), super_114070, *[Stroke_114071, self_114072], **kwargs_114073)
        
        # Obtaining the member '__init__' of a type (line 186)
        init___114075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), super_call_result_114074, '__init__')
        # Calling __init__(args, kwargs) (line 186)
        init___call_result_114078 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), init___114075, *[offset_114076], **kwargs_114077)
        
        
        # Assigning a Name to a Attribute (line 187):
        
        # Assigning a Name to a Attribute (line 187):
        # Getting the type of 'kwargs' (line 187)
        kwargs_114079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'kwargs')
        # Getting the type of 'self' (line 187)
        self_114080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self')
        # Setting the type of the member '_gc' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_114080, '_gc', kwargs_114079)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Stroke.draw_path.__dict__.__setitem__('stypy_localization', localization)
        Stroke.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Stroke.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Stroke.draw_path.__dict__.__setitem__('stypy_function_name', 'Stroke.draw_path')
        Stroke.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        Stroke.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Stroke.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Stroke.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Stroke.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Stroke.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Stroke.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Stroke.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        unicode_114081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'unicode', u'\n        draw the path with updated gc.\n        ')
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to new_gc(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_114084 = {}
        # Getting the type of 'renderer' (line 195)
        renderer_114082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'renderer', False)
        # Obtaining the member 'new_gc' of a type (line 195)
        new_gc_114083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 14), renderer_114082, 'new_gc')
        # Calling new_gc(args, kwargs) (line 195)
        new_gc_call_result_114085 = invoke(stypy.reporting.localization.Localization(__file__, 195, 14), new_gc_114083, *[], **kwargs_114084)
        
        # Assigning a type to the variable 'gc0' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'gc0', new_gc_call_result_114085)
        
        # Call to copy_properties(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'gc' (line 196)
        gc_114088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'gc', False)
        # Processing the call keyword arguments (line 196)
        kwargs_114089 = {}
        # Getting the type of 'gc0' (line 196)
        gc0_114086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'gc0', False)
        # Obtaining the member 'copy_properties' of a type (line 196)
        copy_properties_114087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), gc0_114086, 'copy_properties')
        # Calling copy_properties(args, kwargs) (line 196)
        copy_properties_call_result_114090 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), copy_properties_114087, *[gc_114088], **kwargs_114089)
        
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to _update_gc(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'gc0' (line 198)
        gc0_114093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'gc0', False)
        # Getting the type of 'self' (line 198)
        self_114094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'self', False)
        # Obtaining the member '_gc' of a type (line 198)
        _gc_114095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 35), self_114094, '_gc')
        # Processing the call keyword arguments (line 198)
        kwargs_114096 = {}
        # Getting the type of 'self' (line 198)
        self_114091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 14), 'self', False)
        # Obtaining the member '_update_gc' of a type (line 198)
        _update_gc_114092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 14), self_114091, '_update_gc')
        # Calling _update_gc(args, kwargs) (line 198)
        _update_gc_call_result_114097 = invoke(stypy.reporting.localization.Localization(__file__, 198, 14), _update_gc_114092, *[gc0_114093, _gc_114095], **kwargs_114096)
        
        # Assigning a type to the variable 'gc0' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'gc0', _update_gc_call_result_114097)
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to _offset_transform(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'renderer' (line 199)
        renderer_114100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'renderer', False)
        # Getting the type of 'affine' (line 199)
        affine_114101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 49), 'affine', False)
        # Processing the call keyword arguments (line 199)
        kwargs_114102 = {}
        # Getting the type of 'self' (line 199)
        self_114098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'self', False)
        # Obtaining the member '_offset_transform' of a type (line 199)
        _offset_transform_114099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), self_114098, '_offset_transform')
        # Calling _offset_transform(args, kwargs) (line 199)
        _offset_transform_call_result_114103 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), _offset_transform_114099, *[renderer_114100, affine_114101], **kwargs_114102)
        
        # Assigning a type to the variable 'trans' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'trans', _offset_transform_call_result_114103)
        
        # Call to draw_path(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'gc0' (line 200)
        gc0_114106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'gc0', False)
        # Getting the type of 'tpath' (line 200)
        tpath_114107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'tpath', False)
        # Getting the type of 'trans' (line 200)
        trans_114108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 39), 'trans', False)
        # Getting the type of 'rgbFace' (line 200)
        rgbFace_114109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'rgbFace', False)
        # Processing the call keyword arguments (line 200)
        kwargs_114110 = {}
        # Getting the type of 'renderer' (line 200)
        renderer_114104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'renderer', False)
        # Obtaining the member 'draw_path' of a type (line 200)
        draw_path_114105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), renderer_114104, 'draw_path')
        # Calling draw_path(args, kwargs) (line 200)
        draw_path_call_result_114111 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), draw_path_114105, *[gc0_114106, tpath_114107, trans_114108, rgbFace_114109], **kwargs_114110)
        
        
        # Call to restore(...): (line 201)
        # Processing the call keyword arguments (line 201)
        kwargs_114114 = {}
        # Getting the type of 'gc0' (line 201)
        gc0_114112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'gc0', False)
        # Obtaining the member 'restore' of a type (line 201)
        restore_114113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), gc0_114112, 'restore')
        # Calling restore(args, kwargs) (line 201)
        restore_call_result_114115 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), restore_114113, *[], **kwargs_114114)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_114116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_114116


# Assigning a type to the variable 'Stroke' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'Stroke', Stroke)
# Declaration of the 'withStroke' class
# Getting the type of 'Stroke' (line 204)
Stroke_114117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'Stroke')

class withStroke(Stroke_114117, ):
    unicode_114118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'unicode', u'\n    Adds a simple :class:`Stroke` and then draws the\n    original Artist to avoid needing to call :class:`Normal`.\n\n    ')

    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        withStroke.draw_path.__dict__.__setitem__('stypy_localization', localization)
        withStroke.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        withStroke.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        withStroke.draw_path.__dict__.__setitem__('stypy_function_name', 'withStroke.draw_path')
        withStroke.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        withStroke.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        withStroke.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        withStroke.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        withStroke.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        withStroke.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        withStroke.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'withStroke.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        
        # Call to draw_path(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'self' (line 211)
        self_114121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'self', False)
        # Getting the type of 'renderer' (line 211)
        renderer_114122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'renderer', False)
        # Getting the type of 'gc' (line 211)
        gc_114123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'gc', False)
        # Getting the type of 'tpath' (line 211)
        tpath_114124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 45), 'tpath', False)
        # Getting the type of 'affine' (line 211)
        affine_114125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 52), 'affine', False)
        # Getting the type of 'rgbFace' (line 211)
        rgbFace_114126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 60), 'rgbFace', False)
        # Processing the call keyword arguments (line 211)
        kwargs_114127 = {}
        # Getting the type of 'Stroke' (line 211)
        Stroke_114119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'Stroke', False)
        # Obtaining the member 'draw_path' of a type (line 211)
        draw_path_114120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), Stroke_114119, 'draw_path')
        # Calling draw_path(args, kwargs) (line 211)
        draw_path_call_result_114128 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), draw_path_114120, *[self_114121, renderer_114122, gc_114123, tpath_114124, affine_114125, rgbFace_114126], **kwargs_114127)
        
        
        # Call to draw_path(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'gc' (line 212)
        gc_114131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'gc', False)
        # Getting the type of 'tpath' (line 212)
        tpath_114132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'tpath', False)
        # Getting the type of 'affine' (line 212)
        affine_114133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 38), 'affine', False)
        # Getting the type of 'rgbFace' (line 212)
        rgbFace_114134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'rgbFace', False)
        # Processing the call keyword arguments (line 212)
        kwargs_114135 = {}
        # Getting the type of 'renderer' (line 212)
        renderer_114129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'renderer', False)
        # Obtaining the member 'draw_path' of a type (line 212)
        draw_path_114130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), renderer_114129, 'draw_path')
        # Calling draw_path(args, kwargs) (line 212)
        draw_path_call_result_114136 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), draw_path_114130, *[gc_114131, tpath_114132, affine_114133, rgbFace_114134], **kwargs_114135)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_114137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114137)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_114137


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 204, 0, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'withStroke.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'withStroke' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'withStroke', withStroke)
# Declaration of the 'SimplePatchShadow' class
# Getting the type of 'AbstractPathEffect' (line 215)
AbstractPathEffect_114138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'AbstractPathEffect')

class SimplePatchShadow(AbstractPathEffect_114138, ):
    unicode_114139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 4), 'unicode', u'A simple shadow via a filled patch.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_114140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        int_114141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 31), tuple_114140, int_114141)
        # Adding element type (line 217)
        int_114142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 31), tuple_114140, int_114142)
        
        # Getting the type of 'None' (line 218)
        None_114143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'None')
        # Getting the type of 'None' (line 218)
        None_114144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 44), 'None')
        float_114145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 21), 'float')
        defaults = [tuple_114140, None_114143, None_114144, float_114145]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimplePatchShadow.__init__', ['offset', 'shadow_rgbFace', 'alpha', 'rho'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['offset', 'shadow_rgbFace', 'alpha', 'rho'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_114146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'unicode', u'\n        Parameters\n        ----------\n        offset : pair of floats\n            The offset of the shadow in points.\n        shadow_rgbFace : color\n            The shadow color.\n        alpha : float\n            The alpha transparency of the created shadow patch.\n            Default is 0.3.\n            http://matplotlib.1069221.n5.nabble.com/path-effects-question-td27630.html\n        rho : float\n            A scale factor to apply to the rgbFace color if `shadow_rgbFace`\n            is not specified. Default is 0.3.\n        **kwargs\n            Extra keywords are stored and passed through to\n            :meth:`AbstractPathEffect._update_gc`.\n\n        ')
        
        # Call to __init__(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'offset' (line 239)
        offset_114153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 48), 'offset', False)
        # Processing the call keyword arguments (line 239)
        kwargs_114154 = {}
        
        # Call to super(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'SimplePatchShadow' (line 239)
        SimplePatchShadow_114148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'SimplePatchShadow', False)
        # Getting the type of 'self' (line 239)
        self_114149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 33), 'self', False)
        # Processing the call keyword arguments (line 239)
        kwargs_114150 = {}
        # Getting the type of 'super' (line 239)
        super_114147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'super', False)
        # Calling super(args, kwargs) (line 239)
        super_call_result_114151 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), super_114147, *[SimplePatchShadow_114148, self_114149], **kwargs_114150)
        
        # Obtaining the member '__init__' of a type (line 239)
        init___114152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), super_call_result_114151, '__init__')
        # Calling __init__(args, kwargs) (line 239)
        init___call_result_114155 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), init___114152, *[offset_114153], **kwargs_114154)
        
        
        # Type idiom detected: calculating its left and rigth part (line 241)
        # Getting the type of 'shadow_rgbFace' (line 241)
        shadow_rgbFace_114156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'shadow_rgbFace')
        # Getting the type of 'None' (line 241)
        None_114157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'None')
        
        (may_be_114158, more_types_in_union_114159) = may_be_none(shadow_rgbFace_114156, None_114157)

        if may_be_114158:

            if more_types_in_union_114159:
                # Runtime conditional SSA (line 241)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 242):
            
            # Assigning a Name to a Attribute (line 242):
            # Getting the type of 'shadow_rgbFace' (line 242)
            shadow_rgbFace_114160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 35), 'shadow_rgbFace')
            # Getting the type of 'self' (line 242)
            self_114161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self')
            # Setting the type of the member '_shadow_rgbFace' of a type (line 242)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_114161, '_shadow_rgbFace', shadow_rgbFace_114160)

            if more_types_in_union_114159:
                # Runtime conditional SSA for else branch (line 241)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114158) or more_types_in_union_114159):
            
            # Assigning a Call to a Attribute (line 244):
            
            # Assigning a Call to a Attribute (line 244):
            
            # Call to to_rgba(...): (line 244)
            # Processing the call arguments (line 244)
            # Getting the type of 'shadow_rgbFace' (line 244)
            shadow_rgbFace_114164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 51), 'shadow_rgbFace', False)
            # Processing the call keyword arguments (line 244)
            kwargs_114165 = {}
            # Getting the type of 'mcolors' (line 244)
            mcolors_114162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 35), 'mcolors', False)
            # Obtaining the member 'to_rgba' of a type (line 244)
            to_rgba_114163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 35), mcolors_114162, 'to_rgba')
            # Calling to_rgba(args, kwargs) (line 244)
            to_rgba_call_result_114166 = invoke(stypy.reporting.localization.Localization(__file__, 244, 35), to_rgba_114163, *[shadow_rgbFace_114164], **kwargs_114165)
            
            # Getting the type of 'self' (line 244)
            self_114167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self')
            # Setting the type of the member '_shadow_rgbFace' of a type (line 244)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_114167, '_shadow_rgbFace', to_rgba_call_result_114166)

            if (may_be_114158 and more_types_in_union_114159):
                # SSA join for if statement (line 241)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 246)
        # Getting the type of 'alpha' (line 246)
        alpha_114168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'alpha')
        # Getting the type of 'None' (line 246)
        None_114169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'None')
        
        (may_be_114170, more_types_in_union_114171) = may_be_none(alpha_114168, None_114169)

        if may_be_114170:

            if more_types_in_union_114171:
                # Runtime conditional SSA (line 246)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 247):
            
            # Assigning a Num to a Name (line 247):
            float_114172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'float')
            # Assigning a type to the variable 'alpha' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'alpha', float_114172)

            if more_types_in_union_114171:
                # SSA join for if statement (line 246)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'alpha' (line 249)
        alpha_114173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 22), 'alpha')
        # Getting the type of 'self' (line 249)
        self_114174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member '_alpha' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_114174, '_alpha', alpha_114173)
        
        # Assigning a Name to a Attribute (line 250):
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'rho' (line 250)
        rho_114175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'rho')
        # Getting the type of 'self' (line 250)
        self_114176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member '_rho' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_114176, '_rho', rho_114175)
        
        # Assigning a Name to a Attribute (line 253):
        
        # Assigning a Name to a Attribute (line 253):
        # Getting the type of 'kwargs' (line 253)
        kwargs_114177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'kwargs')
        # Getting the type of 'self' (line 253)
        self_114178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self')
        # Setting the type of the member '_gc' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_114178, '_gc', kwargs_114177)
        
        # Assigning a Call to a Attribute (line 257):
        
        # Assigning a Call to a Attribute (line 257):
        
        # Call to Affine2D(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_114181 = {}
        # Getting the type of 'mtransforms' (line 257)
        mtransforms_114179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 257)
        Affine2D_114180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 28), mtransforms_114179, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 257)
        Affine2D_call_result_114182 = invoke(stypy.reporting.localization.Localization(__file__, 257, 28), Affine2D_114180, *[], **kwargs_114181)
        
        # Getting the type of 'self' (line 257)
        self_114183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self')
        # Setting the type of the member '_offset_tran' of a type (line 257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_114183, '_offset_tran', Affine2D_call_result_114182)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_localization', localization)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_function_name', 'SimplePatchShadow.draw_path')
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SimplePatchShadow.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimplePatchShadow.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        unicode_114184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, (-1)), 'unicode', u'\n        Overrides the standard draw_path to add the shadow offset and\n        necessary color changes for the shadow.\n\n        ')
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to _offset_transform(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'renderer' (line 266)
        renderer_114187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 41), 'renderer', False)
        # Getting the type of 'affine' (line 266)
        affine_114188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 51), 'affine', False)
        # Processing the call keyword arguments (line 266)
        kwargs_114189 = {}
        # Getting the type of 'self' (line 266)
        self_114185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 18), 'self', False)
        # Obtaining the member '_offset_transform' of a type (line 266)
        _offset_transform_114186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 18), self_114185, '_offset_transform')
        # Calling _offset_transform(args, kwargs) (line 266)
        _offset_transform_call_result_114190 = invoke(stypy.reporting.localization.Localization(__file__, 266, 18), _offset_transform_114186, *[renderer_114187, affine_114188], **kwargs_114189)
        
        # Assigning a type to the variable 'affine0' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'affine0', _offset_transform_call_result_114190)
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to new_gc(...): (line 267)
        # Processing the call keyword arguments (line 267)
        kwargs_114193 = {}
        # Getting the type of 'renderer' (line 267)
        renderer_114191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'renderer', False)
        # Obtaining the member 'new_gc' of a type (line 267)
        new_gc_114192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 14), renderer_114191, 'new_gc')
        # Calling new_gc(args, kwargs) (line 267)
        new_gc_call_result_114194 = invoke(stypy.reporting.localization.Localization(__file__, 267, 14), new_gc_114192, *[], **kwargs_114193)
        
        # Assigning a type to the variable 'gc0' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'gc0', new_gc_call_result_114194)
        
        # Call to copy_properties(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'gc' (line 268)
        gc_114197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'gc', False)
        # Processing the call keyword arguments (line 268)
        kwargs_114198 = {}
        # Getting the type of 'gc0' (line 268)
        gc0_114195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'gc0', False)
        # Obtaining the member 'copy_properties' of a type (line 268)
        copy_properties_114196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), gc0_114195, 'copy_properties')
        # Calling copy_properties(args, kwargs) (line 268)
        copy_properties_call_result_114199 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), copy_properties_114196, *[gc_114197], **kwargs_114198)
        
        
        # Type idiom detected: calculating its left and rigth part (line 270)
        # Getting the type of 'self' (line 270)
        self_114200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'self')
        # Obtaining the member '_shadow_rgbFace' of a type (line 270)
        _shadow_rgbFace_114201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 11), self_114200, '_shadow_rgbFace')
        # Getting the type of 'None' (line 270)
        None_114202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 35), 'None')
        
        (may_be_114203, more_types_in_union_114204) = may_be_none(_shadow_rgbFace_114201, None_114202)

        if may_be_114203:

            if more_types_in_union_114204:
                # Runtime conditional SSA (line 270)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Tuple (line 271):
            
            # Assigning a Subscript to a Name (line 271):
            
            # Obtaining the type of the subscript
            int_114205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'int')
            
            # Obtaining the type of the subscript
            int_114206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 47), 'int')
            slice_114207 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 21), None, int_114206, None)
            
            # Evaluating a boolean operation
            # Getting the type of 'rgbFace' (line 271)
            rgbFace_114208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'rgbFace')
            
            # Obtaining an instance of the builtin type 'tuple' (line 271)
            tuple_114209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 271)
            # Adding element type (line 271)
            float_114210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114209, float_114210)
            # Adding element type (line 271)
            float_114211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 37), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114209, float_114211)
            # Adding element type (line 271)
            float_114212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 41), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114209, float_114212)
            
            # Applying the binary operator 'or' (line 271)
            result_or_keyword_114213 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 21), 'or', rgbFace_114208, tuple_114209)
            
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___114214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), result_or_keyword_114213, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_114215 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), getitem___114214, slice_114207)
            
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___114216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), subscript_call_result_114215, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_114217 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), getitem___114216, int_114205)
            
            # Assigning a type to the variable 'tuple_var_assignment_113749' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple_var_assignment_113749', subscript_call_result_114217)
            
            # Assigning a Subscript to a Name (line 271):
            
            # Obtaining the type of the subscript
            int_114218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'int')
            
            # Obtaining the type of the subscript
            int_114219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 47), 'int')
            slice_114220 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 21), None, int_114219, None)
            
            # Evaluating a boolean operation
            # Getting the type of 'rgbFace' (line 271)
            rgbFace_114221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'rgbFace')
            
            # Obtaining an instance of the builtin type 'tuple' (line 271)
            tuple_114222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 271)
            # Adding element type (line 271)
            float_114223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114222, float_114223)
            # Adding element type (line 271)
            float_114224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 37), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114222, float_114224)
            # Adding element type (line 271)
            float_114225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 41), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114222, float_114225)
            
            # Applying the binary operator 'or' (line 271)
            result_or_keyword_114226 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 21), 'or', rgbFace_114221, tuple_114222)
            
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___114227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), result_or_keyword_114226, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_114228 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), getitem___114227, slice_114220)
            
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___114229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), subscript_call_result_114228, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_114230 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), getitem___114229, int_114218)
            
            # Assigning a type to the variable 'tuple_var_assignment_113750' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple_var_assignment_113750', subscript_call_result_114230)
            
            # Assigning a Subscript to a Name (line 271):
            
            # Obtaining the type of the subscript
            int_114231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'int')
            
            # Obtaining the type of the subscript
            int_114232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 47), 'int')
            slice_114233 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 21), None, int_114232, None)
            
            # Evaluating a boolean operation
            # Getting the type of 'rgbFace' (line 271)
            rgbFace_114234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'rgbFace')
            
            # Obtaining an instance of the builtin type 'tuple' (line 271)
            tuple_114235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 271)
            # Adding element type (line 271)
            float_114236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114235, float_114236)
            # Adding element type (line 271)
            float_114237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 37), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114235, float_114237)
            # Adding element type (line 271)
            float_114238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 41), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), tuple_114235, float_114238)
            
            # Applying the binary operator 'or' (line 271)
            result_or_keyword_114239 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 21), 'or', rgbFace_114234, tuple_114235)
            
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___114240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), result_or_keyword_114239, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_114241 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), getitem___114240, slice_114233)
            
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___114242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), subscript_call_result_114241, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_114243 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), getitem___114242, int_114231)
            
            # Assigning a type to the variable 'tuple_var_assignment_113751' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple_var_assignment_113751', subscript_call_result_114243)
            
            # Assigning a Name to a Name (line 271):
            # Getting the type of 'tuple_var_assignment_113749' (line 271)
            tuple_var_assignment_113749_114244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple_var_assignment_113749')
            # Assigning a type to the variable 'r' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'r', tuple_var_assignment_113749_114244)
            
            # Assigning a Name to a Name (line 271):
            # Getting the type of 'tuple_var_assignment_113750' (line 271)
            tuple_var_assignment_113750_114245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple_var_assignment_113750')
            # Assigning a type to the variable 'g' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'g', tuple_var_assignment_113750_114245)
            
            # Assigning a Name to a Name (line 271):
            # Getting the type of 'tuple_var_assignment_113751' (line 271)
            tuple_var_assignment_113751_114246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'tuple_var_assignment_113751')
            # Assigning a type to the variable 'b' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'b', tuple_var_assignment_113751_114246)
            
            # Assigning a Tuple to a Name (line 273):
            
            # Assigning a Tuple to a Name (line 273):
            
            # Obtaining an instance of the builtin type 'tuple' (line 273)
            tuple_114247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 273)
            # Adding element type (line 273)
            # Getting the type of 'r' (line 273)
            r_114248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'r')
            # Getting the type of 'self' (line 273)
            self_114249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 34), 'self')
            # Obtaining the member '_rho' of a type (line 273)
            _rho_114250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 34), self_114249, '_rho')
            # Applying the binary operator '*' (line 273)
            result_mul_114251 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 30), '*', r_114248, _rho_114250)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 30), tuple_114247, result_mul_114251)
            # Adding element type (line 273)
            # Getting the type of 'g' (line 273)
            g_114252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 45), 'g')
            # Getting the type of 'self' (line 273)
            self_114253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 49), 'self')
            # Obtaining the member '_rho' of a type (line 273)
            _rho_114254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 49), self_114253, '_rho')
            # Applying the binary operator '*' (line 273)
            result_mul_114255 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 45), '*', g_114252, _rho_114254)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 30), tuple_114247, result_mul_114255)
            # Adding element type (line 273)
            # Getting the type of 'b' (line 273)
            b_114256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 60), 'b')
            # Getting the type of 'self' (line 273)
            self_114257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 64), 'self')
            # Obtaining the member '_rho' of a type (line 273)
            _rho_114258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 64), self_114257, '_rho')
            # Applying the binary operator '*' (line 273)
            result_mul_114259 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 60), '*', b_114256, _rho_114258)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 30), tuple_114247, result_mul_114259)
            
            # Assigning a type to the variable 'shadow_rgbFace' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'shadow_rgbFace', tuple_114247)

            if more_types_in_union_114204:
                # Runtime conditional SSA for else branch (line 270)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114203) or more_types_in_union_114204):
            
            # Assigning a Attribute to a Name (line 275):
            
            # Assigning a Attribute to a Name (line 275):
            # Getting the type of 'self' (line 275)
            self_114260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'self')
            # Obtaining the member '_shadow_rgbFace' of a type (line 275)
            _shadow_rgbFace_114261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 29), self_114260, '_shadow_rgbFace')
            # Assigning a type to the variable 'shadow_rgbFace' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'shadow_rgbFace', _shadow_rgbFace_114261)

            if (may_be_114203 and more_types_in_union_114204):
                # SSA join for if statement (line 270)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to set_foreground(...): (line 277)
        # Processing the call arguments (line 277)
        unicode_114264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 27), 'unicode', u'none')
        # Processing the call keyword arguments (line 277)
        kwargs_114265 = {}
        # Getting the type of 'gc0' (line 277)
        gc0_114262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'gc0', False)
        # Obtaining the member 'set_foreground' of a type (line 277)
        set_foreground_114263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), gc0_114262, 'set_foreground')
        # Calling set_foreground(args, kwargs) (line 277)
        set_foreground_call_result_114266 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), set_foreground_114263, *[unicode_114264], **kwargs_114265)
        
        
        # Call to set_alpha(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'self' (line 278)
        self_114269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'self', False)
        # Obtaining the member '_alpha' of a type (line 278)
        _alpha_114270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 22), self_114269, '_alpha')
        # Processing the call keyword arguments (line 278)
        kwargs_114271 = {}
        # Getting the type of 'gc0' (line 278)
        gc0_114267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'gc0', False)
        # Obtaining the member 'set_alpha' of a type (line 278)
        set_alpha_114268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), gc0_114267, 'set_alpha')
        # Calling set_alpha(args, kwargs) (line 278)
        set_alpha_call_result_114272 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), set_alpha_114268, *[_alpha_114270], **kwargs_114271)
        
        
        # Call to set_linewidth(...): (line 279)
        # Processing the call arguments (line 279)
        int_114275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'int')
        # Processing the call keyword arguments (line 279)
        kwargs_114276 = {}
        # Getting the type of 'gc0' (line 279)
        gc0_114273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'gc0', False)
        # Obtaining the member 'set_linewidth' of a type (line 279)
        set_linewidth_114274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), gc0_114273, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 279)
        set_linewidth_call_result_114277 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), set_linewidth_114274, *[int_114275], **kwargs_114276)
        
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to _update_gc(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'gc0' (line 281)
        gc0_114280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 30), 'gc0', False)
        # Getting the type of 'self' (line 281)
        self_114281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 35), 'self', False)
        # Obtaining the member '_gc' of a type (line 281)
        _gc_114282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 35), self_114281, '_gc')
        # Processing the call keyword arguments (line 281)
        kwargs_114283 = {}
        # Getting the type of 'self' (line 281)
        self_114278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'self', False)
        # Obtaining the member '_update_gc' of a type (line 281)
        _update_gc_114279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 14), self_114278, '_update_gc')
        # Calling _update_gc(args, kwargs) (line 281)
        _update_gc_call_result_114284 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), _update_gc_114279, *[gc0_114280, _gc_114282], **kwargs_114283)
        
        # Assigning a type to the variable 'gc0' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'gc0', _update_gc_call_result_114284)
        
        # Call to draw_path(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'gc0' (line 282)
        gc0_114287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'gc0', False)
        # Getting the type of 'tpath' (line 282)
        tpath_114288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'tpath', False)
        # Getting the type of 'affine0' (line 282)
        affine0_114289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 39), 'affine0', False)
        # Getting the type of 'shadow_rgbFace' (line 282)
        shadow_rgbFace_114290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 48), 'shadow_rgbFace', False)
        # Processing the call keyword arguments (line 282)
        kwargs_114291 = {}
        # Getting the type of 'renderer' (line 282)
        renderer_114285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'renderer', False)
        # Obtaining the member 'draw_path' of a type (line 282)
        draw_path_114286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), renderer_114285, 'draw_path')
        # Calling draw_path(args, kwargs) (line 282)
        draw_path_call_result_114292 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), draw_path_114286, *[gc0_114287, tpath_114288, affine0_114289, shadow_rgbFace_114290], **kwargs_114291)
        
        
        # Call to restore(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_114295 = {}
        # Getting the type of 'gc0' (line 283)
        gc0_114293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'gc0', False)
        # Obtaining the member 'restore' of a type (line 283)
        restore_114294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), gc0_114293, 'restore')
        # Calling restore(args, kwargs) (line 283)
        restore_call_result_114296 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), restore_114294, *[], **kwargs_114295)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_114297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_114297


# Assigning a type to the variable 'SimplePatchShadow' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'SimplePatchShadow', SimplePatchShadow)
# Declaration of the 'withSimplePatchShadow' class
# Getting the type of 'SimplePatchShadow' (line 286)
SimplePatchShadow_114298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 28), 'SimplePatchShadow')

class withSimplePatchShadow(SimplePatchShadow_114298, ):
    unicode_114299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'unicode', u'\n    Adds a simple :class:`SimplePatchShadow` and then draws the\n    original Artist to avoid needing to call :class:`Normal`.\n\n    ')

    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_localization', localization)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_function_name', 'withSimplePatchShadow.draw_path')
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        withSimplePatchShadow.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'withSimplePatchShadow.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        
        # Call to draw_path(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'self' (line 293)
        self_114302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 36), 'self', False)
        # Getting the type of 'renderer' (line 293)
        renderer_114303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 42), 'renderer', False)
        # Getting the type of 'gc' (line 293)
        gc_114304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 52), 'gc', False)
        # Getting the type of 'tpath' (line 293)
        tpath_114305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 56), 'tpath', False)
        # Getting the type of 'affine' (line 293)
        affine_114306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 63), 'affine', False)
        # Getting the type of 'rgbFace' (line 293)
        rgbFace_114307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 71), 'rgbFace', False)
        # Processing the call keyword arguments (line 293)
        kwargs_114308 = {}
        # Getting the type of 'SimplePatchShadow' (line 293)
        SimplePatchShadow_114300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'SimplePatchShadow', False)
        # Obtaining the member 'draw_path' of a type (line 293)
        draw_path_114301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), SimplePatchShadow_114300, 'draw_path')
        # Calling draw_path(args, kwargs) (line 293)
        draw_path_call_result_114309 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), draw_path_114301, *[self_114302, renderer_114303, gc_114304, tpath_114305, affine_114306, rgbFace_114307], **kwargs_114308)
        
        
        # Call to draw_path(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'gc' (line 294)
        gc_114312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 27), 'gc', False)
        # Getting the type of 'tpath' (line 294)
        tpath_114313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'tpath', False)
        # Getting the type of 'affine' (line 294)
        affine_114314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 38), 'affine', False)
        # Getting the type of 'rgbFace' (line 294)
        rgbFace_114315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'rgbFace', False)
        # Processing the call keyword arguments (line 294)
        kwargs_114316 = {}
        # Getting the type of 'renderer' (line 294)
        renderer_114310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'renderer', False)
        # Obtaining the member 'draw_path' of a type (line 294)
        draw_path_114311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), renderer_114310, 'draw_path')
        # Calling draw_path(args, kwargs) (line 294)
        draw_path_call_result_114317 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), draw_path_114311, *[gc_114312, tpath_114313, affine_114314, rgbFace_114315], **kwargs_114316)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_114318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_114318


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 286, 0, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'withSimplePatchShadow.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'withSimplePatchShadow' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'withSimplePatchShadow', withSimplePatchShadow)
# Declaration of the 'SimpleLineShadow' class
# Getting the type of 'AbstractPathEffect' (line 297)
AbstractPathEffect_114319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'AbstractPathEffect')

class SimpleLineShadow(AbstractPathEffect_114319, ):
    unicode_114320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'unicode', u'A simple shadow via a line.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_114321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        int_114322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 31), tuple_114321, int_114322)
        # Adding element type (line 299)
        int_114323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 31), tuple_114321, int_114323)
        
        unicode_114324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 30), 'unicode', u'k')
        float_114325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 41), 'float')
        float_114326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 50), 'float')
        defaults = [tuple_114321, unicode_114324, float_114325, float_114326]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimpleLineShadow.__init__', ['offset', 'shadow_color', 'alpha', 'rho'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['offset', 'shadow_color', 'alpha', 'rho'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_114327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, (-1)), 'unicode', u"\n        Parameters\n        ----------\n        offset : pair of floats\n            The offset to apply to the path, in points.\n        shadow_color : color\n            The shadow color. Default is black.\n            A value of ``None`` takes the original artist's color\n            with a scale factor of `rho`.\n        alpha : float\n            The alpha transparency of the created shadow patch.\n            Default is 0.3.\n        rho : float\n            A scale factor to apply to the rgbFace color if `shadow_rgbFace`\n            is ``None``. Default is 0.3.\n        **kwargs\n            Extra keywords are stored and passed through to\n            :meth:`AbstractPathEffect._update_gc`.\n\n        ")
        
        # Call to __init__(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'offset' (line 321)
        offset_114334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 47), 'offset', False)
        # Processing the call keyword arguments (line 321)
        kwargs_114335 = {}
        
        # Call to super(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'SimpleLineShadow' (line 321)
        SimpleLineShadow_114329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 14), 'SimpleLineShadow', False)
        # Getting the type of 'self' (line 321)
        self_114330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 32), 'self', False)
        # Processing the call keyword arguments (line 321)
        kwargs_114331 = {}
        # Getting the type of 'super' (line 321)
        super_114328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'super', False)
        # Calling super(args, kwargs) (line 321)
        super_call_result_114332 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), super_114328, *[SimpleLineShadow_114329, self_114330], **kwargs_114331)
        
        # Obtaining the member '__init__' of a type (line 321)
        init___114333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), super_call_result_114332, '__init__')
        # Calling __init__(args, kwargs) (line 321)
        init___call_result_114336 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), init___114333, *[offset_114334], **kwargs_114335)
        
        
        # Type idiom detected: calculating its left and rigth part (line 322)
        # Getting the type of 'shadow_color' (line 322)
        shadow_color_114337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'shadow_color')
        # Getting the type of 'None' (line 322)
        None_114338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 27), 'None')
        
        (may_be_114339, more_types_in_union_114340) = may_be_none(shadow_color_114337, None_114338)

        if may_be_114339:

            if more_types_in_union_114340:
                # Runtime conditional SSA (line 322)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 323):
            
            # Assigning a Name to a Attribute (line 323):
            # Getting the type of 'shadow_color' (line 323)
            shadow_color_114341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 33), 'shadow_color')
            # Getting the type of 'self' (line 323)
            self_114342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'self')
            # Setting the type of the member '_shadow_color' of a type (line 323)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), self_114342, '_shadow_color', shadow_color_114341)

            if more_types_in_union_114340:
                # Runtime conditional SSA for else branch (line 322)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114339) or more_types_in_union_114340):
            
            # Assigning a Call to a Attribute (line 325):
            
            # Assigning a Call to a Attribute (line 325):
            
            # Call to to_rgba(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'shadow_color' (line 325)
            shadow_color_114345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 49), 'shadow_color', False)
            # Processing the call keyword arguments (line 325)
            kwargs_114346 = {}
            # Getting the type of 'mcolors' (line 325)
            mcolors_114343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'mcolors', False)
            # Obtaining the member 'to_rgba' of a type (line 325)
            to_rgba_114344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 33), mcolors_114343, 'to_rgba')
            # Calling to_rgba(args, kwargs) (line 325)
            to_rgba_call_result_114347 = invoke(stypy.reporting.localization.Localization(__file__, 325, 33), to_rgba_114344, *[shadow_color_114345], **kwargs_114346)
            
            # Getting the type of 'self' (line 325)
            self_114348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'self')
            # Setting the type of the member '_shadow_color' of a type (line 325)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), self_114348, '_shadow_color', to_rgba_call_result_114347)

            if (may_be_114339 and more_types_in_union_114340):
                # SSA join for if statement (line 322)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 326):
        
        # Assigning a Name to a Attribute (line 326):
        # Getting the type of 'alpha' (line 326)
        alpha_114349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'alpha')
        # Getting the type of 'self' (line 326)
        self_114350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self')
        # Setting the type of the member '_alpha' of a type (line 326)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_114350, '_alpha', alpha_114349)
        
        # Assigning a Name to a Attribute (line 327):
        
        # Assigning a Name to a Attribute (line 327):
        # Getting the type of 'rho' (line 327)
        rho_114351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'rho')
        # Getting the type of 'self' (line 327)
        self_114352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'self')
        # Setting the type of the member '_rho' of a type (line 327)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), self_114352, '_rho', rho_114351)
        
        # Assigning a Name to a Attribute (line 330):
        
        # Assigning a Name to a Attribute (line 330):
        # Getting the type of 'kwargs' (line 330)
        kwargs_114353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'kwargs')
        # Getting the type of 'self' (line 330)
        self_114354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self')
        # Setting the type of the member '_gc' of a type (line 330)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_114354, '_gc', kwargs_114353)
        
        # Assigning a Call to a Attribute (line 334):
        
        # Assigning a Call to a Attribute (line 334):
        
        # Call to Affine2D(...): (line 334)
        # Processing the call keyword arguments (line 334)
        kwargs_114357 = {}
        # Getting the type of 'mtransforms' (line 334)
        mtransforms_114355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'mtransforms', False)
        # Obtaining the member 'Affine2D' of a type (line 334)
        Affine2D_114356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 28), mtransforms_114355, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 334)
        Affine2D_call_result_114358 = invoke(stypy.reporting.localization.Localization(__file__, 334, 28), Affine2D_114356, *[], **kwargs_114357)
        
        # Getting the type of 'self' (line 334)
        self_114359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'self')
        # Setting the type of the member '_offset_tran' of a type (line 334)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), self_114359, '_offset_tran', Affine2D_call_result_114358)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_localization', localization)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_function_name', 'SimpleLineShadow.draw_path')
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SimpleLineShadow.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SimpleLineShadow.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        unicode_114360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'unicode', u'\n        Overrides the standard draw_path to add the shadow offset and\n        necessary color changes for the shadow.\n\n        ')
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to _offset_transform(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'renderer' (line 343)
        renderer_114363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'renderer', False)
        # Getting the type of 'affine' (line 343)
        affine_114364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 51), 'affine', False)
        # Processing the call keyword arguments (line 343)
        kwargs_114365 = {}
        # Getting the type of 'self' (line 343)
        self_114361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'self', False)
        # Obtaining the member '_offset_transform' of a type (line 343)
        _offset_transform_114362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 18), self_114361, '_offset_transform')
        # Calling _offset_transform(args, kwargs) (line 343)
        _offset_transform_call_result_114366 = invoke(stypy.reporting.localization.Localization(__file__, 343, 18), _offset_transform_114362, *[renderer_114363, affine_114364], **kwargs_114365)
        
        # Assigning a type to the variable 'affine0' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'affine0', _offset_transform_call_result_114366)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to new_gc(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_114369 = {}
        # Getting the type of 'renderer' (line 344)
        renderer_114367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 14), 'renderer', False)
        # Obtaining the member 'new_gc' of a type (line 344)
        new_gc_114368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 14), renderer_114367, 'new_gc')
        # Calling new_gc(args, kwargs) (line 344)
        new_gc_call_result_114370 = invoke(stypy.reporting.localization.Localization(__file__, 344, 14), new_gc_114368, *[], **kwargs_114369)
        
        # Assigning a type to the variable 'gc0' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'gc0', new_gc_call_result_114370)
        
        # Call to copy_properties(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'gc' (line 345)
        gc_114373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 28), 'gc', False)
        # Processing the call keyword arguments (line 345)
        kwargs_114374 = {}
        # Getting the type of 'gc0' (line 345)
        gc0_114371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'gc0', False)
        # Obtaining the member 'copy_properties' of a type (line 345)
        copy_properties_114372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), gc0_114371, 'copy_properties')
        # Calling copy_properties(args, kwargs) (line 345)
        copy_properties_call_result_114375 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), copy_properties_114372, *[gc_114373], **kwargs_114374)
        
        
        # Type idiom detected: calculating its left and rigth part (line 347)
        # Getting the type of 'self' (line 347)
        self_114376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'self')
        # Obtaining the member '_shadow_color' of a type (line 347)
        _shadow_color_114377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 11), self_114376, '_shadow_color')
        # Getting the type of 'None' (line 347)
        None_114378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 33), 'None')
        
        (may_be_114379, more_types_in_union_114380) = may_be_none(_shadow_color_114377, None_114378)

        if may_be_114379:

            if more_types_in_union_114380:
                # Runtime conditional SSA (line 347)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Tuple (line 348):
            
            # Assigning a Subscript to a Name (line 348):
            
            # Obtaining the type of the subscript
            int_114381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'int')
            
            # Obtaining the type of the subscript
            int_114382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 60), 'int')
            slice_114383 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 21), None, int_114382, None)
            
            # Evaluating a boolean operation
            
            # Call to get_foreground(...): (line 348)
            # Processing the call keyword arguments (line 348)
            kwargs_114386 = {}
            # Getting the type of 'gc0' (line 348)
            gc0_114384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'gc0', False)
            # Obtaining the member 'get_foreground' of a type (line 348)
            get_foreground_114385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), gc0_114384, 'get_foreground')
            # Calling get_foreground(args, kwargs) (line 348)
            get_foreground_call_result_114387 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), get_foreground_114385, *[], **kwargs_114386)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 348)
            tuple_114388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 348)
            # Adding element type (line 348)
            float_114389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114388, float_114389)
            # Adding element type (line 348)
            float_114390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 50), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114388, float_114390)
            # Adding element type (line 348)
            float_114391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114388, float_114391)
            
            # Applying the binary operator 'or' (line 348)
            result_or_keyword_114392 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 21), 'or', get_foreground_call_result_114387, tuple_114388)
            
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___114393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), result_or_keyword_114392, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_114394 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), getitem___114393, slice_114383)
            
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___114395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), subscript_call_result_114394, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_114396 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), getitem___114395, int_114381)
            
            # Assigning a type to the variable 'tuple_var_assignment_113752' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'tuple_var_assignment_113752', subscript_call_result_114396)
            
            # Assigning a Subscript to a Name (line 348):
            
            # Obtaining the type of the subscript
            int_114397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'int')
            
            # Obtaining the type of the subscript
            int_114398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 60), 'int')
            slice_114399 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 21), None, int_114398, None)
            
            # Evaluating a boolean operation
            
            # Call to get_foreground(...): (line 348)
            # Processing the call keyword arguments (line 348)
            kwargs_114402 = {}
            # Getting the type of 'gc0' (line 348)
            gc0_114400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'gc0', False)
            # Obtaining the member 'get_foreground' of a type (line 348)
            get_foreground_114401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), gc0_114400, 'get_foreground')
            # Calling get_foreground(args, kwargs) (line 348)
            get_foreground_call_result_114403 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), get_foreground_114401, *[], **kwargs_114402)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 348)
            tuple_114404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 348)
            # Adding element type (line 348)
            float_114405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114404, float_114405)
            # Adding element type (line 348)
            float_114406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 50), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114404, float_114406)
            # Adding element type (line 348)
            float_114407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114404, float_114407)
            
            # Applying the binary operator 'or' (line 348)
            result_or_keyword_114408 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 21), 'or', get_foreground_call_result_114403, tuple_114404)
            
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___114409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), result_or_keyword_114408, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_114410 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), getitem___114409, slice_114399)
            
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___114411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), subscript_call_result_114410, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_114412 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), getitem___114411, int_114397)
            
            # Assigning a type to the variable 'tuple_var_assignment_113753' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'tuple_var_assignment_113753', subscript_call_result_114412)
            
            # Assigning a Subscript to a Name (line 348):
            
            # Obtaining the type of the subscript
            int_114413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'int')
            
            # Obtaining the type of the subscript
            int_114414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 60), 'int')
            slice_114415 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 21), None, int_114414, None)
            
            # Evaluating a boolean operation
            
            # Call to get_foreground(...): (line 348)
            # Processing the call keyword arguments (line 348)
            kwargs_114418 = {}
            # Getting the type of 'gc0' (line 348)
            gc0_114416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'gc0', False)
            # Obtaining the member 'get_foreground' of a type (line 348)
            get_foreground_114417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), gc0_114416, 'get_foreground')
            # Calling get_foreground(args, kwargs) (line 348)
            get_foreground_call_result_114419 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), get_foreground_114417, *[], **kwargs_114418)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 348)
            tuple_114420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 348)
            # Adding element type (line 348)
            float_114421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114420, float_114421)
            # Adding element type (line 348)
            float_114422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 50), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114420, float_114422)
            # Adding element type (line 348)
            float_114423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 46), tuple_114420, float_114423)
            
            # Applying the binary operator 'or' (line 348)
            result_or_keyword_114424 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 21), 'or', get_foreground_call_result_114419, tuple_114420)
            
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___114425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), result_or_keyword_114424, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_114426 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), getitem___114425, slice_114415)
            
            # Obtaining the member '__getitem__' of a type (line 348)
            getitem___114427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), subscript_call_result_114426, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 348)
            subscript_call_result_114428 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), getitem___114427, int_114413)
            
            # Assigning a type to the variable 'tuple_var_assignment_113754' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'tuple_var_assignment_113754', subscript_call_result_114428)
            
            # Assigning a Name to a Name (line 348):
            # Getting the type of 'tuple_var_assignment_113752' (line 348)
            tuple_var_assignment_113752_114429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'tuple_var_assignment_113752')
            # Assigning a type to the variable 'r' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'r', tuple_var_assignment_113752_114429)
            
            # Assigning a Name to a Name (line 348):
            # Getting the type of 'tuple_var_assignment_113753' (line 348)
            tuple_var_assignment_113753_114430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'tuple_var_assignment_113753')
            # Assigning a type to the variable 'g' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'g', tuple_var_assignment_113753_114430)
            
            # Assigning a Name to a Name (line 348):
            # Getting the type of 'tuple_var_assignment_113754' (line 348)
            tuple_var_assignment_113754_114431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'tuple_var_assignment_113754')
            # Assigning a type to the variable 'b' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'b', tuple_var_assignment_113754_114431)
            
            # Assigning a Tuple to a Name (line 350):
            
            # Assigning a Tuple to a Name (line 350):
            
            # Obtaining an instance of the builtin type 'tuple' (line 350)
            tuple_114432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 350)
            # Adding element type (line 350)
            # Getting the type of 'r' (line 350)
            r_114433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 30), 'r')
            # Getting the type of 'self' (line 350)
            self_114434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'self')
            # Obtaining the member '_rho' of a type (line 350)
            _rho_114435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 34), self_114434, '_rho')
            # Applying the binary operator '*' (line 350)
            result_mul_114436 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 30), '*', r_114433, _rho_114435)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 30), tuple_114432, result_mul_114436)
            # Adding element type (line 350)
            # Getting the type of 'g' (line 350)
            g_114437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 45), 'g')
            # Getting the type of 'self' (line 350)
            self_114438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'self')
            # Obtaining the member '_rho' of a type (line 350)
            _rho_114439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 49), self_114438, '_rho')
            # Applying the binary operator '*' (line 350)
            result_mul_114440 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 45), '*', g_114437, _rho_114439)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 30), tuple_114432, result_mul_114440)
            # Adding element type (line 350)
            # Getting the type of 'b' (line 350)
            b_114441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 60), 'b')
            # Getting the type of 'self' (line 350)
            self_114442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 64), 'self')
            # Obtaining the member '_rho' of a type (line 350)
            _rho_114443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 64), self_114442, '_rho')
            # Applying the binary operator '*' (line 350)
            result_mul_114444 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 60), '*', b_114441, _rho_114443)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 30), tuple_114432, result_mul_114444)
            
            # Assigning a type to the variable 'shadow_rgbFace' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'shadow_rgbFace', tuple_114432)

            if more_types_in_union_114380:
                # Runtime conditional SSA for else branch (line 347)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114379) or more_types_in_union_114380):
            
            # Assigning a Attribute to a Name (line 352):
            
            # Assigning a Attribute to a Name (line 352):
            # Getting the type of 'self' (line 352)
            self_114445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'self')
            # Obtaining the member '_shadow_color' of a type (line 352)
            _shadow_color_114446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), self_114445, '_shadow_color')
            # Assigning a type to the variable 'shadow_rgbFace' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'shadow_rgbFace', _shadow_color_114446)

            if (may_be_114379 and more_types_in_union_114380):
                # SSA join for if statement (line 347)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 354):
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'None' (line 354)
        None_114447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 21), 'None')
        # Assigning a type to the variable 'fill_color' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'fill_color', None_114447)
        
        # Call to set_foreground(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'shadow_rgbFace' (line 356)
        shadow_rgbFace_114450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'shadow_rgbFace', False)
        # Processing the call keyword arguments (line 356)
        kwargs_114451 = {}
        # Getting the type of 'gc0' (line 356)
        gc0_114448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'gc0', False)
        # Obtaining the member 'set_foreground' of a type (line 356)
        set_foreground_114449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), gc0_114448, 'set_foreground')
        # Calling set_foreground(args, kwargs) (line 356)
        set_foreground_call_result_114452 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), set_foreground_114449, *[shadow_rgbFace_114450], **kwargs_114451)
        
        
        # Call to set_alpha(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'self' (line 357)
        self_114455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'self', False)
        # Obtaining the member '_alpha' of a type (line 357)
        _alpha_114456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 22), self_114455, '_alpha')
        # Processing the call keyword arguments (line 357)
        kwargs_114457 = {}
        # Getting the type of 'gc0' (line 357)
        gc0_114453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'gc0', False)
        # Obtaining the member 'set_alpha' of a type (line 357)
        set_alpha_114454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), gc0_114453, 'set_alpha')
        # Calling set_alpha(args, kwargs) (line 357)
        set_alpha_call_result_114458 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), set_alpha_114454, *[_alpha_114456], **kwargs_114457)
        
        
        # Assigning a Call to a Name (line 359):
        
        # Assigning a Call to a Name (line 359):
        
        # Call to _update_gc(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'gc0' (line 359)
        gc0_114461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 30), 'gc0', False)
        # Getting the type of 'self' (line 359)
        self_114462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 35), 'self', False)
        # Obtaining the member '_gc' of a type (line 359)
        _gc_114463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 35), self_114462, '_gc')
        # Processing the call keyword arguments (line 359)
        kwargs_114464 = {}
        # Getting the type of 'self' (line 359)
        self_114459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 14), 'self', False)
        # Obtaining the member '_update_gc' of a type (line 359)
        _update_gc_114460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 14), self_114459, '_update_gc')
        # Calling _update_gc(args, kwargs) (line 359)
        _update_gc_call_result_114465 = invoke(stypy.reporting.localization.Localization(__file__, 359, 14), _update_gc_114460, *[gc0_114461, _gc_114463], **kwargs_114464)
        
        # Assigning a type to the variable 'gc0' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'gc0', _update_gc_call_result_114465)
        
        # Call to draw_path(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'gc0' (line 360)
        gc0_114468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 'gc0', False)
        # Getting the type of 'tpath' (line 360)
        tpath_114469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 32), 'tpath', False)
        # Getting the type of 'affine0' (line 360)
        affine0_114470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 39), 'affine0', False)
        # Getting the type of 'fill_color' (line 360)
        fill_color_114471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 48), 'fill_color', False)
        # Processing the call keyword arguments (line 360)
        kwargs_114472 = {}
        # Getting the type of 'renderer' (line 360)
        renderer_114466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'renderer', False)
        # Obtaining the member 'draw_path' of a type (line 360)
        draw_path_114467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), renderer_114466, 'draw_path')
        # Calling draw_path(args, kwargs) (line 360)
        draw_path_call_result_114473 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), draw_path_114467, *[gc0_114468, tpath_114469, affine0_114470, fill_color_114471], **kwargs_114472)
        
        
        # Call to restore(...): (line 361)
        # Processing the call keyword arguments (line 361)
        kwargs_114476 = {}
        # Getting the type of 'gc0' (line 361)
        gc0_114474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'gc0', False)
        # Obtaining the member 'restore' of a type (line 361)
        restore_114475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), gc0_114474, 'restore')
        # Calling restore(args, kwargs) (line 361)
        restore_call_result_114477 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), restore_114475, *[], **kwargs_114476)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_114478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_114478


# Assigning a type to the variable 'SimpleLineShadow' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'SimpleLineShadow', SimpleLineShadow)
# Declaration of the 'PathPatchEffect' class
# Getting the type of 'AbstractPathEffect' (line 364)
AbstractPathEffect_114479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 22), 'AbstractPathEffect')

class PathPatchEffect(AbstractPathEffect_114479, ):
    unicode_114480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, (-1)), 'unicode', u'\n    Draws a :class:`~matplotlib.patches.PathPatch` instance whose Path\n    comes from the original PathEffect artist.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 370)
        tuple_114481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 370)
        # Adding element type (line 370)
        int_114482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 31), tuple_114481, int_114482)
        # Adding element type (line 370)
        int_114483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 31), tuple_114481, int_114483)
        
        defaults = [tuple_114481]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 370, 4, False)
        # Assigning a type to the variable 'self' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathPatchEffect.__init__', ['offset'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_114484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, (-1)), 'unicode', u'\n        Parameters\n        ----------\n        offset : pair of floats\n            The offset to apply to the path, in points.\n        **kwargs :\n            All keyword arguments are passed through to the\n            :class:`~matplotlib.patches.PathPatch` constructor. The\n            properties which cannot be overridden are "path", "clip_box"\n            "transform" and "clip_path".\n        ')
        
        # Call to __init__(...): (line 382)
        # Processing the call keyword arguments (line 382)
        # Getting the type of 'offset' (line 382)
        offset_114491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 53), 'offset', False)
        keyword_114492 = offset_114491
        kwargs_114493 = {'offset': keyword_114492}
        
        # Call to super(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'PathPatchEffect' (line 382)
        PathPatchEffect_114486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 14), 'PathPatchEffect', False)
        # Getting the type of 'self' (line 382)
        self_114487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 31), 'self', False)
        # Processing the call keyword arguments (line 382)
        kwargs_114488 = {}
        # Getting the type of 'super' (line 382)
        super_114485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'super', False)
        # Calling super(args, kwargs) (line 382)
        super_call_result_114489 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), super_114485, *[PathPatchEffect_114486, self_114487], **kwargs_114488)
        
        # Obtaining the member '__init__' of a type (line 382)
        init___114490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), super_call_result_114489, '__init__')
        # Calling __init__(args, kwargs) (line 382)
        init___call_result_114494 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), init___114490, *[], **kwargs_114493)
        
        
        # Assigning a Call to a Attribute (line 383):
        
        # Assigning a Call to a Attribute (line 383):
        
        # Call to PathPatch(...): (line 383)
        # Processing the call arguments (line 383)
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_114497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        
        # Processing the call keyword arguments (line 383)
        # Getting the type of 'kwargs' (line 383)
        kwargs_114498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 46), 'kwargs', False)
        kwargs_114499 = {'kwargs_114498': kwargs_114498}
        # Getting the type of 'mpatches' (line 383)
        mpatches_114495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 21), 'mpatches', False)
        # Obtaining the member 'PathPatch' of a type (line 383)
        PathPatch_114496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 21), mpatches_114495, 'PathPatch')
        # Calling PathPatch(args, kwargs) (line 383)
        PathPatch_call_result_114500 = invoke(stypy.reporting.localization.Localization(__file__, 383, 21), PathPatch_114496, *[list_114497], **kwargs_114499)
        
        # Getting the type of 'self' (line 383)
        self_114501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'patch' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_114501, 'patch', PathPatch_call_result_114500)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_localization', localization)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_function_name', 'PathPatchEffect.draw_path')
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_param_names_list', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'])
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathPatchEffect.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathPatchEffect.draw_path', ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['renderer', 'gc', 'tpath', 'affine', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to _offset_transform(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'renderer' (line 386)
        renderer_114504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 40), 'renderer', False)
        # Getting the type of 'affine' (line 386)
        affine_114505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 50), 'affine', False)
        # Processing the call keyword arguments (line 386)
        kwargs_114506 = {}
        # Getting the type of 'self' (line 386)
        self_114502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'self', False)
        # Obtaining the member '_offset_transform' of a type (line 386)
        _offset_transform_114503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), self_114502, '_offset_transform')
        # Calling _offset_transform(args, kwargs) (line 386)
        _offset_transform_call_result_114507 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), _offset_transform_114503, *[renderer_114504, affine_114505], **kwargs_114506)
        
        # Assigning a type to the variable 'affine' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'affine', _offset_transform_call_result_114507)
        
        # Assigning a Name to a Attribute (line 387):
        
        # Assigning a Name to a Attribute (line 387):
        # Getting the type of 'tpath' (line 387)
        tpath_114508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 27), 'tpath')
        # Getting the type of 'self' (line 387)
        self_114509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'self')
        # Obtaining the member 'patch' of a type (line 387)
        patch_114510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), self_114509, 'patch')
        # Setting the type of the member '_path' of a type (line 387)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), patch_114510, '_path', tpath_114508)
        
        # Call to set_transform(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'affine' (line 388)
        affine_114514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 33), 'affine', False)
        # Processing the call keyword arguments (line 388)
        kwargs_114515 = {}
        # Getting the type of 'self' (line 388)
        self_114511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self', False)
        # Obtaining the member 'patch' of a type (line 388)
        patch_114512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_114511, 'patch')
        # Obtaining the member 'set_transform' of a type (line 388)
        set_transform_114513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), patch_114512, 'set_transform')
        # Calling set_transform(args, kwargs) (line 388)
        set_transform_call_result_114516 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), set_transform_114513, *[affine_114514], **kwargs_114515)
        
        
        # Call to set_clip_box(...): (line 389)
        # Processing the call arguments (line 389)
        
        # Call to get_clip_rectangle(...): (line 389)
        # Processing the call keyword arguments (line 389)
        kwargs_114522 = {}
        # Getting the type of 'gc' (line 389)
        gc_114520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'gc', False)
        # Obtaining the member 'get_clip_rectangle' of a type (line 389)
        get_clip_rectangle_114521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 32), gc_114520, 'get_clip_rectangle')
        # Calling get_clip_rectangle(args, kwargs) (line 389)
        get_clip_rectangle_call_result_114523 = invoke(stypy.reporting.localization.Localization(__file__, 389, 32), get_clip_rectangle_114521, *[], **kwargs_114522)
        
        # Processing the call keyword arguments (line 389)
        kwargs_114524 = {}
        # Getting the type of 'self' (line 389)
        self_114517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self', False)
        # Obtaining the member 'patch' of a type (line 389)
        patch_114518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_114517, 'patch')
        # Obtaining the member 'set_clip_box' of a type (line 389)
        set_clip_box_114519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), patch_114518, 'set_clip_box')
        # Calling set_clip_box(args, kwargs) (line 389)
        set_clip_box_call_result_114525 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), set_clip_box_114519, *[get_clip_rectangle_call_result_114523], **kwargs_114524)
        
        
        # Assigning a Call to a Name (line 390):
        
        # Assigning a Call to a Name (line 390):
        
        # Call to get_clip_path(...): (line 390)
        # Processing the call keyword arguments (line 390)
        kwargs_114528 = {}
        # Getting the type of 'gc' (line 390)
        gc_114526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'gc', False)
        # Obtaining the member 'get_clip_path' of a type (line 390)
        get_clip_path_114527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 20), gc_114526, 'get_clip_path')
        # Calling get_clip_path(args, kwargs) (line 390)
        get_clip_path_call_result_114529 = invoke(stypy.reporting.localization.Localization(__file__, 390, 20), get_clip_path_114527, *[], **kwargs_114528)
        
        # Assigning a type to the variable 'clip_path' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'clip_path', get_clip_path_call_result_114529)
        
        # Getting the type of 'clip_path' (line 391)
        clip_path_114530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'clip_path')
        # Testing the type of an if condition (line 391)
        if_condition_114531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 8), clip_path_114530)
        # Assigning a type to the variable 'if_condition_114531' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'if_condition_114531', if_condition_114531)
        # SSA begins for if statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_clip_path(...): (line 392)
        # Getting the type of 'clip_path' (line 392)
        clip_path_114535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 38), 'clip_path', False)
        # Processing the call keyword arguments (line 392)
        kwargs_114536 = {}
        # Getting the type of 'self' (line 392)
        self_114532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'self', False)
        # Obtaining the member 'patch' of a type (line 392)
        patch_114533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), self_114532, 'patch')
        # Obtaining the member 'set_clip_path' of a type (line 392)
        set_clip_path_114534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), patch_114533, 'set_clip_path')
        # Calling set_clip_path(args, kwargs) (line 392)
        set_clip_path_call_result_114537 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), set_clip_path_114534, *[clip_path_114535], **kwargs_114536)
        
        # SSA join for if statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'renderer' (line 393)
        renderer_114541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'renderer', False)
        # Processing the call keyword arguments (line 393)
        kwargs_114542 = {}
        # Getting the type of 'self' (line 393)
        self_114538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'self', False)
        # Obtaining the member 'patch' of a type (line 393)
        patch_114539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), self_114538, 'patch')
        # Obtaining the member 'draw' of a type (line 393)
        draw_114540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), patch_114539, 'draw')
        # Calling draw(args, kwargs) (line 393)
        draw_call_result_114543 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), draw_114540, *[renderer_114541], **kwargs_114542)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_114544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_114544


# Assigning a type to the variable 'PathPatchEffect' (line 364)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'PathPatchEffect', PathPatchEffect)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

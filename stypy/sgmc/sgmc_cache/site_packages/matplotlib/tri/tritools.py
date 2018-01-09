
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Tools for triangular grids.
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: 
9: from matplotlib.tri import Triangulation
10: import numpy as np
11: 
12: 
13: class TriAnalyzer(object):
14:     '''
15:     Define basic tools for triangular mesh analysis and improvement.
16: 
17:     A TriAnalizer encapsulates a :class:`~matplotlib.tri.Triangulation`
18:     object and provides basic tools for mesh analysis and mesh improvement.
19: 
20:     Parameters
21:     ----------
22:     triangulation : :class:`~matplotlib.tri.Triangulation` object
23:         The encapsulated triangulation to analyze.
24: 
25:     Attributes
26:     ----------
27:     `scale_factors`
28: 
29:     '''
30:     def __init__(self, triangulation):
31:         if not isinstance(triangulation, Triangulation):
32:             raise ValueError("Expected a Triangulation object")
33:         self._triangulation = triangulation
34: 
35:     @property
36:     def scale_factors(self):
37:         '''
38:         Factors to rescale the triangulation into a unit square.
39: 
40:         Returns *k*, tuple of 2 scale factors.
41: 
42:         Returns
43:         -------
44:         k : tuple of 2 floats (kx, ky)
45:             Tuple of floats that would rescale the triangulation :
46:             ``[triangulation.x * kx, triangulation.y * ky]``
47:             fits exactly inside a unit square.
48: 
49:         '''
50:         compressed_triangles = self._triangulation.get_masked_triangles()
51:         node_used = (np.bincount(np.ravel(compressed_triangles),
52:                                  minlength=self._triangulation.x.size) != 0)
53:         return (1 / np.ptp(self._triangulation.x[node_used]),
54:                 1 / np.ptp(self._triangulation.y[node_used]))
55: 
56:     def circle_ratios(self, rescale=True):
57:         '''
58:         Returns a measure of the triangulation triangles flatness.
59: 
60:         The ratio of the incircle radius over the circumcircle radius is a
61:         widely used indicator of a triangle flatness.
62:         It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
63:         triangles. Circle ratios below 0.01 denote very flat triangles.
64: 
65:         To avoid unduly low values due to a difference of scale between the 2
66:         axis, the triangular mesh can first be rescaled to fit inside a unit
67:         square with :attr:`scale_factors` (Only if *rescale* is True, which is
68:         its default value).
69: 
70:         Parameters
71:         ----------
72:         rescale : boolean, optional
73:             If True, a rescaling will be internally performed (based on
74:             :attr:`scale_factors`, so that the (unmasked) triangles fit
75:             exactly inside a unit square mesh. Default is True.
76: 
77:         Returns
78:         -------
79:         circle_ratios : masked array
80:             Ratio of the incircle radius over the
81:             circumcircle radius, for each 'rescaled' triangle of the
82:             encapsulated triangulation.
83:             Values corresponding to masked triangles are masked out.
84: 
85:         '''
86:         # Coords rescaling
87:         if rescale:
88:             (kx, ky) = self.scale_factors
89:         else:
90:             (kx, ky) = (1.0, 1.0)
91:         pts = np.vstack([self._triangulation.x*kx,
92:                          self._triangulation.y*ky]).T
93:         tri_pts = pts[self._triangulation.triangles]
94:         # Computes the 3 side lengths
95:         a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
96:         b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
97:         c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
98:         a = np.sqrt(a[:, 0]**2 + a[:, 1]**2)
99:         b = np.sqrt(b[:, 0]**2 + b[:, 1]**2)
100:         c = np.sqrt(c[:, 0]**2 + c[:, 1]**2)
101:         # circumcircle and incircle radii
102:         s = (a+b+c)*0.5
103:         prod = s*(a+b-s)*(a+c-s)*(b+c-s)
104:         # We have to deal with flat triangles with infinite circum_radius
105:         bool_flat = (prod == 0.)
106:         if np.any(bool_flat):
107:             # Pathologic flow
108:             ntri = tri_pts.shape[0]
109:             circum_radius = np.empty(ntri, dtype=np.float64)
110:             circum_radius[bool_flat] = np.inf
111:             abc = a*b*c
112:             circum_radius[~bool_flat] = abc[~bool_flat] / (
113:                 4.0*np.sqrt(prod[~bool_flat]))
114:         else:
115:             # Normal optimized flow
116:             circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
117:         in_radius = (a*b*c) / (4.0*circum_radius*s)
118:         circle_ratio = in_radius/circum_radius
119:         mask = self._triangulation.mask
120:         if mask is None:
121:             return circle_ratio
122:         else:
123:             return np.ma.array(circle_ratio, mask=mask)
124: 
125:     def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
126:         '''
127:         Eliminates excessively flat border triangles from the triangulation.
128: 
129:         Returns a mask *new_mask* which allows to clean the encapsulated
130:         triangulation from its border-located flat triangles
131:         (according to their :meth:`circle_ratios`).
132:         This mask is meant to be subsequently applied to the triangulation
133:         using :func:`matplotlib.tri.Triangulation.set_mask` .
134:         *new_mask* is an extension of the initial triangulation mask
135:         in the sense that an initially masked triangle will remain masked.
136: 
137:         The *new_mask* array is computed recursively ; at each step flat
138:         triangles are removed only if they share a side with the current
139:         mesh border. Thus no new holes in the triangulated domain will be
140:         created.
141: 
142:         Parameters
143:         ----------
144:         min_circle_ratio : float, optional
145:             Border triangles with incircle/circumcircle radii ratio r/R will
146:             be removed if r/R < *min_circle_ratio*. Default value: 0.01
147:         rescale : boolean, optional
148:             If True, a rescaling will first be internally performed (based on
149:             :attr:`scale_factors` ), so that the (unmasked) triangles fit
150:             exactly inside a unit square mesh. This rescaling accounts for the
151:             difference of scale which might exist between the 2 axis. Default
152:             (and recommended) value is True.
153: 
154:         Returns
155:         -------
156:         new_mask : array-like of booleans
157:             Mask to apply to encapsulated triangulation.
158:             All the initially masked triangles remain masked in the
159:             *new_mask*.
160: 
161:         Notes
162:         -----
163:         The rationale behind this function is that a Delaunay
164:         triangulation - of an unstructured set of points - sometimes contains
165:         almost flat triangles at its border, leading to artifacts in plots
166:         (especially for high-resolution contouring).
167:         Masked with computed *new_mask*, the encapsulated
168:         triangulation would contain no more unmasked border triangles
169:         with a circle ratio below *min_circle_ratio*, thus improving the
170:         mesh quality for subsequent plots or interpolation.
171:         '''
172:         # Recursively computes the mask_current_borders, true if a triangle is
173:         # at the border of the mesh OR touching the border through a chain of
174:         # invalid aspect ratio masked_triangles.
175:         ntri = self._triangulation.triangles.shape[0]
176:         mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
177: 
178:         current_mask = self._triangulation.mask
179:         if current_mask is None:
180:             current_mask = np.zeros(ntri, dtype=bool)
181:         valid_neighbors = np.copy(self._triangulation.neighbors)
182:         renum_neighbors = np.arange(ntri, dtype=np.int32)
183:         nadd = -1
184:         while nadd != 0:
185:             # The active wavefront is the triangles from the border (unmasked
186:             # but with a least 1 neighbor equal to -1
187:             wavefront = ((np.min(valid_neighbors, axis=1) == -1)
188:                          & ~current_mask)
189:             # The element from the active wavefront will be masked if their
190:             # circle ratio is bad.
191:             added_mask = np.logical_and(wavefront, mask_bad_ratio)
192:             current_mask = (added_mask | current_mask)
193:             nadd = np.sum(added_mask)
194: 
195:             # now we have to update the tables valid_neighbors
196:             valid_neighbors[added_mask, :] = -1
197:             renum_neighbors[added_mask] = -1
198:             valid_neighbors = np.where(valid_neighbors == -1, -1,
199:                                        renum_neighbors[valid_neighbors])
200: 
201:         return np.ma.filled(current_mask, True)
202: 
203:     def _get_compressed_triangulation(self, return_tri_renum=False,
204:                                       return_node_renum=False):
205:         '''
206:         Compress (if masked) the encapsulated triangulation.
207: 
208:         Returns minimal-length triangles array (*compressed_triangles*) and
209:         coordinates arrays (*compressed_x*, *compressed_y*) that can still
210:         describe the unmasked triangles of the encapsulated triangulation.
211: 
212:         Parameters
213:         ----------
214:         return_tri_renum : boolean, optional
215:             Indicates whether a renumbering table to translate the triangle
216:             numbers from the encapsulated triangulation numbering into the
217:             new (compressed) renumbering will be returned.
218:         return_node_renum : boolean, optional
219:             Indicates whether a renumbering table to translate the nodes
220:             numbers from the encapsulated triangulation numbering into the
221:             new (compressed) renumbering will be returned.
222: 
223:         Returns
224:         -------
225:         compressed_triangles : array-like
226:             the returned compressed triangulation triangles
227:         compressed_x : array-like
228:             the returned compressed triangulation 1st coordinate
229:         compressed_y : array-like
230:             the returned compressed triangulation 2nd coordinate
231:         tri_renum : array-like of integers
232:             renumbering table to translate the triangle numbers from the
233:             encapsulated triangulation into the new (compressed) renumbering.
234:             -1 for masked triangles (deleted from *compressed_triangles*).
235:             Returned only if *return_tri_renum* is True.
236:         node_renum : array-like of integers
237:             renumbering table to translate the point numbers from the
238:             encapsulated triangulation into the new (compressed) renumbering.
239:             -1 for unused points (i.e. those deleted from *compressed_x* and
240:             *compressed_y*). Returned only if *return_node_renum* is True.
241: 
242:         '''
243:         # Valid triangles and renumbering
244:         tri_mask = self._triangulation.mask
245:         compressed_triangles = self._triangulation.get_masked_triangles()
246:         ntri = self._triangulation.triangles.shape[0]
247:         tri_renum = self._total_to_compress_renum(tri_mask, ntri)
248: 
249:         # Valid nodes and renumbering
250:         node_mask = (np.bincount(np.ravel(compressed_triangles),
251:                                  minlength=self._triangulation.x.size) == 0)
252:         compressed_x = self._triangulation.x[~node_mask]
253:         compressed_y = self._triangulation.y[~node_mask]
254:         node_renum = self._total_to_compress_renum(node_mask)
255: 
256:         # Now renumbering the valid triangles nodes
257:         compressed_triangles = node_renum[compressed_triangles]
258: 
259:         # 4 cases possible for return
260:         if not return_tri_renum:
261:             if not return_node_renum:
262:                 return compressed_triangles, compressed_x, compressed_y
263:             else:
264:                 return (compressed_triangles, compressed_x, compressed_y,
265:                         node_renum)
266:         else:
267:             if not return_node_renum:
268:                 return (compressed_triangles, compressed_x, compressed_y,
269:                         tri_renum)
270:             else:
271:                 return (compressed_triangles, compressed_x, compressed_y,
272:                         tri_renum, node_renum)
273: 
274:     @staticmethod
275:     def _total_to_compress_renum(mask, n=None):
276:         '''
277:         Parameters
278:         ----------
279:         mask : 1d boolean array or None
280:             mask
281:         n : integer
282:             length of the mask. Useful only id mask can be None
283: 
284:         Returns
285:         -------
286:         renum : integer array
287:             array so that (`valid_array` being a compressed array
288:             based on a `masked_array` with mask *mask*) :
289: 
290:                   - For all i such as mask[i] = False:
291:                     valid_array[renum[i]] = masked_array[i]
292:                   - For all i such as mask[i] = True:
293:                     renum[i] = -1 (invalid value)
294: 
295:         '''
296:         if n is None:
297:             n = np.size(mask)
298:         if mask is not None:
299:             renum = -np.ones(n, dtype=np.int32)  # Default num is -1
300:             valid = np.arange(n, dtype=np.int32).compress(~mask, axis=0)
301:             renum[valid] = np.arange(np.size(valid, 0), dtype=np.int32)
302:             return renum
303:         else:
304:             return np.arange(n, dtype=np.int32)
305: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_301757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nTools for triangular grids.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301758 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_301758) is not StypyTypeError):

    if (import_301758 != 'pyd_module'):
        __import__(import_301758)
        sys_modules_301759 = sys.modules[import_301758]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_301759.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_301758)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.tri import Triangulation' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301760 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri')

if (type(import_301760) is not StypyTypeError):

    if (import_301760 != 'pyd_module'):
        __import__(import_301760)
        sys_modules_301761 = sys.modules[import_301760]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri', sys_modules_301761.module_type_store, module_type_store, ['Triangulation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_301761, sys_modules_301761.module_type_store, module_type_store)
    else:
        from matplotlib.tri import Triangulation

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri', None, module_type_store, ['Triangulation'], [Triangulation])

else:
    # Assigning a type to the variable 'matplotlib.tri' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tri', import_301760)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301762 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_301762) is not StypyTypeError):

    if (import_301762 != 'pyd_module'):
        __import__(import_301762)
        sys_modules_301763 = sys.modules[import_301762]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_301763.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_301762)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

# Declaration of the 'TriAnalyzer' class

class TriAnalyzer(object, ):
    unicode_301764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'unicode', u'\n    Define basic tools for triangular mesh analysis and improvement.\n\n    A TriAnalizer encapsulates a :class:`~matplotlib.tri.Triangulation`\n    object and provides basic tools for mesh analysis and mesh improvement.\n\n    Parameters\n    ----------\n    triangulation : :class:`~matplotlib.tri.Triangulation` object\n        The encapsulated triangulation to analyze.\n\n    Attributes\n    ----------\n    `scale_factors`\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriAnalyzer.__init__', ['triangulation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['triangulation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'triangulation' (line 31)
        triangulation_301766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'triangulation', False)
        # Getting the type of 'Triangulation' (line 31)
        Triangulation_301767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'Triangulation', False)
        # Processing the call keyword arguments (line 31)
        kwargs_301768 = {}
        # Getting the type of 'isinstance' (line 31)
        isinstance_301765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 31)
        isinstance_call_result_301769 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), isinstance_301765, *[triangulation_301766, Triangulation_301767], **kwargs_301768)
        
        # Applying the 'not' unary operator (line 31)
        result_not__301770 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), 'not', isinstance_call_result_301769)
        
        # Testing the type of an if condition (line 31)
        if_condition_301771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_not__301770)
        # Assigning a type to the variable 'if_condition_301771' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_301771', if_condition_301771)
        # SSA begins for if statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 32)
        # Processing the call arguments (line 32)
        unicode_301773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'unicode', u'Expected a Triangulation object')
        # Processing the call keyword arguments (line 32)
        kwargs_301774 = {}
        # Getting the type of 'ValueError' (line 32)
        ValueError_301772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 32)
        ValueError_call_result_301775 = invoke(stypy.reporting.localization.Localization(__file__, 32, 18), ValueError_301772, *[unicode_301773], **kwargs_301774)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 32, 12), ValueError_call_result_301775, 'raise parameter', BaseException)
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'triangulation' (line 33)
        triangulation_301776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 'triangulation')
        # Getting the type of 'self' (line 33)
        self_301777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member '_triangulation' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_301777, '_triangulation', triangulation_301776)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def scale_factors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scale_factors'
        module_type_store = module_type_store.open_function_context('scale_factors', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_localization', localization)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_function_name', 'TriAnalyzer.scale_factors')
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_param_names_list', [])
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriAnalyzer.scale_factors.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriAnalyzer.scale_factors', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scale_factors', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scale_factors(...)' code ##################

        unicode_301778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'unicode', u'\n        Factors to rescale the triangulation into a unit square.\n\n        Returns *k*, tuple of 2 scale factors.\n\n        Returns\n        -------\n        k : tuple of 2 floats (kx, ky)\n            Tuple of floats that would rescale the triangulation :\n            ``[triangulation.x * kx, triangulation.y * ky]``\n            fits exactly inside a unit square.\n\n        ')
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to get_masked_triangles(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_301782 = {}
        # Getting the type of 'self' (line 50)
        self_301779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 50)
        _triangulation_301780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 31), self_301779, '_triangulation')
        # Obtaining the member 'get_masked_triangles' of a type (line 50)
        get_masked_triangles_301781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 31), _triangulation_301780, 'get_masked_triangles')
        # Calling get_masked_triangles(args, kwargs) (line 50)
        get_masked_triangles_call_result_301783 = invoke(stypy.reporting.localization.Localization(__file__, 50, 31), get_masked_triangles_301781, *[], **kwargs_301782)
        
        # Assigning a type to the variable 'compressed_triangles' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'compressed_triangles', get_masked_triangles_call_result_301783)
        
        # Assigning a Compare to a Name (line 51):
        
        # Assigning a Compare to a Name (line 51):
        
        
        # Call to bincount(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to ravel(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'compressed_triangles' (line 51)
        compressed_triangles_301788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'compressed_triangles', False)
        # Processing the call keyword arguments (line 51)
        kwargs_301789 = {}
        # Getting the type of 'np' (line 51)
        np_301786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'np', False)
        # Obtaining the member 'ravel' of a type (line 51)
        ravel_301787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 33), np_301786, 'ravel')
        # Calling ravel(args, kwargs) (line 51)
        ravel_call_result_301790 = invoke(stypy.reporting.localization.Localization(__file__, 51, 33), ravel_301787, *[compressed_triangles_301788], **kwargs_301789)
        
        # Processing the call keyword arguments (line 51)
        # Getting the type of 'self' (line 52)
        self_301791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 52)
        _triangulation_301792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), self_301791, '_triangulation')
        # Obtaining the member 'x' of a type (line 52)
        x_301793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), _triangulation_301792, 'x')
        # Obtaining the member 'size' of a type (line 52)
        size_301794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), x_301793, 'size')
        keyword_301795 = size_301794
        kwargs_301796 = {'minlength': keyword_301795}
        # Getting the type of 'np' (line 51)
        np_301784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'np', False)
        # Obtaining the member 'bincount' of a type (line 51)
        bincount_301785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 21), np_301784, 'bincount')
        # Calling bincount(args, kwargs) (line 51)
        bincount_call_result_301797 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), bincount_301785, *[ravel_call_result_301790], **kwargs_301796)
        
        int_301798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 74), 'int')
        # Applying the binary operator '!=' (line 51)
        result_ne_301799 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 21), '!=', bincount_call_result_301797, int_301798)
        
        # Assigning a type to the variable 'node_used' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'node_used', result_ne_301799)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_301800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        int_301801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'int')
        
        # Call to ptp(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining the type of the subscript
        # Getting the type of 'node_used' (line 53)
        node_used_301804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 49), 'node_used', False)
        # Getting the type of 'self' (line 53)
        self_301805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 53)
        _triangulation_301806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 27), self_301805, '_triangulation')
        # Obtaining the member 'x' of a type (line 53)
        x_301807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 27), _triangulation_301806, 'x')
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___301808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 27), x_301807, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_301809 = invoke(stypy.reporting.localization.Localization(__file__, 53, 27), getitem___301808, node_used_301804)
        
        # Processing the call keyword arguments (line 53)
        kwargs_301810 = {}
        # Getting the type of 'np' (line 53)
        np_301802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'np', False)
        # Obtaining the member 'ptp' of a type (line 53)
        ptp_301803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), np_301802, 'ptp')
        # Calling ptp(args, kwargs) (line 53)
        ptp_call_result_301811 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), ptp_301803, *[subscript_call_result_301809], **kwargs_301810)
        
        # Applying the binary operator 'div' (line 53)
        result_div_301812 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 16), 'div', int_301801, ptp_call_result_301811)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), tuple_301800, result_div_301812)
        # Adding element type (line 53)
        int_301813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'int')
        
        # Call to ptp(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining the type of the subscript
        # Getting the type of 'node_used' (line 54)
        node_used_301816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 49), 'node_used', False)
        # Getting the type of 'self' (line 54)
        self_301817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 54)
        _triangulation_301818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 27), self_301817, '_triangulation')
        # Obtaining the member 'y' of a type (line 54)
        y_301819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 27), _triangulation_301818, 'y')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___301820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 27), y_301819, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_301821 = invoke(stypy.reporting.localization.Localization(__file__, 54, 27), getitem___301820, node_used_301816)
        
        # Processing the call keyword arguments (line 54)
        kwargs_301822 = {}
        # Getting the type of 'np' (line 54)
        np_301814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'np', False)
        # Obtaining the member 'ptp' of a type (line 54)
        ptp_301815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), np_301814, 'ptp')
        # Calling ptp(args, kwargs) (line 54)
        ptp_call_result_301823 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), ptp_301815, *[subscript_call_result_301821], **kwargs_301822)
        
        # Applying the binary operator 'div' (line 54)
        result_div_301824 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 16), 'div', int_301813, ptp_call_result_301823)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), tuple_301800, result_div_301824)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', tuple_301800)
        
        # ################# End of 'scale_factors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scale_factors' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_301825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_301825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scale_factors'
        return stypy_return_type_301825


    @norecursion
    def circle_ratios(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 56)
        True_301826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'True')
        defaults = [True_301826]
        # Create a new context for function 'circle_ratios'
        module_type_store = module_type_store.open_function_context('circle_ratios', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_localization', localization)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_function_name', 'TriAnalyzer.circle_ratios')
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_param_names_list', ['rescale'])
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriAnalyzer.circle_ratios.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriAnalyzer.circle_ratios', ['rescale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'circle_ratios', localization, ['rescale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'circle_ratios(...)' code ##################

        unicode_301827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'unicode', u"\n        Returns a measure of the triangulation triangles flatness.\n\n        The ratio of the incircle radius over the circumcircle radius is a\n        widely used indicator of a triangle flatness.\n        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral\n        triangles. Circle ratios below 0.01 denote very flat triangles.\n\n        To avoid unduly low values due to a difference of scale between the 2\n        axis, the triangular mesh can first be rescaled to fit inside a unit\n        square with :attr:`scale_factors` (Only if *rescale* is True, which is\n        its default value).\n\n        Parameters\n        ----------\n        rescale : boolean, optional\n            If True, a rescaling will be internally performed (based on\n            :attr:`scale_factors`, so that the (unmasked) triangles fit\n            exactly inside a unit square mesh. Default is True.\n\n        Returns\n        -------\n        circle_ratios : masked array\n            Ratio of the incircle radius over the\n            circumcircle radius, for each 'rescaled' triangle of the\n            encapsulated triangulation.\n            Values corresponding to masked triangles are masked out.\n\n        ")
        
        # Getting the type of 'rescale' (line 87)
        rescale_301828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'rescale')
        # Testing the type of an if condition (line 87)
        if_condition_301829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), rescale_301828)
        # Assigning a type to the variable 'if_condition_301829' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_301829', if_condition_301829)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 88):
        
        # Assigning a Subscript to a Name (line 88):
        
        # Obtaining the type of the subscript
        int_301830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'int')
        # Getting the type of 'self' (line 88)
        self_301831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'self')
        # Obtaining the member 'scale_factors' of a type (line 88)
        scale_factors_301832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 23), self_301831, 'scale_factors')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___301833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), scale_factors_301832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_301834 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), getitem___301833, int_301830)
        
        # Assigning a type to the variable 'tuple_var_assignment_301753' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'tuple_var_assignment_301753', subscript_call_result_301834)
        
        # Assigning a Subscript to a Name (line 88):
        
        # Obtaining the type of the subscript
        int_301835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'int')
        # Getting the type of 'self' (line 88)
        self_301836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'self')
        # Obtaining the member 'scale_factors' of a type (line 88)
        scale_factors_301837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 23), self_301836, 'scale_factors')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___301838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), scale_factors_301837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_301839 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), getitem___301838, int_301835)
        
        # Assigning a type to the variable 'tuple_var_assignment_301754' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'tuple_var_assignment_301754', subscript_call_result_301839)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'tuple_var_assignment_301753' (line 88)
        tuple_var_assignment_301753_301840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'tuple_var_assignment_301753')
        # Assigning a type to the variable 'kx' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'kx', tuple_var_assignment_301753_301840)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'tuple_var_assignment_301754' (line 88)
        tuple_var_assignment_301754_301841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'tuple_var_assignment_301754')
        # Assigning a type to the variable 'ky' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'ky', tuple_var_assignment_301754_301841)
        # SSA branch for the else part of an if statement (line 87)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Tuple (line 90):
        
        # Assigning a Num to a Name (line 90):
        float_301842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'float')
        # Assigning a type to the variable 'tuple_assignment_301755' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_assignment_301755', float_301842)
        
        # Assigning a Num to a Name (line 90):
        float_301843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'float')
        # Assigning a type to the variable 'tuple_assignment_301756' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_assignment_301756', float_301843)
        
        # Assigning a Name to a Name (line 90):
        # Getting the type of 'tuple_assignment_301755' (line 90)
        tuple_assignment_301755_301844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_assignment_301755')
        # Assigning a type to the variable 'kx' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'kx', tuple_assignment_301755_301844)
        
        # Assigning a Name to a Name (line 90):
        # Getting the type of 'tuple_assignment_301756' (line 90)
        tuple_assignment_301756_301845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_assignment_301756')
        # Assigning a type to the variable 'ky' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'ky', tuple_assignment_301756_301845)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 91):
        
        # Assigning a Attribute to a Name (line 91):
        
        # Call to vstack(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_301848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        # Getting the type of 'self' (line 91)
        self_301849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 91)
        _triangulation_301850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), self_301849, '_triangulation')
        # Obtaining the member 'x' of a type (line 91)
        x_301851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), _triangulation_301850, 'x')
        # Getting the type of 'kx' (line 91)
        kx_301852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'kx', False)
        # Applying the binary operator '*' (line 91)
        result_mul_301853 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 25), '*', x_301851, kx_301852)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 24), list_301848, result_mul_301853)
        # Adding element type (line 91)
        # Getting the type of 'self' (line 92)
        self_301854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 92)
        _triangulation_301855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), self_301854, '_triangulation')
        # Obtaining the member 'y' of a type (line 92)
        y_301856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), _triangulation_301855, 'y')
        # Getting the type of 'ky' (line 92)
        ky_301857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'ky', False)
        # Applying the binary operator '*' (line 92)
        result_mul_301858 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 25), '*', y_301856, ky_301857)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 24), list_301848, result_mul_301858)
        
        # Processing the call keyword arguments (line 91)
        kwargs_301859 = {}
        # Getting the type of 'np' (line 91)
        np_301846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'np', False)
        # Obtaining the member 'vstack' of a type (line 91)
        vstack_301847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 14), np_301846, 'vstack')
        # Calling vstack(args, kwargs) (line 91)
        vstack_call_result_301860 = invoke(stypy.reporting.localization.Localization(__file__, 91, 14), vstack_301847, *[list_301848], **kwargs_301859)
        
        # Obtaining the member 'T' of a type (line 91)
        T_301861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 14), vstack_call_result_301860, 'T')
        # Assigning a type to the variable 'pts' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'pts', T_301861)
        
        # Assigning a Subscript to a Name (line 93):
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 93)
        self_301862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'self')
        # Obtaining the member '_triangulation' of a type (line 93)
        _triangulation_301863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 22), self_301862, '_triangulation')
        # Obtaining the member 'triangles' of a type (line 93)
        triangles_301864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 22), _triangulation_301863, 'triangles')
        # Getting the type of 'pts' (line 93)
        pts_301865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'pts')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___301866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), pts_301865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_301867 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), getitem___301866, triangles_301864)
        
        # Assigning a type to the variable 'tri_pts' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tri_pts', subscript_call_result_301867)
        
        # Assigning a BinOp to a Name (line 95):
        
        # Assigning a BinOp to a Name (line 95):
        
        # Obtaining the type of the subscript
        slice_301868 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 12), None, None, None)
        int_301869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'int')
        slice_301870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 12), None, None, None)
        # Getting the type of 'tri_pts' (line 95)
        tri_pts_301871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'tri_pts')
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___301872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), tri_pts_301871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_301873 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), getitem___301872, (slice_301868, int_301869, slice_301870))
        
        
        # Obtaining the type of the subscript
        slice_301874 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 31), None, None, None)
        int_301875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 42), 'int')
        slice_301876 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 31), None, None, None)
        # Getting the type of 'tri_pts' (line 95)
        tri_pts_301877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'tri_pts')
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___301878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 31), tri_pts_301877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_301879 = invoke(stypy.reporting.localization.Localization(__file__, 95, 31), getitem___301878, (slice_301874, int_301875, slice_301876))
        
        # Applying the binary operator '-' (line 95)
        result_sub_301880 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '-', subscript_call_result_301873, subscript_call_result_301879)
        
        # Assigning a type to the variable 'a' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'a', result_sub_301880)
        
        # Assigning a BinOp to a Name (line 96):
        
        # Assigning a BinOp to a Name (line 96):
        
        # Obtaining the type of the subscript
        slice_301881 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 12), None, None, None)
        int_301882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'int')
        slice_301883 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 12), None, None, None)
        # Getting the type of 'tri_pts' (line 96)
        tri_pts_301884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'tri_pts')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___301885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), tri_pts_301884, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_301886 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), getitem___301885, (slice_301881, int_301882, slice_301883))
        
        
        # Obtaining the type of the subscript
        slice_301887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 31), None, None, None)
        int_301888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'int')
        slice_301889 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 31), None, None, None)
        # Getting the type of 'tri_pts' (line 96)
        tri_pts_301890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'tri_pts')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___301891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 31), tri_pts_301890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_301892 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), getitem___301891, (slice_301887, int_301888, slice_301889))
        
        # Applying the binary operator '-' (line 96)
        result_sub_301893 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 12), '-', subscript_call_result_301886, subscript_call_result_301892)
        
        # Assigning a type to the variable 'b' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'b', result_sub_301893)
        
        # Assigning a BinOp to a Name (line 97):
        
        # Assigning a BinOp to a Name (line 97):
        
        # Obtaining the type of the subscript
        slice_301894 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 12), None, None, None)
        int_301895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'int')
        slice_301896 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 12), None, None, None)
        # Getting the type of 'tri_pts' (line 97)
        tri_pts_301897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'tri_pts')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___301898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), tri_pts_301897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_301899 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), getitem___301898, (slice_301894, int_301895, slice_301896))
        
        
        # Obtaining the type of the subscript
        slice_301900 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 31), None, None, None)
        int_301901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'int')
        slice_301902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 31), None, None, None)
        # Getting the type of 'tri_pts' (line 97)
        tri_pts_301903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'tri_pts')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___301904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), tri_pts_301903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_301905 = invoke(stypy.reporting.localization.Localization(__file__, 97, 31), getitem___301904, (slice_301900, int_301901, slice_301902))
        
        # Applying the binary operator '-' (line 97)
        result_sub_301906 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 12), '-', subscript_call_result_301899, subscript_call_result_301905)
        
        # Assigning a type to the variable 'c' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'c', result_sub_301906)
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to sqrt(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining the type of the subscript
        slice_301909 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 20), None, None, None)
        int_301910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 25), 'int')
        # Getting the type of 'a' (line 98)
        a_301911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___301912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), a_301911, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_301913 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), getitem___301912, (slice_301909, int_301910))
        
        int_301914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'int')
        # Applying the binary operator '**' (line 98)
        result_pow_301915 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 20), '**', subscript_call_result_301913, int_301914)
        
        
        # Obtaining the type of the subscript
        slice_301916 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 33), None, None, None)
        int_301917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 38), 'int')
        # Getting the type of 'a' (line 98)
        a_301918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___301919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 33), a_301918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_301920 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), getitem___301919, (slice_301916, int_301917))
        
        int_301921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'int')
        # Applying the binary operator '**' (line 98)
        result_pow_301922 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 33), '**', subscript_call_result_301920, int_301921)
        
        # Applying the binary operator '+' (line 98)
        result_add_301923 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 20), '+', result_pow_301915, result_pow_301922)
        
        # Processing the call keyword arguments (line 98)
        kwargs_301924 = {}
        # Getting the type of 'np' (line 98)
        np_301907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 98)
        sqrt_301908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), np_301907, 'sqrt')
        # Calling sqrt(args, kwargs) (line 98)
        sqrt_call_result_301925 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), sqrt_301908, *[result_add_301923], **kwargs_301924)
        
        # Assigning a type to the variable 'a' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'a', sqrt_call_result_301925)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to sqrt(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining the type of the subscript
        slice_301928 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 20), None, None, None)
        int_301929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'int')
        # Getting the type of 'b' (line 99)
        b_301930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___301931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 20), b_301930, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 99)
        subscript_call_result_301932 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), getitem___301931, (slice_301928, int_301929))
        
        int_301933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'int')
        # Applying the binary operator '**' (line 99)
        result_pow_301934 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 20), '**', subscript_call_result_301932, int_301933)
        
        
        # Obtaining the type of the subscript
        slice_301935 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 33), None, None, None)
        int_301936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 38), 'int')
        # Getting the type of 'b' (line 99)
        b_301937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'b', False)
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___301938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 33), b_301937, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 99)
        subscript_call_result_301939 = invoke(stypy.reporting.localization.Localization(__file__, 99, 33), getitem___301938, (slice_301935, int_301936))
        
        int_301940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'int')
        # Applying the binary operator '**' (line 99)
        result_pow_301941 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 33), '**', subscript_call_result_301939, int_301940)
        
        # Applying the binary operator '+' (line 99)
        result_add_301942 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 20), '+', result_pow_301934, result_pow_301941)
        
        # Processing the call keyword arguments (line 99)
        kwargs_301943 = {}
        # Getting the type of 'np' (line 99)
        np_301926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 99)
        sqrt_301927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), np_301926, 'sqrt')
        # Calling sqrt(args, kwargs) (line 99)
        sqrt_call_result_301944 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), sqrt_301927, *[result_add_301942], **kwargs_301943)
        
        # Assigning a type to the variable 'b' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'b', sqrt_call_result_301944)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to sqrt(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining the type of the subscript
        slice_301947 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 100, 20), None, None, None)
        int_301948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'int')
        # Getting the type of 'c' (line 100)
        c_301949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___301950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), c_301949, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_301951 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), getitem___301950, (slice_301947, int_301948))
        
        int_301952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'int')
        # Applying the binary operator '**' (line 100)
        result_pow_301953 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 20), '**', subscript_call_result_301951, int_301952)
        
        
        # Obtaining the type of the subscript
        slice_301954 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 100, 33), None, None, None)
        int_301955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 38), 'int')
        # Getting the type of 'c' (line 100)
        c_301956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___301957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 33), c_301956, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_301958 = invoke(stypy.reporting.localization.Localization(__file__, 100, 33), getitem___301957, (slice_301954, int_301955))
        
        int_301959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 42), 'int')
        # Applying the binary operator '**' (line 100)
        result_pow_301960 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 33), '**', subscript_call_result_301958, int_301959)
        
        # Applying the binary operator '+' (line 100)
        result_add_301961 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 20), '+', result_pow_301953, result_pow_301960)
        
        # Processing the call keyword arguments (line 100)
        kwargs_301962 = {}
        # Getting the type of 'np' (line 100)
        np_301945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 100)
        sqrt_301946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), np_301945, 'sqrt')
        # Calling sqrt(args, kwargs) (line 100)
        sqrt_call_result_301963 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), sqrt_301946, *[result_add_301961], **kwargs_301962)
        
        # Assigning a type to the variable 'c' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'c', sqrt_call_result_301963)
        
        # Assigning a BinOp to a Name (line 102):
        
        # Assigning a BinOp to a Name (line 102):
        # Getting the type of 'a' (line 102)
        a_301964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'a')
        # Getting the type of 'b' (line 102)
        b_301965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'b')
        # Applying the binary operator '+' (line 102)
        result_add_301966 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 13), '+', a_301964, b_301965)
        
        # Getting the type of 'c' (line 102)
        c_301967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'c')
        # Applying the binary operator '+' (line 102)
        result_add_301968 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+', result_add_301966, c_301967)
        
        float_301969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'float')
        # Applying the binary operator '*' (line 102)
        result_mul_301970 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 12), '*', result_add_301968, float_301969)
        
        # Assigning a type to the variable 's' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 's', result_mul_301970)
        
        # Assigning a BinOp to a Name (line 103):
        
        # Assigning a BinOp to a Name (line 103):
        # Getting the type of 's' (line 103)
        s_301971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 's')
        # Getting the type of 'a' (line 103)
        a_301972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'a')
        # Getting the type of 'b' (line 103)
        b_301973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'b')
        # Applying the binary operator '+' (line 103)
        result_add_301974 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), '+', a_301972, b_301973)
        
        # Getting the type of 's' (line 103)
        s_301975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 's')
        # Applying the binary operator '-' (line 103)
        result_sub_301976 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 21), '-', result_add_301974, s_301975)
        
        # Applying the binary operator '*' (line 103)
        result_mul_301977 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '*', s_301971, result_sub_301976)
        
        # Getting the type of 'a' (line 103)
        a_301978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'a')
        # Getting the type of 'c' (line 103)
        c_301979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'c')
        # Applying the binary operator '+' (line 103)
        result_add_301980 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 26), '+', a_301978, c_301979)
        
        # Getting the type of 's' (line 103)
        s_301981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 's')
        # Applying the binary operator '-' (line 103)
        result_sub_301982 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 29), '-', result_add_301980, s_301981)
        
        # Applying the binary operator '*' (line 103)
        result_mul_301983 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 24), '*', result_mul_301977, result_sub_301982)
        
        # Getting the type of 'b' (line 103)
        b_301984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'b')
        # Getting the type of 'c' (line 103)
        c_301985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'c')
        # Applying the binary operator '+' (line 103)
        result_add_301986 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 34), '+', b_301984, c_301985)
        
        # Getting the type of 's' (line 103)
        s_301987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 's')
        # Applying the binary operator '-' (line 103)
        result_sub_301988 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 37), '-', result_add_301986, s_301987)
        
        # Applying the binary operator '*' (line 103)
        result_mul_301989 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 32), '*', result_mul_301983, result_sub_301988)
        
        # Assigning a type to the variable 'prod' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'prod', result_mul_301989)
        
        # Assigning a Compare to a Name (line 105):
        
        # Assigning a Compare to a Name (line 105):
        
        # Getting the type of 'prod' (line 105)
        prod_301990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'prod')
        float_301991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'float')
        # Applying the binary operator '==' (line 105)
        result_eq_301992 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 21), '==', prod_301990, float_301991)
        
        # Assigning a type to the variable 'bool_flat' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'bool_flat', result_eq_301992)
        
        
        # Call to any(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'bool_flat' (line 106)
        bool_flat_301995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'bool_flat', False)
        # Processing the call keyword arguments (line 106)
        kwargs_301996 = {}
        # Getting the type of 'np' (line 106)
        np_301993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 106)
        any_301994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), np_301993, 'any')
        # Calling any(args, kwargs) (line 106)
        any_call_result_301997 = invoke(stypy.reporting.localization.Localization(__file__, 106, 11), any_301994, *[bool_flat_301995], **kwargs_301996)
        
        # Testing the type of an if condition (line 106)
        if_condition_301998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), any_call_result_301997)
        # Assigning a type to the variable 'if_condition_301998' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_301998', if_condition_301998)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 108):
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_301999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 33), 'int')
        # Getting the type of 'tri_pts' (line 108)
        tri_pts_302000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'tri_pts')
        # Obtaining the member 'shape' of a type (line 108)
        shape_302001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), tri_pts_302000, 'shape')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___302002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), shape_302001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_302003 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), getitem___302002, int_301999)
        
        # Assigning a type to the variable 'ntri' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'ntri', subscript_call_result_302003)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to empty(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'ntri' (line 109)
        ntri_302006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'ntri', False)
        # Processing the call keyword arguments (line 109)
        # Getting the type of 'np' (line 109)
        np_302007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'np', False)
        # Obtaining the member 'float64' of a type (line 109)
        float64_302008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 49), np_302007, 'float64')
        keyword_302009 = float64_302008
        kwargs_302010 = {'dtype': keyword_302009}
        # Getting the type of 'np' (line 109)
        np_302004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'np', False)
        # Obtaining the member 'empty' of a type (line 109)
        empty_302005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 28), np_302004, 'empty')
        # Calling empty(args, kwargs) (line 109)
        empty_call_result_302011 = invoke(stypy.reporting.localization.Localization(__file__, 109, 28), empty_302005, *[ntri_302006], **kwargs_302010)
        
        # Assigning a type to the variable 'circum_radius' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'circum_radius', empty_call_result_302011)
        
        # Assigning a Attribute to a Subscript (line 110):
        
        # Assigning a Attribute to a Subscript (line 110):
        # Getting the type of 'np' (line 110)
        np_302012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'np')
        # Obtaining the member 'inf' of a type (line 110)
        inf_302013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 39), np_302012, 'inf')
        # Getting the type of 'circum_radius' (line 110)
        circum_radius_302014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'circum_radius')
        # Getting the type of 'bool_flat' (line 110)
        bool_flat_302015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'bool_flat')
        # Storing an element on a container (line 110)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), circum_radius_302014, (bool_flat_302015, inf_302013))
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        # Getting the type of 'a' (line 111)
        a_302016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'a')
        # Getting the type of 'b' (line 111)
        b_302017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'b')
        # Applying the binary operator '*' (line 111)
        result_mul_302018 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 18), '*', a_302016, b_302017)
        
        # Getting the type of 'c' (line 111)
        c_302019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'c')
        # Applying the binary operator '*' (line 111)
        result_mul_302020 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 21), '*', result_mul_302018, c_302019)
        
        # Assigning a type to the variable 'abc' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'abc', result_mul_302020)
        
        # Assigning a BinOp to a Subscript (line 112):
        
        # Assigning a BinOp to a Subscript (line 112):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'bool_flat' (line 112)
        bool_flat_302021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'bool_flat')
        # Applying the '~' unary operator (line 112)
        result_inv_302022 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 44), '~', bool_flat_302021)
        
        # Getting the type of 'abc' (line 112)
        abc_302023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'abc')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___302024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 40), abc_302023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_302025 = invoke(stypy.reporting.localization.Localization(__file__, 112, 40), getitem___302024, result_inv_302022)
        
        float_302026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'float')
        
        # Call to sqrt(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'bool_flat' (line 113)
        bool_flat_302029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'bool_flat', False)
        # Applying the '~' unary operator (line 113)
        result_inv_302030 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 33), '~', bool_flat_302029)
        
        # Getting the type of 'prod' (line 113)
        prod_302031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'prod', False)
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___302032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), prod_302031, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_302033 = invoke(stypy.reporting.localization.Localization(__file__, 113, 28), getitem___302032, result_inv_302030)
        
        # Processing the call keyword arguments (line 113)
        kwargs_302034 = {}
        # Getting the type of 'np' (line 113)
        np_302027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 113)
        sqrt_302028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), np_302027, 'sqrt')
        # Calling sqrt(args, kwargs) (line 113)
        sqrt_call_result_302035 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), sqrt_302028, *[subscript_call_result_302033], **kwargs_302034)
        
        # Applying the binary operator '*' (line 113)
        result_mul_302036 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 16), '*', float_302026, sqrt_call_result_302035)
        
        # Applying the binary operator 'div' (line 112)
        result_div_302037 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 40), 'div', subscript_call_result_302025, result_mul_302036)
        
        # Getting the type of 'circum_radius' (line 112)
        circum_radius_302038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'circum_radius')
        
        # Getting the type of 'bool_flat' (line 112)
        bool_flat_302039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'bool_flat')
        # Applying the '~' unary operator (line 112)
        result_inv_302040 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 26), '~', bool_flat_302039)
        
        # Storing an element on a container (line 112)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 12), circum_radius_302038, (result_inv_302040, result_div_302037))
        # SSA branch for the else part of an if statement (line 106)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 116):
        
        # Assigning a BinOp to a Name (line 116):
        # Getting the type of 'a' (line 116)
        a_302041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'a')
        # Getting the type of 'b' (line 116)
        b_302042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'b')
        # Applying the binary operator '*' (line 116)
        result_mul_302043 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 29), '*', a_302041, b_302042)
        
        # Getting the type of 'c' (line 116)
        c_302044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'c')
        # Applying the binary operator '*' (line 116)
        result_mul_302045 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 32), '*', result_mul_302043, c_302044)
        
        float_302046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 39), 'float')
        
        # Call to sqrt(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'prod' (line 116)
        prod_302049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 51), 'prod', False)
        # Processing the call keyword arguments (line 116)
        kwargs_302050 = {}
        # Getting the type of 'np' (line 116)
        np_302047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 43), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 116)
        sqrt_302048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 43), np_302047, 'sqrt')
        # Calling sqrt(args, kwargs) (line 116)
        sqrt_call_result_302051 = invoke(stypy.reporting.localization.Localization(__file__, 116, 43), sqrt_302048, *[prod_302049], **kwargs_302050)
        
        # Applying the binary operator '*' (line 116)
        result_mul_302052 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 39), '*', float_302046, sqrt_call_result_302051)
        
        # Applying the binary operator 'div' (line 116)
        result_div_302053 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 28), 'div', result_mul_302045, result_mul_302052)
        
        # Assigning a type to the variable 'circum_radius' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'circum_radius', result_div_302053)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 117):
        
        # Assigning a BinOp to a Name (line 117):
        # Getting the type of 'a' (line 117)
        a_302054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'a')
        # Getting the type of 'b' (line 117)
        b_302055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'b')
        # Applying the binary operator '*' (line 117)
        result_mul_302056 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 21), '*', a_302054, b_302055)
        
        # Getting the type of 'c' (line 117)
        c_302057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'c')
        # Applying the binary operator '*' (line 117)
        result_mul_302058 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 24), '*', result_mul_302056, c_302057)
        
        float_302059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'float')
        # Getting the type of 'circum_radius' (line 117)
        circum_radius_302060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 35), 'circum_radius')
        # Applying the binary operator '*' (line 117)
        result_mul_302061 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 31), '*', float_302059, circum_radius_302060)
        
        # Getting the type of 's' (line 117)
        s_302062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 49), 's')
        # Applying the binary operator '*' (line 117)
        result_mul_302063 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 48), '*', result_mul_302061, s_302062)
        
        # Applying the binary operator 'div' (line 117)
        result_div_302064 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 20), 'div', result_mul_302058, result_mul_302063)
        
        # Assigning a type to the variable 'in_radius' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'in_radius', result_div_302064)
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        # Getting the type of 'in_radius' (line 118)
        in_radius_302065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'in_radius')
        # Getting the type of 'circum_radius' (line 118)
        circum_radius_302066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'circum_radius')
        # Applying the binary operator 'div' (line 118)
        result_div_302067 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 23), 'div', in_radius_302065, circum_radius_302066)
        
        # Assigning a type to the variable 'circle_ratio' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'circle_ratio', result_div_302067)
        
        # Assigning a Attribute to a Name (line 119):
        
        # Assigning a Attribute to a Name (line 119):
        # Getting the type of 'self' (line 119)
        self_302068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'self')
        # Obtaining the member '_triangulation' of a type (line 119)
        _triangulation_302069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), self_302068, '_triangulation')
        # Obtaining the member 'mask' of a type (line 119)
        mask_302070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), _triangulation_302069, 'mask')
        # Assigning a type to the variable 'mask' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'mask', mask_302070)
        
        # Type idiom detected: calculating its left and rigth part (line 120)
        # Getting the type of 'mask' (line 120)
        mask_302071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'mask')
        # Getting the type of 'None' (line 120)
        None_302072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'None')
        
        (may_be_302073, more_types_in_union_302074) = may_be_none(mask_302071, None_302072)

        if may_be_302073:

            if more_types_in_union_302074:
                # Runtime conditional SSA (line 120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'circle_ratio' (line 121)
            circle_ratio_302075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'circle_ratio')
            # Assigning a type to the variable 'stypy_return_type' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'stypy_return_type', circle_ratio_302075)

            if more_types_in_union_302074:
                # Runtime conditional SSA for else branch (line 120)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_302073) or more_types_in_union_302074):
            
            # Call to array(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'circle_ratio' (line 123)
            circle_ratio_302079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'circle_ratio', False)
            # Processing the call keyword arguments (line 123)
            # Getting the type of 'mask' (line 123)
            mask_302080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 50), 'mask', False)
            keyword_302081 = mask_302080
            kwargs_302082 = {'mask': keyword_302081}
            # Getting the type of 'np' (line 123)
            np_302076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'np', False)
            # Obtaining the member 'ma' of a type (line 123)
            ma_302077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 19), np_302076, 'ma')
            # Obtaining the member 'array' of a type (line 123)
            array_302078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 19), ma_302077, 'array')
            # Calling array(args, kwargs) (line 123)
            array_call_result_302083 = invoke(stypy.reporting.localization.Localization(__file__, 123, 19), array_302078, *[circle_ratio_302079], **kwargs_302082)
            
            # Assigning a type to the variable 'stypy_return_type' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'stypy_return_type', array_call_result_302083)

            if (may_be_302073 and more_types_in_union_302074):
                # SSA join for if statement (line 120)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'circle_ratios(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'circle_ratios' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_302084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'circle_ratios'
        return stypy_return_type_302084


    @norecursion
    def get_flat_tri_mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_302085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 49), 'float')
        # Getting the type of 'True' (line 125)
        True_302086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 63), 'True')
        defaults = [float_302085, True_302086]
        # Create a new context for function 'get_flat_tri_mask'
        module_type_store = module_type_store.open_function_context('get_flat_tri_mask', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_localization', localization)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_function_name', 'TriAnalyzer.get_flat_tri_mask')
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_param_names_list', ['min_circle_ratio', 'rescale'])
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriAnalyzer.get_flat_tri_mask.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriAnalyzer.get_flat_tri_mask', ['min_circle_ratio', 'rescale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flat_tri_mask', localization, ['min_circle_ratio', 'rescale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flat_tri_mask(...)' code ##################

        unicode_302087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'unicode', u'\n        Eliminates excessively flat border triangles from the triangulation.\n\n        Returns a mask *new_mask* which allows to clean the encapsulated\n        triangulation from its border-located flat triangles\n        (according to their :meth:`circle_ratios`).\n        This mask is meant to be subsequently applied to the triangulation\n        using :func:`matplotlib.tri.Triangulation.set_mask` .\n        *new_mask* is an extension of the initial triangulation mask\n        in the sense that an initially masked triangle will remain masked.\n\n        The *new_mask* array is computed recursively ; at each step flat\n        triangles are removed only if they share a side with the current\n        mesh border. Thus no new holes in the triangulated domain will be\n        created.\n\n        Parameters\n        ----------\n        min_circle_ratio : float, optional\n            Border triangles with incircle/circumcircle radii ratio r/R will\n            be removed if r/R < *min_circle_ratio*. Default value: 0.01\n        rescale : boolean, optional\n            If True, a rescaling will first be internally performed (based on\n            :attr:`scale_factors` ), so that the (unmasked) triangles fit\n            exactly inside a unit square mesh. This rescaling accounts for the\n            difference of scale which might exist between the 2 axis. Default\n            (and recommended) value is True.\n\n        Returns\n        -------\n        new_mask : array-like of booleans\n            Mask to apply to encapsulated triangulation.\n            All the initially masked triangles remain masked in the\n            *new_mask*.\n\n        Notes\n        -----\n        The rationale behind this function is that a Delaunay\n        triangulation - of an unstructured set of points - sometimes contains\n        almost flat triangles at its border, leading to artifacts in plots\n        (especially for high-resolution contouring).\n        Masked with computed *new_mask*, the encapsulated\n        triangulation would contain no more unmasked border triangles\n        with a circle ratio below *min_circle_ratio*, thus improving the\n        mesh quality for subsequent plots or interpolation.\n        ')
        
        # Assigning a Subscript to a Name (line 175):
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        int_302088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 51), 'int')
        # Getting the type of 'self' (line 175)
        self_302089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'self')
        # Obtaining the member '_triangulation' of a type (line 175)
        _triangulation_302090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), self_302089, '_triangulation')
        # Obtaining the member 'triangles' of a type (line 175)
        triangles_302091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), _triangulation_302090, 'triangles')
        # Obtaining the member 'shape' of a type (line 175)
        shape_302092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), triangles_302091, 'shape')
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___302093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), shape_302092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_302094 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), getitem___302093, int_302088)
        
        # Assigning a type to the variable 'ntri' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ntri', subscript_call_result_302094)
        
        # Assigning a Compare to a Name (line 176):
        
        # Assigning a Compare to a Name (line 176):
        
        
        # Call to circle_ratios(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'rescale' (line 176)
        rescale_302097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'rescale', False)
        # Processing the call keyword arguments (line 176)
        kwargs_302098 = {}
        # Getting the type of 'self' (line 176)
        self_302095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'self', False)
        # Obtaining the member 'circle_ratios' of a type (line 176)
        circle_ratios_302096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), self_302095, 'circle_ratios')
        # Calling circle_ratios(args, kwargs) (line 176)
        circle_ratios_call_result_302099 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), circle_ratios_302096, *[rescale_302097], **kwargs_302098)
        
        # Getting the type of 'min_circle_ratio' (line 176)
        min_circle_ratio_302100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 55), 'min_circle_ratio')
        # Applying the binary operator '<' (line 176)
        result_lt_302101 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 25), '<', circle_ratios_call_result_302099, min_circle_ratio_302100)
        
        # Assigning a type to the variable 'mask_bad_ratio' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'mask_bad_ratio', result_lt_302101)
        
        # Assigning a Attribute to a Name (line 178):
        
        # Assigning a Attribute to a Name (line 178):
        # Getting the type of 'self' (line 178)
        self_302102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'self')
        # Obtaining the member '_triangulation' of a type (line 178)
        _triangulation_302103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 23), self_302102, '_triangulation')
        # Obtaining the member 'mask' of a type (line 178)
        mask_302104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 23), _triangulation_302103, 'mask')
        # Assigning a type to the variable 'current_mask' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'current_mask', mask_302104)
        
        # Type idiom detected: calculating its left and rigth part (line 179)
        # Getting the type of 'current_mask' (line 179)
        current_mask_302105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'current_mask')
        # Getting the type of 'None' (line 179)
        None_302106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'None')
        
        (may_be_302107, more_types_in_union_302108) = may_be_none(current_mask_302105, None_302106)

        if may_be_302107:

            if more_types_in_union_302108:
                # Runtime conditional SSA (line 179)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 180):
            
            # Assigning a Call to a Name (line 180):
            
            # Call to zeros(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'ntri' (line 180)
            ntri_302111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 36), 'ntri', False)
            # Processing the call keyword arguments (line 180)
            # Getting the type of 'bool' (line 180)
            bool_302112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 48), 'bool', False)
            keyword_302113 = bool_302112
            kwargs_302114 = {'dtype': keyword_302113}
            # Getting the type of 'np' (line 180)
            np_302109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'np', False)
            # Obtaining the member 'zeros' of a type (line 180)
            zeros_302110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), np_302109, 'zeros')
            # Calling zeros(args, kwargs) (line 180)
            zeros_call_result_302115 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), zeros_302110, *[ntri_302111], **kwargs_302114)
            
            # Assigning a type to the variable 'current_mask' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'current_mask', zeros_call_result_302115)

            if more_types_in_union_302108:
                # SSA join for if statement (line 179)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to copy(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'self' (line 181)
        self_302118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 34), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 181)
        _triangulation_302119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 34), self_302118, '_triangulation')
        # Obtaining the member 'neighbors' of a type (line 181)
        neighbors_302120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 34), _triangulation_302119, 'neighbors')
        # Processing the call keyword arguments (line 181)
        kwargs_302121 = {}
        # Getting the type of 'np' (line 181)
        np_302116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'np', False)
        # Obtaining the member 'copy' of a type (line 181)
        copy_302117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 26), np_302116, 'copy')
        # Calling copy(args, kwargs) (line 181)
        copy_call_result_302122 = invoke(stypy.reporting.localization.Localization(__file__, 181, 26), copy_302117, *[neighbors_302120], **kwargs_302121)
        
        # Assigning a type to the variable 'valid_neighbors' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'valid_neighbors', copy_call_result_302122)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to arange(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'ntri' (line 182)
        ntri_302125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 36), 'ntri', False)
        # Processing the call keyword arguments (line 182)
        # Getting the type of 'np' (line 182)
        np_302126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 48), 'np', False)
        # Obtaining the member 'int32' of a type (line 182)
        int32_302127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 48), np_302126, 'int32')
        keyword_302128 = int32_302127
        kwargs_302129 = {'dtype': keyword_302128}
        # Getting the type of 'np' (line 182)
        np_302123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'np', False)
        # Obtaining the member 'arange' of a type (line 182)
        arange_302124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 26), np_302123, 'arange')
        # Calling arange(args, kwargs) (line 182)
        arange_call_result_302130 = invoke(stypy.reporting.localization.Localization(__file__, 182, 26), arange_302124, *[ntri_302125], **kwargs_302129)
        
        # Assigning a type to the variable 'renum_neighbors' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'renum_neighbors', arange_call_result_302130)
        
        # Assigning a Num to a Name (line 183):
        
        # Assigning a Num to a Name (line 183):
        int_302131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'int')
        # Assigning a type to the variable 'nadd' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'nadd', int_302131)
        
        
        # Getting the type of 'nadd' (line 184)
        nadd_302132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'nadd')
        int_302133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 22), 'int')
        # Applying the binary operator '!=' (line 184)
        result_ne_302134 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 14), '!=', nadd_302132, int_302133)
        
        # Testing the type of an if condition (line 184)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_ne_302134)
        # SSA begins for while statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 187):
        
        # Assigning a BinOp to a Name (line 187):
        
        
        # Call to min(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'valid_neighbors' (line 187)
        valid_neighbors_302137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'valid_neighbors', False)
        # Processing the call keyword arguments (line 187)
        int_302138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 55), 'int')
        keyword_302139 = int_302138
        kwargs_302140 = {'axis': keyword_302139}
        # Getting the type of 'np' (line 187)
        np_302135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'np', False)
        # Obtaining the member 'min' of a type (line 187)
        min_302136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 26), np_302135, 'min')
        # Calling min(args, kwargs) (line 187)
        min_call_result_302141 = invoke(stypy.reporting.localization.Localization(__file__, 187, 26), min_302136, *[valid_neighbors_302137], **kwargs_302140)
        
        int_302142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 61), 'int')
        # Applying the binary operator '==' (line 187)
        result_eq_302143 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 26), '==', min_call_result_302141, int_302142)
        
        
        # Getting the type of 'current_mask' (line 188)
        current_mask_302144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'current_mask')
        # Applying the '~' unary operator (line 188)
        result_inv_302145 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 27), '~', current_mask_302144)
        
        # Applying the binary operator '&' (line 187)
        result_and__302146 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 25), '&', result_eq_302143, result_inv_302145)
        
        # Assigning a type to the variable 'wavefront' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'wavefront', result_and__302146)
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to logical_and(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'wavefront' (line 191)
        wavefront_302149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 40), 'wavefront', False)
        # Getting the type of 'mask_bad_ratio' (line 191)
        mask_bad_ratio_302150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 51), 'mask_bad_ratio', False)
        # Processing the call keyword arguments (line 191)
        kwargs_302151 = {}
        # Getting the type of 'np' (line 191)
        np_302147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'np', False)
        # Obtaining the member 'logical_and' of a type (line 191)
        logical_and_302148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), np_302147, 'logical_and')
        # Calling logical_and(args, kwargs) (line 191)
        logical_and_call_result_302152 = invoke(stypy.reporting.localization.Localization(__file__, 191, 25), logical_and_302148, *[wavefront_302149, mask_bad_ratio_302150], **kwargs_302151)
        
        # Assigning a type to the variable 'added_mask' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'added_mask', logical_and_call_result_302152)
        
        # Assigning a BinOp to a Name (line 192):
        
        # Assigning a BinOp to a Name (line 192):
        # Getting the type of 'added_mask' (line 192)
        added_mask_302153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'added_mask')
        # Getting the type of 'current_mask' (line 192)
        current_mask_302154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 41), 'current_mask')
        # Applying the binary operator '|' (line 192)
        result_or__302155 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 28), '|', added_mask_302153, current_mask_302154)
        
        # Assigning a type to the variable 'current_mask' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'current_mask', result_or__302155)
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to sum(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'added_mask' (line 193)
        added_mask_302158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'added_mask', False)
        # Processing the call keyword arguments (line 193)
        kwargs_302159 = {}
        # Getting the type of 'np' (line 193)
        np_302156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'np', False)
        # Obtaining the member 'sum' of a type (line 193)
        sum_302157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 19), np_302156, 'sum')
        # Calling sum(args, kwargs) (line 193)
        sum_call_result_302160 = invoke(stypy.reporting.localization.Localization(__file__, 193, 19), sum_302157, *[added_mask_302158], **kwargs_302159)
        
        # Assigning a type to the variable 'nadd' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'nadd', sum_call_result_302160)
        
        # Assigning a Num to a Subscript (line 196):
        
        # Assigning a Num to a Subscript (line 196):
        int_302161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 45), 'int')
        # Getting the type of 'valid_neighbors' (line 196)
        valid_neighbors_302162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'valid_neighbors')
        # Getting the type of 'added_mask' (line 196)
        added_mask_302163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'added_mask')
        slice_302164 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 196, 12), None, None, None)
        # Storing an element on a container (line 196)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 12), valid_neighbors_302162, ((added_mask_302163, slice_302164), int_302161))
        
        # Assigning a Num to a Subscript (line 197):
        
        # Assigning a Num to a Subscript (line 197):
        int_302165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 42), 'int')
        # Getting the type of 'renum_neighbors' (line 197)
        renum_neighbors_302166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'renum_neighbors')
        # Getting the type of 'added_mask' (line 197)
        added_mask_302167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'added_mask')
        # Storing an element on a container (line 197)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), renum_neighbors_302166, (added_mask_302167, int_302165))
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to where(...): (line 198)
        # Processing the call arguments (line 198)
        
        # Getting the type of 'valid_neighbors' (line 198)
        valid_neighbors_302170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 39), 'valid_neighbors', False)
        int_302171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 58), 'int')
        # Applying the binary operator '==' (line 198)
        result_eq_302172 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 39), '==', valid_neighbors_302170, int_302171)
        
        int_302173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 62), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'valid_neighbors' (line 199)
        valid_neighbors_302174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 55), 'valid_neighbors', False)
        # Getting the type of 'renum_neighbors' (line 199)
        renum_neighbors_302175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'renum_neighbors', False)
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___302176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), renum_neighbors_302175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_302177 = invoke(stypy.reporting.localization.Localization(__file__, 199, 39), getitem___302176, valid_neighbors_302174)
        
        # Processing the call keyword arguments (line 198)
        kwargs_302178 = {}
        # Getting the type of 'np' (line 198)
        np_302168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'np', False)
        # Obtaining the member 'where' of a type (line 198)
        where_302169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 30), np_302168, 'where')
        # Calling where(args, kwargs) (line 198)
        where_call_result_302179 = invoke(stypy.reporting.localization.Localization(__file__, 198, 30), where_302169, *[result_eq_302172, int_302173, subscript_call_result_302177], **kwargs_302178)
        
        # Assigning a type to the variable 'valid_neighbors' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'valid_neighbors', where_call_result_302179)
        # SSA join for while statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to filled(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'current_mask' (line 201)
        current_mask_302183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 28), 'current_mask', False)
        # Getting the type of 'True' (line 201)
        True_302184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 42), 'True', False)
        # Processing the call keyword arguments (line 201)
        kwargs_302185 = {}
        # Getting the type of 'np' (line 201)
        np_302180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'np', False)
        # Obtaining the member 'ma' of a type (line 201)
        ma_302181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), np_302180, 'ma')
        # Obtaining the member 'filled' of a type (line 201)
        filled_302182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), ma_302181, 'filled')
        # Calling filled(args, kwargs) (line 201)
        filled_call_result_302186 = invoke(stypy.reporting.localization.Localization(__file__, 201, 15), filled_302182, *[current_mask_302183, True_302184], **kwargs_302185)
        
        # Assigning a type to the variable 'stypy_return_type' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', filled_call_result_302186)
        
        # ################# End of 'get_flat_tri_mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flat_tri_mask' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_302187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flat_tri_mask'
        return stypy_return_type_302187


    @norecursion
    def _get_compressed_triangulation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 203)
        False_302188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 61), 'False')
        # Getting the type of 'False' (line 204)
        False_302189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 56), 'False')
        defaults = [False_302188, False_302189]
        # Create a new context for function '_get_compressed_triangulation'
        module_type_store = module_type_store.open_function_context('_get_compressed_triangulation', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_localization', localization)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_function_name', 'TriAnalyzer._get_compressed_triangulation')
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_param_names_list', ['return_tri_renum', 'return_node_renum'])
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriAnalyzer._get_compressed_triangulation.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriAnalyzer._get_compressed_triangulation', ['return_tri_renum', 'return_node_renum'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_compressed_triangulation', localization, ['return_tri_renum', 'return_node_renum'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_compressed_triangulation(...)' code ##################

        unicode_302190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, (-1)), 'unicode', u'\n        Compress (if masked) the encapsulated triangulation.\n\n        Returns minimal-length triangles array (*compressed_triangles*) and\n        coordinates arrays (*compressed_x*, *compressed_y*) that can still\n        describe the unmasked triangles of the encapsulated triangulation.\n\n        Parameters\n        ----------\n        return_tri_renum : boolean, optional\n            Indicates whether a renumbering table to translate the triangle\n            numbers from the encapsulated triangulation numbering into the\n            new (compressed) renumbering will be returned.\n        return_node_renum : boolean, optional\n            Indicates whether a renumbering table to translate the nodes\n            numbers from the encapsulated triangulation numbering into the\n            new (compressed) renumbering will be returned.\n\n        Returns\n        -------\n        compressed_triangles : array-like\n            the returned compressed triangulation triangles\n        compressed_x : array-like\n            the returned compressed triangulation 1st coordinate\n        compressed_y : array-like\n            the returned compressed triangulation 2nd coordinate\n        tri_renum : array-like of integers\n            renumbering table to translate the triangle numbers from the\n            encapsulated triangulation into the new (compressed) renumbering.\n            -1 for masked triangles (deleted from *compressed_triangles*).\n            Returned only if *return_tri_renum* is True.\n        node_renum : array-like of integers\n            renumbering table to translate the point numbers from the\n            encapsulated triangulation into the new (compressed) renumbering.\n            -1 for unused points (i.e. those deleted from *compressed_x* and\n            *compressed_y*). Returned only if *return_node_renum* is True.\n\n        ')
        
        # Assigning a Attribute to a Name (line 244):
        
        # Assigning a Attribute to a Name (line 244):
        # Getting the type of 'self' (line 244)
        self_302191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'self')
        # Obtaining the member '_triangulation' of a type (line 244)
        _triangulation_302192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), self_302191, '_triangulation')
        # Obtaining the member 'mask' of a type (line 244)
        mask_302193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), _triangulation_302192, 'mask')
        # Assigning a type to the variable 'tri_mask' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'tri_mask', mask_302193)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to get_masked_triangles(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_302197 = {}
        # Getting the type of 'self' (line 245)
        self_302194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 31), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 245)
        _triangulation_302195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 31), self_302194, '_triangulation')
        # Obtaining the member 'get_masked_triangles' of a type (line 245)
        get_masked_triangles_302196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 31), _triangulation_302195, 'get_masked_triangles')
        # Calling get_masked_triangles(args, kwargs) (line 245)
        get_masked_triangles_call_result_302198 = invoke(stypy.reporting.localization.Localization(__file__, 245, 31), get_masked_triangles_302196, *[], **kwargs_302197)
        
        # Assigning a type to the variable 'compressed_triangles' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'compressed_triangles', get_masked_triangles_call_result_302198)
        
        # Assigning a Subscript to a Name (line 246):
        
        # Assigning a Subscript to a Name (line 246):
        
        # Obtaining the type of the subscript
        int_302199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 51), 'int')
        # Getting the type of 'self' (line 246)
        self_302200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'self')
        # Obtaining the member '_triangulation' of a type (line 246)
        _triangulation_302201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), self_302200, '_triangulation')
        # Obtaining the member 'triangles' of a type (line 246)
        triangles_302202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), _triangulation_302201, 'triangles')
        # Obtaining the member 'shape' of a type (line 246)
        shape_302203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), triangles_302202, 'shape')
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___302204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), shape_302203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_302205 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), getitem___302204, int_302199)
        
        # Assigning a type to the variable 'ntri' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'ntri', subscript_call_result_302205)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to _total_to_compress_renum(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'tri_mask' (line 247)
        tri_mask_302208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 50), 'tri_mask', False)
        # Getting the type of 'ntri' (line 247)
        ntri_302209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 60), 'ntri', False)
        # Processing the call keyword arguments (line 247)
        kwargs_302210 = {}
        # Getting the type of 'self' (line 247)
        self_302206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'self', False)
        # Obtaining the member '_total_to_compress_renum' of a type (line 247)
        _total_to_compress_renum_302207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), self_302206, '_total_to_compress_renum')
        # Calling _total_to_compress_renum(args, kwargs) (line 247)
        _total_to_compress_renum_call_result_302211 = invoke(stypy.reporting.localization.Localization(__file__, 247, 20), _total_to_compress_renum_302207, *[tri_mask_302208, ntri_302209], **kwargs_302210)
        
        # Assigning a type to the variable 'tri_renum' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tri_renum', _total_to_compress_renum_call_result_302211)
        
        # Assigning a Compare to a Name (line 250):
        
        # Assigning a Compare to a Name (line 250):
        
        
        # Call to bincount(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to ravel(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'compressed_triangles' (line 250)
        compressed_triangles_302216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 42), 'compressed_triangles', False)
        # Processing the call keyword arguments (line 250)
        kwargs_302217 = {}
        # Getting the type of 'np' (line 250)
        np_302214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 33), 'np', False)
        # Obtaining the member 'ravel' of a type (line 250)
        ravel_302215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 33), np_302214, 'ravel')
        # Calling ravel(args, kwargs) (line 250)
        ravel_call_result_302218 = invoke(stypy.reporting.localization.Localization(__file__, 250, 33), ravel_302215, *[compressed_triangles_302216], **kwargs_302217)
        
        # Processing the call keyword arguments (line 250)
        # Getting the type of 'self' (line 251)
        self_302219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 43), 'self', False)
        # Obtaining the member '_triangulation' of a type (line 251)
        _triangulation_302220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), self_302219, '_triangulation')
        # Obtaining the member 'x' of a type (line 251)
        x_302221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), _triangulation_302220, 'x')
        # Obtaining the member 'size' of a type (line 251)
        size_302222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), x_302221, 'size')
        keyword_302223 = size_302222
        kwargs_302224 = {'minlength': keyword_302223}
        # Getting the type of 'np' (line 250)
        np_302212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 21), 'np', False)
        # Obtaining the member 'bincount' of a type (line 250)
        bincount_302213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 21), np_302212, 'bincount')
        # Calling bincount(args, kwargs) (line 250)
        bincount_call_result_302225 = invoke(stypy.reporting.localization.Localization(__file__, 250, 21), bincount_302213, *[ravel_call_result_302218], **kwargs_302224)
        
        int_302226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 74), 'int')
        # Applying the binary operator '==' (line 250)
        result_eq_302227 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 21), '==', bincount_call_result_302225, int_302226)
        
        # Assigning a type to the variable 'node_mask' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'node_mask', result_eq_302227)
        
        # Assigning a Subscript to a Name (line 252):
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'node_mask' (line 252)
        node_mask_302228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 46), 'node_mask')
        # Applying the '~' unary operator (line 252)
        result_inv_302229 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 45), '~', node_mask_302228)
        
        # Getting the type of 'self' (line 252)
        self_302230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'self')
        # Obtaining the member '_triangulation' of a type (line 252)
        _triangulation_302231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 23), self_302230, '_triangulation')
        # Obtaining the member 'x' of a type (line 252)
        x_302232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 23), _triangulation_302231, 'x')
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___302233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 23), x_302232, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_302234 = invoke(stypy.reporting.localization.Localization(__file__, 252, 23), getitem___302233, result_inv_302229)
        
        # Assigning a type to the variable 'compressed_x' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'compressed_x', subscript_call_result_302234)
        
        # Assigning a Subscript to a Name (line 253):
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'node_mask' (line 253)
        node_mask_302235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 46), 'node_mask')
        # Applying the '~' unary operator (line 253)
        result_inv_302236 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 45), '~', node_mask_302235)
        
        # Getting the type of 'self' (line 253)
        self_302237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 'self')
        # Obtaining the member '_triangulation' of a type (line 253)
        _triangulation_302238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 23), self_302237, '_triangulation')
        # Obtaining the member 'y' of a type (line 253)
        y_302239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 23), _triangulation_302238, 'y')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___302240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 23), y_302239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_302241 = invoke(stypy.reporting.localization.Localization(__file__, 253, 23), getitem___302240, result_inv_302236)
        
        # Assigning a type to the variable 'compressed_y' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'compressed_y', subscript_call_result_302241)
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to _total_to_compress_renum(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'node_mask' (line 254)
        node_mask_302244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 51), 'node_mask', False)
        # Processing the call keyword arguments (line 254)
        kwargs_302245 = {}
        # Getting the type of 'self' (line 254)
        self_302242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'self', False)
        # Obtaining the member '_total_to_compress_renum' of a type (line 254)
        _total_to_compress_renum_302243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 21), self_302242, '_total_to_compress_renum')
        # Calling _total_to_compress_renum(args, kwargs) (line 254)
        _total_to_compress_renum_call_result_302246 = invoke(stypy.reporting.localization.Localization(__file__, 254, 21), _total_to_compress_renum_302243, *[node_mask_302244], **kwargs_302245)
        
        # Assigning a type to the variable 'node_renum' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'node_renum', _total_to_compress_renum_call_result_302246)
        
        # Assigning a Subscript to a Name (line 257):
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        # Getting the type of 'compressed_triangles' (line 257)
        compressed_triangles_302247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 42), 'compressed_triangles')
        # Getting the type of 'node_renum' (line 257)
        node_renum_302248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 31), 'node_renum')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___302249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 31), node_renum_302248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_302250 = invoke(stypy.reporting.localization.Localization(__file__, 257, 31), getitem___302249, compressed_triangles_302247)
        
        # Assigning a type to the variable 'compressed_triangles' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'compressed_triangles', subscript_call_result_302250)
        
        
        # Getting the type of 'return_tri_renum' (line 260)
        return_tri_renum_302251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'return_tri_renum')
        # Applying the 'not' unary operator (line 260)
        result_not__302252 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), 'not', return_tri_renum_302251)
        
        # Testing the type of an if condition (line 260)
        if_condition_302253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), result_not__302252)
        # Assigning a type to the variable 'if_condition_302253' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_302253', if_condition_302253)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'return_node_renum' (line 261)
        return_node_renum_302254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'return_node_renum')
        # Applying the 'not' unary operator (line 261)
        result_not__302255 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 15), 'not', return_node_renum_302254)
        
        # Testing the type of an if condition (line 261)
        if_condition_302256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 12), result_not__302255)
        # Assigning a type to the variable 'if_condition_302256' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'if_condition_302256', if_condition_302256)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 262)
        tuple_302257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 262)
        # Adding element type (line 262)
        # Getting the type of 'compressed_triangles' (line 262)
        compressed_triangles_302258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'compressed_triangles')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 23), tuple_302257, compressed_triangles_302258)
        # Adding element type (line 262)
        # Getting the type of 'compressed_x' (line 262)
        compressed_x_302259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 45), 'compressed_x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 23), tuple_302257, compressed_x_302259)
        # Adding element type (line 262)
        # Getting the type of 'compressed_y' (line 262)
        compressed_y_302260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 59), 'compressed_y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 23), tuple_302257, compressed_y_302260)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'stypy_return_type', tuple_302257)
        # SSA branch for the else part of an if statement (line 261)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 264)
        tuple_302261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 264)
        # Adding element type (line 264)
        # Getting the type of 'compressed_triangles' (line 264)
        compressed_triangles_302262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'compressed_triangles')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_302261, compressed_triangles_302262)
        # Adding element type (line 264)
        # Getting the type of 'compressed_x' (line 264)
        compressed_x_302263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 46), 'compressed_x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_302261, compressed_x_302263)
        # Adding element type (line 264)
        # Getting the type of 'compressed_y' (line 264)
        compressed_y_302264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 60), 'compressed_y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_302261, compressed_y_302264)
        # Adding element type (line 264)
        # Getting the type of 'node_renum' (line 265)
        node_renum_302265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'node_renum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 24), tuple_302261, node_renum_302265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'stypy_return_type', tuple_302261)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 260)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'return_node_renum' (line 267)
        return_node_renum_302266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'return_node_renum')
        # Applying the 'not' unary operator (line 267)
        result_not__302267 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 15), 'not', return_node_renum_302266)
        
        # Testing the type of an if condition (line 267)
        if_condition_302268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 12), result_not__302267)
        # Assigning a type to the variable 'if_condition_302268' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'if_condition_302268', if_condition_302268)
        # SSA begins for if statement (line 267)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 268)
        tuple_302269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 268)
        # Adding element type (line 268)
        # Getting the type of 'compressed_triangles' (line 268)
        compressed_triangles_302270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'compressed_triangles')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), tuple_302269, compressed_triangles_302270)
        # Adding element type (line 268)
        # Getting the type of 'compressed_x' (line 268)
        compressed_x_302271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 46), 'compressed_x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), tuple_302269, compressed_x_302271)
        # Adding element type (line 268)
        # Getting the type of 'compressed_y' (line 268)
        compressed_y_302272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 60), 'compressed_y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), tuple_302269, compressed_y_302272)
        # Adding element type (line 268)
        # Getting the type of 'tri_renum' (line 269)
        tri_renum_302273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'tri_renum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), tuple_302269, tri_renum_302273)
        
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'stypy_return_type', tuple_302269)
        # SSA branch for the else part of an if statement (line 267)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 271)
        tuple_302274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 271)
        # Adding element type (line 271)
        # Getting the type of 'compressed_triangles' (line 271)
        compressed_triangles_302275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'compressed_triangles')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_302274, compressed_triangles_302275)
        # Adding element type (line 271)
        # Getting the type of 'compressed_x' (line 271)
        compressed_x_302276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 46), 'compressed_x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_302274, compressed_x_302276)
        # Adding element type (line 271)
        # Getting the type of 'compressed_y' (line 271)
        compressed_y_302277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 60), 'compressed_y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_302274, compressed_y_302277)
        # Adding element type (line 271)
        # Getting the type of 'tri_renum' (line 272)
        tri_renum_302278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'tri_renum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_302274, tri_renum_302278)
        # Adding element type (line 271)
        # Getting the type of 'node_renum' (line 272)
        node_renum_302279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'node_renum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_302274, node_renum_302279)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'stypy_return_type', tuple_302274)
        # SSA join for if statement (line 267)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_get_compressed_triangulation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_compressed_triangulation' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_302280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_compressed_triangulation'
        return stypy_return_type_302280


    @staticmethod
    @norecursion
    def _total_to_compress_renum(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 275)
        None_302281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 41), 'None')
        defaults = [None_302281]
        # Create a new context for function '_total_to_compress_renum'
        module_type_store = module_type_store.open_function_context('_total_to_compress_renum', 274, 4, False)
        
        # Passed parameters checking function
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_localization', localization)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_type_of_self', None)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_function_name', '_total_to_compress_renum')
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_param_names_list', ['mask', 'n'])
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriAnalyzer._total_to_compress_renum.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '_total_to_compress_renum', ['mask', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_total_to_compress_renum', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_total_to_compress_renum(...)' code ##################

        unicode_302282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, (-1)), 'unicode', u'\n        Parameters\n        ----------\n        mask : 1d boolean array or None\n            mask\n        n : integer\n            length of the mask. Useful only id mask can be None\n\n        Returns\n        -------\n        renum : integer array\n            array so that (`valid_array` being a compressed array\n            based on a `masked_array` with mask *mask*) :\n\n                  - For all i such as mask[i] = False:\n                    valid_array[renum[i]] = masked_array[i]\n                  - For all i such as mask[i] = True:\n                    renum[i] = -1 (invalid value)\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 296)
        # Getting the type of 'n' (line 296)
        n_302283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'n')
        # Getting the type of 'None' (line 296)
        None_302284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'None')
        
        (may_be_302285, more_types_in_union_302286) = may_be_none(n_302283, None_302284)

        if may_be_302285:

            if more_types_in_union_302286:
                # Runtime conditional SSA (line 296)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 297):
            
            # Assigning a Call to a Name (line 297):
            
            # Call to size(...): (line 297)
            # Processing the call arguments (line 297)
            # Getting the type of 'mask' (line 297)
            mask_302289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'mask', False)
            # Processing the call keyword arguments (line 297)
            kwargs_302290 = {}
            # Getting the type of 'np' (line 297)
            np_302287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'np', False)
            # Obtaining the member 'size' of a type (line 297)
            size_302288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), np_302287, 'size')
            # Calling size(args, kwargs) (line 297)
            size_call_result_302291 = invoke(stypy.reporting.localization.Localization(__file__, 297, 16), size_302288, *[mask_302289], **kwargs_302290)
            
            # Assigning a type to the variable 'n' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'n', size_call_result_302291)

            if more_types_in_union_302286:
                # SSA join for if statement (line 296)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 298)
        # Getting the type of 'mask' (line 298)
        mask_302292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'mask')
        # Getting the type of 'None' (line 298)
        None_302293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'None')
        
        (may_be_302294, more_types_in_union_302295) = may_not_be_none(mask_302292, None_302293)

        if may_be_302294:

            if more_types_in_union_302295:
                # Runtime conditional SSA (line 298)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a UnaryOp to a Name (line 299):
            
            # Assigning a UnaryOp to a Name (line 299):
            
            
            # Call to ones(...): (line 299)
            # Processing the call arguments (line 299)
            # Getting the type of 'n' (line 299)
            n_302298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'n', False)
            # Processing the call keyword arguments (line 299)
            # Getting the type of 'np' (line 299)
            np_302299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 38), 'np', False)
            # Obtaining the member 'int32' of a type (line 299)
            int32_302300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 38), np_302299, 'int32')
            keyword_302301 = int32_302300
            kwargs_302302 = {'dtype': keyword_302301}
            # Getting the type of 'np' (line 299)
            np_302296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 21), 'np', False)
            # Obtaining the member 'ones' of a type (line 299)
            ones_302297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 21), np_302296, 'ones')
            # Calling ones(args, kwargs) (line 299)
            ones_call_result_302303 = invoke(stypy.reporting.localization.Localization(__file__, 299, 21), ones_302297, *[n_302298], **kwargs_302302)
            
            # Applying the 'usub' unary operator (line 299)
            result___neg___302304 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 20), 'usub', ones_call_result_302303)
            
            # Assigning a type to the variable 'renum' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'renum', result___neg___302304)
            
            # Assigning a Call to a Name (line 300):
            
            # Assigning a Call to a Name (line 300):
            
            # Call to compress(...): (line 300)
            # Processing the call arguments (line 300)
            
            # Getting the type of 'mask' (line 300)
            mask_302314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 59), 'mask', False)
            # Applying the '~' unary operator (line 300)
            result_inv_302315 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 58), '~', mask_302314)
            
            # Processing the call keyword arguments (line 300)
            int_302316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 70), 'int')
            keyword_302317 = int_302316
            kwargs_302318 = {'axis': keyword_302317}
            
            # Call to arange(...): (line 300)
            # Processing the call arguments (line 300)
            # Getting the type of 'n' (line 300)
            n_302307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 30), 'n', False)
            # Processing the call keyword arguments (line 300)
            # Getting the type of 'np' (line 300)
            np_302308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 39), 'np', False)
            # Obtaining the member 'int32' of a type (line 300)
            int32_302309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 39), np_302308, 'int32')
            keyword_302310 = int32_302309
            kwargs_302311 = {'dtype': keyword_302310}
            # Getting the type of 'np' (line 300)
            np_302305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'np', False)
            # Obtaining the member 'arange' of a type (line 300)
            arange_302306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), np_302305, 'arange')
            # Calling arange(args, kwargs) (line 300)
            arange_call_result_302312 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), arange_302306, *[n_302307], **kwargs_302311)
            
            # Obtaining the member 'compress' of a type (line 300)
            compress_302313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), arange_call_result_302312, 'compress')
            # Calling compress(args, kwargs) (line 300)
            compress_call_result_302319 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), compress_302313, *[result_inv_302315], **kwargs_302318)
            
            # Assigning a type to the variable 'valid' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'valid', compress_call_result_302319)
            
            # Assigning a Call to a Subscript (line 301):
            
            # Assigning a Call to a Subscript (line 301):
            
            # Call to arange(...): (line 301)
            # Processing the call arguments (line 301)
            
            # Call to size(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 'valid' (line 301)
            valid_302324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 45), 'valid', False)
            int_302325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 52), 'int')
            # Processing the call keyword arguments (line 301)
            kwargs_302326 = {}
            # Getting the type of 'np' (line 301)
            np_302322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 37), 'np', False)
            # Obtaining the member 'size' of a type (line 301)
            size_302323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 37), np_302322, 'size')
            # Calling size(args, kwargs) (line 301)
            size_call_result_302327 = invoke(stypy.reporting.localization.Localization(__file__, 301, 37), size_302323, *[valid_302324, int_302325], **kwargs_302326)
            
            # Processing the call keyword arguments (line 301)
            # Getting the type of 'np' (line 301)
            np_302328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 62), 'np', False)
            # Obtaining the member 'int32' of a type (line 301)
            int32_302329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 62), np_302328, 'int32')
            keyword_302330 = int32_302329
            kwargs_302331 = {'dtype': keyword_302330}
            # Getting the type of 'np' (line 301)
            np_302320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 'np', False)
            # Obtaining the member 'arange' of a type (line 301)
            arange_302321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 27), np_302320, 'arange')
            # Calling arange(args, kwargs) (line 301)
            arange_call_result_302332 = invoke(stypy.reporting.localization.Localization(__file__, 301, 27), arange_302321, *[size_call_result_302327], **kwargs_302331)
            
            # Getting the type of 'renum' (line 301)
            renum_302333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'renum')
            # Getting the type of 'valid' (line 301)
            valid_302334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 18), 'valid')
            # Storing an element on a container (line 301)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), renum_302333, (valid_302334, arange_call_result_302332))
            # Getting the type of 'renum' (line 302)
            renum_302335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'renum')
            # Assigning a type to the variable 'stypy_return_type' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'stypy_return_type', renum_302335)

            if more_types_in_union_302295:
                # Runtime conditional SSA for else branch (line 298)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_302294) or more_types_in_union_302295):
            
            # Call to arange(...): (line 304)
            # Processing the call arguments (line 304)
            # Getting the type of 'n' (line 304)
            n_302338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'n', False)
            # Processing the call keyword arguments (line 304)
            # Getting the type of 'np' (line 304)
            np_302339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'np', False)
            # Obtaining the member 'int32' of a type (line 304)
            int32_302340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 38), np_302339, 'int32')
            keyword_302341 = int32_302340
            kwargs_302342 = {'dtype': keyword_302341}
            # Getting the type of 'np' (line 304)
            np_302336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'np', False)
            # Obtaining the member 'arange' of a type (line 304)
            arange_302337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 19), np_302336, 'arange')
            # Calling arange(args, kwargs) (line 304)
            arange_call_result_302343 = invoke(stypy.reporting.localization.Localization(__file__, 304, 19), arange_302337, *[n_302338], **kwargs_302342)
            
            # Assigning a type to the variable 'stypy_return_type' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'stypy_return_type', arange_call_result_302343)

            if (may_be_302294 and more_types_in_union_302295):
                # SSA join for if statement (line 298)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_total_to_compress_renum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_total_to_compress_renum' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_302344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302344)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_total_to_compress_renum'
        return stypy_return_type_302344


# Assigning a type to the variable 'TriAnalyzer' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TriAnalyzer', TriAnalyzer)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

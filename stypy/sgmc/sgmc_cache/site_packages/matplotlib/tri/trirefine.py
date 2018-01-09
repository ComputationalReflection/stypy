
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Mesh refinement for triangular grids.
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: 
9: import numpy as np
10: from matplotlib.tri.triangulation import Triangulation
11: import matplotlib.tri.triinterpolate
12: 
13: 
14: class TriRefiner(object):
15:     '''
16:     Abstract base class for classes implementing mesh refinement.
17: 
18:     A TriRefiner encapsulates a Triangulation object and provides tools for
19:     mesh refinement and interpolation.
20: 
21:     Derived classes must implements:
22: 
23:         - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
24:           the optional keyword arguments *kwargs* are defined in each
25:           TriRefiner concrete implementation, and which returns :
26: 
27:               - a refined triangulation
28:               - optionally (depending on *return_tri_index*), for each
29:                 point of the refined triangulation: the index of
30:                 the initial triangulation triangle to which it belongs.
31: 
32:         - ``refine_field(z, triinterpolator=None, **kwargs)`` , where:
33: 
34:               - *z* array of field values (to refine) defined at the base
35:                 triangulation nodes
36:               - *triinterpolator* is a
37:                 :class:`~matplotlib.tri.TriInterpolator` (optional)
38:               - the other optional keyword arguments *kwargs* are defined in
39:                 each TriRefiner concrete implementation
40: 
41:           and which returns (as a tuple) a refined triangular mesh and the
42:           interpolated values of the field at the refined triangulation nodes.
43: 
44:     '''
45:     def __init__(self, triangulation):
46:         if not isinstance(triangulation, Triangulation):
47:             raise ValueError("Expected a Triangulation object")
48:         self._triangulation = triangulation
49: 
50: 
51: class UniformTriRefiner(TriRefiner):
52:     '''
53:     Uniform mesh refinement by recursive subdivisions.
54: 
55:     Parameters
56:     ----------
57:     triangulation : :class:`~matplotlib.tri.Triangulation`
58:                      The encapsulated triangulation (to be refined)
59:     '''
60: #    See Also
61: #    --------
62: #    :class:`~matplotlib.tri.CubicTriInterpolator` and
63: #    :class:`~matplotlib.tri.TriAnalyzer`.
64: #    '''
65:     def __init__(self, triangulation):
66:         TriRefiner.__init__(self, triangulation)
67: 
68:     def refine_triangulation(self, return_tri_index=False, subdiv=3):
69:         '''
70:         Computes an uniformly refined triangulation *refi_triangulation* of
71:         the encapsulated :attr:`triangulation`.
72: 
73:         This function refines the encapsulated triangulation by splitting each
74:         father triangle into 4 child sub-triangles built on the edges midside
75:         nodes, recursively (level of recursion *subdiv*).
76:         In the end, each triangle is hence divided into ``4**subdiv``
77:         child triangles.
78:         The default value for *subdiv* is 3 resulting in 64 refined
79:         subtriangles for each triangle of the initial triangulation.
80: 
81:         Parameters
82:         ----------
83:         return_tri_index : boolean, optional
84:             Boolean indicating whether an index table indicating the father
85:             triangle index of each point will be returned. Default value
86:             False.
87:         subdiv : integer, optional
88:             Recursion level for the subdivision. Defaults value 3.
89:             Each triangle will be divided into ``4**subdiv`` child triangles.
90: 
91:         Returns
92:         -------
93:         refi_triangulation : :class:`~matplotlib.tri.Triangulation`
94:             The returned refined triangulation
95:         found_index : array-like of integers
96:             Index of the initial triangulation containing triangle, for each
97:             point of *refi_triangulation*.
98:             Returned only if *return_tri_index* is set to True.
99: 
100:         '''
101:         refi_triangulation = self._triangulation
102:         ntri = refi_triangulation.triangles.shape[0]
103: 
104:         # Computes the triangulation ancestors numbers in the reference
105:         # triangulation.
106:         ancestors = np.arange(ntri, dtype=np.int32)
107:         for _ in range(subdiv):
108:             refi_triangulation, ancestors = self._refine_triangulation_once(
109:                 refi_triangulation, ancestors)
110:         refi_npts = refi_triangulation.x.shape[0]
111:         refi_triangles = refi_triangulation.triangles
112: 
113:         # Now we compute found_index table if needed
114:         if return_tri_index:
115:             # We have to initialize found_index with -1 because some nodes
116:             # may very well belong to no triangle at all, e.g., in case of
117:             # Delaunay Triangulation with DuplicatePointWarning.
118:             found_index = - np.ones(refi_npts, dtype=np.int32)
119:             tri_mask = self._triangulation.mask
120:             if tri_mask is None:
121:                 found_index[refi_triangles] = np.repeat(ancestors,
122:                                                         3).reshape(-1, 3)
123:             else:
124:                 # There is a subtlety here: we want to avoid whenever possible
125:                 # that refined points container is a masked triangle (which
126:                 # would result in artifacts in plots).
127:                 # So we impose the numbering from masked ancestors first,
128:                 # then overwrite it with unmasked ancestor numbers.
129:                 ancestor_mask = tri_mask[ancestors]
130:                 found_index[refi_triangles[ancestor_mask, :]
131:                             ] = np.repeat(ancestors[ancestor_mask],
132:                                           3).reshape(-1, 3)
133:                 found_index[refi_triangles[~ancestor_mask, :]
134:                             ] = np.repeat(ancestors[~ancestor_mask],
135:                                           3).reshape(-1, 3)
136:             return refi_triangulation, found_index
137:         else:
138:             return refi_triangulation
139: 
140:     def refine_field(self, z, triinterpolator=None, subdiv=3):
141:         '''
142:         Refines a field defined on the encapsulated triangulation.
143: 
144:         Returns *refi_tri* (refined triangulation), *refi_z* (interpolated
145:         values of the field at the node of the refined triangulation).
146: 
147:         Parameters
148:         ----------
149:         z : 1d-array-like of length ``n_points``
150:             Values of the field to refine, defined at the nodes of the
151:             encapsulated triangulation. (``n_points`` is the number of points
152:             in the initial triangulation)
153:         triinterpolator : :class:`~matplotlib.tri.TriInterpolator`, optional
154:             Interpolator used for field interpolation. If not specified,
155:             a :class:`~matplotlib.tri.CubicTriInterpolator` will
156:             be used.
157:         subdiv : integer, optional
158:             Recursion level for the subdivision. Defaults to 3.
159:             Each triangle will be divided into ``4**subdiv`` child triangles.
160: 
161:         Returns
162:         -------
163:         refi_tri : :class:`~matplotlib.tri.Triangulation` object
164:                      The returned refined triangulation
165:         refi_z : 1d array of length: *refi_tri* node count.
166:                    The returned interpolated field (at *refi_tri* nodes)
167:         '''
168:         if triinterpolator is None:
169:             interp = matplotlib.tri.CubicTriInterpolator(
170:                 self._triangulation, z)
171:         else:
172:             if not isinstance(triinterpolator,
173:                               matplotlib.tri.TriInterpolator):
174:                 raise ValueError("Expected a TriInterpolator object")
175:             interp = triinterpolator
176: 
177:         refi_tri, found_index = self.refine_triangulation(
178:             subdiv=subdiv, return_tri_index=True)
179:         refi_z = interp._interpolate_multikeys(
180:             refi_tri.x, refi_tri.y, tri_index=found_index)[0]
181:         return refi_tri, refi_z
182: 
183:     @staticmethod
184:     def _refine_triangulation_once(triangulation, ancestors=None):
185:         '''
186:         This function refines a matplotlib.tri *triangulation* by splitting
187:         each triangle into 4 child-masked_triangles built on the edges midside
188:         nodes.
189:         The masked triangles, if present, are also splitted but their children
190:         returned masked.
191: 
192:         If *ancestors* is not provided, returns only a new triangulation:
193:         child_triangulation.
194: 
195:         If the array-like key table *ancestor* is given, it shall be of shape
196:         (ntri,) where ntri is the number of *triangulation* masked_triangles.
197:         In this case, the function returns
198:         (child_triangulation, child_ancestors)
199:         child_ancestors is defined so that the 4 child masked_triangles share
200:         the same index as their father: child_ancestors.shape = (4 * ntri,).
201: 
202:         '''
203:         x = triangulation.x
204:         y = triangulation.y
205: 
206:         #    According to tri.triangulation doc:
207:         #         neighbors[i,j] is the triangle that is the neighbor
208:         #         to the edge from point index masked_triangles[i,j] to point
209:         #         index masked_triangles[i,(j+1)%3].
210:         neighbors = triangulation.neighbors
211:         triangles = triangulation.triangles
212:         npts = np.shape(x)[0]
213:         ntri = np.shape(triangles)[0]
214:         if ancestors is not None:
215:             ancestors = np.asarray(ancestors)
216:             if np.shape(ancestors) != (ntri,):
217:                 raise ValueError(
218:                     "Incompatible shapes provide for triangulation"
219:                     ".masked_triangles and ancestors: {0} and {1}".format(
220:                         np.shape(triangles), np.shape(ancestors)))
221: 
222:         # Initiating tables refi_x and refi_y of the refined triangulation
223:         # points
224:         # hint: each apex is shared by 2 masked_triangles except the borders.
225:         borders = np.sum(neighbors == -1)
226:         added_pts = (3*ntri + borders) // 2
227:         refi_npts = npts + added_pts
228:         refi_x = np.zeros(refi_npts)
229:         refi_y = np.zeros(refi_npts)
230: 
231:         # First part of refi_x, refi_y is just the initial points
232:         refi_x[:npts] = x
233:         refi_y[:npts] = y
234: 
235:         # Second part contains the edge midside nodes.
236:         # Each edge belongs to 1 triangle (if border edge) or is shared by 2
237:         # masked_triangles (interior edge).
238:         # We first build 2 * ntri arrays of edge starting nodes (edge_elems,
239:         # edge_apexes) ; we then extract only the masters to avoid overlaps.
240:         # The so-called 'master' is the triangle with biggest index
241:         # The 'slave' is the triangle with lower index
242:         # (can be -1 if border edge)
243:         # For slave and master we will identify the apex pointing to the edge
244:         # start
245:         edge_elems = np.ravel(np.vstack([np.arange(ntri, dtype=np.int32),
246:                                          np.arange(ntri, dtype=np.int32),
247:                                          np.arange(ntri, dtype=np.int32)]))
248:         edge_apexes = np.ravel(np.vstack([np.zeros(ntri, dtype=np.int32),
249:                                           np.ones(ntri, dtype=np.int32),
250:                                           np.ones(ntri, dtype=np.int32)*2]))
251:         edge_neighbors = neighbors[edge_elems, edge_apexes]
252:         mask_masters = (edge_elems > edge_neighbors)
253: 
254:         # Identifying the "masters" and adding to refi_x, refi_y vec
255:         masters = edge_elems[mask_masters]
256:         apex_masters = edge_apexes[mask_masters]
257:         x_add = (x[triangles[masters, apex_masters]] +
258:                  x[triangles[masters, (apex_masters+1) % 3]]) * 0.5
259:         y_add = (y[triangles[masters, apex_masters]] +
260:                  y[triangles[masters, (apex_masters+1) % 3]]) * 0.5
261:         refi_x[npts:] = x_add
262:         refi_y[npts:] = y_add
263: 
264:         # Building the new masked_triangles ; each old masked_triangles hosts
265:         # 4 new masked_triangles
266:         # there are 6 pts to identify per 'old' triangle, 3 new_pt_corner and
267:         # 3 new_pt_midside
268:         new_pt_corner = triangles
269: 
270:         # What is the index in refi_x, refi_y of point at middle of apex iapex
271:         #  of elem ielem ?
272:         # If ielem is the apex master: simple count, given the way refi_x was
273:         #  built.
274:         # If ielem is the apex slave: yet we do not know ; but we will soon
275:         # using the neighbors table.
276:         new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
277:         cum_sum = npts
278:         for imid in range(3):
279:             mask_st_loc = (imid == apex_masters)
280:             n_masters_loc = np.sum(mask_st_loc)
281:             elem_masters_loc = masters[mask_st_loc]
282:             new_pt_midside[:, imid][elem_masters_loc] = np.arange(
283:                 n_masters_loc, dtype=np.int32) + cum_sum
284:             cum_sum += n_masters_loc
285: 
286:         # Now dealing with slave elems.
287:         # for each slave element we identify the master and then the inode
288:         # onces slave_masters is indentified, slave_masters_apex is such that:
289:         # neighbors[slaves_masters, slave_masters_apex] == slaves
290:         mask_slaves = np.logical_not(mask_masters)
291:         slaves = edge_elems[mask_slaves]
292:         slaves_masters = edge_neighbors[mask_slaves]
293:         diff_table = np.abs(neighbors[slaves_masters, :] -
294:                             np.outer(slaves, np.ones(3, dtype=np.int32)))
295:         slave_masters_apex = np.argmin(diff_table, axis=1)
296:         slaves_apex = edge_apexes[mask_slaves]
297:         new_pt_midside[slaves, slaves_apex] = new_pt_midside[
298:             slaves_masters, slave_masters_apex]
299: 
300:         # Builds the 4 child masked_triangles
301:         child_triangles = np.empty([ntri*4, 3], dtype=np.int32)
302:         child_triangles[0::4, :] = np.vstack([
303:             new_pt_corner[:, 0], new_pt_midside[:, 0],
304:             new_pt_midside[:, 2]]).T
305:         child_triangles[1::4, :] = np.vstack([
306:             new_pt_corner[:, 1], new_pt_midside[:, 1],
307:             new_pt_midside[:, 0]]).T
308:         child_triangles[2::4, :] = np.vstack([
309:             new_pt_corner[:, 2], new_pt_midside[:, 2],
310:             new_pt_midside[:, 1]]).T
311:         child_triangles[3::4, :] = np.vstack([
312:             new_pt_midside[:, 0], new_pt_midside[:, 1],
313:             new_pt_midside[:, 2]]).T
314:         child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
315: 
316:         # Builds the child mask
317:         if triangulation.mask is not None:
318:             child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
319: 
320:         if ancestors is None:
321:             return child_triangulation
322:         else:
323:             return child_triangulation, np.repeat(ancestors, 4)
324: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_301028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nMesh refinement for triangular grids.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301029 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_301029) is not StypyTypeError):

    if (import_301029 != 'pyd_module'):
        __import__(import_301029)
        sys_modules_301030 = sys.modules[import_301029]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_301030.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_301029)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301031 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_301031) is not StypyTypeError):

    if (import_301031 != 'pyd_module'):
        __import__(import_301031)
        sys_modules_301032 = sys.modules[import_301031]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_301032.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_301031)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.tri.triangulation import Triangulation' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301033 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.triangulation')

if (type(import_301033) is not StypyTypeError):

    if (import_301033 != 'pyd_module'):
        __import__(import_301033)
        sys_modules_301034 = sys.modules[import_301033]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.triangulation', sys_modules_301034.module_type_store, module_type_store, ['Triangulation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_301034, sys_modules_301034.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triangulation import Triangulation

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.triangulation', None, module_type_store, ['Triangulation'], [Triangulation])

else:
    # Assigning a type to the variable 'matplotlib.tri.triangulation' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.tri.triangulation', import_301033)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import matplotlib.tri.triinterpolate' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_301035 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.triinterpolate')

if (type(import_301035) is not StypyTypeError):

    if (import_301035 != 'pyd_module'):
        __import__(import_301035)
        sys_modules_301036 = sys.modules[import_301035]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.triinterpolate', sys_modules_301036.module_type_store, module_type_store)
    else:
        import matplotlib.tri.triinterpolate

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.triinterpolate', matplotlib.tri.triinterpolate, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.tri.triinterpolate' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.tri.triinterpolate', import_301035)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

# Declaration of the 'TriRefiner' class

class TriRefiner(object, ):
    unicode_301037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'unicode', u'\n    Abstract base class for classes implementing mesh refinement.\n\n    A TriRefiner encapsulates a Triangulation object and provides tools for\n    mesh refinement and interpolation.\n\n    Derived classes must implements:\n\n        - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where\n          the optional keyword arguments *kwargs* are defined in each\n          TriRefiner concrete implementation, and which returns :\n\n              - a refined triangulation\n              - optionally (depending on *return_tri_index*), for each\n                point of the refined triangulation: the index of\n                the initial triangulation triangle to which it belongs.\n\n        - ``refine_field(z, triinterpolator=None, **kwargs)`` , where:\n\n              - *z* array of field values (to refine) defined at the base\n                triangulation nodes\n              - *triinterpolator* is a\n                :class:`~matplotlib.tri.TriInterpolator` (optional)\n              - the other optional keyword arguments *kwargs* are defined in\n                each TriRefiner concrete implementation\n\n          and which returns (as a tuple) a refined triangular mesh and the\n          interpolated values of the field at the refined triangulation nodes.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriRefiner.__init__', ['triangulation'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to isinstance(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'triangulation' (line 46)
        triangulation_301039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'triangulation', False)
        # Getting the type of 'Triangulation' (line 46)
        Triangulation_301040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 41), 'Triangulation', False)
        # Processing the call keyword arguments (line 46)
        kwargs_301041 = {}
        # Getting the type of 'isinstance' (line 46)
        isinstance_301038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 46)
        isinstance_call_result_301042 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), isinstance_301038, *[triangulation_301039, Triangulation_301040], **kwargs_301041)
        
        # Applying the 'not' unary operator (line 46)
        result_not__301043 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 11), 'not', isinstance_call_result_301042)
        
        # Testing the type of an if condition (line 46)
        if_condition_301044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), result_not__301043)
        # Assigning a type to the variable 'if_condition_301044' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_301044', if_condition_301044)
        # SSA begins for if statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 47)
        # Processing the call arguments (line 47)
        unicode_301046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'unicode', u'Expected a Triangulation object')
        # Processing the call keyword arguments (line 47)
        kwargs_301047 = {}
        # Getting the type of 'ValueError' (line 47)
        ValueError_301045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 47)
        ValueError_call_result_301048 = invoke(stypy.reporting.localization.Localization(__file__, 47, 18), ValueError_301045, *[unicode_301046], **kwargs_301047)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 47, 12), ValueError_call_result_301048, 'raise parameter', BaseException)
        # SSA join for if statement (line 46)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'triangulation' (line 48)
        triangulation_301049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'triangulation')
        # Getting the type of 'self' (line 48)
        self_301050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member '_triangulation' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_301050, '_triangulation', triangulation_301049)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TriRefiner' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TriRefiner', TriRefiner)
# Declaration of the 'UniformTriRefiner' class
# Getting the type of 'TriRefiner' (line 51)
TriRefiner_301051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'TriRefiner')

class UniformTriRefiner(TriRefiner_301051, ):
    unicode_301052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'unicode', u'\n    Uniform mesh refinement by recursive subdivisions.\n\n    Parameters\n    ----------\n    triangulation : :class:`~matplotlib.tri.Triangulation`\n                     The encapsulated triangulation (to be refined)\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UniformTriRefiner.__init__', ['triangulation'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'self' (line 66)
        self_301055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'self', False)
        # Getting the type of 'triangulation' (line 66)
        triangulation_301056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'triangulation', False)
        # Processing the call keyword arguments (line 66)
        kwargs_301057 = {}
        # Getting the type of 'TriRefiner' (line 66)
        TriRefiner_301053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'TriRefiner', False)
        # Obtaining the member '__init__' of a type (line 66)
        init___301054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), TriRefiner_301053, '__init__')
        # Calling __init__(args, kwargs) (line 66)
        init___call_result_301058 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), init___301054, *[self_301055, triangulation_301056], **kwargs_301057)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def refine_triangulation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 68)
        False_301059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 52), 'False')
        int_301060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 66), 'int')
        defaults = [False_301059, int_301060]
        # Create a new context for function 'refine_triangulation'
        module_type_store = module_type_store.open_function_context('refine_triangulation', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_localization', localization)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_type_store', module_type_store)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_function_name', 'UniformTriRefiner.refine_triangulation')
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_param_names_list', ['return_tri_index', 'subdiv'])
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_varargs_param_name', None)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_call_defaults', defaults)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_call_varargs', varargs)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UniformTriRefiner.refine_triangulation.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UniformTriRefiner.refine_triangulation', ['return_tri_index', 'subdiv'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'refine_triangulation', localization, ['return_tri_index', 'subdiv'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'refine_triangulation(...)' code ##################

        unicode_301061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'unicode', u'\n        Computes an uniformly refined triangulation *refi_triangulation* of\n        the encapsulated :attr:`triangulation`.\n\n        This function refines the encapsulated triangulation by splitting each\n        father triangle into 4 child sub-triangles built on the edges midside\n        nodes, recursively (level of recursion *subdiv*).\n        In the end, each triangle is hence divided into ``4**subdiv``\n        child triangles.\n        The default value for *subdiv* is 3 resulting in 64 refined\n        subtriangles for each triangle of the initial triangulation.\n\n        Parameters\n        ----------\n        return_tri_index : boolean, optional\n            Boolean indicating whether an index table indicating the father\n            triangle index of each point will be returned. Default value\n            False.\n        subdiv : integer, optional\n            Recursion level for the subdivision. Defaults value 3.\n            Each triangle will be divided into ``4**subdiv`` child triangles.\n\n        Returns\n        -------\n        refi_triangulation : :class:`~matplotlib.tri.Triangulation`\n            The returned refined triangulation\n        found_index : array-like of integers\n            Index of the initial triangulation containing triangle, for each\n            point of *refi_triangulation*.\n            Returned only if *return_tri_index* is set to True.\n\n        ')
        
        # Assigning a Attribute to a Name (line 101):
        
        # Assigning a Attribute to a Name (line 101):
        # Getting the type of 'self' (line 101)
        self_301062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'self')
        # Obtaining the member '_triangulation' of a type (line 101)
        _triangulation_301063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 29), self_301062, '_triangulation')
        # Assigning a type to the variable 'refi_triangulation' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'refi_triangulation', _triangulation_301063)
        
        # Assigning a Subscript to a Name (line 102):
        
        # Assigning a Subscript to a Name (line 102):
        
        # Obtaining the type of the subscript
        int_301064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 50), 'int')
        # Getting the type of 'refi_triangulation' (line 102)
        refi_triangulation_301065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'refi_triangulation')
        # Obtaining the member 'triangles' of a type (line 102)
        triangles_301066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), refi_triangulation_301065, 'triangles')
        # Obtaining the member 'shape' of a type (line 102)
        shape_301067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), triangles_301066, 'shape')
        # Obtaining the member '__getitem__' of a type (line 102)
        getitem___301068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), shape_301067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 102)
        subscript_call_result_301069 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___301068, int_301064)
        
        # Assigning a type to the variable 'ntri' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'ntri', subscript_call_result_301069)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to arange(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'ntri' (line 106)
        ntri_301072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'ntri', False)
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'np' (line 106)
        np_301073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'np', False)
        # Obtaining the member 'int32' of a type (line 106)
        int32_301074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 42), np_301073, 'int32')
        keyword_301075 = int32_301074
        kwargs_301076 = {'dtype': keyword_301075}
        # Getting the type of 'np' (line 106)
        np_301070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'np', False)
        # Obtaining the member 'arange' of a type (line 106)
        arange_301071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), np_301070, 'arange')
        # Calling arange(args, kwargs) (line 106)
        arange_call_result_301077 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), arange_301071, *[ntri_301072], **kwargs_301076)
        
        # Assigning a type to the variable 'ancestors' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'ancestors', arange_call_result_301077)
        
        
        # Call to range(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'subdiv' (line 107)
        subdiv_301079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'subdiv', False)
        # Processing the call keyword arguments (line 107)
        kwargs_301080 = {}
        # Getting the type of 'range' (line 107)
        range_301078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'range', False)
        # Calling range(args, kwargs) (line 107)
        range_call_result_301081 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), range_301078, *[subdiv_301079], **kwargs_301080)
        
        # Testing the type of a for loop iterable (line 107)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 107, 8), range_call_result_301081)
        # Getting the type of the for loop variable (line 107)
        for_loop_var_301082 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 107, 8), range_call_result_301081)
        # Assigning a type to the variable '_' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), '_', for_loop_var_301082)
        # SSA begins for a for statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 108):
        
        # Assigning a Call to a Name:
        
        # Call to _refine_triangulation_once(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'refi_triangulation' (line 109)
        refi_triangulation_301085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'refi_triangulation', False)
        # Getting the type of 'ancestors' (line 109)
        ancestors_301086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 36), 'ancestors', False)
        # Processing the call keyword arguments (line 108)
        kwargs_301087 = {}
        # Getting the type of 'self' (line 108)
        self_301083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'self', False)
        # Obtaining the member '_refine_triangulation_once' of a type (line 108)
        _refine_triangulation_once_301084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 44), self_301083, '_refine_triangulation_once')
        # Calling _refine_triangulation_once(args, kwargs) (line 108)
        _refine_triangulation_once_call_result_301088 = invoke(stypy.reporting.localization.Localization(__file__, 108, 44), _refine_triangulation_once_301084, *[refi_triangulation_301085, ancestors_301086], **kwargs_301087)
        
        # Assigning a type to the variable 'call_assignment_301022' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301022', _refine_triangulation_once_call_result_301088)
        
        # Assigning a Call to a Name (line 108):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_301091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
        # Processing the call keyword arguments
        kwargs_301092 = {}
        # Getting the type of 'call_assignment_301022' (line 108)
        call_assignment_301022_301089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301022', False)
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___301090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), call_assignment_301022_301089, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_301093 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___301090, *[int_301091], **kwargs_301092)
        
        # Assigning a type to the variable 'call_assignment_301023' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301023', getitem___call_result_301093)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'call_assignment_301023' (line 108)
        call_assignment_301023_301094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301023')
        # Assigning a type to the variable 'refi_triangulation' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'refi_triangulation', call_assignment_301023_301094)
        
        # Assigning a Call to a Name (line 108):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_301097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
        # Processing the call keyword arguments
        kwargs_301098 = {}
        # Getting the type of 'call_assignment_301022' (line 108)
        call_assignment_301022_301095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301022', False)
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___301096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), call_assignment_301022_301095, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_301099 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___301096, *[int_301097], **kwargs_301098)
        
        # Assigning a type to the variable 'call_assignment_301024' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301024', getitem___call_result_301099)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'call_assignment_301024' (line 108)
        call_assignment_301024_301100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_assignment_301024')
        # Assigning a type to the variable 'ancestors' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'ancestors', call_assignment_301024_301100)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 110):
        
        # Assigning a Subscript to a Name (line 110):
        
        # Obtaining the type of the subscript
        int_301101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 47), 'int')
        # Getting the type of 'refi_triangulation' (line 110)
        refi_triangulation_301102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'refi_triangulation')
        # Obtaining the member 'x' of a type (line 110)
        x_301103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), refi_triangulation_301102, 'x')
        # Obtaining the member 'shape' of a type (line 110)
        shape_301104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), x_301103, 'shape')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___301105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), shape_301104, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_301106 = invoke(stypy.reporting.localization.Localization(__file__, 110, 20), getitem___301105, int_301101)
        
        # Assigning a type to the variable 'refi_npts' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'refi_npts', subscript_call_result_301106)
        
        # Assigning a Attribute to a Name (line 111):
        
        # Assigning a Attribute to a Name (line 111):
        # Getting the type of 'refi_triangulation' (line 111)
        refi_triangulation_301107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'refi_triangulation')
        # Obtaining the member 'triangles' of a type (line 111)
        triangles_301108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), refi_triangulation_301107, 'triangles')
        # Assigning a type to the variable 'refi_triangles' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'refi_triangles', triangles_301108)
        
        # Getting the type of 'return_tri_index' (line 114)
        return_tri_index_301109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'return_tri_index')
        # Testing the type of an if condition (line 114)
        if_condition_301110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 8), return_tri_index_301109)
        # Assigning a type to the variable 'if_condition_301110' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'if_condition_301110', if_condition_301110)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 118):
        
        # Assigning a UnaryOp to a Name (line 118):
        
        
        # Call to ones(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'refi_npts' (line 118)
        refi_npts_301113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'refi_npts', False)
        # Processing the call keyword arguments (line 118)
        # Getting the type of 'np' (line 118)
        np_301114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 53), 'np', False)
        # Obtaining the member 'int32' of a type (line 118)
        int32_301115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 53), np_301114, 'int32')
        keyword_301116 = int32_301115
        kwargs_301117 = {'dtype': keyword_301116}
        # Getting the type of 'np' (line 118)
        np_301111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'np', False)
        # Obtaining the member 'ones' of a type (line 118)
        ones_301112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 28), np_301111, 'ones')
        # Calling ones(args, kwargs) (line 118)
        ones_call_result_301118 = invoke(stypy.reporting.localization.Localization(__file__, 118, 28), ones_301112, *[refi_npts_301113], **kwargs_301117)
        
        # Applying the 'usub' unary operator (line 118)
        result___neg___301119 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 26), 'usub', ones_call_result_301118)
        
        # Assigning a type to the variable 'found_index' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'found_index', result___neg___301119)
        
        # Assigning a Attribute to a Name (line 119):
        
        # Assigning a Attribute to a Name (line 119):
        # Getting the type of 'self' (line 119)
        self_301120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'self')
        # Obtaining the member '_triangulation' of a type (line 119)
        _triangulation_301121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), self_301120, '_triangulation')
        # Obtaining the member 'mask' of a type (line 119)
        mask_301122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), _triangulation_301121, 'mask')
        # Assigning a type to the variable 'tri_mask' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'tri_mask', mask_301122)
        
        # Type idiom detected: calculating its left and rigth part (line 120)
        # Getting the type of 'tri_mask' (line 120)
        tri_mask_301123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'tri_mask')
        # Getting the type of 'None' (line 120)
        None_301124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'None')
        
        (may_be_301125, more_types_in_union_301126) = may_be_none(tri_mask_301123, None_301124)

        if may_be_301125:

            if more_types_in_union_301126:
                # Runtime conditional SSA (line 120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Subscript (line 121):
            
            # Assigning a Call to a Subscript (line 121):
            
            # Call to reshape(...): (line 121)
            # Processing the call arguments (line 121)
            int_301134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 67), 'int')
            int_301135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 71), 'int')
            # Processing the call keyword arguments (line 121)
            kwargs_301136 = {}
            
            # Call to repeat(...): (line 121)
            # Processing the call arguments (line 121)
            # Getting the type of 'ancestors' (line 121)
            ancestors_301129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 56), 'ancestors', False)
            int_301130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 56), 'int')
            # Processing the call keyword arguments (line 121)
            kwargs_301131 = {}
            # Getting the type of 'np' (line 121)
            np_301127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'np', False)
            # Obtaining the member 'repeat' of a type (line 121)
            repeat_301128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 46), np_301127, 'repeat')
            # Calling repeat(args, kwargs) (line 121)
            repeat_call_result_301132 = invoke(stypy.reporting.localization.Localization(__file__, 121, 46), repeat_301128, *[ancestors_301129, int_301130], **kwargs_301131)
            
            # Obtaining the member 'reshape' of a type (line 121)
            reshape_301133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 46), repeat_call_result_301132, 'reshape')
            # Calling reshape(args, kwargs) (line 121)
            reshape_call_result_301137 = invoke(stypy.reporting.localization.Localization(__file__, 121, 46), reshape_301133, *[int_301134, int_301135], **kwargs_301136)
            
            # Getting the type of 'found_index' (line 121)
            found_index_301138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'found_index')
            # Getting the type of 'refi_triangles' (line 121)
            refi_triangles_301139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'refi_triangles')
            # Storing an element on a container (line 121)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 16), found_index_301138, (refi_triangles_301139, reshape_call_result_301137))

            if more_types_in_union_301126:
                # Runtime conditional SSA for else branch (line 120)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_301125) or more_types_in_union_301126):
            
            # Assigning a Subscript to a Name (line 129):
            
            # Assigning a Subscript to a Name (line 129):
            
            # Obtaining the type of the subscript
            # Getting the type of 'ancestors' (line 129)
            ancestors_301140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'ancestors')
            # Getting the type of 'tri_mask' (line 129)
            tri_mask_301141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'tri_mask')
            # Obtaining the member '__getitem__' of a type (line 129)
            getitem___301142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 32), tri_mask_301141, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 129)
            subscript_call_result_301143 = invoke(stypy.reporting.localization.Localization(__file__, 129, 32), getitem___301142, ancestors_301140)
            
            # Assigning a type to the variable 'ancestor_mask' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'ancestor_mask', subscript_call_result_301143)
            
            # Assigning a Call to a Subscript (line 130):
            
            # Assigning a Call to a Subscript (line 130):
            
            # Call to reshape(...): (line 131)
            # Processing the call arguments (line 131)
            int_301154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 53), 'int')
            int_301155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 57), 'int')
            # Processing the call keyword arguments (line 131)
            kwargs_301156 = {}
            
            # Call to repeat(...): (line 131)
            # Processing the call arguments (line 131)
            
            # Obtaining the type of the subscript
            # Getting the type of 'ancestor_mask' (line 131)
            ancestor_mask_301146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 52), 'ancestor_mask', False)
            # Getting the type of 'ancestors' (line 131)
            ancestors_301147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 42), 'ancestors', False)
            # Obtaining the member '__getitem__' of a type (line 131)
            getitem___301148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 42), ancestors_301147, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 131)
            subscript_call_result_301149 = invoke(stypy.reporting.localization.Localization(__file__, 131, 42), getitem___301148, ancestor_mask_301146)
            
            int_301150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 42), 'int')
            # Processing the call keyword arguments (line 131)
            kwargs_301151 = {}
            # Getting the type of 'np' (line 131)
            np_301144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 32), 'np', False)
            # Obtaining the member 'repeat' of a type (line 131)
            repeat_301145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 32), np_301144, 'repeat')
            # Calling repeat(args, kwargs) (line 131)
            repeat_call_result_301152 = invoke(stypy.reporting.localization.Localization(__file__, 131, 32), repeat_301145, *[subscript_call_result_301149, int_301150], **kwargs_301151)
            
            # Obtaining the member 'reshape' of a type (line 131)
            reshape_301153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 32), repeat_call_result_301152, 'reshape')
            # Calling reshape(args, kwargs) (line 131)
            reshape_call_result_301157 = invoke(stypy.reporting.localization.Localization(__file__, 131, 32), reshape_301153, *[int_301154, int_301155], **kwargs_301156)
            
            # Getting the type of 'found_index' (line 130)
            found_index_301158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'found_index')
            
            # Obtaining the type of the subscript
            # Getting the type of 'ancestor_mask' (line 130)
            ancestor_mask_301159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'ancestor_mask')
            slice_301160 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 28), None, None, None)
            # Getting the type of 'refi_triangles' (line 130)
            refi_triangles_301161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'refi_triangles')
            # Obtaining the member '__getitem__' of a type (line 130)
            getitem___301162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 28), refi_triangles_301161, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
            subscript_call_result_301163 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), getitem___301162, (ancestor_mask_301159, slice_301160))
            
            # Storing an element on a container (line 130)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), found_index_301158, (subscript_call_result_301163, reshape_call_result_301157))
            
            # Assigning a Call to a Subscript (line 133):
            
            # Assigning a Call to a Subscript (line 133):
            
            # Call to reshape(...): (line 134)
            # Processing the call arguments (line 134)
            int_301175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 53), 'int')
            int_301176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 57), 'int')
            # Processing the call keyword arguments (line 134)
            kwargs_301177 = {}
            
            # Call to repeat(...): (line 134)
            # Processing the call arguments (line 134)
            
            # Obtaining the type of the subscript
            
            # Getting the type of 'ancestor_mask' (line 134)
            ancestor_mask_301166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 53), 'ancestor_mask', False)
            # Applying the '~' unary operator (line 134)
            result_inv_301167 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 52), '~', ancestor_mask_301166)
            
            # Getting the type of 'ancestors' (line 134)
            ancestors_301168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'ancestors', False)
            # Obtaining the member '__getitem__' of a type (line 134)
            getitem___301169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 42), ancestors_301168, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 134)
            subscript_call_result_301170 = invoke(stypy.reporting.localization.Localization(__file__, 134, 42), getitem___301169, result_inv_301167)
            
            int_301171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 42), 'int')
            # Processing the call keyword arguments (line 134)
            kwargs_301172 = {}
            # Getting the type of 'np' (line 134)
            np_301164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'np', False)
            # Obtaining the member 'repeat' of a type (line 134)
            repeat_301165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), np_301164, 'repeat')
            # Calling repeat(args, kwargs) (line 134)
            repeat_call_result_301173 = invoke(stypy.reporting.localization.Localization(__file__, 134, 32), repeat_301165, *[subscript_call_result_301170, int_301171], **kwargs_301172)
            
            # Obtaining the member 'reshape' of a type (line 134)
            reshape_301174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), repeat_call_result_301173, 'reshape')
            # Calling reshape(args, kwargs) (line 134)
            reshape_call_result_301178 = invoke(stypy.reporting.localization.Localization(__file__, 134, 32), reshape_301174, *[int_301175, int_301176], **kwargs_301177)
            
            # Getting the type of 'found_index' (line 133)
            found_index_301179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'found_index')
            
            # Obtaining the type of the subscript
            
            # Getting the type of 'ancestor_mask' (line 133)
            ancestor_mask_301180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'ancestor_mask')
            # Applying the '~' unary operator (line 133)
            result_inv_301181 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 43), '~', ancestor_mask_301180)
            
            slice_301182 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 133, 28), None, None, None)
            # Getting the type of 'refi_triangles' (line 133)
            refi_triangles_301183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'refi_triangles')
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___301184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 28), refi_triangles_301183, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_301185 = invoke(stypy.reporting.localization.Localization(__file__, 133, 28), getitem___301184, (result_inv_301181, slice_301182))
            
            # Storing an element on a container (line 133)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 16), found_index_301179, (subscript_call_result_301185, reshape_call_result_301178))

            if (may_be_301125 and more_types_in_union_301126):
                # SSA join for if statement (line 120)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_301186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        # Getting the type of 'refi_triangulation' (line 136)
        refi_triangulation_301187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'refi_triangulation')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), tuple_301186, refi_triangulation_301187)
        # Adding element type (line 136)
        # Getting the type of 'found_index' (line 136)
        found_index_301188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 39), 'found_index')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), tuple_301186, found_index_301188)
        
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'stypy_return_type', tuple_301186)
        # SSA branch for the else part of an if statement (line 114)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'refi_triangulation' (line 138)
        refi_triangulation_301189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'refi_triangulation')
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', refi_triangulation_301189)
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'refine_triangulation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'refine_triangulation' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_301190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_301190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'refine_triangulation'
        return stypy_return_type_301190


    @norecursion
    def refine_field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 140)
        None_301191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 46), 'None')
        int_301192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 59), 'int')
        defaults = [None_301191, int_301192]
        # Create a new context for function 'refine_field'
        module_type_store = module_type_store.open_function_context('refine_field', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_localization', localization)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_type_store', module_type_store)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_function_name', 'UniformTriRefiner.refine_field')
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_param_names_list', ['z', 'triinterpolator', 'subdiv'])
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_varargs_param_name', None)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_call_defaults', defaults)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_call_varargs', varargs)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UniformTriRefiner.refine_field.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UniformTriRefiner.refine_field', ['z', 'triinterpolator', 'subdiv'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'refine_field', localization, ['z', 'triinterpolator', 'subdiv'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'refine_field(...)' code ##################

        unicode_301193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, (-1)), 'unicode', u'\n        Refines a field defined on the encapsulated triangulation.\n\n        Returns *refi_tri* (refined triangulation), *refi_z* (interpolated\n        values of the field at the node of the refined triangulation).\n\n        Parameters\n        ----------\n        z : 1d-array-like of length ``n_points``\n            Values of the field to refine, defined at the nodes of the\n            encapsulated triangulation. (``n_points`` is the number of points\n            in the initial triangulation)\n        triinterpolator : :class:`~matplotlib.tri.TriInterpolator`, optional\n            Interpolator used for field interpolation. If not specified,\n            a :class:`~matplotlib.tri.CubicTriInterpolator` will\n            be used.\n        subdiv : integer, optional\n            Recursion level for the subdivision. Defaults to 3.\n            Each triangle will be divided into ``4**subdiv`` child triangles.\n\n        Returns\n        -------\n        refi_tri : :class:`~matplotlib.tri.Triangulation` object\n                     The returned refined triangulation\n        refi_z : 1d array of length: *refi_tri* node count.\n                   The returned interpolated field (at *refi_tri* nodes)\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 168)
        # Getting the type of 'triinterpolator' (line 168)
        triinterpolator_301194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'triinterpolator')
        # Getting the type of 'None' (line 168)
        None_301195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'None')
        
        (may_be_301196, more_types_in_union_301197) = may_be_none(triinterpolator_301194, None_301195)

        if may_be_301196:

            if more_types_in_union_301197:
                # Runtime conditional SSA (line 168)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 169):
            
            # Assigning a Call to a Name (line 169):
            
            # Call to CubicTriInterpolator(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 'self' (line 170)
            self_301201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'self', False)
            # Obtaining the member '_triangulation' of a type (line 170)
            _triangulation_301202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), self_301201, '_triangulation')
            # Getting the type of 'z' (line 170)
            z_301203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 37), 'z', False)
            # Processing the call keyword arguments (line 169)
            kwargs_301204 = {}
            # Getting the type of 'matplotlib' (line 169)
            matplotlib_301198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'matplotlib', False)
            # Obtaining the member 'tri' of a type (line 169)
            tri_301199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), matplotlib_301198, 'tri')
            # Obtaining the member 'CubicTriInterpolator' of a type (line 169)
            CubicTriInterpolator_301200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), tri_301199, 'CubicTriInterpolator')
            # Calling CubicTriInterpolator(args, kwargs) (line 169)
            CubicTriInterpolator_call_result_301205 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), CubicTriInterpolator_301200, *[_triangulation_301202, z_301203], **kwargs_301204)
            
            # Assigning a type to the variable 'interp' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'interp', CubicTriInterpolator_call_result_301205)

            if more_types_in_union_301197:
                # Runtime conditional SSA for else branch (line 168)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_301196) or more_types_in_union_301197):
            
            
            
            # Call to isinstance(...): (line 172)
            # Processing the call arguments (line 172)
            # Getting the type of 'triinterpolator' (line 172)
            triinterpolator_301207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 30), 'triinterpolator', False)
            # Getting the type of 'matplotlib' (line 173)
            matplotlib_301208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'matplotlib', False)
            # Obtaining the member 'tri' of a type (line 173)
            tri_301209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 30), matplotlib_301208, 'tri')
            # Obtaining the member 'TriInterpolator' of a type (line 173)
            TriInterpolator_301210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 30), tri_301209, 'TriInterpolator')
            # Processing the call keyword arguments (line 172)
            kwargs_301211 = {}
            # Getting the type of 'isinstance' (line 172)
            isinstance_301206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 172)
            isinstance_call_result_301212 = invoke(stypy.reporting.localization.Localization(__file__, 172, 19), isinstance_301206, *[triinterpolator_301207, TriInterpolator_301210], **kwargs_301211)
            
            # Applying the 'not' unary operator (line 172)
            result_not__301213 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), 'not', isinstance_call_result_301212)
            
            # Testing the type of an if condition (line 172)
            if_condition_301214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 12), result_not__301213)
            # Assigning a type to the variable 'if_condition_301214' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'if_condition_301214', if_condition_301214)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 174)
            # Processing the call arguments (line 174)
            unicode_301216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 33), 'unicode', u'Expected a TriInterpolator object')
            # Processing the call keyword arguments (line 174)
            kwargs_301217 = {}
            # Getting the type of 'ValueError' (line 174)
            ValueError_301215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 174)
            ValueError_call_result_301218 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), ValueError_301215, *[unicode_301216], **kwargs_301217)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 174, 16), ValueError_call_result_301218, 'raise parameter', BaseException)
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 175):
            
            # Assigning a Name to a Name (line 175):
            # Getting the type of 'triinterpolator' (line 175)
            triinterpolator_301219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'triinterpolator')
            # Assigning a type to the variable 'interp' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'interp', triinterpolator_301219)

            if (may_be_301196 and more_types_in_union_301197):
                # SSA join for if statement (line 168)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 177):
        
        # Assigning a Call to a Name:
        
        # Call to refine_triangulation(...): (line 177)
        # Processing the call keyword arguments (line 177)
        # Getting the type of 'subdiv' (line 178)
        subdiv_301222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'subdiv', False)
        keyword_301223 = subdiv_301222
        # Getting the type of 'True' (line 178)
        True_301224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'True', False)
        keyword_301225 = True_301224
        kwargs_301226 = {'subdiv': keyword_301223, 'return_tri_index': keyword_301225}
        # Getting the type of 'self' (line 177)
        self_301220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'self', False)
        # Obtaining the member 'refine_triangulation' of a type (line 177)
        refine_triangulation_301221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 32), self_301220, 'refine_triangulation')
        # Calling refine_triangulation(args, kwargs) (line 177)
        refine_triangulation_call_result_301227 = invoke(stypy.reporting.localization.Localization(__file__, 177, 32), refine_triangulation_301221, *[], **kwargs_301226)
        
        # Assigning a type to the variable 'call_assignment_301025' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301025', refine_triangulation_call_result_301227)
        
        # Assigning a Call to a Name (line 177):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_301230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        # Processing the call keyword arguments
        kwargs_301231 = {}
        # Getting the type of 'call_assignment_301025' (line 177)
        call_assignment_301025_301228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301025', False)
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___301229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), call_assignment_301025_301228, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_301232 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___301229, *[int_301230], **kwargs_301231)
        
        # Assigning a type to the variable 'call_assignment_301026' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301026', getitem___call_result_301232)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'call_assignment_301026' (line 177)
        call_assignment_301026_301233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301026')
        # Assigning a type to the variable 'refi_tri' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'refi_tri', call_assignment_301026_301233)
        
        # Assigning a Call to a Name (line 177):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_301236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        # Processing the call keyword arguments
        kwargs_301237 = {}
        # Getting the type of 'call_assignment_301025' (line 177)
        call_assignment_301025_301234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301025', False)
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___301235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), call_assignment_301025_301234, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_301238 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___301235, *[int_301236], **kwargs_301237)
        
        # Assigning a type to the variable 'call_assignment_301027' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301027', getitem___call_result_301238)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'call_assignment_301027' (line 177)
        call_assignment_301027_301239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_301027')
        # Assigning a type to the variable 'found_index' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'found_index', call_assignment_301027_301239)
        
        # Assigning a Subscript to a Name (line 179):
        
        # Assigning a Subscript to a Name (line 179):
        
        # Obtaining the type of the subscript
        int_301240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 59), 'int')
        
        # Call to _interpolate_multikeys(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'refi_tri' (line 180)
        refi_tri_301243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'refi_tri', False)
        # Obtaining the member 'x' of a type (line 180)
        x_301244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), refi_tri_301243, 'x')
        # Getting the type of 'refi_tri' (line 180)
        refi_tri_301245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'refi_tri', False)
        # Obtaining the member 'y' of a type (line 180)
        y_301246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 24), refi_tri_301245, 'y')
        # Processing the call keyword arguments (line 179)
        # Getting the type of 'found_index' (line 180)
        found_index_301247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 46), 'found_index', False)
        keyword_301248 = found_index_301247
        kwargs_301249 = {'tri_index': keyword_301248}
        # Getting the type of 'interp' (line 179)
        interp_301241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'interp', False)
        # Obtaining the member '_interpolate_multikeys' of a type (line 179)
        _interpolate_multikeys_301242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 17), interp_301241, '_interpolate_multikeys')
        # Calling _interpolate_multikeys(args, kwargs) (line 179)
        _interpolate_multikeys_call_result_301250 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), _interpolate_multikeys_301242, *[x_301244, y_301246], **kwargs_301249)
        
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___301251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 17), _interpolate_multikeys_call_result_301250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_301252 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), getitem___301251, int_301240)
        
        # Assigning a type to the variable 'refi_z' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'refi_z', subscript_call_result_301252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_301253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        # Getting the type of 'refi_tri' (line 181)
        refi_tri_301254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'refi_tri')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 15), tuple_301253, refi_tri_301254)
        # Adding element type (line 181)
        # Getting the type of 'refi_z' (line 181)
        refi_z_301255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'refi_z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 15), tuple_301253, refi_z_301255)
        
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', tuple_301253)
        
        # ################# End of 'refine_field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'refine_field' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_301256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_301256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'refine_field'
        return stypy_return_type_301256


    @staticmethod
    @norecursion
    def _refine_triangulation_once(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 184)
        None_301257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 60), 'None')
        defaults = [None_301257]
        # Create a new context for function '_refine_triangulation_once'
        module_type_store = module_type_store.open_function_context('_refine_triangulation_once', 183, 4, False)
        
        # Passed parameters checking function
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_localization', localization)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_type_of_self', None)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_type_store', module_type_store)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_function_name', '_refine_triangulation_once')
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_param_names_list', ['triangulation', 'ancestors'])
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_varargs_param_name', None)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_call_defaults', defaults)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_call_varargs', varargs)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UniformTriRefiner._refine_triangulation_once.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '_refine_triangulation_once', ['triangulation', 'ancestors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_refine_triangulation_once', localization, ['ancestors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_refine_triangulation_once(...)' code ##################

        unicode_301258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'unicode', u'\n        This function refines a matplotlib.tri *triangulation* by splitting\n        each triangle into 4 child-masked_triangles built on the edges midside\n        nodes.\n        The masked triangles, if present, are also splitted but their children\n        returned masked.\n\n        If *ancestors* is not provided, returns only a new triangulation:\n        child_triangulation.\n\n        If the array-like key table *ancestor* is given, it shall be of shape\n        (ntri,) where ntri is the number of *triangulation* masked_triangles.\n        In this case, the function returns\n        (child_triangulation, child_ancestors)\n        child_ancestors is defined so that the 4 child masked_triangles share\n        the same index as their father: child_ancestors.shape = (4 * ntri,).\n\n        ')
        
        # Assigning a Attribute to a Name (line 203):
        
        # Assigning a Attribute to a Name (line 203):
        # Getting the type of 'triangulation' (line 203)
        triangulation_301259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'triangulation')
        # Obtaining the member 'x' of a type (line 203)
        x_301260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), triangulation_301259, 'x')
        # Assigning a type to the variable 'x' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'x', x_301260)
        
        # Assigning a Attribute to a Name (line 204):
        
        # Assigning a Attribute to a Name (line 204):
        # Getting the type of 'triangulation' (line 204)
        triangulation_301261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'triangulation')
        # Obtaining the member 'y' of a type (line 204)
        y_301262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), triangulation_301261, 'y')
        # Assigning a type to the variable 'y' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'y', y_301262)
        
        # Assigning a Attribute to a Name (line 210):
        
        # Assigning a Attribute to a Name (line 210):
        # Getting the type of 'triangulation' (line 210)
        triangulation_301263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'triangulation')
        # Obtaining the member 'neighbors' of a type (line 210)
        neighbors_301264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), triangulation_301263, 'neighbors')
        # Assigning a type to the variable 'neighbors' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'neighbors', neighbors_301264)
        
        # Assigning a Attribute to a Name (line 211):
        
        # Assigning a Attribute to a Name (line 211):
        # Getting the type of 'triangulation' (line 211)
        triangulation_301265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'triangulation')
        # Obtaining the member 'triangles' of a type (line 211)
        triangles_301266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 20), triangulation_301265, 'triangles')
        # Assigning a type to the variable 'triangles' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'triangles', triangles_301266)
        
        # Assigning a Subscript to a Name (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_301267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 27), 'int')
        
        # Call to shape(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_301270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'x', False)
        # Processing the call keyword arguments (line 212)
        kwargs_301271 = {}
        # Getting the type of 'np' (line 212)
        np_301268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'np', False)
        # Obtaining the member 'shape' of a type (line 212)
        shape_301269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), np_301268, 'shape')
        # Calling shape(args, kwargs) (line 212)
        shape_call_result_301272 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), shape_301269, *[x_301270], **kwargs_301271)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___301273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), shape_call_result_301272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_301274 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), getitem___301273, int_301267)
        
        # Assigning a type to the variable 'npts' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'npts', subscript_call_result_301274)
        
        # Assigning a Subscript to a Name (line 213):
        
        # Assigning a Subscript to a Name (line 213):
        
        # Obtaining the type of the subscript
        int_301275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 35), 'int')
        
        # Call to shape(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'triangles' (line 213)
        triangles_301278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'triangles', False)
        # Processing the call keyword arguments (line 213)
        kwargs_301279 = {}
        # Getting the type of 'np' (line 213)
        np_301276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'np', False)
        # Obtaining the member 'shape' of a type (line 213)
        shape_301277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), np_301276, 'shape')
        # Calling shape(args, kwargs) (line 213)
        shape_call_result_301280 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), shape_301277, *[triangles_301278], **kwargs_301279)
        
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___301281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), shape_call_result_301280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_301282 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), getitem___301281, int_301275)
        
        # Assigning a type to the variable 'ntri' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'ntri', subscript_call_result_301282)
        
        # Type idiom detected: calculating its left and rigth part (line 214)
        # Getting the type of 'ancestors' (line 214)
        ancestors_301283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'ancestors')
        # Getting the type of 'None' (line 214)
        None_301284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'None')
        
        (may_be_301285, more_types_in_union_301286) = may_not_be_none(ancestors_301283, None_301284)

        if may_be_301285:

            if more_types_in_union_301286:
                # Runtime conditional SSA (line 214)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 215):
            
            # Assigning a Call to a Name (line 215):
            
            # Call to asarray(...): (line 215)
            # Processing the call arguments (line 215)
            # Getting the type of 'ancestors' (line 215)
            ancestors_301289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 35), 'ancestors', False)
            # Processing the call keyword arguments (line 215)
            kwargs_301290 = {}
            # Getting the type of 'np' (line 215)
            np_301287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 215)
            asarray_301288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 24), np_301287, 'asarray')
            # Calling asarray(args, kwargs) (line 215)
            asarray_call_result_301291 = invoke(stypy.reporting.localization.Localization(__file__, 215, 24), asarray_301288, *[ancestors_301289], **kwargs_301290)
            
            # Assigning a type to the variable 'ancestors' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'ancestors', asarray_call_result_301291)
            
            
            
            # Call to shape(...): (line 216)
            # Processing the call arguments (line 216)
            # Getting the type of 'ancestors' (line 216)
            ancestors_301294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'ancestors', False)
            # Processing the call keyword arguments (line 216)
            kwargs_301295 = {}
            # Getting the type of 'np' (line 216)
            np_301292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'np', False)
            # Obtaining the member 'shape' of a type (line 216)
            shape_301293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), np_301292, 'shape')
            # Calling shape(args, kwargs) (line 216)
            shape_call_result_301296 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), shape_301293, *[ancestors_301294], **kwargs_301295)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 216)
            tuple_301297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 216)
            # Adding element type (line 216)
            # Getting the type of 'ntri' (line 216)
            ntri_301298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'ntri')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 39), tuple_301297, ntri_301298)
            
            # Applying the binary operator '!=' (line 216)
            result_ne_301299 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 15), '!=', shape_call_result_301296, tuple_301297)
            
            # Testing the type of an if condition (line 216)
            if_condition_301300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 12), result_ne_301299)
            # Assigning a type to the variable 'if_condition_301300' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'if_condition_301300', if_condition_301300)
            # SSA begins for if statement (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 217)
            # Processing the call arguments (line 217)
            
            # Call to format(...): (line 218)
            # Processing the call arguments (line 218)
            
            # Call to shape(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'triangles' (line 220)
            triangles_301306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'triangles', False)
            # Processing the call keyword arguments (line 220)
            kwargs_301307 = {}
            # Getting the type of 'np' (line 220)
            np_301304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'np', False)
            # Obtaining the member 'shape' of a type (line 220)
            shape_301305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 24), np_301304, 'shape')
            # Calling shape(args, kwargs) (line 220)
            shape_call_result_301308 = invoke(stypy.reporting.localization.Localization(__file__, 220, 24), shape_301305, *[triangles_301306], **kwargs_301307)
            
            
            # Call to shape(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'ancestors' (line 220)
            ancestors_301311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 54), 'ancestors', False)
            # Processing the call keyword arguments (line 220)
            kwargs_301312 = {}
            # Getting the type of 'np' (line 220)
            np_301309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 45), 'np', False)
            # Obtaining the member 'shape' of a type (line 220)
            shape_301310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 45), np_301309, 'shape')
            # Calling shape(args, kwargs) (line 220)
            shape_call_result_301313 = invoke(stypy.reporting.localization.Localization(__file__, 220, 45), shape_301310, *[ancestors_301311], **kwargs_301312)
            
            # Processing the call keyword arguments (line 218)
            kwargs_301314 = {}
            unicode_301302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'unicode', u'Incompatible shapes provide for triangulation.masked_triangles and ancestors: {0} and {1}')
            # Obtaining the member 'format' of a type (line 218)
            format_301303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 20), unicode_301302, 'format')
            # Calling format(args, kwargs) (line 218)
            format_call_result_301315 = invoke(stypy.reporting.localization.Localization(__file__, 218, 20), format_301303, *[shape_call_result_301308, shape_call_result_301313], **kwargs_301314)
            
            # Processing the call keyword arguments (line 217)
            kwargs_301316 = {}
            # Getting the type of 'ValueError' (line 217)
            ValueError_301301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 217)
            ValueError_call_result_301317 = invoke(stypy.reporting.localization.Localization(__file__, 217, 22), ValueError_301301, *[format_call_result_301315], **kwargs_301316)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 217, 16), ValueError_call_result_301317, 'raise parameter', BaseException)
            # SSA join for if statement (line 216)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_301286:
                # SSA join for if statement (line 214)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to sum(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Getting the type of 'neighbors' (line 225)
        neighbors_301320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 25), 'neighbors', False)
        int_301321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 38), 'int')
        # Applying the binary operator '==' (line 225)
        result_eq_301322 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 25), '==', neighbors_301320, int_301321)
        
        # Processing the call keyword arguments (line 225)
        kwargs_301323 = {}
        # Getting the type of 'np' (line 225)
        np_301318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 18), 'np', False)
        # Obtaining the member 'sum' of a type (line 225)
        sum_301319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 18), np_301318, 'sum')
        # Calling sum(args, kwargs) (line 225)
        sum_call_result_301324 = invoke(stypy.reporting.localization.Localization(__file__, 225, 18), sum_301319, *[result_eq_301322], **kwargs_301323)
        
        # Assigning a type to the variable 'borders' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'borders', sum_call_result_301324)
        
        # Assigning a BinOp to a Name (line 226):
        
        # Assigning a BinOp to a Name (line 226):
        int_301325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 21), 'int')
        # Getting the type of 'ntri' (line 226)
        ntri_301326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'ntri')
        # Applying the binary operator '*' (line 226)
        result_mul_301327 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 21), '*', int_301325, ntri_301326)
        
        # Getting the type of 'borders' (line 226)
        borders_301328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), 'borders')
        # Applying the binary operator '+' (line 226)
        result_add_301329 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 21), '+', result_mul_301327, borders_301328)
        
        int_301330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 42), 'int')
        # Applying the binary operator '//' (line 226)
        result_floordiv_301331 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 20), '//', result_add_301329, int_301330)
        
        # Assigning a type to the variable 'added_pts' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'added_pts', result_floordiv_301331)
        
        # Assigning a BinOp to a Name (line 227):
        
        # Assigning a BinOp to a Name (line 227):
        # Getting the type of 'npts' (line 227)
        npts_301332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'npts')
        # Getting the type of 'added_pts' (line 227)
        added_pts_301333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'added_pts')
        # Applying the binary operator '+' (line 227)
        result_add_301334 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 20), '+', npts_301332, added_pts_301333)
        
        # Assigning a type to the variable 'refi_npts' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'refi_npts', result_add_301334)
        
        # Assigning a Call to a Name (line 228):
        
        # Assigning a Call to a Name (line 228):
        
        # Call to zeros(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'refi_npts' (line 228)
        refi_npts_301337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'refi_npts', False)
        # Processing the call keyword arguments (line 228)
        kwargs_301338 = {}
        # Getting the type of 'np' (line 228)
        np_301335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 228)
        zeros_301336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 17), np_301335, 'zeros')
        # Calling zeros(args, kwargs) (line 228)
        zeros_call_result_301339 = invoke(stypy.reporting.localization.Localization(__file__, 228, 17), zeros_301336, *[refi_npts_301337], **kwargs_301338)
        
        # Assigning a type to the variable 'refi_x' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'refi_x', zeros_call_result_301339)
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to zeros(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'refi_npts' (line 229)
        refi_npts_301342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'refi_npts', False)
        # Processing the call keyword arguments (line 229)
        kwargs_301343 = {}
        # Getting the type of 'np' (line 229)
        np_301340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 229)
        zeros_301341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 17), np_301340, 'zeros')
        # Calling zeros(args, kwargs) (line 229)
        zeros_call_result_301344 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), zeros_301341, *[refi_npts_301342], **kwargs_301343)
        
        # Assigning a type to the variable 'refi_y' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'refi_y', zeros_call_result_301344)
        
        # Assigning a Name to a Subscript (line 232):
        
        # Assigning a Name to a Subscript (line 232):
        # Getting the type of 'x' (line 232)
        x_301345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'x')
        # Getting the type of 'refi_x' (line 232)
        refi_x_301346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'refi_x')
        # Getting the type of 'npts' (line 232)
        npts_301347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'npts')
        slice_301348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 232, 8), None, npts_301347, None)
        # Storing an element on a container (line 232)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 8), refi_x_301346, (slice_301348, x_301345))
        
        # Assigning a Name to a Subscript (line 233):
        
        # Assigning a Name to a Subscript (line 233):
        # Getting the type of 'y' (line 233)
        y_301349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'y')
        # Getting the type of 'refi_y' (line 233)
        refi_y_301350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'refi_y')
        # Getting the type of 'npts' (line 233)
        npts_301351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'npts')
        slice_301352 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 233, 8), None, npts_301351, None)
        # Storing an element on a container (line 233)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 8), refi_y_301350, (slice_301352, y_301349))
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to ravel(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Call to vstack(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_301357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        
        # Call to arange(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'ntri' (line 245)
        ntri_301360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 51), 'ntri', False)
        # Processing the call keyword arguments (line 245)
        # Getting the type of 'np' (line 245)
        np_301361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 63), 'np', False)
        # Obtaining the member 'int32' of a type (line 245)
        int32_301362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 63), np_301361, 'int32')
        keyword_301363 = int32_301362
        kwargs_301364 = {'dtype': keyword_301363}
        # Getting the type of 'np' (line 245)
        np_301358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'np', False)
        # Obtaining the member 'arange' of a type (line 245)
        arange_301359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 41), np_301358, 'arange')
        # Calling arange(args, kwargs) (line 245)
        arange_call_result_301365 = invoke(stypy.reporting.localization.Localization(__file__, 245, 41), arange_301359, *[ntri_301360], **kwargs_301364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 40), list_301357, arange_call_result_301365)
        # Adding element type (line 245)
        
        # Call to arange(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'ntri' (line 246)
        ntri_301368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 51), 'ntri', False)
        # Processing the call keyword arguments (line 246)
        # Getting the type of 'np' (line 246)
        np_301369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 63), 'np', False)
        # Obtaining the member 'int32' of a type (line 246)
        int32_301370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 63), np_301369, 'int32')
        keyword_301371 = int32_301370
        kwargs_301372 = {'dtype': keyword_301371}
        # Getting the type of 'np' (line 246)
        np_301366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 41), 'np', False)
        # Obtaining the member 'arange' of a type (line 246)
        arange_301367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 41), np_301366, 'arange')
        # Calling arange(args, kwargs) (line 246)
        arange_call_result_301373 = invoke(stypy.reporting.localization.Localization(__file__, 246, 41), arange_301367, *[ntri_301368], **kwargs_301372)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 40), list_301357, arange_call_result_301373)
        # Adding element type (line 245)
        
        # Call to arange(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'ntri' (line 247)
        ntri_301376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'ntri', False)
        # Processing the call keyword arguments (line 247)
        # Getting the type of 'np' (line 247)
        np_301377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 63), 'np', False)
        # Obtaining the member 'int32' of a type (line 247)
        int32_301378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 63), np_301377, 'int32')
        keyword_301379 = int32_301378
        kwargs_301380 = {'dtype': keyword_301379}
        # Getting the type of 'np' (line 247)
        np_301374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 41), 'np', False)
        # Obtaining the member 'arange' of a type (line 247)
        arange_301375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 41), np_301374, 'arange')
        # Calling arange(args, kwargs) (line 247)
        arange_call_result_301381 = invoke(stypy.reporting.localization.Localization(__file__, 247, 41), arange_301375, *[ntri_301376], **kwargs_301380)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 40), list_301357, arange_call_result_301381)
        
        # Processing the call keyword arguments (line 245)
        kwargs_301382 = {}
        # Getting the type of 'np' (line 245)
        np_301355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'np', False)
        # Obtaining the member 'vstack' of a type (line 245)
        vstack_301356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 30), np_301355, 'vstack')
        # Calling vstack(args, kwargs) (line 245)
        vstack_call_result_301383 = invoke(stypy.reporting.localization.Localization(__file__, 245, 30), vstack_301356, *[list_301357], **kwargs_301382)
        
        # Processing the call keyword arguments (line 245)
        kwargs_301384 = {}
        # Getting the type of 'np' (line 245)
        np_301353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'np', False)
        # Obtaining the member 'ravel' of a type (line 245)
        ravel_301354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), np_301353, 'ravel')
        # Calling ravel(args, kwargs) (line 245)
        ravel_call_result_301385 = invoke(stypy.reporting.localization.Localization(__file__, 245, 21), ravel_301354, *[vstack_call_result_301383], **kwargs_301384)
        
        # Assigning a type to the variable 'edge_elems' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'edge_elems', ravel_call_result_301385)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to ravel(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Call to vstack(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_301390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        
        # Call to zeros(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'ntri' (line 248)
        ntri_301393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 51), 'ntri', False)
        # Processing the call keyword arguments (line 248)
        # Getting the type of 'np' (line 248)
        np_301394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 63), 'np', False)
        # Obtaining the member 'int32' of a type (line 248)
        int32_301395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 63), np_301394, 'int32')
        keyword_301396 = int32_301395
        kwargs_301397 = {'dtype': keyword_301396}
        # Getting the type of 'np' (line 248)
        np_301391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'np', False)
        # Obtaining the member 'zeros' of a type (line 248)
        zeros_301392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 42), np_301391, 'zeros')
        # Calling zeros(args, kwargs) (line 248)
        zeros_call_result_301398 = invoke(stypy.reporting.localization.Localization(__file__, 248, 42), zeros_301392, *[ntri_301393], **kwargs_301397)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 41), list_301390, zeros_call_result_301398)
        # Adding element type (line 248)
        
        # Call to ones(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'ntri' (line 249)
        ntri_301401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 50), 'ntri', False)
        # Processing the call keyword arguments (line 249)
        # Getting the type of 'np' (line 249)
        np_301402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 62), 'np', False)
        # Obtaining the member 'int32' of a type (line 249)
        int32_301403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 62), np_301402, 'int32')
        keyword_301404 = int32_301403
        kwargs_301405 = {'dtype': keyword_301404}
        # Getting the type of 'np' (line 249)
        np_301399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'np', False)
        # Obtaining the member 'ones' of a type (line 249)
        ones_301400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 42), np_301399, 'ones')
        # Calling ones(args, kwargs) (line 249)
        ones_call_result_301406 = invoke(stypy.reporting.localization.Localization(__file__, 249, 42), ones_301400, *[ntri_301401], **kwargs_301405)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 41), list_301390, ones_call_result_301406)
        # Adding element type (line 248)
        
        # Call to ones(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'ntri' (line 250)
        ntri_301409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 50), 'ntri', False)
        # Processing the call keyword arguments (line 250)
        # Getting the type of 'np' (line 250)
        np_301410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 62), 'np', False)
        # Obtaining the member 'int32' of a type (line 250)
        int32_301411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 62), np_301410, 'int32')
        keyword_301412 = int32_301411
        kwargs_301413 = {'dtype': keyword_301412}
        # Getting the type of 'np' (line 250)
        np_301407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 42), 'np', False)
        # Obtaining the member 'ones' of a type (line 250)
        ones_301408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 42), np_301407, 'ones')
        # Calling ones(args, kwargs) (line 250)
        ones_call_result_301414 = invoke(stypy.reporting.localization.Localization(__file__, 250, 42), ones_301408, *[ntri_301409], **kwargs_301413)
        
        int_301415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 72), 'int')
        # Applying the binary operator '*' (line 250)
        result_mul_301416 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 42), '*', ones_call_result_301414, int_301415)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 41), list_301390, result_mul_301416)
        
        # Processing the call keyword arguments (line 248)
        kwargs_301417 = {}
        # Getting the type of 'np' (line 248)
        np_301388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 'np', False)
        # Obtaining the member 'vstack' of a type (line 248)
        vstack_301389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 31), np_301388, 'vstack')
        # Calling vstack(args, kwargs) (line 248)
        vstack_call_result_301418 = invoke(stypy.reporting.localization.Localization(__file__, 248, 31), vstack_301389, *[list_301390], **kwargs_301417)
        
        # Processing the call keyword arguments (line 248)
        kwargs_301419 = {}
        # Getting the type of 'np' (line 248)
        np_301386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'np', False)
        # Obtaining the member 'ravel' of a type (line 248)
        ravel_301387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 22), np_301386, 'ravel')
        # Calling ravel(args, kwargs) (line 248)
        ravel_call_result_301420 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), ravel_301387, *[vstack_call_result_301418], **kwargs_301419)
        
        # Assigning a type to the variable 'edge_apexes' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'edge_apexes', ravel_call_result_301420)
        
        # Assigning a Subscript to a Name (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 251)
        tuple_301421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 251)
        # Adding element type (line 251)
        # Getting the type of 'edge_elems' (line 251)
        edge_elems_301422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 35), 'edge_elems')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 35), tuple_301421, edge_elems_301422)
        # Adding element type (line 251)
        # Getting the type of 'edge_apexes' (line 251)
        edge_apexes_301423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 47), 'edge_apexes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 35), tuple_301421, edge_apexes_301423)
        
        # Getting the type of 'neighbors' (line 251)
        neighbors_301424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 25), 'neighbors')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___301425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 25), neighbors_301424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_301426 = invoke(stypy.reporting.localization.Localization(__file__, 251, 25), getitem___301425, tuple_301421)
        
        # Assigning a type to the variable 'edge_neighbors' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'edge_neighbors', subscript_call_result_301426)
        
        # Assigning a Compare to a Name (line 252):
        
        # Assigning a Compare to a Name (line 252):
        
        # Getting the type of 'edge_elems' (line 252)
        edge_elems_301427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'edge_elems')
        # Getting the type of 'edge_neighbors' (line 252)
        edge_neighbors_301428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 37), 'edge_neighbors')
        # Applying the binary operator '>' (line 252)
        result_gt_301429 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 24), '>', edge_elems_301427, edge_neighbors_301428)
        
        # Assigning a type to the variable 'mask_masters' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'mask_masters', result_gt_301429)
        
        # Assigning a Subscript to a Name (line 255):
        
        # Assigning a Subscript to a Name (line 255):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask_masters' (line 255)
        mask_masters_301430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 29), 'mask_masters')
        # Getting the type of 'edge_elems' (line 255)
        edge_elems_301431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 18), 'edge_elems')
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___301432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 18), edge_elems_301431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_301433 = invoke(stypy.reporting.localization.Localization(__file__, 255, 18), getitem___301432, mask_masters_301430)
        
        # Assigning a type to the variable 'masters' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'masters', subscript_call_result_301433)
        
        # Assigning a Subscript to a Name (line 256):
        
        # Assigning a Subscript to a Name (line 256):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask_masters' (line 256)
        mask_masters_301434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'mask_masters')
        # Getting the type of 'edge_apexes' (line 256)
        edge_apexes_301435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'edge_apexes')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___301436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 23), edge_apexes_301435, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_301437 = invoke(stypy.reporting.localization.Localization(__file__, 256, 23), getitem___301436, mask_masters_301434)
        
        # Assigning a type to the variable 'apex_masters' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'apex_masters', subscript_call_result_301437)
        
        # Assigning a BinOp to a Name (line 257):
        
        # Assigning a BinOp to a Name (line 257):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 257)
        tuple_301438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 257)
        # Adding element type (line 257)
        # Getting the type of 'masters' (line 257)
        masters_301439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 29), 'masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 29), tuple_301438, masters_301439)
        # Adding element type (line 257)
        # Getting the type of 'apex_masters' (line 257)
        apex_masters_301440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'apex_masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 29), tuple_301438, apex_masters_301440)
        
        # Getting the type of 'triangles' (line 257)
        triangles_301441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 'triangles')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___301442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 19), triangles_301441, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_301443 = invoke(stypy.reporting.localization.Localization(__file__, 257, 19), getitem___301442, tuple_301438)
        
        # Getting the type of 'x' (line 257)
        x_301444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'x')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___301445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 17), x_301444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_301446 = invoke(stypy.reporting.localization.Localization(__file__, 257, 17), getitem___301445, subscript_call_result_301443)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 258)
        tuple_301447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 258)
        # Adding element type (line 258)
        # Getting the type of 'masters' (line 258)
        masters_301448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 29), tuple_301447, masters_301448)
        # Adding element type (line 258)
        # Getting the type of 'apex_masters' (line 258)
        apex_masters_301449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 39), 'apex_masters')
        int_301450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 52), 'int')
        # Applying the binary operator '+' (line 258)
        result_add_301451 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 39), '+', apex_masters_301449, int_301450)
        
        int_301452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 57), 'int')
        # Applying the binary operator '%' (line 258)
        result_mod_301453 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 38), '%', result_add_301451, int_301452)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 29), tuple_301447, result_mod_301453)
        
        # Getting the type of 'triangles' (line 258)
        triangles_301454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'triangles')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___301455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), triangles_301454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_301456 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), getitem___301455, tuple_301447)
        
        # Getting the type of 'x' (line 258)
        x_301457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'x')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___301458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 17), x_301457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_301459 = invoke(stypy.reporting.localization.Localization(__file__, 258, 17), getitem___301458, subscript_call_result_301456)
        
        # Applying the binary operator '+' (line 257)
        result_add_301460 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 17), '+', subscript_call_result_301446, subscript_call_result_301459)
        
        float_301461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 64), 'float')
        # Applying the binary operator '*' (line 257)
        result_mul_301462 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 16), '*', result_add_301460, float_301461)
        
        # Assigning a type to the variable 'x_add' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'x_add', result_mul_301462)
        
        # Assigning a BinOp to a Name (line 259):
        
        # Assigning a BinOp to a Name (line 259):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_301463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        # Getting the type of 'masters' (line 259)
        masters_301464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 29), 'masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 29), tuple_301463, masters_301464)
        # Adding element type (line 259)
        # Getting the type of 'apex_masters' (line 259)
        apex_masters_301465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 38), 'apex_masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 29), tuple_301463, apex_masters_301465)
        
        # Getting the type of 'triangles' (line 259)
        triangles_301466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 19), 'triangles')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___301467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 19), triangles_301466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_301468 = invoke(stypy.reporting.localization.Localization(__file__, 259, 19), getitem___301467, tuple_301463)
        
        # Getting the type of 'y' (line 259)
        y_301469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 17), 'y')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___301470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 17), y_301469, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_301471 = invoke(stypy.reporting.localization.Localization(__file__, 259, 17), getitem___301470, subscript_call_result_301468)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 260)
        tuple_301472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 260)
        # Adding element type (line 260)
        # Getting the type of 'masters' (line 260)
        masters_301473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 29), tuple_301472, masters_301473)
        # Adding element type (line 260)
        # Getting the type of 'apex_masters' (line 260)
        apex_masters_301474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'apex_masters')
        int_301475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 52), 'int')
        # Applying the binary operator '+' (line 260)
        result_add_301476 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 39), '+', apex_masters_301474, int_301475)
        
        int_301477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 57), 'int')
        # Applying the binary operator '%' (line 260)
        result_mod_301478 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 38), '%', result_add_301476, int_301477)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 29), tuple_301472, result_mod_301478)
        
        # Getting the type of 'triangles' (line 260)
        triangles_301479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'triangles')
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___301480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 19), triangles_301479, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_301481 = invoke(stypy.reporting.localization.Localization(__file__, 260, 19), getitem___301480, tuple_301472)
        
        # Getting the type of 'y' (line 260)
        y_301482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'y')
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___301483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 17), y_301482, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_301484 = invoke(stypy.reporting.localization.Localization(__file__, 260, 17), getitem___301483, subscript_call_result_301481)
        
        # Applying the binary operator '+' (line 259)
        result_add_301485 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 17), '+', subscript_call_result_301471, subscript_call_result_301484)
        
        float_301486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 64), 'float')
        # Applying the binary operator '*' (line 259)
        result_mul_301487 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 16), '*', result_add_301485, float_301486)
        
        # Assigning a type to the variable 'y_add' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'y_add', result_mul_301487)
        
        # Assigning a Name to a Subscript (line 261):
        
        # Assigning a Name to a Subscript (line 261):
        # Getting the type of 'x_add' (line 261)
        x_add_301488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'x_add')
        # Getting the type of 'refi_x' (line 261)
        refi_x_301489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'refi_x')
        # Getting the type of 'npts' (line 261)
        npts_301490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'npts')
        slice_301491 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 261, 8), npts_301490, None, None)
        # Storing an element on a container (line 261)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), refi_x_301489, (slice_301491, x_add_301488))
        
        # Assigning a Name to a Subscript (line 262):
        
        # Assigning a Name to a Subscript (line 262):
        # Getting the type of 'y_add' (line 262)
        y_add_301492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'y_add')
        # Getting the type of 'refi_y' (line 262)
        refi_y_301493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'refi_y')
        # Getting the type of 'npts' (line 262)
        npts_301494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'npts')
        slice_301495 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 262, 8), npts_301494, None, None)
        # Storing an element on a container (line 262)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), refi_y_301493, (slice_301495, y_add_301492))
        
        # Assigning a Name to a Name (line 268):
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'triangles' (line 268)
        triangles_301496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'triangles')
        # Assigning a type to the variable 'new_pt_corner' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'new_pt_corner', triangles_301496)
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to empty(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_301499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        # Getting the type of 'ntri' (line 276)
        ntri_301500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'ntri', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 34), list_301499, ntri_301500)
        # Adding element type (line 276)
        int_301501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 34), list_301499, int_301501)
        
        # Processing the call keyword arguments (line 276)
        # Getting the type of 'np' (line 276)
        np_301502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 51), 'np', False)
        # Obtaining the member 'int32' of a type (line 276)
        int32_301503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 51), np_301502, 'int32')
        keyword_301504 = int32_301503
        kwargs_301505 = {'dtype': keyword_301504}
        # Getting the type of 'np' (line 276)
        np_301497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 25), 'np', False)
        # Obtaining the member 'empty' of a type (line 276)
        empty_301498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 25), np_301497, 'empty')
        # Calling empty(args, kwargs) (line 276)
        empty_call_result_301506 = invoke(stypy.reporting.localization.Localization(__file__, 276, 25), empty_301498, *[list_301499], **kwargs_301505)
        
        # Assigning a type to the variable 'new_pt_midside' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'new_pt_midside', empty_call_result_301506)
        
        # Assigning a Name to a Name (line 277):
        
        # Assigning a Name to a Name (line 277):
        # Getting the type of 'npts' (line 277)
        npts_301507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'npts')
        # Assigning a type to the variable 'cum_sum' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'cum_sum', npts_301507)
        
        
        # Call to range(...): (line 278)
        # Processing the call arguments (line 278)
        int_301509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 26), 'int')
        # Processing the call keyword arguments (line 278)
        kwargs_301510 = {}
        # Getting the type of 'range' (line 278)
        range_301508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'range', False)
        # Calling range(args, kwargs) (line 278)
        range_call_result_301511 = invoke(stypy.reporting.localization.Localization(__file__, 278, 20), range_301508, *[int_301509], **kwargs_301510)
        
        # Testing the type of a for loop iterable (line 278)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 278, 8), range_call_result_301511)
        # Getting the type of the for loop variable (line 278)
        for_loop_var_301512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 278, 8), range_call_result_301511)
        # Assigning a type to the variable 'imid' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'imid', for_loop_var_301512)
        # SSA begins for a for statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Compare to a Name (line 279):
        
        # Assigning a Compare to a Name (line 279):
        
        # Getting the type of 'imid' (line 279)
        imid_301513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 27), 'imid')
        # Getting the type of 'apex_masters' (line 279)
        apex_masters_301514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'apex_masters')
        # Applying the binary operator '==' (line 279)
        result_eq_301515 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 27), '==', imid_301513, apex_masters_301514)
        
        # Assigning a type to the variable 'mask_st_loc' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'mask_st_loc', result_eq_301515)
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to sum(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'mask_st_loc' (line 280)
        mask_st_loc_301518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'mask_st_loc', False)
        # Processing the call keyword arguments (line 280)
        kwargs_301519 = {}
        # Getting the type of 'np' (line 280)
        np_301516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'np', False)
        # Obtaining the member 'sum' of a type (line 280)
        sum_301517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 28), np_301516, 'sum')
        # Calling sum(args, kwargs) (line 280)
        sum_call_result_301520 = invoke(stypy.reporting.localization.Localization(__file__, 280, 28), sum_301517, *[mask_st_loc_301518], **kwargs_301519)
        
        # Assigning a type to the variable 'n_masters_loc' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'n_masters_loc', sum_call_result_301520)
        
        # Assigning a Subscript to a Name (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask_st_loc' (line 281)
        mask_st_loc_301521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 39), 'mask_st_loc')
        # Getting the type of 'masters' (line 281)
        masters_301522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'masters')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___301523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 31), masters_301522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_301524 = invoke(stypy.reporting.localization.Localization(__file__, 281, 31), getitem___301523, mask_st_loc_301521)
        
        # Assigning a type to the variable 'elem_masters_loc' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'elem_masters_loc', subscript_call_result_301524)
        
        # Assigning a BinOp to a Subscript (line 282):
        
        # Assigning a BinOp to a Subscript (line 282):
        
        # Call to arange(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'n_masters_loc' (line 283)
        n_masters_loc_301527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'n_masters_loc', False)
        # Processing the call keyword arguments (line 282)
        # Getting the type of 'np' (line 283)
        np_301528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 37), 'np', False)
        # Obtaining the member 'int32' of a type (line 283)
        int32_301529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 37), np_301528, 'int32')
        keyword_301530 = int32_301529
        kwargs_301531 = {'dtype': keyword_301530}
        # Getting the type of 'np' (line 282)
        np_301525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 56), 'np', False)
        # Obtaining the member 'arange' of a type (line 282)
        arange_301526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 56), np_301525, 'arange')
        # Calling arange(args, kwargs) (line 282)
        arange_call_result_301532 = invoke(stypy.reporting.localization.Localization(__file__, 282, 56), arange_301526, *[n_masters_loc_301527], **kwargs_301531)
        
        # Getting the type of 'cum_sum' (line 283)
        cum_sum_301533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 49), 'cum_sum')
        # Applying the binary operator '+' (line 282)
        result_add_301534 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 56), '+', arange_call_result_301532, cum_sum_301533)
        
        
        # Obtaining the type of the subscript
        slice_301535 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 282, 12), None, None, None)
        # Getting the type of 'imid' (line 282)
        imid_301536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 30), 'imid')
        # Getting the type of 'new_pt_midside' (line 282)
        new_pt_midside_301537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'new_pt_midside')
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___301538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), new_pt_midside_301537, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_301539 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), getitem___301538, (slice_301535, imid_301536))
        
        # Getting the type of 'elem_masters_loc' (line 282)
        elem_masters_loc_301540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'elem_masters_loc')
        # Storing an element on a container (line 282)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 12), subscript_call_result_301539, (elem_masters_loc_301540, result_add_301534))
        
        # Getting the type of 'cum_sum' (line 284)
        cum_sum_301541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'cum_sum')
        # Getting the type of 'n_masters_loc' (line 284)
        n_masters_loc_301542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'n_masters_loc')
        # Applying the binary operator '+=' (line 284)
        result_iadd_301543 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 12), '+=', cum_sum_301541, n_masters_loc_301542)
        # Assigning a type to the variable 'cum_sum' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'cum_sum', result_iadd_301543)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to logical_not(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'mask_masters' (line 290)
        mask_masters_301546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 37), 'mask_masters', False)
        # Processing the call keyword arguments (line 290)
        kwargs_301547 = {}
        # Getting the type of 'np' (line 290)
        np_301544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'np', False)
        # Obtaining the member 'logical_not' of a type (line 290)
        logical_not_301545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 22), np_301544, 'logical_not')
        # Calling logical_not(args, kwargs) (line 290)
        logical_not_call_result_301548 = invoke(stypy.reporting.localization.Localization(__file__, 290, 22), logical_not_301545, *[mask_masters_301546], **kwargs_301547)
        
        # Assigning a type to the variable 'mask_slaves' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'mask_slaves', logical_not_call_result_301548)
        
        # Assigning a Subscript to a Name (line 291):
        
        # Assigning a Subscript to a Name (line 291):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask_slaves' (line 291)
        mask_slaves_301549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'mask_slaves')
        # Getting the type of 'edge_elems' (line 291)
        edge_elems_301550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'edge_elems')
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___301551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 17), edge_elems_301550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_301552 = invoke(stypy.reporting.localization.Localization(__file__, 291, 17), getitem___301551, mask_slaves_301549)
        
        # Assigning a type to the variable 'slaves' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'slaves', subscript_call_result_301552)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask_slaves' (line 292)
        mask_slaves_301553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 40), 'mask_slaves')
        # Getting the type of 'edge_neighbors' (line 292)
        edge_neighbors_301554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'edge_neighbors')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___301555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 25), edge_neighbors_301554, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_301556 = invoke(stypy.reporting.localization.Localization(__file__, 292, 25), getitem___301555, mask_slaves_301553)
        
        # Assigning a type to the variable 'slaves_masters' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'slaves_masters', subscript_call_result_301556)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to abs(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Obtaining the type of the subscript
        # Getting the type of 'slaves_masters' (line 293)
        slaves_masters_301559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 38), 'slaves_masters', False)
        slice_301560 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 293, 28), None, None, None)
        # Getting the type of 'neighbors' (line 293)
        neighbors_301561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 28), 'neighbors', False)
        # Obtaining the member '__getitem__' of a type (line 293)
        getitem___301562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 28), neighbors_301561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 293)
        subscript_call_result_301563 = invoke(stypy.reporting.localization.Localization(__file__, 293, 28), getitem___301562, (slaves_masters_301559, slice_301560))
        
        
        # Call to outer(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'slaves' (line 294)
        slaves_301566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 37), 'slaves', False)
        
        # Call to ones(...): (line 294)
        # Processing the call arguments (line 294)
        int_301569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 53), 'int')
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'np' (line 294)
        np_301570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 62), 'np', False)
        # Obtaining the member 'int32' of a type (line 294)
        int32_301571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 62), np_301570, 'int32')
        keyword_301572 = int32_301571
        kwargs_301573 = {'dtype': keyword_301572}
        # Getting the type of 'np' (line 294)
        np_301567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 45), 'np', False)
        # Obtaining the member 'ones' of a type (line 294)
        ones_301568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 45), np_301567, 'ones')
        # Calling ones(args, kwargs) (line 294)
        ones_call_result_301574 = invoke(stypy.reporting.localization.Localization(__file__, 294, 45), ones_301568, *[int_301569], **kwargs_301573)
        
        # Processing the call keyword arguments (line 294)
        kwargs_301575 = {}
        # Getting the type of 'np' (line 294)
        np_301564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 28), 'np', False)
        # Obtaining the member 'outer' of a type (line 294)
        outer_301565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 28), np_301564, 'outer')
        # Calling outer(args, kwargs) (line 294)
        outer_call_result_301576 = invoke(stypy.reporting.localization.Localization(__file__, 294, 28), outer_301565, *[slaves_301566, ones_call_result_301574], **kwargs_301575)
        
        # Applying the binary operator '-' (line 293)
        result_sub_301577 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 28), '-', subscript_call_result_301563, outer_call_result_301576)
        
        # Processing the call keyword arguments (line 293)
        kwargs_301578 = {}
        # Getting the type of 'np' (line 293)
        np_301557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'np', False)
        # Obtaining the member 'abs' of a type (line 293)
        abs_301558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), np_301557, 'abs')
        # Calling abs(args, kwargs) (line 293)
        abs_call_result_301579 = invoke(stypy.reporting.localization.Localization(__file__, 293, 21), abs_301558, *[result_sub_301577], **kwargs_301578)
        
        # Assigning a type to the variable 'diff_table' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'diff_table', abs_call_result_301579)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to argmin(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'diff_table' (line 295)
        diff_table_301582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'diff_table', False)
        # Processing the call keyword arguments (line 295)
        int_301583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 56), 'int')
        keyword_301584 = int_301583
        kwargs_301585 = {'axis': keyword_301584}
        # Getting the type of 'np' (line 295)
        np_301580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'np', False)
        # Obtaining the member 'argmin' of a type (line 295)
        argmin_301581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 29), np_301580, 'argmin')
        # Calling argmin(args, kwargs) (line 295)
        argmin_call_result_301586 = invoke(stypy.reporting.localization.Localization(__file__, 295, 29), argmin_301581, *[diff_table_301582], **kwargs_301585)
        
        # Assigning a type to the variable 'slave_masters_apex' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'slave_masters_apex', argmin_call_result_301586)
        
        # Assigning a Subscript to a Name (line 296):
        
        # Assigning a Subscript to a Name (line 296):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask_slaves' (line 296)
        mask_slaves_301587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'mask_slaves')
        # Getting the type of 'edge_apexes' (line 296)
        edge_apexes_301588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'edge_apexes')
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___301589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 22), edge_apexes_301588, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_301590 = invoke(stypy.reporting.localization.Localization(__file__, 296, 22), getitem___301589, mask_slaves_301587)
        
        # Assigning a type to the variable 'slaves_apex' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'slaves_apex', subscript_call_result_301590)
        
        # Assigning a Subscript to a Subscript (line 297):
        
        # Assigning a Subscript to a Subscript (line 297):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_301591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 'slaves_masters' (line 298)
        slaves_masters_301592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'slaves_masters')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 12), tuple_301591, slaves_masters_301592)
        # Adding element type (line 298)
        # Getting the type of 'slave_masters_apex' (line 298)
        slave_masters_apex_301593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 28), 'slave_masters_apex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 12), tuple_301591, slave_masters_apex_301593)
        
        # Getting the type of 'new_pt_midside' (line 297)
        new_pt_midside_301594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 46), 'new_pt_midside')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___301595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 46), new_pt_midside_301594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_301596 = invoke(stypy.reporting.localization.Localization(__file__, 297, 46), getitem___301595, tuple_301591)
        
        # Getting the type of 'new_pt_midside' (line 297)
        new_pt_midside_301597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'new_pt_midside')
        
        # Obtaining an instance of the builtin type 'tuple' (line 297)
        tuple_301598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 297)
        # Adding element type (line 297)
        # Getting the type of 'slaves' (line 297)
        slaves_301599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'slaves')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 23), tuple_301598, slaves_301599)
        # Adding element type (line 297)
        # Getting the type of 'slaves_apex' (line 297)
        slaves_apex_301600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 'slaves_apex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 23), tuple_301598, slaves_apex_301600)
        
        # Storing an element on a container (line 297)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 8), new_pt_midside_301597, (tuple_301598, subscript_call_result_301596))
        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to empty(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_301603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        # Getting the type of 'ntri' (line 301)
        ntri_301604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'ntri', False)
        int_301605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 41), 'int')
        # Applying the binary operator '*' (line 301)
        result_mul_301606 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 36), '*', ntri_301604, int_301605)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 35), list_301603, result_mul_301606)
        # Adding element type (line 301)
        int_301607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 35), list_301603, int_301607)
        
        # Processing the call keyword arguments (line 301)
        # Getting the type of 'np' (line 301)
        np_301608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 54), 'np', False)
        # Obtaining the member 'int32' of a type (line 301)
        int32_301609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 54), np_301608, 'int32')
        keyword_301610 = int32_301609
        kwargs_301611 = {'dtype': keyword_301610}
        # Getting the type of 'np' (line 301)
        np_301601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 26), 'np', False)
        # Obtaining the member 'empty' of a type (line 301)
        empty_301602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 26), np_301601, 'empty')
        # Calling empty(args, kwargs) (line 301)
        empty_call_result_301612 = invoke(stypy.reporting.localization.Localization(__file__, 301, 26), empty_301602, *[list_301603], **kwargs_301611)
        
        # Assigning a type to the variable 'child_triangles' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'child_triangles', empty_call_result_301612)
        
        # Assigning a Attribute to a Subscript (line 302):
        
        # Assigning a Attribute to a Subscript (line 302):
        
        # Call to vstack(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_301615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        
        # Obtaining the type of the subscript
        slice_301616 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 12), None, None, None)
        int_301617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'int')
        # Getting the type of 'new_pt_corner' (line 303)
        new_pt_corner_301618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'new_pt_corner', False)
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___301619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), new_pt_corner_301618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_301620 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), getitem___301619, (slice_301616, int_301617))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 45), list_301615, subscript_call_result_301620)
        # Adding element type (line 302)
        
        # Obtaining the type of the subscript
        slice_301621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 33), None, None, None)
        int_301622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 51), 'int')
        # Getting the type of 'new_pt_midside' (line 303)
        new_pt_midside_301623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 33), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___301624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 33), new_pt_midside_301623, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_301625 = invoke(stypy.reporting.localization.Localization(__file__, 303, 33), getitem___301624, (slice_301621, int_301622))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 45), list_301615, subscript_call_result_301625)
        # Adding element type (line 302)
        
        # Obtaining the type of the subscript
        slice_301626 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 12), None, None, None)
        int_301627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 30), 'int')
        # Getting the type of 'new_pt_midside' (line 304)
        new_pt_midside_301628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___301629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), new_pt_midside_301628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_301630 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), getitem___301629, (slice_301626, int_301627))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 45), list_301615, subscript_call_result_301630)
        
        # Processing the call keyword arguments (line 302)
        kwargs_301631 = {}
        # Getting the type of 'np' (line 302)
        np_301613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 35), 'np', False)
        # Obtaining the member 'vstack' of a type (line 302)
        vstack_301614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 35), np_301613, 'vstack')
        # Calling vstack(args, kwargs) (line 302)
        vstack_call_result_301632 = invoke(stypy.reporting.localization.Localization(__file__, 302, 35), vstack_301614, *[list_301615], **kwargs_301631)
        
        # Obtaining the member 'T' of a type (line 302)
        T_301633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 35), vstack_call_result_301632, 'T')
        # Getting the type of 'child_triangles' (line 302)
        child_triangles_301634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'child_triangles')
        int_301635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 24), 'int')
        int_301636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 27), 'int')
        slice_301637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 302, 8), int_301635, None, int_301636)
        slice_301638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 302, 8), None, None, None)
        # Storing an element on a container (line 302)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 8), child_triangles_301634, ((slice_301637, slice_301638), T_301633))
        
        # Assigning a Attribute to a Subscript (line 305):
        
        # Assigning a Attribute to a Subscript (line 305):
        
        # Call to vstack(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_301641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        
        # Obtaining the type of the subscript
        slice_301642 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 306, 12), None, None, None)
        int_301643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 29), 'int')
        # Getting the type of 'new_pt_corner' (line 306)
        new_pt_corner_301644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'new_pt_corner', False)
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___301645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), new_pt_corner_301644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_301646 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___301645, (slice_301642, int_301643))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 45), list_301641, subscript_call_result_301646)
        # Adding element type (line 305)
        
        # Obtaining the type of the subscript
        slice_301647 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 306, 33), None, None, None)
        int_301648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 51), 'int')
        # Getting the type of 'new_pt_midside' (line 306)
        new_pt_midside_301649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___301650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 33), new_pt_midside_301649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_301651 = invoke(stypy.reporting.localization.Localization(__file__, 306, 33), getitem___301650, (slice_301647, int_301648))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 45), list_301641, subscript_call_result_301651)
        # Adding element type (line 305)
        
        # Obtaining the type of the subscript
        slice_301652 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 307, 12), None, None, None)
        int_301653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'int')
        # Getting the type of 'new_pt_midside' (line 307)
        new_pt_midside_301654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___301655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), new_pt_midside_301654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_301656 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), getitem___301655, (slice_301652, int_301653))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 45), list_301641, subscript_call_result_301656)
        
        # Processing the call keyword arguments (line 305)
        kwargs_301657 = {}
        # Getting the type of 'np' (line 305)
        np_301639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 35), 'np', False)
        # Obtaining the member 'vstack' of a type (line 305)
        vstack_301640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 35), np_301639, 'vstack')
        # Calling vstack(args, kwargs) (line 305)
        vstack_call_result_301658 = invoke(stypy.reporting.localization.Localization(__file__, 305, 35), vstack_301640, *[list_301641], **kwargs_301657)
        
        # Obtaining the member 'T' of a type (line 305)
        T_301659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 35), vstack_call_result_301658, 'T')
        # Getting the type of 'child_triangles' (line 305)
        child_triangles_301660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'child_triangles')
        int_301661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 24), 'int')
        int_301662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 27), 'int')
        slice_301663 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 305, 8), int_301661, None, int_301662)
        slice_301664 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 305, 8), None, None, None)
        # Storing an element on a container (line 305)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 8), child_triangles_301660, ((slice_301663, slice_301664), T_301659))
        
        # Assigning a Attribute to a Subscript (line 308):
        
        # Assigning a Attribute to a Subscript (line 308):
        
        # Call to vstack(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_301667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        # Adding element type (line 308)
        
        # Obtaining the type of the subscript
        slice_301668 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 309, 12), None, None, None)
        int_301669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'int')
        # Getting the type of 'new_pt_corner' (line 309)
        new_pt_corner_301670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'new_pt_corner', False)
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___301671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), new_pt_corner_301670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_301672 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), getitem___301671, (slice_301668, int_301669))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 45), list_301667, subscript_call_result_301672)
        # Adding element type (line 308)
        
        # Obtaining the type of the subscript
        slice_301673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 309, 33), None, None, None)
        int_301674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 51), 'int')
        # Getting the type of 'new_pt_midside' (line 309)
        new_pt_midside_301675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___301676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 33), new_pt_midside_301675, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_301677 = invoke(stypy.reporting.localization.Localization(__file__, 309, 33), getitem___301676, (slice_301673, int_301674))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 45), list_301667, subscript_call_result_301677)
        # Adding element type (line 308)
        
        # Obtaining the type of the subscript
        slice_301678 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 310, 12), None, None, None)
        int_301679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 30), 'int')
        # Getting the type of 'new_pt_midside' (line 310)
        new_pt_midside_301680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___301681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), new_pt_midside_301680, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_301682 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), getitem___301681, (slice_301678, int_301679))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 45), list_301667, subscript_call_result_301682)
        
        # Processing the call keyword arguments (line 308)
        kwargs_301683 = {}
        # Getting the type of 'np' (line 308)
        np_301665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'np', False)
        # Obtaining the member 'vstack' of a type (line 308)
        vstack_301666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 35), np_301665, 'vstack')
        # Calling vstack(args, kwargs) (line 308)
        vstack_call_result_301684 = invoke(stypy.reporting.localization.Localization(__file__, 308, 35), vstack_301666, *[list_301667], **kwargs_301683)
        
        # Obtaining the member 'T' of a type (line 308)
        T_301685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 35), vstack_call_result_301684, 'T')
        # Getting the type of 'child_triangles' (line 308)
        child_triangles_301686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'child_triangles')
        int_301687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'int')
        int_301688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 27), 'int')
        slice_301689 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 308, 8), int_301687, None, int_301688)
        slice_301690 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 308, 8), None, None, None)
        # Storing an element on a container (line 308)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 8), child_triangles_301686, ((slice_301689, slice_301690), T_301685))
        
        # Assigning a Attribute to a Subscript (line 311):
        
        # Assigning a Attribute to a Subscript (line 311):
        
        # Call to vstack(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_301693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        
        # Obtaining the type of the subscript
        slice_301694 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 12), None, None, None)
        int_301695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 30), 'int')
        # Getting the type of 'new_pt_midside' (line 312)
        new_pt_midside_301696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___301697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), new_pt_midside_301696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_301698 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), getitem___301697, (slice_301694, int_301695))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 45), list_301693, subscript_call_result_301698)
        # Adding element type (line 311)
        
        # Obtaining the type of the subscript
        slice_301699 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 34), None, None, None)
        int_301700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 52), 'int')
        # Getting the type of 'new_pt_midside' (line 312)
        new_pt_midside_301701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___301702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 34), new_pt_midside_301701, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_301703 = invoke(stypy.reporting.localization.Localization(__file__, 312, 34), getitem___301702, (slice_301699, int_301700))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 45), list_301693, subscript_call_result_301703)
        # Adding element type (line 311)
        
        # Obtaining the type of the subscript
        slice_301704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 313, 12), None, None, None)
        int_301705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 30), 'int')
        # Getting the type of 'new_pt_midside' (line 313)
        new_pt_midside_301706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'new_pt_midside', False)
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___301707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), new_pt_midside_301706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_301708 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), getitem___301707, (slice_301704, int_301705))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 45), list_301693, subscript_call_result_301708)
        
        # Processing the call keyword arguments (line 311)
        kwargs_301709 = {}
        # Getting the type of 'np' (line 311)
        np_301691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 35), 'np', False)
        # Obtaining the member 'vstack' of a type (line 311)
        vstack_301692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 35), np_301691, 'vstack')
        # Calling vstack(args, kwargs) (line 311)
        vstack_call_result_301710 = invoke(stypy.reporting.localization.Localization(__file__, 311, 35), vstack_301692, *[list_301693], **kwargs_301709)
        
        # Obtaining the member 'T' of a type (line 311)
        T_301711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 35), vstack_call_result_301710, 'T')
        # Getting the type of 'child_triangles' (line 311)
        child_triangles_301712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'child_triangles')
        int_301713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 24), 'int')
        int_301714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 27), 'int')
        slice_301715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 8), int_301713, None, int_301714)
        slice_301716 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 8), None, None, None)
        # Storing an element on a container (line 311)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), child_triangles_301712, ((slice_301715, slice_301716), T_301711))
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to Triangulation(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'refi_x' (line 314)
        refi_x_301718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 44), 'refi_x', False)
        # Getting the type of 'refi_y' (line 314)
        refi_y_301719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 52), 'refi_y', False)
        # Getting the type of 'child_triangles' (line 314)
        child_triangles_301720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 60), 'child_triangles', False)
        # Processing the call keyword arguments (line 314)
        kwargs_301721 = {}
        # Getting the type of 'Triangulation' (line 314)
        Triangulation_301717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 30), 'Triangulation', False)
        # Calling Triangulation(args, kwargs) (line 314)
        Triangulation_call_result_301722 = invoke(stypy.reporting.localization.Localization(__file__, 314, 30), Triangulation_301717, *[refi_x_301718, refi_y_301719, child_triangles_301720], **kwargs_301721)
        
        # Assigning a type to the variable 'child_triangulation' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'child_triangulation', Triangulation_call_result_301722)
        
        
        # Getting the type of 'triangulation' (line 317)
        triangulation_301723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'triangulation')
        # Obtaining the member 'mask' of a type (line 317)
        mask_301724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 11), triangulation_301723, 'mask')
        # Getting the type of 'None' (line 317)
        None_301725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 37), 'None')
        # Applying the binary operator 'isnot' (line 317)
        result_is_not_301726 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), 'isnot', mask_301724, None_301725)
        
        # Testing the type of an if condition (line 317)
        if_condition_301727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_is_not_301726)
        # Assigning a type to the variable 'if_condition_301727' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_301727', if_condition_301727)
        # SSA begins for if statement (line 317)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_mask(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Call to repeat(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'triangulation' (line 318)
        triangulation_301732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 51), 'triangulation', False)
        # Obtaining the member 'mask' of a type (line 318)
        mask_301733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 51), triangulation_301732, 'mask')
        int_301734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 71), 'int')
        # Processing the call keyword arguments (line 318)
        kwargs_301735 = {}
        # Getting the type of 'np' (line 318)
        np_301730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 41), 'np', False)
        # Obtaining the member 'repeat' of a type (line 318)
        repeat_301731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 41), np_301730, 'repeat')
        # Calling repeat(args, kwargs) (line 318)
        repeat_call_result_301736 = invoke(stypy.reporting.localization.Localization(__file__, 318, 41), repeat_301731, *[mask_301733, int_301734], **kwargs_301735)
        
        # Processing the call keyword arguments (line 318)
        kwargs_301737 = {}
        # Getting the type of 'child_triangulation' (line 318)
        child_triangulation_301728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'child_triangulation', False)
        # Obtaining the member 'set_mask' of a type (line 318)
        set_mask_301729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), child_triangulation_301728, 'set_mask')
        # Calling set_mask(args, kwargs) (line 318)
        set_mask_call_result_301738 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), set_mask_301729, *[repeat_call_result_301736], **kwargs_301737)
        
        # SSA join for if statement (line 317)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 320)
        # Getting the type of 'ancestors' (line 320)
        ancestors_301739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'ancestors')
        # Getting the type of 'None' (line 320)
        None_301740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'None')
        
        (may_be_301741, more_types_in_union_301742) = may_be_none(ancestors_301739, None_301740)

        if may_be_301741:

            if more_types_in_union_301742:
                # Runtime conditional SSA (line 320)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'child_triangulation' (line 321)
            child_triangulation_301743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'child_triangulation')
            # Assigning a type to the variable 'stypy_return_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'stypy_return_type', child_triangulation_301743)

            if more_types_in_union_301742:
                # Runtime conditional SSA for else branch (line 320)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_301741) or more_types_in_union_301742):
            
            # Obtaining an instance of the builtin type 'tuple' (line 323)
            tuple_301744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 323)
            # Adding element type (line 323)
            # Getting the type of 'child_triangulation' (line 323)
            child_triangulation_301745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'child_triangulation')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 19), tuple_301744, child_triangulation_301745)
            # Adding element type (line 323)
            
            # Call to repeat(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'ancestors' (line 323)
            ancestors_301748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 50), 'ancestors', False)
            int_301749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 61), 'int')
            # Processing the call keyword arguments (line 323)
            kwargs_301750 = {}
            # Getting the type of 'np' (line 323)
            np_301746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 40), 'np', False)
            # Obtaining the member 'repeat' of a type (line 323)
            repeat_301747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 40), np_301746, 'repeat')
            # Calling repeat(args, kwargs) (line 323)
            repeat_call_result_301751 = invoke(stypy.reporting.localization.Localization(__file__, 323, 40), repeat_301747, *[ancestors_301748, int_301749], **kwargs_301750)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 19), tuple_301744, repeat_call_result_301751)
            
            # Assigning a type to the variable 'stypy_return_type' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'stypy_return_type', tuple_301744)

            if (may_be_301741 and more_types_in_union_301742):
                # SSA join for if statement (line 320)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_refine_triangulation_once(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_refine_triangulation_once' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_301752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_301752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_refine_triangulation_once'
        return stypy_return_type_301752


# Assigning a type to the variable 'UniformTriRefiner' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'UniformTriRefiner', UniformTriRefiner)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

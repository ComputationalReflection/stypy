
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Spherical Voronoi Code
3: 
4: .. versionadded:: 0.18.0
5: 
6: '''
7: #
8: # Copyright (C)  Tyler Reddy, Ross Hemsley, Edd Edmondson,
9: #                Nikolai Nowaczyk, Joe Pitt-Francis, 2015.
10: #
11: # Distributed under the same BSD license as Scipy.
12: #
13: 
14: import numpy as np
15: import numpy.matlib
16: import scipy
17: import itertools
18: from . import _voronoi
19: from scipy.spatial.distance import pdist
20: 
21: __all__ = ['SphericalVoronoi']
22: 
23: def sphere_check(points, radius, center):
24:     ''' Determines distance of generators from theoretical sphere
25:     surface.
26: 
27:     '''
28:     actual_squared_radii = (((points[...,0] - center[0]) ** 2) +
29:                             ((points[...,1] - center[1]) ** 2) +
30:                             ((points[...,2] - center[2]) ** 2))
31:     max_discrepancy = (np.sqrt(actual_squared_radii) - radius).max()
32:     return abs(max_discrepancy)
33: 
34: def calc_circumcenters(tetrahedrons):
35:     ''' Calculates the cirumcenters of the circumspheres of tetrahedrons.
36: 
37:     An implementation based on
38:     http://mathworld.wolfram.com/Circumsphere.html
39: 
40:     Parameters
41:     ----------
42:     tetrahedrons : an array of shape (N, 4, 3)
43:         consisting of N tetrahedrons defined by 4 points in 3D
44: 
45:     Returns
46:     ----------
47:     circumcenters : an array of shape (N, 3)
48:         consisting of the N circumcenters of the tetrahedrons in 3D
49: 
50:     '''
51: 
52:     num = tetrahedrons.shape[0]
53:     a = np.concatenate((tetrahedrons, np.ones((num, 4, 1))), axis=2)
54: 
55:     sums = np.sum(tetrahedrons ** 2, axis=2)
56:     d = np.concatenate((sums[:, :, np.newaxis], a), axis=2)
57: 
58:     dx = np.delete(d, 1, axis=2)
59:     dy = np.delete(d, 2, axis=2)
60:     dz = np.delete(d, 3, axis=2)
61: 
62:     dx = np.linalg.det(dx)
63:     dy = -np.linalg.det(dy)
64:     dz = np.linalg.det(dz)
65:     a = np.linalg.det(a)
66: 
67:     nominator = np.vstack((dx, dy, dz))
68:     denominator = 2*a
69:     return (nominator / denominator).T
70: 
71: 
72: def project_to_sphere(points, center, radius):
73:     '''
74:     Projects the elements of points onto the sphere defined
75:     by center and radius.
76: 
77:     Parameters
78:     ----------
79:     points : array of floats of shape (npoints, ndim)
80:              consisting of the points in a space of dimension ndim
81:     center : array of floats of shape (ndim,)
82:             the center of the sphere to project on
83:     radius : float
84:             the radius of the sphere to project on
85: 
86:     returns: array of floats of shape (npoints, ndim)
87:             the points projected onto the sphere
88:     '''
89: 
90:     lengths = scipy.spatial.distance.cdist(points, np.array([center]))
91:     return (points - center) / lengths * radius + center
92: 
93: 
94: class SphericalVoronoi:
95:     ''' Voronoi diagrams on the surface of a sphere.
96: 
97:     .. versionadded:: 0.18.0
98: 
99:     Parameters
100:     ----------
101:     points : ndarray of floats, shape (npoints, 3)
102:         Coordinates of points to construct a spherical
103:         Voronoi diagram from
104:     radius : float, optional
105:         Radius of the sphere (Default: 1)
106:     center : ndarray of floats, shape (3,)
107:         Center of sphere (Default: origin)
108:     threshold : float
109:         Threshold for detecting duplicate points and
110:         mismatches between points and sphere parameters.
111:         (Default: 1e-06)
112: 
113:     Attributes
114:     ----------
115:     points : double array of shape (npoints, 3)
116:             the points in 3D to generate the Voronoi diagram from
117:     radius : double
118:             radius of the sphere
119:             Default: None (forces estimation, which is less precise)
120:     center : double array of shape (3,)
121:             center of the sphere
122:             Default: None (assumes sphere is centered at origin)
123:     vertices : double array of shape (nvertices, 3)
124:             Voronoi vertices corresponding to points
125:     regions : list of list of integers of shape (npoints, _ )
126:             the n-th entry is a list consisting of the indices
127:             of the vertices belonging to the n-th point in points
128: 
129:     Raises
130:     ------
131:     ValueError
132:         If there are duplicates in `points`.
133:         If the provided `radius` is not consistent with `points`.
134: 
135:     Notes
136:     ----------
137:     The spherical Voronoi diagram algorithm proceeds as follows. The Convex
138:     Hull of the input points (generators) is calculated, and is equivalent to
139:     their Delaunay triangulation on the surface of the sphere [Caroli]_.
140:     A 3D Delaunay tetrahedralization is obtained by including the origin of
141:     the coordinate system as the fourth vertex of each simplex of the Convex
142:     Hull. The circumcenters of all tetrahedra in the system are calculated and
143:     projected to the surface of the sphere, producing the Voronoi vertices.
144:     The Delaunay tetrahedralization neighbour information is then used to
145:     order the Voronoi region vertices around each generator. The latter
146:     approach is substantially less sensitive to floating point issues than
147:     angle-based methods of Voronoi region vertex sorting.
148: 
149:     The surface area of spherical polygons is calculated by decomposing them
150:     into triangles and using L'Huilier's Theorem to calculate the spherical
151:     excess of each triangle [Weisstein]_. The sum of the spherical excesses is
152:     multiplied by the square of the sphere radius to obtain the surface area
153:     of the spherical polygon. For nearly-degenerate spherical polygons an area
154:     of approximately 0 is returned by default, rather than attempting the
155:     unstable calculation.
156: 
157:     Empirical assessment of spherical Voronoi algorithm performance suggests
158:     quadratic time complexity (loglinear is optimal, but algorithms are more
159:     challenging to implement). The reconstitution of the surface area of the
160:     sphere, measured as the sum of the surface areas of all Voronoi regions,
161:     is closest to 100 % for larger (>> 10) numbers of generators.
162: 
163:     References
164:     ----------
165: 
166:     .. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of
167:                 points on or close to a sphere. Research Report RR-7004, 2009.
168:     .. [Weisstein] "L'Huilier's Theorem." From MathWorld -- A Wolfram Web
169:                 Resource. http://mathworld.wolfram.com/LHuiliersTheorem.html
170: 
171:     See Also
172:     --------
173:     Voronoi : Conventional Voronoi diagrams in N dimensions.
174: 
175:     Examples
176:     --------
177: 
178:     >>> from matplotlib import colors
179:     >>> from mpl_toolkits.mplot3d.art3d import Poly3DCollection
180:     >>> import matplotlib.pyplot as plt
181:     >>> from scipy.spatial import SphericalVoronoi
182:     >>> from mpl_toolkits.mplot3d import proj3d
183:     >>> # set input data
184:     >>> points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
185:     ...                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])
186:     >>> center = np.array([0, 0, 0])
187:     >>> radius = 1
188:     >>> # calculate spherical Voronoi diagram
189:     >>> sv = SphericalVoronoi(points, radius, center)
190:     >>> # sort vertices (optional, helpful for plotting)
191:     >>> sv.sort_vertices_of_regions()
192:     >>> # generate plot
193:     >>> fig = plt.figure()
194:     >>> ax = fig.add_subplot(111, projection='3d')
195:     >>> # plot the unit sphere for reference (optional)
196:     >>> u = np.linspace(0, 2 * np.pi, 100)
197:     >>> v = np.linspace(0, np.pi, 100)
198:     >>> x = np.outer(np.cos(u), np.sin(v))
199:     >>> y = np.outer(np.sin(u), np.sin(v))
200:     >>> z = np.outer(np.ones(np.size(u)), np.cos(v))
201:     >>> ax.plot_surface(x, y, z, color='y', alpha=0.1)
202:     >>> # plot generator points
203:     >>> ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
204:     >>> # plot Voronoi vertices
205:     >>> ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
206:     ...                    c='g')
207:     >>> # indicate Voronoi regions (as Euclidean polygons)
208:     >>> for region in sv.regions:
209:     ...    random_color = colors.rgb2hex(np.random.rand(3))
210:     ...    polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
211:     ...    polygon.set_color(random_color)
212:     ...    ax.add_collection3d(polygon)
213:     >>> plt.show()
214: 
215:     '''
216: 
217:     def __init__(self, points, radius=None, center=None, threshold=1e-06):
218:         '''
219:         Initializes the object and starts the computation of the Voronoi
220:         diagram.
221: 
222:         points : The generator points of the Voronoi diagram assumed to be
223:          all on the sphere with radius supplied by the radius parameter and
224:          center supplied by the center parameter.
225:         radius : The radius of the sphere. Will default to 1 if not supplied.
226:         center : The center of the sphere. Will default to the origin if not
227:          supplied.
228:         '''
229: 
230:         self.points = points
231:         if np.any(center):
232:             self.center = center
233:         else:
234:             self.center = np.zeros(3)
235:         if radius:
236:             self.radius = radius
237:         else:
238:             self.radius = 1
239: 
240:         if pdist(self.points).min() <= threshold * self.radius:
241:             raise ValueError("Duplicate generators present.")
242: 
243:         max_discrepancy = sphere_check(self.points,
244:                                        self.radius,
245:                                        self.center)
246:         if max_discrepancy >= threshold * self.radius:
247:             raise ValueError("Radius inconsistent with generators.")
248:         self.vertices = None
249:         self.regions = None
250:         self._tri = None
251:         self._calc_vertices_regions()
252: 
253:     def _calc_vertices_regions(self):
254:         '''
255:         Calculates the Voronoi vertices and regions of the generators stored
256:         in self.points. The vertices will be stored in self.vertices and the
257:         regions in self.regions.
258: 
259:         This algorithm was discussed at PyData London 2015 by
260:         Tyler Reddy, Ross Hemsley and Nikolai Nowaczyk
261:         '''
262: 
263:         # perform 3D Delaunay triangulation on data set
264:         # (here ConvexHull can also be used, and is faster)
265:         self._tri = scipy.spatial.ConvexHull(self.points)
266: 
267:         # add the center to each of the simplices in tri to get the same
268:         # tetrahedrons we'd have gotten from Delaunay tetrahedralization
269:         # tetrahedrons will have shape: (2N-4, 4, 3)
270:         tetrahedrons = self._tri.points[self._tri.simplices]
271:         tetrahedrons = np.insert(
272:             tetrahedrons,
273:             3,
274:             np.array([self.center]),
275:             axis=1
276:         )
277: 
278:         # produce circumcenters of tetrahedrons from 3D Delaunay
279:         # circumcenters will have shape: (2N-4, 3)
280:         circumcenters = calc_circumcenters(tetrahedrons)
281: 
282:         # project tetrahedron circumcenters to the surface of the sphere
283:         # self.vertices will have shape: (2N-4, 3)
284:         self.vertices = project_to_sphere(
285:             circumcenters,
286:             self.center,
287:             self.radius
288:         )
289: 
290:         # calculate regions from triangulation
291:         # simplex_indices will have shape: (2N-4,)
292:         simplex_indices = np.arange(self._tri.simplices.shape[0])
293:         # tri_indices will have shape: (6N-12,)
294:         tri_indices = np.column_stack([simplex_indices, simplex_indices,
295:             simplex_indices]).ravel()
296:         # point_indices will have shape: (6N-12,)
297:         point_indices = self._tri.simplices.ravel()
298: 
299:         # array_associations will have shape: (6N-12, 2)
300:         array_associations = np.dstack((point_indices, tri_indices))[0]
301:         array_associations = array_associations[np.lexsort((
302:                                                 array_associations[...,1],
303:                                                 array_associations[...,0]))]
304:         array_associations = array_associations.astype(np.intp)
305: 
306:         # group by generator indices to produce
307:         # unsorted regions in nested list
308:         groups = []
309:         for k, g in itertools.groupby(array_associations,
310:                                       lambda t: t[0]):
311:             groups.append(list(list(zip(*list(g)))[1]))
312: 
313:         self.regions = groups
314: 
315:     def sort_vertices_of_regions(self):
316:         '''
317:          For each region in regions, it sorts the indices of the Voronoi
318:          vertices such that the resulting points are in a clockwise or
319:          counterclockwise order around the generator point.
320: 
321:          This is done as follows: Recall that the n-th region in regions
322:          surrounds the n-th generator in points and that the k-th
323:          Voronoi vertex in vertices is the projected circumcenter of the
324:          tetrahedron obtained by the k-th triangle in _tri.simplices (and the
325:          origin). For each region n, we choose the first triangle (=Voronoi
326:          vertex) in _tri.simplices and a vertex of that triangle not equal to
327:          the center n. These determine a unique neighbor of that triangle,
328:          which is then chosen as the second triangle. The second triangle
329:          will have a unique vertex not equal to the current vertex or the
330:          center. This determines a unique neighbor of the second triangle,
331:          which is then chosen as the third triangle and so forth. We proceed
332:          through all the triangles (=Voronoi vertices) belonging to the
333:          generator in points and obtain a sorted version of the vertices
334:          of its surrounding region.
335:         '''
336: 
337:         _voronoi.sort_vertices_of_regions(self._tri.simplices,
338:                                                    self.regions)
339: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_470944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nSpherical Voronoi Code\n\n.. versionadded:: 0.18.0\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_470945) is not StypyTypeError):

    if (import_470945 != 'pyd_module'):
        __import__(import_470945)
        sys_modules_470946 = sys.modules[import_470945]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', sys_modules_470946.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_470945)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import numpy.matlib' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470947 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.matlib')

if (type(import_470947) is not StypyTypeError):

    if (import_470947 != 'pyd_module'):
        __import__(import_470947)
        sys_modules_470948 = sys.modules[import_470947]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.matlib', sys_modules_470948.module_type_store, module_type_store)
    else:
        import numpy.matlib

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.matlib', numpy.matlib, module_type_store)

else:
    # Assigning a type to the variable 'numpy.matlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.matlib', import_470947)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import scipy' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy')

if (type(import_470949) is not StypyTypeError):

    if (import_470949 != 'pyd_module'):
        __import__(import_470949)
        sys_modules_470950 = sys.modules[import_470949]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy', sys_modules_470950.module_type_store, module_type_store)
    else:
        import scipy

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy', scipy, module_type_store)

else:
    # Assigning a type to the variable 'scipy' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy', import_470949)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import itertools' statement (line 17)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.spatial import _voronoi' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470951 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.spatial')

if (type(import_470951) is not StypyTypeError):

    if (import_470951 != 'pyd_module'):
        __import__(import_470951)
        sys_modules_470952 = sys.modules[import_470951]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.spatial', sys_modules_470952.module_type_store, module_type_store, ['_voronoi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_470952, sys_modules_470952.module_type_store, module_type_store)
    else:
        from scipy.spatial import _voronoi

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.spatial', None, module_type_store, ['_voronoi'], [_voronoi])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.spatial', import_470951)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.spatial.distance import pdist' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470953 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.spatial.distance')

if (type(import_470953) is not StypyTypeError):

    if (import_470953 != 'pyd_module'):
        __import__(import_470953)
        sys_modules_470954 = sys.modules[import_470953]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.spatial.distance', sys_modules_470954.module_type_store, module_type_store, ['pdist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_470954, sys_modules_470954.module_type_store, module_type_store)
    else:
        from scipy.spatial.distance import pdist

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.spatial.distance', None, module_type_store, ['pdist'], [pdist])

else:
    # Assigning a type to the variable 'scipy.spatial.distance' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.spatial.distance', import_470953)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')


# Assigning a List to a Name (line 21):
__all__ = ['SphericalVoronoi']
module_type_store.set_exportable_members(['SphericalVoronoi'])

# Obtaining an instance of the builtin type 'list' (line 21)
list_470955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_470956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'SphericalVoronoi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_470955, str_470956)

# Assigning a type to the variable '__all__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__all__', list_470955)

@norecursion
def sphere_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sphere_check'
    module_type_store = module_type_store.open_function_context('sphere_check', 23, 0, False)
    
    # Passed parameters checking function
    sphere_check.stypy_localization = localization
    sphere_check.stypy_type_of_self = None
    sphere_check.stypy_type_store = module_type_store
    sphere_check.stypy_function_name = 'sphere_check'
    sphere_check.stypy_param_names_list = ['points', 'radius', 'center']
    sphere_check.stypy_varargs_param_name = None
    sphere_check.stypy_kwargs_param_name = None
    sphere_check.stypy_call_defaults = defaults
    sphere_check.stypy_call_varargs = varargs
    sphere_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sphere_check', ['points', 'radius', 'center'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sphere_check', localization, ['points', 'radius', 'center'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sphere_check(...)' code ##################

    str_470957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', ' Determines distance of generators from theoretical sphere\n    surface.\n\n    ')
    
    # Assigning a BinOp to a Name (line 28):
    
    # Obtaining the type of the subscript
    Ellipsis_470958 = Ellipsis
    int_470959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 41), 'int')
    # Getting the type of 'points' (line 28)
    points_470960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'points')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___470961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 30), points_470960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_470962 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), getitem___470961, (Ellipsis_470958, int_470959))
    
    
    # Obtaining the type of the subscript
    int_470963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 53), 'int')
    # Getting the type of 'center' (line 28)
    center_470964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'center')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___470965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 46), center_470964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_470966 = invoke(stypy.reporting.localization.Localization(__file__, 28, 46), getitem___470965, int_470963)
    
    # Applying the binary operator '-' (line 28)
    result_sub_470967 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 30), '-', subscript_call_result_470962, subscript_call_result_470966)
    
    int_470968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 60), 'int')
    # Applying the binary operator '**' (line 28)
    result_pow_470969 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 29), '**', result_sub_470967, int_470968)
    
    
    # Obtaining the type of the subscript
    Ellipsis_470970 = Ellipsis
    int_470971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 41), 'int')
    # Getting the type of 'points' (line 29)
    points_470972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'points')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___470973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), points_470972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_470974 = invoke(stypy.reporting.localization.Localization(__file__, 29, 30), getitem___470973, (Ellipsis_470970, int_470971))
    
    
    # Obtaining the type of the subscript
    int_470975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 53), 'int')
    # Getting the type of 'center' (line 29)
    center_470976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 46), 'center')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___470977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 46), center_470976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_470978 = invoke(stypy.reporting.localization.Localization(__file__, 29, 46), getitem___470977, int_470975)
    
    # Applying the binary operator '-' (line 29)
    result_sub_470979 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 30), '-', subscript_call_result_470974, subscript_call_result_470978)
    
    int_470980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 60), 'int')
    # Applying the binary operator '**' (line 29)
    result_pow_470981 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 29), '**', result_sub_470979, int_470980)
    
    # Applying the binary operator '+' (line 28)
    result_add_470982 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 28), '+', result_pow_470969, result_pow_470981)
    
    
    # Obtaining the type of the subscript
    Ellipsis_470983 = Ellipsis
    int_470984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'int')
    # Getting the type of 'points' (line 30)
    points_470985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'points')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___470986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 30), points_470985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_470987 = invoke(stypy.reporting.localization.Localization(__file__, 30, 30), getitem___470986, (Ellipsis_470983, int_470984))
    
    
    # Obtaining the type of the subscript
    int_470988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 53), 'int')
    # Getting the type of 'center' (line 30)
    center_470989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 46), 'center')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___470990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 46), center_470989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_470991 = invoke(stypy.reporting.localization.Localization(__file__, 30, 46), getitem___470990, int_470988)
    
    # Applying the binary operator '-' (line 30)
    result_sub_470992 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 30), '-', subscript_call_result_470987, subscript_call_result_470991)
    
    int_470993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 60), 'int')
    # Applying the binary operator '**' (line 30)
    result_pow_470994 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 29), '**', result_sub_470992, int_470993)
    
    # Applying the binary operator '+' (line 29)
    result_add_470995 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 63), '+', result_add_470982, result_pow_470994)
    
    # Assigning a type to the variable 'actual_squared_radii' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'actual_squared_radii', result_add_470995)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to max(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_471004 = {}
    
    # Call to sqrt(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'actual_squared_radii' (line 31)
    actual_squared_radii_470998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'actual_squared_radii', False)
    # Processing the call keyword arguments (line 31)
    kwargs_470999 = {}
    # Getting the type of 'np' (line 31)
    np_470996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 31)
    sqrt_470997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 23), np_470996, 'sqrt')
    # Calling sqrt(args, kwargs) (line 31)
    sqrt_call_result_471000 = invoke(stypy.reporting.localization.Localization(__file__, 31, 23), sqrt_470997, *[actual_squared_radii_470998], **kwargs_470999)
    
    # Getting the type of 'radius' (line 31)
    radius_471001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 55), 'radius', False)
    # Applying the binary operator '-' (line 31)
    result_sub_471002 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 23), '-', sqrt_call_result_471000, radius_471001)
    
    # Obtaining the member 'max' of a type (line 31)
    max_471003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 23), result_sub_471002, 'max')
    # Calling max(args, kwargs) (line 31)
    max_call_result_471005 = invoke(stypy.reporting.localization.Localization(__file__, 31, 23), max_471003, *[], **kwargs_471004)
    
    # Assigning a type to the variable 'max_discrepancy' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'max_discrepancy', max_call_result_471005)
    
    # Call to abs(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'max_discrepancy' (line 32)
    max_discrepancy_471007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'max_discrepancy', False)
    # Processing the call keyword arguments (line 32)
    kwargs_471008 = {}
    # Getting the type of 'abs' (line 32)
    abs_471006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 32)
    abs_call_result_471009 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), abs_471006, *[max_discrepancy_471007], **kwargs_471008)
    
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', abs_call_result_471009)
    
    # ################# End of 'sphere_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sphere_check' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_471010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_471010)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sphere_check'
    return stypy_return_type_471010

# Assigning a type to the variable 'sphere_check' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'sphere_check', sphere_check)

@norecursion
def calc_circumcenters(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'calc_circumcenters'
    module_type_store = module_type_store.open_function_context('calc_circumcenters', 34, 0, False)
    
    # Passed parameters checking function
    calc_circumcenters.stypy_localization = localization
    calc_circumcenters.stypy_type_of_self = None
    calc_circumcenters.stypy_type_store = module_type_store
    calc_circumcenters.stypy_function_name = 'calc_circumcenters'
    calc_circumcenters.stypy_param_names_list = ['tetrahedrons']
    calc_circumcenters.stypy_varargs_param_name = None
    calc_circumcenters.stypy_kwargs_param_name = None
    calc_circumcenters.stypy_call_defaults = defaults
    calc_circumcenters.stypy_call_varargs = varargs
    calc_circumcenters.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'calc_circumcenters', ['tetrahedrons'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'calc_circumcenters', localization, ['tetrahedrons'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'calc_circumcenters(...)' code ##################

    str_471011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', ' Calculates the cirumcenters of the circumspheres of tetrahedrons.\n\n    An implementation based on\n    http://mathworld.wolfram.com/Circumsphere.html\n\n    Parameters\n    ----------\n    tetrahedrons : an array of shape (N, 4, 3)\n        consisting of N tetrahedrons defined by 4 points in 3D\n\n    Returns\n    ----------\n    circumcenters : an array of shape (N, 3)\n        consisting of the N circumcenters of the tetrahedrons in 3D\n\n    ')
    
    # Assigning a Subscript to a Name (line 52):
    
    # Obtaining the type of the subscript
    int_471012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'int')
    # Getting the type of 'tetrahedrons' (line 52)
    tetrahedrons_471013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'tetrahedrons')
    # Obtaining the member 'shape' of a type (line 52)
    shape_471014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 10), tetrahedrons_471013, 'shape')
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___471015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 10), shape_471014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_471016 = invoke(stypy.reporting.localization.Localization(__file__, 52, 10), getitem___471015, int_471012)
    
    # Assigning a type to the variable 'num' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'num', subscript_call_result_471016)
    
    # Assigning a Call to a Name (line 53):
    
    # Call to concatenate(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_471019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'tetrahedrons' (line 53)
    tetrahedrons_471020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'tetrahedrons', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 24), tuple_471019, tetrahedrons_471020)
    # Adding element type (line 53)
    
    # Call to ones(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_471023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'num' (line 53)
    num_471024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 47), 'num', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 47), tuple_471023, num_471024)
    # Adding element type (line 53)
    int_471025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 47), tuple_471023, int_471025)
    # Adding element type (line 53)
    int_471026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 47), tuple_471023, int_471026)
    
    # Processing the call keyword arguments (line 53)
    kwargs_471027 = {}
    # Getting the type of 'np' (line 53)
    np_471021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 38), 'np', False)
    # Obtaining the member 'ones' of a type (line 53)
    ones_471022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 38), np_471021, 'ones')
    # Calling ones(args, kwargs) (line 53)
    ones_call_result_471028 = invoke(stypy.reporting.localization.Localization(__file__, 53, 38), ones_471022, *[tuple_471023], **kwargs_471027)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 24), tuple_471019, ones_call_result_471028)
    
    # Processing the call keyword arguments (line 53)
    int_471029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 66), 'int')
    keyword_471030 = int_471029
    kwargs_471031 = {'axis': keyword_471030}
    # Getting the type of 'np' (line 53)
    np_471017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 53)
    concatenate_471018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), np_471017, 'concatenate')
    # Calling concatenate(args, kwargs) (line 53)
    concatenate_call_result_471032 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), concatenate_471018, *[tuple_471019], **kwargs_471031)
    
    # Assigning a type to the variable 'a' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'a', concatenate_call_result_471032)
    
    # Assigning a Call to a Name (line 55):
    
    # Call to sum(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'tetrahedrons' (line 55)
    tetrahedrons_471035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'tetrahedrons', False)
    int_471036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
    # Applying the binary operator '**' (line 55)
    result_pow_471037 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 18), '**', tetrahedrons_471035, int_471036)
    
    # Processing the call keyword arguments (line 55)
    int_471038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
    keyword_471039 = int_471038
    kwargs_471040 = {'axis': keyword_471039}
    # Getting the type of 'np' (line 55)
    np_471033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'np', False)
    # Obtaining the member 'sum' of a type (line 55)
    sum_471034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), np_471033, 'sum')
    # Calling sum(args, kwargs) (line 55)
    sum_call_result_471041 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), sum_471034, *[result_pow_471037], **kwargs_471040)
    
    # Assigning a type to the variable 'sums' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'sums', sum_call_result_471041)
    
    # Assigning a Call to a Name (line 56):
    
    # Call to concatenate(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_471044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    
    # Obtaining the type of the subscript
    slice_471045 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 24), None, None, None)
    slice_471046 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 24), None, None, None)
    # Getting the type of 'np' (line 56)
    np_471047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'np', False)
    # Obtaining the member 'newaxis' of a type (line 56)
    newaxis_471048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 35), np_471047, 'newaxis')
    # Getting the type of 'sums' (line 56)
    sums_471049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'sums', False)
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___471050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 24), sums_471049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_471051 = invoke(stypy.reporting.localization.Localization(__file__, 56, 24), getitem___471050, (slice_471045, slice_471046, newaxis_471048))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), tuple_471044, subscript_call_result_471051)
    # Adding element type (line 56)
    # Getting the type of 'a' (line 56)
    a_471052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 48), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), tuple_471044, a_471052)
    
    # Processing the call keyword arguments (line 56)
    int_471053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 57), 'int')
    keyword_471054 = int_471053
    kwargs_471055 = {'axis': keyword_471054}
    # Getting the type of 'np' (line 56)
    np_471042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 56)
    concatenate_471043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), np_471042, 'concatenate')
    # Calling concatenate(args, kwargs) (line 56)
    concatenate_call_result_471056 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), concatenate_471043, *[tuple_471044], **kwargs_471055)
    
    # Assigning a type to the variable 'd' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'd', concatenate_call_result_471056)
    
    # Assigning a Call to a Name (line 58):
    
    # Call to delete(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'd' (line 58)
    d_471059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'd', False)
    int_471060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'int')
    # Processing the call keyword arguments (line 58)
    int_471061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 30), 'int')
    keyword_471062 = int_471061
    kwargs_471063 = {'axis': keyword_471062}
    # Getting the type of 'np' (line 58)
    np_471057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'np', False)
    # Obtaining the member 'delete' of a type (line 58)
    delete_471058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 9), np_471057, 'delete')
    # Calling delete(args, kwargs) (line 58)
    delete_call_result_471064 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), delete_471058, *[d_471059, int_471060], **kwargs_471063)
    
    # Assigning a type to the variable 'dx' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'dx', delete_call_result_471064)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to delete(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'd' (line 59)
    d_471067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'd', False)
    int_471068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'int')
    # Processing the call keyword arguments (line 59)
    int_471069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'int')
    keyword_471070 = int_471069
    kwargs_471071 = {'axis': keyword_471070}
    # Getting the type of 'np' (line 59)
    np_471065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'np', False)
    # Obtaining the member 'delete' of a type (line 59)
    delete_471066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 9), np_471065, 'delete')
    # Calling delete(args, kwargs) (line 59)
    delete_call_result_471072 = invoke(stypy.reporting.localization.Localization(__file__, 59, 9), delete_471066, *[d_471067, int_471068], **kwargs_471071)
    
    # Assigning a type to the variable 'dy' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'dy', delete_call_result_471072)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to delete(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'd' (line 60)
    d_471075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'd', False)
    int_471076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'int')
    # Processing the call keyword arguments (line 60)
    int_471077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 30), 'int')
    keyword_471078 = int_471077
    kwargs_471079 = {'axis': keyword_471078}
    # Getting the type of 'np' (line 60)
    np_471073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 9), 'np', False)
    # Obtaining the member 'delete' of a type (line 60)
    delete_471074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 9), np_471073, 'delete')
    # Calling delete(args, kwargs) (line 60)
    delete_call_result_471080 = invoke(stypy.reporting.localization.Localization(__file__, 60, 9), delete_471074, *[d_471075, int_471076], **kwargs_471079)
    
    # Assigning a type to the variable 'dz' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'dz', delete_call_result_471080)
    
    # Assigning a Call to a Name (line 62):
    
    # Call to det(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'dx' (line 62)
    dx_471084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'dx', False)
    # Processing the call keyword arguments (line 62)
    kwargs_471085 = {}
    # Getting the type of 'np' (line 62)
    np_471081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'np', False)
    # Obtaining the member 'linalg' of a type (line 62)
    linalg_471082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), np_471081, 'linalg')
    # Obtaining the member 'det' of a type (line 62)
    det_471083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), linalg_471082, 'det')
    # Calling det(args, kwargs) (line 62)
    det_call_result_471086 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), det_471083, *[dx_471084], **kwargs_471085)
    
    # Assigning a type to the variable 'dx' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'dx', det_call_result_471086)
    
    # Assigning a UnaryOp to a Name (line 63):
    
    
    # Call to det(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'dy' (line 63)
    dy_471090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'dy', False)
    # Processing the call keyword arguments (line 63)
    kwargs_471091 = {}
    # Getting the type of 'np' (line 63)
    np_471087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 10), 'np', False)
    # Obtaining the member 'linalg' of a type (line 63)
    linalg_471088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 10), np_471087, 'linalg')
    # Obtaining the member 'det' of a type (line 63)
    det_471089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 10), linalg_471088, 'det')
    # Calling det(args, kwargs) (line 63)
    det_call_result_471092 = invoke(stypy.reporting.localization.Localization(__file__, 63, 10), det_471089, *[dy_471090], **kwargs_471091)
    
    # Applying the 'usub' unary operator (line 63)
    result___neg___471093 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 9), 'usub', det_call_result_471092)
    
    # Assigning a type to the variable 'dy' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'dy', result___neg___471093)
    
    # Assigning a Call to a Name (line 64):
    
    # Call to det(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'dz' (line 64)
    dz_471097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'dz', False)
    # Processing the call keyword arguments (line 64)
    kwargs_471098 = {}
    # Getting the type of 'np' (line 64)
    np_471094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'np', False)
    # Obtaining the member 'linalg' of a type (line 64)
    linalg_471095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 9), np_471094, 'linalg')
    # Obtaining the member 'det' of a type (line 64)
    det_471096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 9), linalg_471095, 'det')
    # Calling det(args, kwargs) (line 64)
    det_call_result_471099 = invoke(stypy.reporting.localization.Localization(__file__, 64, 9), det_471096, *[dz_471097], **kwargs_471098)
    
    # Assigning a type to the variable 'dz' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'dz', det_call_result_471099)
    
    # Assigning a Call to a Name (line 65):
    
    # Call to det(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'a' (line 65)
    a_471103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'a', False)
    # Processing the call keyword arguments (line 65)
    kwargs_471104 = {}
    # Getting the type of 'np' (line 65)
    np_471100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'np', False)
    # Obtaining the member 'linalg' of a type (line 65)
    linalg_471101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), np_471100, 'linalg')
    # Obtaining the member 'det' of a type (line 65)
    det_471102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), linalg_471101, 'det')
    # Calling det(args, kwargs) (line 65)
    det_call_result_471105 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), det_471102, *[a_471103], **kwargs_471104)
    
    # Assigning a type to the variable 'a' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'a', det_call_result_471105)
    
    # Assigning a Call to a Name (line 67):
    
    # Call to vstack(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_471108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'dx' (line 67)
    dx_471109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 'dx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_471108, dx_471109)
    # Adding element type (line 67)
    # Getting the type of 'dy' (line 67)
    dy_471110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'dy', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_471108, dy_471110)
    # Adding element type (line 67)
    # Getting the type of 'dz' (line 67)
    dz_471111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'dz', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_471108, dz_471111)
    
    # Processing the call keyword arguments (line 67)
    kwargs_471112 = {}
    # Getting the type of 'np' (line 67)
    np_471106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'np', False)
    # Obtaining the member 'vstack' of a type (line 67)
    vstack_471107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), np_471106, 'vstack')
    # Calling vstack(args, kwargs) (line 67)
    vstack_call_result_471113 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), vstack_471107, *[tuple_471108], **kwargs_471112)
    
    # Assigning a type to the variable 'nominator' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'nominator', vstack_call_result_471113)
    
    # Assigning a BinOp to a Name (line 68):
    int_471114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'int')
    # Getting the type of 'a' (line 68)
    a_471115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'a')
    # Applying the binary operator '*' (line 68)
    result_mul_471116 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 18), '*', int_471114, a_471115)
    
    # Assigning a type to the variable 'denominator' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'denominator', result_mul_471116)
    # Getting the type of 'nominator' (line 69)
    nominator_471117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'nominator')
    # Getting the type of 'denominator' (line 69)
    denominator_471118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'denominator')
    # Applying the binary operator 'div' (line 69)
    result_div_471119 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 12), 'div', nominator_471117, denominator_471118)
    
    # Obtaining the member 'T' of a type (line 69)
    T_471120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), result_div_471119, 'T')
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type', T_471120)
    
    # ################# End of 'calc_circumcenters(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'calc_circumcenters' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_471121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_471121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'calc_circumcenters'
    return stypy_return_type_471121

# Assigning a type to the variable 'calc_circumcenters' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'calc_circumcenters', calc_circumcenters)

@norecursion
def project_to_sphere(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'project_to_sphere'
    module_type_store = module_type_store.open_function_context('project_to_sphere', 72, 0, False)
    
    # Passed parameters checking function
    project_to_sphere.stypy_localization = localization
    project_to_sphere.stypy_type_of_self = None
    project_to_sphere.stypy_type_store = module_type_store
    project_to_sphere.stypy_function_name = 'project_to_sphere'
    project_to_sphere.stypy_param_names_list = ['points', 'center', 'radius']
    project_to_sphere.stypy_varargs_param_name = None
    project_to_sphere.stypy_kwargs_param_name = None
    project_to_sphere.stypy_call_defaults = defaults
    project_to_sphere.stypy_call_varargs = varargs
    project_to_sphere.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'project_to_sphere', ['points', 'center', 'radius'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'project_to_sphere', localization, ['points', 'center', 'radius'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'project_to_sphere(...)' code ##################

    str_471122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', '\n    Projects the elements of points onto the sphere defined\n    by center and radius.\n\n    Parameters\n    ----------\n    points : array of floats of shape (npoints, ndim)\n             consisting of the points in a space of dimension ndim\n    center : array of floats of shape (ndim,)\n            the center of the sphere to project on\n    radius : float\n            the radius of the sphere to project on\n\n    returns: array of floats of shape (npoints, ndim)\n            the points projected onto the sphere\n    ')
    
    # Assigning a Call to a Name (line 90):
    
    # Call to cdist(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'points' (line 90)
    points_471127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 43), 'points', False)
    
    # Call to array(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_471130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 60), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'center' (line 90)
    center_471131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 61), 'center', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 60), list_471130, center_471131)
    
    # Processing the call keyword arguments (line 90)
    kwargs_471132 = {}
    # Getting the type of 'np' (line 90)
    np_471128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 51), 'np', False)
    # Obtaining the member 'array' of a type (line 90)
    array_471129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 51), np_471128, 'array')
    # Calling array(args, kwargs) (line 90)
    array_call_result_471133 = invoke(stypy.reporting.localization.Localization(__file__, 90, 51), array_471129, *[list_471130], **kwargs_471132)
    
    # Processing the call keyword arguments (line 90)
    kwargs_471134 = {}
    # Getting the type of 'scipy' (line 90)
    scipy_471123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'scipy', False)
    # Obtaining the member 'spatial' of a type (line 90)
    spatial_471124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 14), scipy_471123, 'spatial')
    # Obtaining the member 'distance' of a type (line 90)
    distance_471125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 14), spatial_471124, 'distance')
    # Obtaining the member 'cdist' of a type (line 90)
    cdist_471126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 14), distance_471125, 'cdist')
    # Calling cdist(args, kwargs) (line 90)
    cdist_call_result_471135 = invoke(stypy.reporting.localization.Localization(__file__, 90, 14), cdist_471126, *[points_471127, array_call_result_471133], **kwargs_471134)
    
    # Assigning a type to the variable 'lengths' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'lengths', cdist_call_result_471135)
    # Getting the type of 'points' (line 91)
    points_471136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'points')
    # Getting the type of 'center' (line 91)
    center_471137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'center')
    # Applying the binary operator '-' (line 91)
    result_sub_471138 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '-', points_471136, center_471137)
    
    # Getting the type of 'lengths' (line 91)
    lengths_471139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'lengths')
    # Applying the binary operator 'div' (line 91)
    result_div_471140 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), 'div', result_sub_471138, lengths_471139)
    
    # Getting the type of 'radius' (line 91)
    radius_471141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'radius')
    # Applying the binary operator '*' (line 91)
    result_mul_471142 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 39), '*', result_div_471140, radius_471141)
    
    # Getting the type of 'center' (line 91)
    center_471143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 50), 'center')
    # Applying the binary operator '+' (line 91)
    result_add_471144 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '+', result_mul_471142, center_471143)
    
    # Assigning a type to the variable 'stypy_return_type' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type', result_add_471144)
    
    # ################# End of 'project_to_sphere(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'project_to_sphere' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_471145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_471145)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'project_to_sphere'
    return stypy_return_type_471145

# Assigning a type to the variable 'project_to_sphere' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'project_to_sphere', project_to_sphere)
# Declaration of the 'SphericalVoronoi' class

class SphericalVoronoi:
    str_471146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, (-1)), 'str', ' Voronoi diagrams on the surface of a sphere.\n\n    .. versionadded:: 0.18.0\n\n    Parameters\n    ----------\n    points : ndarray of floats, shape (npoints, 3)\n        Coordinates of points to construct a spherical\n        Voronoi diagram from\n    radius : float, optional\n        Radius of the sphere (Default: 1)\n    center : ndarray of floats, shape (3,)\n        Center of sphere (Default: origin)\n    threshold : float\n        Threshold for detecting duplicate points and\n        mismatches between points and sphere parameters.\n        (Default: 1e-06)\n\n    Attributes\n    ----------\n    points : double array of shape (npoints, 3)\n            the points in 3D to generate the Voronoi diagram from\n    radius : double\n            radius of the sphere\n            Default: None (forces estimation, which is less precise)\n    center : double array of shape (3,)\n            center of the sphere\n            Default: None (assumes sphere is centered at origin)\n    vertices : double array of shape (nvertices, 3)\n            Voronoi vertices corresponding to points\n    regions : list of list of integers of shape (npoints, _ )\n            the n-th entry is a list consisting of the indices\n            of the vertices belonging to the n-th point in points\n\n    Raises\n    ------\n    ValueError\n        If there are duplicates in `points`.\n        If the provided `radius` is not consistent with `points`.\n\n    Notes\n    ----------\n    The spherical Voronoi diagram algorithm proceeds as follows. The Convex\n    Hull of the input points (generators) is calculated, and is equivalent to\n    their Delaunay triangulation on the surface of the sphere [Caroli]_.\n    A 3D Delaunay tetrahedralization is obtained by including the origin of\n    the coordinate system as the fourth vertex of each simplex of the Convex\n    Hull. The circumcenters of all tetrahedra in the system are calculated and\n    projected to the surface of the sphere, producing the Voronoi vertices.\n    The Delaunay tetrahedralization neighbour information is then used to\n    order the Voronoi region vertices around each generator. The latter\n    approach is substantially less sensitive to floating point issues than\n    angle-based methods of Voronoi region vertex sorting.\n\n    The surface area of spherical polygons is calculated by decomposing them\n    into triangles and using L\'Huilier\'s Theorem to calculate the spherical\n    excess of each triangle [Weisstein]_. The sum of the spherical excesses is\n    multiplied by the square of the sphere radius to obtain the surface area\n    of the spherical polygon. For nearly-degenerate spherical polygons an area\n    of approximately 0 is returned by default, rather than attempting the\n    unstable calculation.\n\n    Empirical assessment of spherical Voronoi algorithm performance suggests\n    quadratic time complexity (loglinear is optimal, but algorithms are more\n    challenging to implement). The reconstitution of the surface area of the\n    sphere, measured as the sum of the surface areas of all Voronoi regions,\n    is closest to 100 % for larger (>> 10) numbers of generators.\n\n    References\n    ----------\n\n    .. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of\n                points on or close to a sphere. Research Report RR-7004, 2009.\n    .. [Weisstein] "L\'Huilier\'s Theorem." From MathWorld -- A Wolfram Web\n                Resource. http://mathworld.wolfram.com/LHuiliersTheorem.html\n\n    See Also\n    --------\n    Voronoi : Conventional Voronoi diagrams in N dimensions.\n\n    Examples\n    --------\n\n    >>> from matplotlib import colors\n    >>> from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.spatial import SphericalVoronoi\n    >>> from mpl_toolkits.mplot3d import proj3d\n    >>> # set input data\n    >>> points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],\n    ...                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])\n    >>> center = np.array([0, 0, 0])\n    >>> radius = 1\n    >>> # calculate spherical Voronoi diagram\n    >>> sv = SphericalVoronoi(points, radius, center)\n    >>> # sort vertices (optional, helpful for plotting)\n    >>> sv.sort_vertices_of_regions()\n    >>> # generate plot\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111, projection=\'3d\')\n    >>> # plot the unit sphere for reference (optional)\n    >>> u = np.linspace(0, 2 * np.pi, 100)\n    >>> v = np.linspace(0, np.pi, 100)\n    >>> x = np.outer(np.cos(u), np.sin(v))\n    >>> y = np.outer(np.sin(u), np.sin(v))\n    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))\n    >>> ax.plot_surface(x, y, z, color=\'y\', alpha=0.1)\n    >>> # plot generator points\n    >>> ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=\'b\')\n    >>> # plot Voronoi vertices\n    >>> ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],\n    ...                    c=\'g\')\n    >>> # indicate Voronoi regions (as Euclidean polygons)\n    >>> for region in sv.regions:\n    ...    random_color = colors.rgb2hex(np.random.rand(3))\n    ...    polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)\n    ...    polygon.set_color(random_color)\n    ...    ax.add_collection3d(polygon)\n    >>> plt.show()\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 217)
        None_471147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 38), 'None')
        # Getting the type of 'None' (line 217)
        None_471148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 51), 'None')
        float_471149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 67), 'float')
        defaults = [None_471147, None_471148, float_471149]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalVoronoi.__init__', ['points', 'radius', 'center', 'threshold'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['points', 'radius', 'center', 'threshold'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_471150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'str', '\n        Initializes the object and starts the computation of the Voronoi\n        diagram.\n\n        points : The generator points of the Voronoi diagram assumed to be\n         all on the sphere with radius supplied by the radius parameter and\n         center supplied by the center parameter.\n        radius : The radius of the sphere. Will default to 1 if not supplied.\n        center : The center of the sphere. Will default to the origin if not\n         supplied.\n        ')
        
        # Assigning a Name to a Attribute (line 230):
        # Getting the type of 'points' (line 230)
        points_471151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'points')
        # Getting the type of 'self' (line 230)
        self_471152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'points' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_471152, 'points', points_471151)
        
        
        # Call to any(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'center' (line 231)
        center_471155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'center', False)
        # Processing the call keyword arguments (line 231)
        kwargs_471156 = {}
        # Getting the type of 'np' (line 231)
        np_471153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 231)
        any_471154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 11), np_471153, 'any')
        # Calling any(args, kwargs) (line 231)
        any_call_result_471157 = invoke(stypy.reporting.localization.Localization(__file__, 231, 11), any_471154, *[center_471155], **kwargs_471156)
        
        # Testing the type of an if condition (line 231)
        if_condition_471158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), any_call_result_471157)
        # Assigning a type to the variable 'if_condition_471158' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_471158', if_condition_471158)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 'center' (line 232)
        center_471159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'center')
        # Getting the type of 'self' (line 232)
        self_471160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self')
        # Setting the type of the member 'center' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_471160, 'center', center_471159)
        # SSA branch for the else part of an if statement (line 231)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 234):
        
        # Call to zeros(...): (line 234)
        # Processing the call arguments (line 234)
        int_471163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 35), 'int')
        # Processing the call keyword arguments (line 234)
        kwargs_471164 = {}
        # Getting the type of 'np' (line 234)
        np_471161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'np', False)
        # Obtaining the member 'zeros' of a type (line 234)
        zeros_471162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 26), np_471161, 'zeros')
        # Calling zeros(args, kwargs) (line 234)
        zeros_call_result_471165 = invoke(stypy.reporting.localization.Localization(__file__, 234, 26), zeros_471162, *[int_471163], **kwargs_471164)
        
        # Getting the type of 'self' (line 234)
        self_471166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'self')
        # Setting the type of the member 'center' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), self_471166, 'center', zeros_call_result_471165)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'radius' (line 235)
        radius_471167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'radius')
        # Testing the type of an if condition (line 235)
        if_condition_471168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), radius_471167)
        # Assigning a type to the variable 'if_condition_471168' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_471168', if_condition_471168)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 236):
        # Getting the type of 'radius' (line 236)
        radius_471169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 26), 'radius')
        # Getting the type of 'self' (line 236)
        self_471170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'self')
        # Setting the type of the member 'radius' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), self_471170, 'radius', radius_471169)
        # SSA branch for the else part of an if statement (line 235)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 238):
        int_471171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'int')
        # Getting the type of 'self' (line 238)
        self_471172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'self')
        # Setting the type of the member 'radius' of a type (line 238)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), self_471172, 'radius', int_471171)
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to min(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_471179 = {}
        
        # Call to pdist(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_471174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'self', False)
        # Obtaining the member 'points' of a type (line 240)
        points_471175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 17), self_471174, 'points')
        # Processing the call keyword arguments (line 240)
        kwargs_471176 = {}
        # Getting the type of 'pdist' (line 240)
        pdist_471173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'pdist', False)
        # Calling pdist(args, kwargs) (line 240)
        pdist_call_result_471177 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), pdist_471173, *[points_471175], **kwargs_471176)
        
        # Obtaining the member 'min' of a type (line 240)
        min_471178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), pdist_call_result_471177, 'min')
        # Calling min(args, kwargs) (line 240)
        min_call_result_471180 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), min_471178, *[], **kwargs_471179)
        
        # Getting the type of 'threshold' (line 240)
        threshold_471181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 39), 'threshold')
        # Getting the type of 'self' (line 240)
        self_471182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 51), 'self')
        # Obtaining the member 'radius' of a type (line 240)
        radius_471183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 51), self_471182, 'radius')
        # Applying the binary operator '*' (line 240)
        result_mul_471184 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 39), '*', threshold_471181, radius_471183)
        
        # Applying the binary operator '<=' (line 240)
        result_le_471185 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), '<=', min_call_result_471180, result_mul_471184)
        
        # Testing the type of an if condition (line 240)
        if_condition_471186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_le_471185)
        # Assigning a type to the variable 'if_condition_471186' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_471186', if_condition_471186)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 241)
        # Processing the call arguments (line 241)
        str_471188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'str', 'Duplicate generators present.')
        # Processing the call keyword arguments (line 241)
        kwargs_471189 = {}
        # Getting the type of 'ValueError' (line 241)
        ValueError_471187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 241)
        ValueError_call_result_471190 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), ValueError_471187, *[str_471188], **kwargs_471189)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 12), ValueError_call_result_471190, 'raise parameter', BaseException)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 243):
        
        # Call to sphere_check(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'self' (line 243)
        self_471192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 39), 'self', False)
        # Obtaining the member 'points' of a type (line 243)
        points_471193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 39), self_471192, 'points')
        # Getting the type of 'self' (line 244)
        self_471194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 39), 'self', False)
        # Obtaining the member 'radius' of a type (line 244)
        radius_471195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 39), self_471194, 'radius')
        # Getting the type of 'self' (line 245)
        self_471196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 39), 'self', False)
        # Obtaining the member 'center' of a type (line 245)
        center_471197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 39), self_471196, 'center')
        # Processing the call keyword arguments (line 243)
        kwargs_471198 = {}
        # Getting the type of 'sphere_check' (line 243)
        sphere_check_471191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 26), 'sphere_check', False)
        # Calling sphere_check(args, kwargs) (line 243)
        sphere_check_call_result_471199 = invoke(stypy.reporting.localization.Localization(__file__, 243, 26), sphere_check_471191, *[points_471193, radius_471195, center_471197], **kwargs_471198)
        
        # Assigning a type to the variable 'max_discrepancy' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'max_discrepancy', sphere_check_call_result_471199)
        
        
        # Getting the type of 'max_discrepancy' (line 246)
        max_discrepancy_471200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'max_discrepancy')
        # Getting the type of 'threshold' (line 246)
        threshold_471201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'threshold')
        # Getting the type of 'self' (line 246)
        self_471202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'self')
        # Obtaining the member 'radius' of a type (line 246)
        radius_471203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 42), self_471202, 'radius')
        # Applying the binary operator '*' (line 246)
        result_mul_471204 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 30), '*', threshold_471201, radius_471203)
        
        # Applying the binary operator '>=' (line 246)
        result_ge_471205 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 11), '>=', max_discrepancy_471200, result_mul_471204)
        
        # Testing the type of an if condition (line 246)
        if_condition_471206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), result_ge_471205)
        # Assigning a type to the variable 'if_condition_471206' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_471206', if_condition_471206)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 247)
        # Processing the call arguments (line 247)
        str_471208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'str', 'Radius inconsistent with generators.')
        # Processing the call keyword arguments (line 247)
        kwargs_471209 = {}
        # Getting the type of 'ValueError' (line 247)
        ValueError_471207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 247)
        ValueError_call_result_471210 = invoke(stypy.reporting.localization.Localization(__file__, 247, 18), ValueError_471207, *[str_471208], **kwargs_471209)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 12), ValueError_call_result_471210, 'raise parameter', BaseException)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'None' (line 248)
        None_471211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'None')
        # Getting the type of 'self' (line 248)
        self_471212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'vertices' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_471212, 'vertices', None_471211)
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'None' (line 249)
        None_471213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 23), 'None')
        # Getting the type of 'self' (line 249)
        self_471214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'regions' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_471214, 'regions', None_471213)
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'None' (line 250)
        None_471215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'None')
        # Getting the type of 'self' (line 250)
        self_471216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member '_tri' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_471216, '_tri', None_471215)
        
        # Call to _calc_vertices_regions(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_471219 = {}
        # Getting the type of 'self' (line 251)
        self_471217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self', False)
        # Obtaining the member '_calc_vertices_regions' of a type (line 251)
        _calc_vertices_regions_471218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_471217, '_calc_vertices_regions')
        # Calling _calc_vertices_regions(args, kwargs) (line 251)
        _calc_vertices_regions_call_result_471220 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), _calc_vertices_regions_471218, *[], **kwargs_471219)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _calc_vertices_regions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_calc_vertices_regions'
        module_type_store = module_type_store.open_function_context('_calc_vertices_regions', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_localization', localization)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_type_store', module_type_store)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_function_name', 'SphericalVoronoi._calc_vertices_regions')
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_param_names_list', [])
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_varargs_param_name', None)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_call_defaults', defaults)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_call_varargs', varargs)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SphericalVoronoi._calc_vertices_regions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalVoronoi._calc_vertices_regions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_calc_vertices_regions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_calc_vertices_regions(...)' code ##################

        str_471221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, (-1)), 'str', '\n        Calculates the Voronoi vertices and regions of the generators stored\n        in self.points. The vertices will be stored in self.vertices and the\n        regions in self.regions.\n\n        This algorithm was discussed at PyData London 2015 by\n        Tyler Reddy, Ross Hemsley and Nikolai Nowaczyk\n        ')
        
        # Assigning a Call to a Attribute (line 265):
        
        # Call to ConvexHull(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'self' (line 265)
        self_471225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 45), 'self', False)
        # Obtaining the member 'points' of a type (line 265)
        points_471226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 45), self_471225, 'points')
        # Processing the call keyword arguments (line 265)
        kwargs_471227 = {}
        # Getting the type of 'scipy' (line 265)
        scipy_471222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'scipy', False)
        # Obtaining the member 'spatial' of a type (line 265)
        spatial_471223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), scipy_471222, 'spatial')
        # Obtaining the member 'ConvexHull' of a type (line 265)
        ConvexHull_471224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), spatial_471223, 'ConvexHull')
        # Calling ConvexHull(args, kwargs) (line 265)
        ConvexHull_call_result_471228 = invoke(stypy.reporting.localization.Localization(__file__, 265, 20), ConvexHull_471224, *[points_471226], **kwargs_471227)
        
        # Getting the type of 'self' (line 265)
        self_471229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member '_tri' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_471229, '_tri', ConvexHull_call_result_471228)
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 270)
        self_471230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 40), 'self')
        # Obtaining the member '_tri' of a type (line 270)
        _tri_471231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 40), self_471230, '_tri')
        # Obtaining the member 'simplices' of a type (line 270)
        simplices_471232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 40), _tri_471231, 'simplices')
        # Getting the type of 'self' (line 270)
        self_471233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'self')
        # Obtaining the member '_tri' of a type (line 270)
        _tri_471234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), self_471233, '_tri')
        # Obtaining the member 'points' of a type (line 270)
        points_471235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), _tri_471234, 'points')
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___471236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), points_471235, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_471237 = invoke(stypy.reporting.localization.Localization(__file__, 270, 23), getitem___471236, simplices_471232)
        
        # Assigning a type to the variable 'tetrahedrons' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tetrahedrons', subscript_call_result_471237)
        
        # Assigning a Call to a Name (line 271):
        
        # Call to insert(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'tetrahedrons' (line 272)
        tetrahedrons_471240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tetrahedrons', False)
        int_471241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 12), 'int')
        
        # Call to array(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_471244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'self' (line 274)
        self_471245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'self', False)
        # Obtaining the member 'center' of a type (line 274)
        center_471246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), self_471245, 'center')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 21), list_471244, center_471246)
        
        # Processing the call keyword arguments (line 274)
        kwargs_471247 = {}
        # Getting the type of 'np' (line 274)
        np_471242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 274)
        array_471243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), np_471242, 'array')
        # Calling array(args, kwargs) (line 274)
        array_call_result_471248 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), array_471243, *[list_471244], **kwargs_471247)
        
        # Processing the call keyword arguments (line 271)
        int_471249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 17), 'int')
        keyword_471250 = int_471249
        kwargs_471251 = {'axis': keyword_471250}
        # Getting the type of 'np' (line 271)
        np_471238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'np', False)
        # Obtaining the member 'insert' of a type (line 271)
        insert_471239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 23), np_471238, 'insert')
        # Calling insert(args, kwargs) (line 271)
        insert_call_result_471252 = invoke(stypy.reporting.localization.Localization(__file__, 271, 23), insert_471239, *[tetrahedrons_471240, int_471241, array_call_result_471248], **kwargs_471251)
        
        # Assigning a type to the variable 'tetrahedrons' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'tetrahedrons', insert_call_result_471252)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to calc_circumcenters(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'tetrahedrons' (line 280)
        tetrahedrons_471254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 43), 'tetrahedrons', False)
        # Processing the call keyword arguments (line 280)
        kwargs_471255 = {}
        # Getting the type of 'calc_circumcenters' (line 280)
        calc_circumcenters_471253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'calc_circumcenters', False)
        # Calling calc_circumcenters(args, kwargs) (line 280)
        calc_circumcenters_call_result_471256 = invoke(stypy.reporting.localization.Localization(__file__, 280, 24), calc_circumcenters_471253, *[tetrahedrons_471254], **kwargs_471255)
        
        # Assigning a type to the variable 'circumcenters' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'circumcenters', calc_circumcenters_call_result_471256)
        
        # Assigning a Call to a Attribute (line 284):
        
        # Call to project_to_sphere(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'circumcenters' (line 285)
        circumcenters_471258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'circumcenters', False)
        # Getting the type of 'self' (line 286)
        self_471259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'self', False)
        # Obtaining the member 'center' of a type (line 286)
        center_471260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), self_471259, 'center')
        # Getting the type of 'self' (line 287)
        self_471261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'self', False)
        # Obtaining the member 'radius' of a type (line 287)
        radius_471262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), self_471261, 'radius')
        # Processing the call keyword arguments (line 284)
        kwargs_471263 = {}
        # Getting the type of 'project_to_sphere' (line 284)
        project_to_sphere_471257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'project_to_sphere', False)
        # Calling project_to_sphere(args, kwargs) (line 284)
        project_to_sphere_call_result_471264 = invoke(stypy.reporting.localization.Localization(__file__, 284, 24), project_to_sphere_471257, *[circumcenters_471258, center_471260, radius_471262], **kwargs_471263)
        
        # Getting the type of 'self' (line 284)
        self_471265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self')
        # Setting the type of the member 'vertices' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_471265, 'vertices', project_to_sphere_call_result_471264)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to arange(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Obtaining the type of the subscript
        int_471268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 62), 'int')
        # Getting the type of 'self' (line 292)
        self_471269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 36), 'self', False)
        # Obtaining the member '_tri' of a type (line 292)
        _tri_471270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), self_471269, '_tri')
        # Obtaining the member 'simplices' of a type (line 292)
        simplices_471271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), _tri_471270, 'simplices')
        # Obtaining the member 'shape' of a type (line 292)
        shape_471272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), simplices_471271, 'shape')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___471273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), shape_471272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_471274 = invoke(stypy.reporting.localization.Localization(__file__, 292, 36), getitem___471273, int_471268)
        
        # Processing the call keyword arguments (line 292)
        kwargs_471275 = {}
        # Getting the type of 'np' (line 292)
        np_471266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'np', False)
        # Obtaining the member 'arange' of a type (line 292)
        arange_471267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 26), np_471266, 'arange')
        # Calling arange(args, kwargs) (line 292)
        arange_call_result_471276 = invoke(stypy.reporting.localization.Localization(__file__, 292, 26), arange_471267, *[subscript_call_result_471274], **kwargs_471275)
        
        # Assigning a type to the variable 'simplex_indices' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'simplex_indices', arange_call_result_471276)
        
        # Assigning a Call to a Name (line 294):
        
        # Call to ravel(...): (line 294)
        # Processing the call keyword arguments (line 294)
        kwargs_471286 = {}
        
        # Call to column_stack(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_471279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        # Getting the type of 'simplex_indices' (line 294)
        simplex_indices_471280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'simplex_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 38), list_471279, simplex_indices_471280)
        # Adding element type (line 294)
        # Getting the type of 'simplex_indices' (line 294)
        simplex_indices_471281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 56), 'simplex_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 38), list_471279, simplex_indices_471281)
        # Adding element type (line 294)
        # Getting the type of 'simplex_indices' (line 295)
        simplex_indices_471282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'simplex_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 38), list_471279, simplex_indices_471282)
        
        # Processing the call keyword arguments (line 294)
        kwargs_471283 = {}
        # Getting the type of 'np' (line 294)
        np_471277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'np', False)
        # Obtaining the member 'column_stack' of a type (line 294)
        column_stack_471278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 22), np_471277, 'column_stack')
        # Calling column_stack(args, kwargs) (line 294)
        column_stack_call_result_471284 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), column_stack_471278, *[list_471279], **kwargs_471283)
        
        # Obtaining the member 'ravel' of a type (line 294)
        ravel_471285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 22), column_stack_call_result_471284, 'ravel')
        # Calling ravel(args, kwargs) (line 294)
        ravel_call_result_471287 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), ravel_471285, *[], **kwargs_471286)
        
        # Assigning a type to the variable 'tri_indices' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'tri_indices', ravel_call_result_471287)
        
        # Assigning a Call to a Name (line 297):
        
        # Call to ravel(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_471292 = {}
        # Getting the type of 'self' (line 297)
        self_471288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'self', False)
        # Obtaining the member '_tri' of a type (line 297)
        _tri_471289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 24), self_471288, '_tri')
        # Obtaining the member 'simplices' of a type (line 297)
        simplices_471290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 24), _tri_471289, 'simplices')
        # Obtaining the member 'ravel' of a type (line 297)
        ravel_471291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 24), simplices_471290, 'ravel')
        # Calling ravel(args, kwargs) (line 297)
        ravel_call_result_471293 = invoke(stypy.reporting.localization.Localization(__file__, 297, 24), ravel_471291, *[], **kwargs_471292)
        
        # Assigning a type to the variable 'point_indices' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'point_indices', ravel_call_result_471293)
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_471294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 69), 'int')
        
        # Call to dstack(...): (line 300)
        # Processing the call arguments (line 300)
        
        # Obtaining an instance of the builtin type 'tuple' (line 300)
        tuple_471297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 300)
        # Adding element type (line 300)
        # Getting the type of 'point_indices' (line 300)
        point_indices_471298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'point_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 40), tuple_471297, point_indices_471298)
        # Adding element type (line 300)
        # Getting the type of 'tri_indices' (line 300)
        tri_indices_471299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 55), 'tri_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 40), tuple_471297, tri_indices_471299)
        
        # Processing the call keyword arguments (line 300)
        kwargs_471300 = {}
        # Getting the type of 'np' (line 300)
        np_471295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 29), 'np', False)
        # Obtaining the member 'dstack' of a type (line 300)
        dstack_471296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 29), np_471295, 'dstack')
        # Calling dstack(args, kwargs) (line 300)
        dstack_call_result_471301 = invoke(stypy.reporting.localization.Localization(__file__, 300, 29), dstack_471296, *[tuple_471297], **kwargs_471300)
        
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___471302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 29), dstack_call_result_471301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_471303 = invoke(stypy.reporting.localization.Localization(__file__, 300, 29), getitem___471302, int_471294)
        
        # Assigning a type to the variable 'array_associations' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'array_associations', subscript_call_result_471303)
        
        # Assigning a Subscript to a Name (line 301):
        
        # Obtaining the type of the subscript
        
        # Call to lexsort(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'tuple' (line 302)
        tuple_471306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 302)
        # Adding element type (line 302)
        
        # Obtaining the type of the subscript
        Ellipsis_471307 = Ellipsis
        int_471308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 71), 'int')
        # Getting the type of 'array_associations' (line 302)
        array_associations_471309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 48), 'array_associations', False)
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___471310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 48), array_associations_471309, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_471311 = invoke(stypy.reporting.localization.Localization(__file__, 302, 48), getitem___471310, (Ellipsis_471307, int_471308))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 48), tuple_471306, subscript_call_result_471311)
        # Adding element type (line 302)
        
        # Obtaining the type of the subscript
        Ellipsis_471312 = Ellipsis
        int_471313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 71), 'int')
        # Getting the type of 'array_associations' (line 303)
        array_associations_471314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 48), 'array_associations', False)
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___471315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 48), array_associations_471314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_471316 = invoke(stypy.reporting.localization.Localization(__file__, 303, 48), getitem___471315, (Ellipsis_471312, int_471313))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 48), tuple_471306, subscript_call_result_471316)
        
        # Processing the call keyword arguments (line 301)
        kwargs_471317 = {}
        # Getting the type of 'np' (line 301)
        np_471304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 48), 'np', False)
        # Obtaining the member 'lexsort' of a type (line 301)
        lexsort_471305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 48), np_471304, 'lexsort')
        # Calling lexsort(args, kwargs) (line 301)
        lexsort_call_result_471318 = invoke(stypy.reporting.localization.Localization(__file__, 301, 48), lexsort_471305, *[tuple_471306], **kwargs_471317)
        
        # Getting the type of 'array_associations' (line 301)
        array_associations_471319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 29), 'array_associations')
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___471320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 29), array_associations_471319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_471321 = invoke(stypy.reporting.localization.Localization(__file__, 301, 29), getitem___471320, lexsort_call_result_471318)
        
        # Assigning a type to the variable 'array_associations' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'array_associations', subscript_call_result_471321)
        
        # Assigning a Call to a Name (line 304):
        
        # Call to astype(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'np' (line 304)
        np_471324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 55), 'np', False)
        # Obtaining the member 'intp' of a type (line 304)
        intp_471325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 55), np_471324, 'intp')
        # Processing the call keyword arguments (line 304)
        kwargs_471326 = {}
        # Getting the type of 'array_associations' (line 304)
        array_associations_471322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'array_associations', False)
        # Obtaining the member 'astype' of a type (line 304)
        astype_471323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 29), array_associations_471322, 'astype')
        # Calling astype(args, kwargs) (line 304)
        astype_call_result_471327 = invoke(stypy.reporting.localization.Localization(__file__, 304, 29), astype_471323, *[intp_471325], **kwargs_471326)
        
        # Assigning a type to the variable 'array_associations' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'array_associations', astype_call_result_471327)
        
        # Assigning a List to a Name (line 308):
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_471328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        
        # Assigning a type to the variable 'groups' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'groups', list_471328)
        
        
        # Call to groupby(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'array_associations' (line 309)
        array_associations_471331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 38), 'array_associations', False)

        @norecursion
        def _stypy_temp_lambda_241(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_241'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_241', 310, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_241.stypy_localization = localization
            _stypy_temp_lambda_241.stypy_type_of_self = None
            _stypy_temp_lambda_241.stypy_type_store = module_type_store
            _stypy_temp_lambda_241.stypy_function_name = '_stypy_temp_lambda_241'
            _stypy_temp_lambda_241.stypy_param_names_list = ['t']
            _stypy_temp_lambda_241.stypy_varargs_param_name = None
            _stypy_temp_lambda_241.stypy_kwargs_param_name = None
            _stypy_temp_lambda_241.stypy_call_defaults = defaults
            _stypy_temp_lambda_241.stypy_call_varargs = varargs
            _stypy_temp_lambda_241.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_241', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_241', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_471332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 50), 'int')
            # Getting the type of 't' (line 310)
            t_471333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 48), 't', False)
            # Obtaining the member '__getitem__' of a type (line 310)
            getitem___471334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 48), t_471333, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 310)
            subscript_call_result_471335 = invoke(stypy.reporting.localization.Localization(__file__, 310, 48), getitem___471334, int_471332)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), 'stypy_return_type', subscript_call_result_471335)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_241' in the type store
            # Getting the type of 'stypy_return_type' (line 310)
            stypy_return_type_471336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_471336)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_241'
            return stypy_return_type_471336

        # Assigning a type to the variable '_stypy_temp_lambda_241' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), '_stypy_temp_lambda_241', _stypy_temp_lambda_241)
        # Getting the type of '_stypy_temp_lambda_241' (line 310)
        _stypy_temp_lambda_241_471337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), '_stypy_temp_lambda_241')
        # Processing the call keyword arguments (line 309)
        kwargs_471338 = {}
        # Getting the type of 'itertools' (line 309)
        itertools_471329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'itertools', False)
        # Obtaining the member 'groupby' of a type (line 309)
        groupby_471330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 20), itertools_471329, 'groupby')
        # Calling groupby(args, kwargs) (line 309)
        groupby_call_result_471339 = invoke(stypy.reporting.localization.Localization(__file__, 309, 20), groupby_471330, *[array_associations_471331, _stypy_temp_lambda_241_471337], **kwargs_471338)
        
        # Testing the type of a for loop iterable (line 309)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 309, 8), groupby_call_result_471339)
        # Getting the type of the for loop variable (line 309)
        for_loop_var_471340 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 309, 8), groupby_call_result_471339)
        # Assigning a type to the variable 'k' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 8), for_loop_var_471340))
        # Assigning a type to the variable 'g' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'g', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 8), for_loop_var_471340))
        # SSA begins for a for statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Call to list(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining the type of the subscript
        int_471344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 51), 'int')
        
        # Call to list(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Call to zip(...): (line 311)
        
        # Call to list(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'g' (line 311)
        g_471348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 46), 'g', False)
        # Processing the call keyword arguments (line 311)
        kwargs_471349 = {}
        # Getting the type of 'list' (line 311)
        list_471347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 41), 'list', False)
        # Calling list(args, kwargs) (line 311)
        list_call_result_471350 = invoke(stypy.reporting.localization.Localization(__file__, 311, 41), list_471347, *[g_471348], **kwargs_471349)
        
        # Processing the call keyword arguments (line 311)
        kwargs_471351 = {}
        # Getting the type of 'zip' (line 311)
        zip_471346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'zip', False)
        # Calling zip(args, kwargs) (line 311)
        zip_call_result_471352 = invoke(stypy.reporting.localization.Localization(__file__, 311, 36), zip_471346, *[list_call_result_471350], **kwargs_471351)
        
        # Processing the call keyword arguments (line 311)
        kwargs_471353 = {}
        # Getting the type of 'list' (line 311)
        list_471345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 31), 'list', False)
        # Calling list(args, kwargs) (line 311)
        list_call_result_471354 = invoke(stypy.reporting.localization.Localization(__file__, 311, 31), list_471345, *[zip_call_result_471352], **kwargs_471353)
        
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___471355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 31), list_call_result_471354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_471356 = invoke(stypy.reporting.localization.Localization(__file__, 311, 31), getitem___471355, int_471344)
        
        # Processing the call keyword arguments (line 311)
        kwargs_471357 = {}
        # Getting the type of 'list' (line 311)
        list_471343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 26), 'list', False)
        # Calling list(args, kwargs) (line 311)
        list_call_result_471358 = invoke(stypy.reporting.localization.Localization(__file__, 311, 26), list_471343, *[subscript_call_result_471356], **kwargs_471357)
        
        # Processing the call keyword arguments (line 311)
        kwargs_471359 = {}
        # Getting the type of 'groups' (line 311)
        groups_471341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'groups', False)
        # Obtaining the member 'append' of a type (line 311)
        append_471342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), groups_471341, 'append')
        # Calling append(args, kwargs) (line 311)
        append_call_result_471360 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), append_471342, *[list_call_result_471358], **kwargs_471359)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 313):
        # Getting the type of 'groups' (line 313)
        groups_471361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'groups')
        # Getting the type of 'self' (line 313)
        self_471362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self')
        # Setting the type of the member 'regions' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_471362, 'regions', groups_471361)
        
        # ################# End of '_calc_vertices_regions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_calc_vertices_regions' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_471363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_471363)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_calc_vertices_regions'
        return stypy_return_type_471363


    @norecursion
    def sort_vertices_of_regions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sort_vertices_of_regions'
        module_type_store = module_type_store.open_function_context('sort_vertices_of_regions', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_localization', localization)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_type_store', module_type_store)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_function_name', 'SphericalVoronoi.sort_vertices_of_regions')
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_param_names_list', [])
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_varargs_param_name', None)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_call_defaults', defaults)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_call_varargs', varargs)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SphericalVoronoi.sort_vertices_of_regions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalVoronoi.sort_vertices_of_regions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sort_vertices_of_regions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sort_vertices_of_regions(...)' code ##################

        str_471364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, (-1)), 'str', '\n         For each region in regions, it sorts the indices of the Voronoi\n         vertices such that the resulting points are in a clockwise or\n         counterclockwise order around the generator point.\n\n         This is done as follows: Recall that the n-th region in regions\n         surrounds the n-th generator in points and that the k-th\n         Voronoi vertex in vertices is the projected circumcenter of the\n         tetrahedron obtained by the k-th triangle in _tri.simplices (and the\n         origin). For each region n, we choose the first triangle (=Voronoi\n         vertex) in _tri.simplices and a vertex of that triangle not equal to\n         the center n. These determine a unique neighbor of that triangle,\n         which is then chosen as the second triangle. The second triangle\n         will have a unique vertex not equal to the current vertex or the\n         center. This determines a unique neighbor of the second triangle,\n         which is then chosen as the third triangle and so forth. We proceed\n         through all the triangles (=Voronoi vertices) belonging to the\n         generator in points and obtain a sorted version of the vertices\n         of its surrounding region.\n        ')
        
        # Call to sort_vertices_of_regions(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'self' (line 337)
        self_471367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 42), 'self', False)
        # Obtaining the member '_tri' of a type (line 337)
        _tri_471368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 42), self_471367, '_tri')
        # Obtaining the member 'simplices' of a type (line 337)
        simplices_471369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 42), _tri_471368, 'simplices')
        # Getting the type of 'self' (line 338)
        self_471370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 51), 'self', False)
        # Obtaining the member 'regions' of a type (line 338)
        regions_471371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 51), self_471370, 'regions')
        # Processing the call keyword arguments (line 337)
        kwargs_471372 = {}
        # Getting the type of '_voronoi' (line 337)
        _voronoi_471365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), '_voronoi', False)
        # Obtaining the member 'sort_vertices_of_regions' of a type (line 337)
        sort_vertices_of_regions_471366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), _voronoi_471365, 'sort_vertices_of_regions')
        # Calling sort_vertices_of_regions(args, kwargs) (line 337)
        sort_vertices_of_regions_call_result_471373 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), sort_vertices_of_regions_471366, *[simplices_471369, regions_471371], **kwargs_471372)
        
        
        # ################# End of 'sort_vertices_of_regions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sort_vertices_of_regions' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_471374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_471374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sort_vertices_of_regions'
        return stypy_return_type_471374


# Assigning a type to the variable 'SphericalVoronoi' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'SphericalVoronoi', SphericalVoronoi)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

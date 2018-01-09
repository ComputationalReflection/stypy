
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =============================================================
3: Spatial algorithms and data structures (:mod:`scipy.spatial`)
4: =============================================================
5: 
6: .. currentmodule:: scipy.spatial
7: 
8: Nearest-neighbor Queries
9: ========================
10: .. autosummary::
11:    :toctree: generated/
12: 
13:    KDTree      -- class for efficient nearest-neighbor queries
14:    cKDTree     -- class for efficient nearest-neighbor queries (faster impl.)
15:    distance    -- module containing many different distance measures
16:    Rectangle
17: 
18: Delaunay Triangulation, Convex Hulls and Voronoi Diagrams
19: =========================================================
20: 
21: .. autosummary::
22:    :toctree: generated/
23: 
24:    Delaunay    -- compute Delaunay triangulation of input points
25:    ConvexHull  -- compute a convex hull for input points
26:    Voronoi     -- compute a Voronoi diagram hull from input points
27:    SphericalVoronoi -- compute a Voronoi diagram from input points on the surface of a sphere
28:    HalfspaceIntersection -- compute the intersection points of input halfspaces
29: 
30: Plotting Helpers
31: ================
32: 
33: .. autosummary::
34:    :toctree: generated/
35: 
36:    delaunay_plot_2d     -- plot 2-D triangulation
37:    convex_hull_plot_2d  -- plot 2-D convex hull
38:    voronoi_plot_2d      -- plot 2-D voronoi diagram
39: 
40: .. seealso:: :ref:`Tutorial <qhulltutorial>`
41: 
42: 
43: Simplex representation
44: ======================
45: The simplices (triangles, tetrahedra, ...) appearing in the Delaunay
46: tesselation (N-dim simplices), convex hull facets, and Voronoi ridges
47: (N-1 dim simplices) are represented in the following scheme::
48: 
49:     tess = Delaunay(points)
50:     hull = ConvexHull(points)
51:     voro = Voronoi(points)
52: 
53:     # coordinates of the j-th vertex of the i-th simplex
54:     tess.points[tess.simplices[i, j], :]        # tesselation element
55:     hull.points[hull.simplices[i, j], :]        # convex hull facet
56:     voro.vertices[voro.ridge_vertices[i, j], :] # ridge between Voronoi cells
57: 
58: For Delaunay triangulations and convex hulls, the neighborhood
59: structure of the simplices satisfies the condition:
60: 
61:     ``tess.neighbors[i,j]`` is the neighboring simplex of the i-th
62:     simplex, opposite to the j-vertex. It is -1 in case of no
63:     neighbor.
64: 
65: Convex hull facets also define a hyperplane equation::
66: 
67:     (hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0
68: 
69: Similar hyperplane equations for the Delaunay triangulation correspond
70: to the convex hull facets on the corresponding N+1 dimensional
71: paraboloid.
72: 
73: The Delaunay triangulation objects offer a method for locating the
74: simplex containing a given point, and barycentric coordinate
75: computations.
76: 
77: Functions
78: ---------
79: 
80: .. autosummary::
81:    :toctree: generated/
82: 
83:    tsearch
84:    distance_matrix
85:    minkowski_distance
86:    minkowski_distance_p
87:    procrustes
88: 
89: '''
90: 
91: from __future__ import division, print_function, absolute_import
92: 
93: from .kdtree import *
94: from .ckdtree import *
95: from .qhull import *
96: from ._spherical_voronoi import SphericalVoronoi
97: from ._plotutils import *
98: from ._procrustes import procrustes
99: 
100: __all__ = [s for s in dir() if not s.startswith('_')]
101: __all__ += ['distance']
102: 
103: from . import distance
104: 
105: from scipy._lib._testutils import PytestTester
106: test = PytestTester(__name__)
107: del PytestTester
108: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_471375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n=============================================================\nSpatial algorithms and data structures (:mod:`scipy.spatial`)\n=============================================================\n\n.. currentmodule:: scipy.spatial\n\nNearest-neighbor Queries\n========================\n.. autosummary::\n   :toctree: generated/\n\n   KDTree      -- class for efficient nearest-neighbor queries\n   cKDTree     -- class for efficient nearest-neighbor queries (faster impl.)\n   distance    -- module containing many different distance measures\n   Rectangle\n\nDelaunay Triangulation, Convex Hulls and Voronoi Diagrams\n=========================================================\n\n.. autosummary::\n   :toctree: generated/\n\n   Delaunay    -- compute Delaunay triangulation of input points\n   ConvexHull  -- compute a convex hull for input points\n   Voronoi     -- compute a Voronoi diagram hull from input points\n   SphericalVoronoi -- compute a Voronoi diagram from input points on the surface of a sphere\n   HalfspaceIntersection -- compute the intersection points of input halfspaces\n\nPlotting Helpers\n================\n\n.. autosummary::\n   :toctree: generated/\n\n   delaunay_plot_2d     -- plot 2-D triangulation\n   convex_hull_plot_2d  -- plot 2-D convex hull\n   voronoi_plot_2d      -- plot 2-D voronoi diagram\n\n.. seealso:: :ref:`Tutorial <qhulltutorial>`\n\n\nSimplex representation\n======================\nThe simplices (triangles, tetrahedra, ...) appearing in the Delaunay\ntesselation (N-dim simplices), convex hull facets, and Voronoi ridges\n(N-1 dim simplices) are represented in the following scheme::\n\n    tess = Delaunay(points)\n    hull = ConvexHull(points)\n    voro = Voronoi(points)\n\n    # coordinates of the j-th vertex of the i-th simplex\n    tess.points[tess.simplices[i, j], :]        # tesselation element\n    hull.points[hull.simplices[i, j], :]        # convex hull facet\n    voro.vertices[voro.ridge_vertices[i, j], :] # ridge between Voronoi cells\n\nFor Delaunay triangulations and convex hulls, the neighborhood\nstructure of the simplices satisfies the condition:\n\n    ``tess.neighbors[i,j]`` is the neighboring simplex of the i-th\n    simplex, opposite to the j-vertex. It is -1 in case of no\n    neighbor.\n\nConvex hull facets also define a hyperplane equation::\n\n    (hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0\n\nSimilar hyperplane equations for the Delaunay triangulation correspond\nto the convex hull facets on the corresponding N+1 dimensional\nparaboloid.\n\nThe Delaunay triangulation objects offer a method for locating the\nsimplex containing a given point, and barycentric coordinate\ncomputations.\n\nFunctions\n---------\n\n.. autosummary::\n   :toctree: generated/\n\n   tsearch\n   distance_matrix\n   minkowski_distance\n   minkowski_distance_p\n   procrustes\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 93, 0))

# 'from scipy.spatial.kdtree import ' statement (line 93)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471376 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.spatial.kdtree')

if (type(import_471376) is not StypyTypeError):

    if (import_471376 != 'pyd_module'):
        __import__(import_471376)
        sys_modules_471377 = sys.modules[import_471376]
        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.spatial.kdtree', sys_modules_471377.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 93, 0), __file__, sys_modules_471377, sys_modules_471377.module_type_store, module_type_store)
    else:
        from scipy.spatial.kdtree import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.spatial.kdtree', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.spatial.kdtree' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.spatial.kdtree', import_471376)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 0))

# 'from scipy.spatial.ckdtree import ' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471378 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.spatial.ckdtree')

if (type(import_471378) is not StypyTypeError):

    if (import_471378 != 'pyd_module'):
        __import__(import_471378)
        sys_modules_471379 = sys.modules[import_471378]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.spatial.ckdtree', sys_modules_471379.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 0), __file__, sys_modules_471379, sys_modules_471379.module_type_store, module_type_store)
    else:
        from scipy.spatial.ckdtree import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.spatial.ckdtree', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.spatial.ckdtree' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.spatial.ckdtree', import_471378)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 95, 0))

# 'from scipy.spatial.qhull import ' statement (line 95)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471380 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 95, 0), 'scipy.spatial.qhull')

if (type(import_471380) is not StypyTypeError):

    if (import_471380 != 'pyd_module'):
        __import__(import_471380)
        sys_modules_471381 = sys.modules[import_471380]
        import_from_module(stypy.reporting.localization.Localization(__file__, 95, 0), 'scipy.spatial.qhull', sys_modules_471381.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 95, 0), __file__, sys_modules_471381, sys_modules_471381.module_type_store, module_type_store)
    else:
        from scipy.spatial.qhull import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 95, 0), 'scipy.spatial.qhull', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.spatial.qhull' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'scipy.spatial.qhull', import_471380)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 96, 0))

# 'from scipy.spatial._spherical_voronoi import SphericalVoronoi' statement (line 96)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471382 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 96, 0), 'scipy.spatial._spherical_voronoi')

if (type(import_471382) is not StypyTypeError):

    if (import_471382 != 'pyd_module'):
        __import__(import_471382)
        sys_modules_471383 = sys.modules[import_471382]
        import_from_module(stypy.reporting.localization.Localization(__file__, 96, 0), 'scipy.spatial._spherical_voronoi', sys_modules_471383.module_type_store, module_type_store, ['SphericalVoronoi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 96, 0), __file__, sys_modules_471383, sys_modules_471383.module_type_store, module_type_store)
    else:
        from scipy.spatial._spherical_voronoi import SphericalVoronoi

        import_from_module(stypy.reporting.localization.Localization(__file__, 96, 0), 'scipy.spatial._spherical_voronoi', None, module_type_store, ['SphericalVoronoi'], [SphericalVoronoi])

else:
    # Assigning a type to the variable 'scipy.spatial._spherical_voronoi' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'scipy.spatial._spherical_voronoi', import_471382)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 97, 0))

# 'from scipy.spatial._plotutils import ' statement (line 97)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471384 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.spatial._plotutils')

if (type(import_471384) is not StypyTypeError):

    if (import_471384 != 'pyd_module'):
        __import__(import_471384)
        sys_modules_471385 = sys.modules[import_471384]
        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.spatial._plotutils', sys_modules_471385.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 97, 0), __file__, sys_modules_471385, sys_modules_471385.module_type_store, module_type_store)
    else:
        from scipy.spatial._plotutils import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.spatial._plotutils', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.spatial._plotutils' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.spatial._plotutils', import_471384)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 98, 0))

# 'from scipy.spatial._procrustes import procrustes' statement (line 98)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471386 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.spatial._procrustes')

if (type(import_471386) is not StypyTypeError):

    if (import_471386 != 'pyd_module'):
        __import__(import_471386)
        sys_modules_471387 = sys.modules[import_471386]
        import_from_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.spatial._procrustes', sys_modules_471387.module_type_store, module_type_store, ['procrustes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 98, 0), __file__, sys_modules_471387, sys_modules_471387.module_type_store, module_type_store)
    else:
        from scipy.spatial._procrustes import procrustes

        import_from_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.spatial._procrustes', None, module_type_store, ['procrustes'], [procrustes])

else:
    # Assigning a type to the variable 'scipy.spatial._procrustes' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.spatial._procrustes', import_471386)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')


# Assigning a ListComp to a Name (line 100):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 100)
# Processing the call keyword arguments (line 100)
kwargs_471396 = {}
# Getting the type of 'dir' (line 100)
dir_471395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'dir', False)
# Calling dir(args, kwargs) (line 100)
dir_call_result_471397 = invoke(stypy.reporting.localization.Localization(__file__, 100, 22), dir_471395, *[], **kwargs_471396)

comprehension_471398 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 11), dir_call_result_471397)
# Assigning a type to the variable 's' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 's', comprehension_471398)


# Call to startswith(...): (line 100)
# Processing the call arguments (line 100)
str_471391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 48), 'str', '_')
# Processing the call keyword arguments (line 100)
kwargs_471392 = {}
# Getting the type of 's' (line 100)
s_471389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 100)
startswith_471390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 35), s_471389, 'startswith')
# Calling startswith(args, kwargs) (line 100)
startswith_call_result_471393 = invoke(stypy.reporting.localization.Localization(__file__, 100, 35), startswith_471390, *[str_471391], **kwargs_471392)

# Applying the 'not' unary operator (line 100)
result_not__471394 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 31), 'not', startswith_call_result_471393)

# Getting the type of 's' (line 100)
s_471388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 's')
list_471399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 11), list_471399, s_471388)
# Assigning a type to the variable '__all__' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), '__all__', list_471399)

# Getting the type of '__all__' (line 101)
all___471400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '__all__')

# Obtaining an instance of the builtin type 'list' (line 101)
list_471401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)
# Adding element type (line 101)
str_471402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'str', 'distance')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 11), list_471401, str_471402)

# Applying the binary operator '+=' (line 101)
result_iadd_471403 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 0), '+=', all___471400, list_471401)
# Assigning a type to the variable '__all__' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '__all__', result_iadd_471403)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 103, 0))

# 'from scipy.spatial import distance' statement (line 103)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471404 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.spatial')

if (type(import_471404) is not StypyTypeError):

    if (import_471404 != 'pyd_module'):
        __import__(import_471404)
        sys_modules_471405 = sys.modules[import_471404]
        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.spatial', sys_modules_471405.module_type_store, module_type_store, ['distance'])
        nest_module(stypy.reporting.localization.Localization(__file__, 103, 0), __file__, sys_modules_471405, sys_modules_471405.module_type_store, module_type_store)
    else:
        from scipy.spatial import distance

        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.spatial', None, module_type_store, ['distance'], [distance])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.spatial', import_471404)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 105, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 105)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_471406 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy._lib._testutils')

if (type(import_471406) is not StypyTypeError):

    if (import_471406 != 'pyd_module'):
        __import__(import_471406)
        sys_modules_471407 = sys.modules[import_471406]
        import_from_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy._lib._testutils', sys_modules_471407.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 105, 0), __file__, sys_modules_471407, sys_modules_471407.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy._lib._testutils', import_471406)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')


# Assigning a Call to a Name (line 106):

# Call to PytestTester(...): (line 106)
# Processing the call arguments (line 106)
# Getting the type of '__name__' (line 106)
name___471409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), '__name__', False)
# Processing the call keyword arguments (line 106)
kwargs_471410 = {}
# Getting the type of 'PytestTester' (line 106)
PytestTester_471408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 106)
PytestTester_call_result_471411 = invoke(stypy.reporting.localization.Localization(__file__, 106, 7), PytestTester_471408, *[name___471409], **kwargs_471410)

# Assigning a type to the variable 'test' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'test', PytestTester_call_result_471411)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 107, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

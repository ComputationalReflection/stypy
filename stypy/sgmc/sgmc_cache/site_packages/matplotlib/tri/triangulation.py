
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import matplotlib._tri as _tri
7: import matplotlib._qhull as _qhull
8: import numpy as np
9: 
10: 
11: class Triangulation(object):
12:     '''
13:     An unstructured triangular grid consisting of npoints points and
14:     ntri triangles.  The triangles can either be specified by the user
15:     or automatically generated using a Delaunay triangulation.
16: 
17:     Parameters
18:     ----------
19:     x, y : array_like of shape (npoints)
20:         Coordinates of grid points.
21:     triangles : integer array_like of shape (ntri, 3), optional
22:         For each triangle, the indices of the three points that make
23:         up the triangle, ordered in an anticlockwise manner.  If not
24:         specified, the Delaunay triangulation is calculated.
25:     mask : boolean array_like of shape (ntri), optional
26:         Which triangles are masked out.
27: 
28:     Attributes
29:     ----------
30:     `edges`
31:     `neighbors`
32:     is_delaunay : bool
33:         Whether the Triangulation is a calculated Delaunay
34:         triangulation (where `triangles` was not specified) or not.
35: 
36:     Notes
37:     -----
38:     For a Triangulation to be valid it must not have duplicate points,
39:     triangles formed from colinear points, or overlapping triangles.
40:     '''
41:     def __init__(self, x, y, triangles=None, mask=None):
42:         self.x = np.asarray(x, dtype=np.float64)
43:         self.y = np.asarray(y, dtype=np.float64)
44:         if self.x.shape != self.y.shape or self.x.ndim != 1:
45:             raise ValueError("x and y must be equal-length 1-D arrays")
46: 
47:         self.mask = None
48:         self._edges = None
49:         self._neighbors = None
50:         self.is_delaunay = False
51: 
52:         if triangles is None:
53:             # No triangulation specified, so use matplotlib._qhull to obtain
54:             # Delaunay triangulation.
55:             self.triangles, self._neighbors = _qhull.delaunay(x, y)
56:             self.is_delaunay = True
57:         else:
58:             # Triangulation specified. Copy, since we may correct triangle
59:             # orientation.
60:             self.triangles = np.array(triangles, dtype=np.int32, order='C')
61:             if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
62:                 raise ValueError('triangles must be a (?,3) array')
63:             if self.triangles.max() >= len(self.x):
64:                 raise ValueError('triangles max element is out of bounds')
65:             if self.triangles.min() < 0:
66:                 raise ValueError('triangles min element is out of bounds')
67: 
68:         if mask is not None:
69:             self.mask = np.asarray(mask, dtype=bool)
70:             if self.mask.shape != (self.triangles.shape[0],):
71:                 raise ValueError('mask array must have same length as '
72:                                  'triangles array')
73: 
74:         # Underlying C++ object is not created until first needed.
75:         self._cpp_triangulation = None
76: 
77:         # Default TriFinder not created until needed.
78:         self._trifinder = None
79: 
80:     def calculate_plane_coefficients(self, z):
81:         '''
82:         Calculate plane equation coefficients for all unmasked triangles from
83:         the point (x,y) coordinates and specified z-array of shape (npoints).
84:         Returned array has shape (npoints,3) and allows z-value at (x,y)
85:         position in triangle tri to be calculated using
86:         z = array[tri,0]*x + array[tri,1]*y + array[tri,2].
87:         '''
88:         return self.get_cpp_triangulation().calculate_plane_coefficients(z)
89: 
90:     @property
91:     def edges(self):
92:         '''
93:         Return integer array of shape (nedges,2) containing all edges of
94:         non-masked triangles.
95: 
96:         Each edge is the start point index and end point index.  Each
97:         edge (start,end and end,start) appears only once.
98:         '''
99:         if self._edges is None:
100:             self._edges = self.get_cpp_triangulation().get_edges()
101:         return self._edges
102: 
103:     def get_cpp_triangulation(self):
104:         # Return the underlying C++ Triangulation object, creating it
105:         # if necessary.
106:         if self._cpp_triangulation is None:
107:             self._cpp_triangulation = _tri.Triangulation(
108:                 self.x, self.y, self.triangles, self.mask, self._edges,
109:                 self._neighbors, not self.is_delaunay)
110:         return self._cpp_triangulation
111: 
112:     def get_masked_triangles(self):
113:         '''
114:         Return an array of triangles that are not masked.
115:         '''
116:         if self.mask is not None:
117:             return self.triangles.compress(1 - self.mask, axis=0)
118:         else:
119:             return self.triangles
120: 
121:     @staticmethod
122:     def get_from_args_and_kwargs(*args, **kwargs):
123:         '''
124:         Return a Triangulation object from the args and kwargs, and
125:         the remaining args and kwargs with the consumed values removed.
126: 
127:         There are two alternatives: either the first argument is a
128:         Triangulation object, in which case it is returned, or the args
129:         and kwargs are sufficient to create a new Triangulation to
130:         return.  In the latter case, see Triangulation.__init__ for
131:         the possible args and kwargs.
132:         '''
133:         if isinstance(args[0], Triangulation):
134:             triangulation = args[0]
135:             args = args[1:]
136:         else:
137:             x = args[0]
138:             y = args[1]
139:             args = args[2:]  # Consumed first two args.
140: 
141:             # Check triangles in kwargs then args.
142:             triangles = kwargs.pop('triangles', None)
143:             from_args = False
144:             if triangles is None and len(args) > 0:
145:                 triangles = args[0]
146:                 from_args = True
147: 
148:             if triangles is not None:
149:                 try:
150:                     triangles = np.asarray(triangles, dtype=np.int32)
151:                 except ValueError:
152:                     triangles = None
153: 
154:             if triangles is not None and (triangles.ndim != 2 or
155:                                           triangles.shape[1] != 3):
156:                 triangles = None
157: 
158:             if triangles is not None and from_args:
159:                 args = args[1:]  # Consumed first item in args.
160: 
161:             # Check for mask in kwargs.
162:             mask = kwargs.pop('mask', None)
163: 
164:             triangulation = Triangulation(x, y, triangles, mask)
165:         return triangulation, args, kwargs
166: 
167:     def get_trifinder(self):
168:         '''
169:         Return the default :class:`matplotlib.tri.TriFinder` of this
170:         triangulation, creating it if necessary.  This allows the same
171:         TriFinder object to be easily shared.
172:         '''
173:         if self._trifinder is None:
174:             # Default TriFinder class.
175:             from matplotlib.tri.trifinder import TrapezoidMapTriFinder
176:             self._trifinder = TrapezoidMapTriFinder(self)
177:         return self._trifinder
178: 
179:     @property
180:     def neighbors(self):
181:         '''
182:         Return integer array of shape (ntri,3) containing neighbor
183:         triangles.
184: 
185:         For each triangle, the indices of the three triangles that
186:         share the same edges, or -1 if there is no such neighboring
187:         triangle.  neighbors[i,j] is the triangle that is the neighbor
188:         to the edge from point index triangles[i,j] to point index
189:         triangles[i,(j+1)%3].
190:         '''
191:         if self._neighbors is None:
192:             self._neighbors = self.get_cpp_triangulation().get_neighbors()
193:         return self._neighbors
194: 
195:     def set_mask(self, mask):
196:         '''
197:         Set or clear the mask array.  This is either None, or a boolean
198:         array of shape (ntri).
199:         '''
200:         if mask is None:
201:             self.mask = None
202:         else:
203:             self.mask = np.asarray(mask, dtype=bool)
204:             if self.mask.shape != (self.triangles.shape[0],):
205:                 raise ValueError('mask array must have same length as '
206:                                  'triangles array')
207: 
208:         # Set mask in C++ Triangulation.
209:         if self._cpp_triangulation is not None:
210:             self._cpp_triangulation.set_mask(self.mask)
211: 
212:         # Clear derived fields so they are recalculated when needed.
213:         self._edges = None
214:         self._neighbors = None
215: 
216:         # Recalculate TriFinder if it exists.
217:         if self._trifinder is not None:
218:             self._trifinder._initialize()
219: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294360 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_294360) is not StypyTypeError):

    if (import_294360 != 'pyd_module'):
        __import__(import_294360)
        sys_modules_294361 = sys.modules[import_294360]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_294361.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_294360)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import matplotlib._tri' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294362 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib._tri')

if (type(import_294362) is not StypyTypeError):

    if (import_294362 != 'pyd_module'):
        __import__(import_294362)
        sys_modules_294363 = sys.modules[import_294362]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), '_tri', sys_modules_294363.module_type_store, module_type_store)
    else:
        import matplotlib._tri as _tri

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), '_tri', matplotlib._tri, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib._tri' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib._tri', import_294362)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import matplotlib._qhull' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib._qhull')

if (type(import_294364) is not StypyTypeError):

    if (import_294364 != 'pyd_module'):
        __import__(import_294364)
        sys_modules_294365 = sys.modules[import_294364]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), '_qhull', sys_modules_294365.module_type_store, module_type_store)
    else:
        import matplotlib._qhull as _qhull

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), '_qhull', matplotlib._qhull, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib._qhull' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib._qhull', import_294364)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_294366) is not StypyTypeError):

    if (import_294366 != 'pyd_module'):
        __import__(import_294366)
        sys_modules_294367 = sys.modules[import_294366]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_294367.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_294366)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

# Declaration of the 'Triangulation' class

class Triangulation(object, ):
    unicode_294368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'unicode', u'\n    An unstructured triangular grid consisting of npoints points and\n    ntri triangles.  The triangles can either be specified by the user\n    or automatically generated using a Delaunay triangulation.\n\n    Parameters\n    ----------\n    x, y : array_like of shape (npoints)\n        Coordinates of grid points.\n    triangles : integer array_like of shape (ntri, 3), optional\n        For each triangle, the indices of the three points that make\n        up the triangle, ordered in an anticlockwise manner.  If not\n        specified, the Delaunay triangulation is calculated.\n    mask : boolean array_like of shape (ntri), optional\n        Which triangles are masked out.\n\n    Attributes\n    ----------\n    `edges`\n    `neighbors`\n    is_delaunay : bool\n        Whether the Triangulation is a calculated Delaunay\n        triangulation (where `triangles` was not specified) or not.\n\n    Notes\n    -----\n    For a Triangulation to be valid it must not have duplicate points,\n    triangles formed from colinear points, or overlapping triangles.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 41)
        None_294369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'None')
        # Getting the type of 'None' (line 41)
        None_294370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 50), 'None')
        defaults = [None_294369, None_294370]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.__init__', ['x', 'y', 'triangles', 'mask'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'triangles', 'mask'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 42):
        
        # Assigning a Call to a Attribute (line 42):
        
        # Call to asarray(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'x' (line 42)
        x_294373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'x', False)
        # Processing the call keyword arguments (line 42)
        # Getting the type of 'np' (line 42)
        np_294374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'np', False)
        # Obtaining the member 'float64' of a type (line 42)
        float64_294375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 37), np_294374, 'float64')
        keyword_294376 = float64_294375
        kwargs_294377 = {'dtype': keyword_294376}
        # Getting the type of 'np' (line 42)
        np_294371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'np', False)
        # Obtaining the member 'asarray' of a type (line 42)
        asarray_294372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 17), np_294371, 'asarray')
        # Calling asarray(args, kwargs) (line 42)
        asarray_call_result_294378 = invoke(stypy.reporting.localization.Localization(__file__, 42, 17), asarray_294372, *[x_294373], **kwargs_294377)
        
        # Getting the type of 'self' (line 42)
        self_294379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'x' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_294379, 'x', asarray_call_result_294378)
        
        # Assigning a Call to a Attribute (line 43):
        
        # Assigning a Call to a Attribute (line 43):
        
        # Call to asarray(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'y' (line 43)
        y_294382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'y', False)
        # Processing the call keyword arguments (line 43)
        # Getting the type of 'np' (line 43)
        np_294383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 37), 'np', False)
        # Obtaining the member 'float64' of a type (line 43)
        float64_294384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 37), np_294383, 'float64')
        keyword_294385 = float64_294384
        kwargs_294386 = {'dtype': keyword_294385}
        # Getting the type of 'np' (line 43)
        np_294380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'np', False)
        # Obtaining the member 'asarray' of a type (line 43)
        asarray_294381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), np_294380, 'asarray')
        # Calling asarray(args, kwargs) (line 43)
        asarray_call_result_294387 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), asarray_294381, *[y_294382], **kwargs_294386)
        
        # Getting the type of 'self' (line 43)
        self_294388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'y' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_294388, 'y', asarray_call_result_294387)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 44)
        self_294389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'self')
        # Obtaining the member 'x' of a type (line 44)
        x_294390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), self_294389, 'x')
        # Obtaining the member 'shape' of a type (line 44)
        shape_294391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), x_294390, 'shape')
        # Getting the type of 'self' (line 44)
        self_294392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'self')
        # Obtaining the member 'y' of a type (line 44)
        y_294393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), self_294392, 'y')
        # Obtaining the member 'shape' of a type (line 44)
        shape_294394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), y_294393, 'shape')
        # Applying the binary operator '!=' (line 44)
        result_ne_294395 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '!=', shape_294391, shape_294394)
        
        
        # Getting the type of 'self' (line 44)
        self_294396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 43), 'self')
        # Obtaining the member 'x' of a type (line 44)
        x_294397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 43), self_294396, 'x')
        # Obtaining the member 'ndim' of a type (line 44)
        ndim_294398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 43), x_294397, 'ndim')
        int_294399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 58), 'int')
        # Applying the binary operator '!=' (line 44)
        result_ne_294400 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 43), '!=', ndim_294398, int_294399)
        
        # Applying the binary operator 'or' (line 44)
        result_or_keyword_294401 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'or', result_ne_294395, result_ne_294400)
        
        # Testing the type of an if condition (line 44)
        if_condition_294402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_or_keyword_294401)
        # Assigning a type to the variable 'if_condition_294402' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_294402', if_condition_294402)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 45)
        # Processing the call arguments (line 45)
        unicode_294404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'unicode', u'x and y must be equal-length 1-D arrays')
        # Processing the call keyword arguments (line 45)
        kwargs_294405 = {}
        # Getting the type of 'ValueError' (line 45)
        ValueError_294403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 45)
        ValueError_call_result_294406 = invoke(stypy.reporting.localization.Localization(__file__, 45, 18), ValueError_294403, *[unicode_294404], **kwargs_294405)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 45, 12), ValueError_call_result_294406, 'raise parameter', BaseException)
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'None' (line 47)
        None_294407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'None')
        # Getting the type of 'self' (line 47)
        self_294408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'mask' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_294408, 'mask', None_294407)
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'None' (line 48)
        None_294409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'None')
        # Getting the type of 'self' (line 48)
        self_294410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member '_edges' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_294410, '_edges', None_294409)
        
        # Assigning a Name to a Attribute (line 49):
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'None' (line 49)
        None_294411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'None')
        # Getting the type of 'self' (line 49)
        self_294412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member '_neighbors' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_294412, '_neighbors', None_294411)
        
        # Assigning a Name to a Attribute (line 50):
        
        # Assigning a Name to a Attribute (line 50):
        # Getting the type of 'False' (line 50)
        False_294413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'False')
        # Getting the type of 'self' (line 50)
        self_294414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self')
        # Setting the type of the member 'is_delaunay' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_294414, 'is_delaunay', False_294413)
        
        # Type idiom detected: calculating its left and rigth part (line 52)
        # Getting the type of 'triangles' (line 52)
        triangles_294415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'triangles')
        # Getting the type of 'None' (line 52)
        None_294416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'None')
        
        (may_be_294417, more_types_in_union_294418) = may_be_none(triangles_294415, None_294416)

        if may_be_294417:

            if more_types_in_union_294418:
                # Runtime conditional SSA (line 52)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 55):
            
            # Assigning a Call to a Name:
            
            # Call to delaunay(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'x' (line 55)
            x_294421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 62), 'x', False)
            # Getting the type of 'y' (line 55)
            y_294422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 65), 'y', False)
            # Processing the call keyword arguments (line 55)
            kwargs_294423 = {}
            # Getting the type of '_qhull' (line 55)
            _qhull_294419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), '_qhull', False)
            # Obtaining the member 'delaunay' of a type (line 55)
            delaunay_294420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 46), _qhull_294419, 'delaunay')
            # Calling delaunay(args, kwargs) (line 55)
            delaunay_call_result_294424 = invoke(stypy.reporting.localization.Localization(__file__, 55, 46), delaunay_294420, *[x_294421, y_294422], **kwargs_294423)
            
            # Assigning a type to the variable 'call_assignment_294357' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294357', delaunay_call_result_294424)
            
            # Assigning a Call to a Name (line 55):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_294427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'int')
            # Processing the call keyword arguments
            kwargs_294428 = {}
            # Getting the type of 'call_assignment_294357' (line 55)
            call_assignment_294357_294425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294357', False)
            # Obtaining the member '__getitem__' of a type (line 55)
            getitem___294426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), call_assignment_294357_294425, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_294429 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294426, *[int_294427], **kwargs_294428)
            
            # Assigning a type to the variable 'call_assignment_294358' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294358', getitem___call_result_294429)
            
            # Assigning a Name to a Attribute (line 55):
            # Getting the type of 'call_assignment_294358' (line 55)
            call_assignment_294358_294430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294358')
            # Getting the type of 'self' (line 55)
            self_294431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'self')
            # Setting the type of the member 'triangles' of a type (line 55)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), self_294431, 'triangles', call_assignment_294358_294430)
            
            # Assigning a Call to a Name (line 55):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_294434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'int')
            # Processing the call keyword arguments
            kwargs_294435 = {}
            # Getting the type of 'call_assignment_294357' (line 55)
            call_assignment_294357_294432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294357', False)
            # Obtaining the member '__getitem__' of a type (line 55)
            getitem___294433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), call_assignment_294357_294432, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_294436 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294433, *[int_294434], **kwargs_294435)
            
            # Assigning a type to the variable 'call_assignment_294359' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294359', getitem___call_result_294436)
            
            # Assigning a Name to a Attribute (line 55):
            # Getting the type of 'call_assignment_294359' (line 55)
            call_assignment_294359_294437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'call_assignment_294359')
            # Getting the type of 'self' (line 55)
            self_294438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'self')
            # Setting the type of the member '_neighbors' of a type (line 55)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 28), self_294438, '_neighbors', call_assignment_294359_294437)
            
            # Assigning a Name to a Attribute (line 56):
            
            # Assigning a Name to a Attribute (line 56):
            # Getting the type of 'True' (line 56)
            True_294439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'True')
            # Getting the type of 'self' (line 56)
            self_294440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self')
            # Setting the type of the member 'is_delaunay' of a type (line 56)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_294440, 'is_delaunay', True_294439)

            if more_types_in_union_294418:
                # Runtime conditional SSA for else branch (line 52)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_294417) or more_types_in_union_294418):
            
            # Assigning a Call to a Attribute (line 60):
            
            # Assigning a Call to a Attribute (line 60):
            
            # Call to array(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'triangles' (line 60)
            triangles_294443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'triangles', False)
            # Processing the call keyword arguments (line 60)
            # Getting the type of 'np' (line 60)
            np_294444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 55), 'np', False)
            # Obtaining the member 'int32' of a type (line 60)
            int32_294445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 55), np_294444, 'int32')
            keyword_294446 = int32_294445
            unicode_294447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 71), 'unicode', u'C')
            keyword_294448 = unicode_294447
            kwargs_294449 = {'dtype': keyword_294446, 'order': keyword_294448}
            # Getting the type of 'np' (line 60)
            np_294441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'np', False)
            # Obtaining the member 'array' of a type (line 60)
            array_294442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), np_294441, 'array')
            # Calling array(args, kwargs) (line 60)
            array_call_result_294450 = invoke(stypy.reporting.localization.Localization(__file__, 60, 29), array_294442, *[triangles_294443], **kwargs_294449)
            
            # Getting the type of 'self' (line 60)
            self_294451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self')
            # Setting the type of the member 'triangles' of a type (line 60)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), self_294451, 'triangles', array_call_result_294450)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'self' (line 61)
            self_294452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'self')
            # Obtaining the member 'triangles' of a type (line 61)
            triangles_294453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), self_294452, 'triangles')
            # Obtaining the member 'ndim' of a type (line 61)
            ndim_294454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), triangles_294453, 'ndim')
            int_294455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'int')
            # Applying the binary operator '!=' (line 61)
            result_ne_294456 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 15), '!=', ndim_294454, int_294455)
            
            
            
            # Obtaining the type of the subscript
            int_294457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 64), 'int')
            # Getting the type of 'self' (line 61)
            self_294458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 43), 'self')
            # Obtaining the member 'triangles' of a type (line 61)
            triangles_294459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 43), self_294458, 'triangles')
            # Obtaining the member 'shape' of a type (line 61)
            shape_294460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 43), triangles_294459, 'shape')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___294461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 43), shape_294460, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_294462 = invoke(stypy.reporting.localization.Localization(__file__, 61, 43), getitem___294461, int_294457)
            
            int_294463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 70), 'int')
            # Applying the binary operator '!=' (line 61)
            result_ne_294464 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 43), '!=', subscript_call_result_294462, int_294463)
            
            # Applying the binary operator 'or' (line 61)
            result_or_keyword_294465 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 15), 'or', result_ne_294456, result_ne_294464)
            
            # Testing the type of an if condition (line 61)
            if_condition_294466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 12), result_or_keyword_294465)
            # Assigning a type to the variable 'if_condition_294466' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'if_condition_294466', if_condition_294466)
            # SSA begins for if statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 62)
            # Processing the call arguments (line 62)
            unicode_294468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'unicode', u'triangles must be a (?,3) array')
            # Processing the call keyword arguments (line 62)
            kwargs_294469 = {}
            # Getting the type of 'ValueError' (line 62)
            ValueError_294467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 62)
            ValueError_call_result_294470 = invoke(stypy.reporting.localization.Localization(__file__, 62, 22), ValueError_294467, *[unicode_294468], **kwargs_294469)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 62, 16), ValueError_call_result_294470, 'raise parameter', BaseException)
            # SSA join for if statement (line 61)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to max(...): (line 63)
            # Processing the call keyword arguments (line 63)
            kwargs_294474 = {}
            # Getting the type of 'self' (line 63)
            self_294471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'self', False)
            # Obtaining the member 'triangles' of a type (line 63)
            triangles_294472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 15), self_294471, 'triangles')
            # Obtaining the member 'max' of a type (line 63)
            max_294473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 15), triangles_294472, 'max')
            # Calling max(args, kwargs) (line 63)
            max_call_result_294475 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), max_294473, *[], **kwargs_294474)
            
            
            # Call to len(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'self' (line 63)
            self_294477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 43), 'self', False)
            # Obtaining the member 'x' of a type (line 63)
            x_294478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 43), self_294477, 'x')
            # Processing the call keyword arguments (line 63)
            kwargs_294479 = {}
            # Getting the type of 'len' (line 63)
            len_294476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 39), 'len', False)
            # Calling len(args, kwargs) (line 63)
            len_call_result_294480 = invoke(stypy.reporting.localization.Localization(__file__, 63, 39), len_294476, *[x_294478], **kwargs_294479)
            
            # Applying the binary operator '>=' (line 63)
            result_ge_294481 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 15), '>=', max_call_result_294475, len_call_result_294480)
            
            # Testing the type of an if condition (line 63)
            if_condition_294482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 12), result_ge_294481)
            # Assigning a type to the variable 'if_condition_294482' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'if_condition_294482', if_condition_294482)
            # SSA begins for if statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 64)
            # Processing the call arguments (line 64)
            unicode_294484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'unicode', u'triangles max element is out of bounds')
            # Processing the call keyword arguments (line 64)
            kwargs_294485 = {}
            # Getting the type of 'ValueError' (line 64)
            ValueError_294483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 64)
            ValueError_call_result_294486 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), ValueError_294483, *[unicode_294484], **kwargs_294485)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 64, 16), ValueError_call_result_294486, 'raise parameter', BaseException)
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to min(...): (line 65)
            # Processing the call keyword arguments (line 65)
            kwargs_294490 = {}
            # Getting the type of 'self' (line 65)
            self_294487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'self', False)
            # Obtaining the member 'triangles' of a type (line 65)
            triangles_294488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), self_294487, 'triangles')
            # Obtaining the member 'min' of a type (line 65)
            min_294489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), triangles_294488, 'min')
            # Calling min(args, kwargs) (line 65)
            min_call_result_294491 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), min_294489, *[], **kwargs_294490)
            
            int_294492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'int')
            # Applying the binary operator '<' (line 65)
            result_lt_294493 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 15), '<', min_call_result_294491, int_294492)
            
            # Testing the type of an if condition (line 65)
            if_condition_294494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 12), result_lt_294493)
            # Assigning a type to the variable 'if_condition_294494' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'if_condition_294494', if_condition_294494)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 66)
            # Processing the call arguments (line 66)
            unicode_294496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'unicode', u'triangles min element is out of bounds')
            # Processing the call keyword arguments (line 66)
            kwargs_294497 = {}
            # Getting the type of 'ValueError' (line 66)
            ValueError_294495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 66)
            ValueError_call_result_294498 = invoke(stypy.reporting.localization.Localization(__file__, 66, 22), ValueError_294495, *[unicode_294496], **kwargs_294497)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 66, 16), ValueError_call_result_294498, 'raise parameter', BaseException)
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_294417 and more_types_in_union_294418):
                # SSA join for if statement (line 52)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 68)
        # Getting the type of 'mask' (line 68)
        mask_294499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'mask')
        # Getting the type of 'None' (line 68)
        None_294500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'None')
        
        (may_be_294501, more_types_in_union_294502) = may_not_be_none(mask_294499, None_294500)

        if may_be_294501:

            if more_types_in_union_294502:
                # Runtime conditional SSA (line 68)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 69):
            
            # Assigning a Call to a Attribute (line 69):
            
            # Call to asarray(...): (line 69)
            # Processing the call arguments (line 69)
            # Getting the type of 'mask' (line 69)
            mask_294505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'mask', False)
            # Processing the call keyword arguments (line 69)
            # Getting the type of 'bool' (line 69)
            bool_294506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 47), 'bool', False)
            keyword_294507 = bool_294506
            kwargs_294508 = {'dtype': keyword_294507}
            # Getting the type of 'np' (line 69)
            np_294503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 69)
            asarray_294504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 24), np_294503, 'asarray')
            # Calling asarray(args, kwargs) (line 69)
            asarray_call_result_294509 = invoke(stypy.reporting.localization.Localization(__file__, 69, 24), asarray_294504, *[mask_294505], **kwargs_294508)
            
            # Getting the type of 'self' (line 69)
            self_294510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self')
            # Setting the type of the member 'mask' of a type (line 69)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), self_294510, 'mask', asarray_call_result_294509)
            
            
            # Getting the type of 'self' (line 70)
            self_294511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'self')
            # Obtaining the member 'mask' of a type (line 70)
            mask_294512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), self_294511, 'mask')
            # Obtaining the member 'shape' of a type (line 70)
            shape_294513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), mask_294512, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 70)
            tuple_294514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 70)
            # Adding element type (line 70)
            
            # Obtaining the type of the subscript
            int_294515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 56), 'int')
            # Getting the type of 'self' (line 70)
            self_294516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'self')
            # Obtaining the member 'triangles' of a type (line 70)
            triangles_294517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 35), self_294516, 'triangles')
            # Obtaining the member 'shape' of a type (line 70)
            shape_294518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 35), triangles_294517, 'shape')
            # Obtaining the member '__getitem__' of a type (line 70)
            getitem___294519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 35), shape_294518, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 70)
            subscript_call_result_294520 = invoke(stypy.reporting.localization.Localization(__file__, 70, 35), getitem___294519, int_294515)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 35), tuple_294514, subscript_call_result_294520)
            
            # Applying the binary operator '!=' (line 70)
            result_ne_294521 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), '!=', shape_294513, tuple_294514)
            
            # Testing the type of an if condition (line 70)
            if_condition_294522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_294521)
            # Assigning a type to the variable 'if_condition_294522' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_294522', if_condition_294522)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 71)
            # Processing the call arguments (line 71)
            unicode_294524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'unicode', u'mask array must have same length as triangles array')
            # Processing the call keyword arguments (line 71)
            kwargs_294525 = {}
            # Getting the type of 'ValueError' (line 71)
            ValueError_294523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 71)
            ValueError_call_result_294526 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), ValueError_294523, *[unicode_294524], **kwargs_294525)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 16), ValueError_call_result_294526, 'raise parameter', BaseException)
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_294502:
                # SSA join for if statement (line 68)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 75):
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'None' (line 75)
        None_294527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'None')
        # Getting the type of 'self' (line 75)
        self_294528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member '_cpp_triangulation' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_294528, '_cpp_triangulation', None_294527)
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'None' (line 78)
        None_294529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'None')
        # Getting the type of 'self' (line 78)
        self_294530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member '_trifinder' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_294530, '_trifinder', None_294529)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def calculate_plane_coefficients(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'calculate_plane_coefficients'
        module_type_store = module_type_store.open_function_context('calculate_plane_coefficients', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_function_name', 'Triangulation.calculate_plane_coefficients')
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_param_names_list', ['z'])
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.calculate_plane_coefficients.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.calculate_plane_coefficients', ['z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'calculate_plane_coefficients', localization, ['z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'calculate_plane_coefficients(...)' code ##################

        unicode_294531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'unicode', u'\n        Calculate plane equation coefficients for all unmasked triangles from\n        the point (x,y) coordinates and specified z-array of shape (npoints).\n        Returned array has shape (npoints,3) and allows z-value at (x,y)\n        position in triangle tri to be calculated using\n        z = array[tri,0]*x + array[tri,1]*y + array[tri,2].\n        ')
        
        # Call to calculate_plane_coefficients(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'z' (line 88)
        z_294537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 73), 'z', False)
        # Processing the call keyword arguments (line 88)
        kwargs_294538 = {}
        
        # Call to get_cpp_triangulation(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_294534 = {}
        # Getting the type of 'self' (line 88)
        self_294532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'self', False)
        # Obtaining the member 'get_cpp_triangulation' of a type (line 88)
        get_cpp_triangulation_294533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), self_294532, 'get_cpp_triangulation')
        # Calling get_cpp_triangulation(args, kwargs) (line 88)
        get_cpp_triangulation_call_result_294535 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), get_cpp_triangulation_294533, *[], **kwargs_294534)
        
        # Obtaining the member 'calculate_plane_coefficients' of a type (line 88)
        calculate_plane_coefficients_294536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), get_cpp_triangulation_call_result_294535, 'calculate_plane_coefficients')
        # Calling calculate_plane_coefficients(args, kwargs) (line 88)
        calculate_plane_coefficients_call_result_294539 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), calculate_plane_coefficients_294536, *[z_294537], **kwargs_294538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', calculate_plane_coefficients_call_result_294539)
        
        # ################# End of 'calculate_plane_coefficients(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'calculate_plane_coefficients' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_294540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'calculate_plane_coefficients'
        return stypy_return_type_294540


    @norecursion
    def edges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'edges'
        module_type_store = module_type_store.open_function_context('edges', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.edges.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.edges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.edges.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.edges.__dict__.__setitem__('stypy_function_name', 'Triangulation.edges')
        Triangulation.edges.__dict__.__setitem__('stypy_param_names_list', [])
        Triangulation.edges.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.edges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.edges.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.edges.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.edges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.edges.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.edges', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'edges', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'edges(...)' code ##################

        unicode_294541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'unicode', u'\n        Return integer array of shape (nedges,2) containing all edges of\n        non-masked triangles.\n\n        Each edge is the start point index and end point index.  Each\n        edge (start,end and end,start) appears only once.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 99)
        # Getting the type of 'self' (line 99)
        self_294542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'self')
        # Obtaining the member '_edges' of a type (line 99)
        _edges_294543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), self_294542, '_edges')
        # Getting the type of 'None' (line 99)
        None_294544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'None')
        
        (may_be_294545, more_types_in_union_294546) = may_be_none(_edges_294543, None_294544)

        if may_be_294545:

            if more_types_in_union_294546:
                # Runtime conditional SSA (line 99)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 100):
            
            # Assigning a Call to a Attribute (line 100):
            
            # Call to get_edges(...): (line 100)
            # Processing the call keyword arguments (line 100)
            kwargs_294552 = {}
            
            # Call to get_cpp_triangulation(...): (line 100)
            # Processing the call keyword arguments (line 100)
            kwargs_294549 = {}
            # Getting the type of 'self' (line 100)
            self_294547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'self', False)
            # Obtaining the member 'get_cpp_triangulation' of a type (line 100)
            get_cpp_triangulation_294548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 26), self_294547, 'get_cpp_triangulation')
            # Calling get_cpp_triangulation(args, kwargs) (line 100)
            get_cpp_triangulation_call_result_294550 = invoke(stypy.reporting.localization.Localization(__file__, 100, 26), get_cpp_triangulation_294548, *[], **kwargs_294549)
            
            # Obtaining the member 'get_edges' of a type (line 100)
            get_edges_294551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 26), get_cpp_triangulation_call_result_294550, 'get_edges')
            # Calling get_edges(args, kwargs) (line 100)
            get_edges_call_result_294553 = invoke(stypy.reporting.localization.Localization(__file__, 100, 26), get_edges_294551, *[], **kwargs_294552)
            
            # Getting the type of 'self' (line 100)
            self_294554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
            # Setting the type of the member '_edges' of a type (line 100)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_294554, '_edges', get_edges_call_result_294553)

            if more_types_in_union_294546:
                # SSA join for if statement (line 99)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 101)
        self_294555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'self')
        # Obtaining the member '_edges' of a type (line 101)
        _edges_294556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), self_294555, '_edges')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', _edges_294556)
        
        # ################# End of 'edges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'edges' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_294557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'edges'
        return stypy_return_type_294557


    @norecursion
    def get_cpp_triangulation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_cpp_triangulation'
        module_type_store = module_type_store.open_function_context('get_cpp_triangulation', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_function_name', 'Triangulation.get_cpp_triangulation')
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_param_names_list', [])
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.get_cpp_triangulation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.get_cpp_triangulation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_cpp_triangulation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_cpp_triangulation(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 106)
        # Getting the type of 'self' (line 106)
        self_294558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'self')
        # Obtaining the member '_cpp_triangulation' of a type (line 106)
        _cpp_triangulation_294559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), self_294558, '_cpp_triangulation')
        # Getting the type of 'None' (line 106)
        None_294560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'None')
        
        (may_be_294561, more_types_in_union_294562) = may_be_none(_cpp_triangulation_294559, None_294560)

        if may_be_294561:

            if more_types_in_union_294562:
                # Runtime conditional SSA (line 106)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 107):
            
            # Assigning a Call to a Attribute (line 107):
            
            # Call to Triangulation(...): (line 107)
            # Processing the call arguments (line 107)
            # Getting the type of 'self' (line 108)
            self_294565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'self', False)
            # Obtaining the member 'x' of a type (line 108)
            x_294566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), self_294565, 'x')
            # Getting the type of 'self' (line 108)
            self_294567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'self', False)
            # Obtaining the member 'y' of a type (line 108)
            y_294568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), self_294567, 'y')
            # Getting the type of 'self' (line 108)
            self_294569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'self', False)
            # Obtaining the member 'triangles' of a type (line 108)
            triangles_294570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 32), self_294569, 'triangles')
            # Getting the type of 'self' (line 108)
            self_294571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 48), 'self', False)
            # Obtaining the member 'mask' of a type (line 108)
            mask_294572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 48), self_294571, 'mask')
            # Getting the type of 'self' (line 108)
            self_294573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 59), 'self', False)
            # Obtaining the member '_edges' of a type (line 108)
            _edges_294574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 59), self_294573, '_edges')
            # Getting the type of 'self' (line 109)
            self_294575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self', False)
            # Obtaining the member '_neighbors' of a type (line 109)
            _neighbors_294576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_294575, '_neighbors')
            
            # Getting the type of 'self' (line 109)
            self_294577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'self', False)
            # Obtaining the member 'is_delaunay' of a type (line 109)
            is_delaunay_294578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 37), self_294577, 'is_delaunay')
            # Applying the 'not' unary operator (line 109)
            result_not__294579 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 33), 'not', is_delaunay_294578)
            
            # Processing the call keyword arguments (line 107)
            kwargs_294580 = {}
            # Getting the type of '_tri' (line 107)
            _tri_294563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), '_tri', False)
            # Obtaining the member 'Triangulation' of a type (line 107)
            Triangulation_294564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 38), _tri_294563, 'Triangulation')
            # Calling Triangulation(args, kwargs) (line 107)
            Triangulation_call_result_294581 = invoke(stypy.reporting.localization.Localization(__file__, 107, 38), Triangulation_294564, *[x_294566, y_294568, triangles_294570, mask_294572, _edges_294574, _neighbors_294576, result_not__294579], **kwargs_294580)
            
            # Getting the type of 'self' (line 107)
            self_294582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self')
            # Setting the type of the member '_cpp_triangulation' of a type (line 107)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_294582, '_cpp_triangulation', Triangulation_call_result_294581)

            if more_types_in_union_294562:
                # SSA join for if statement (line 106)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 110)
        self_294583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'self')
        # Obtaining the member '_cpp_triangulation' of a type (line 110)
        _cpp_triangulation_294584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), self_294583, '_cpp_triangulation')
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', _cpp_triangulation_294584)
        
        # ################# End of 'get_cpp_triangulation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_cpp_triangulation' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_294585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294585)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_cpp_triangulation'
        return stypy_return_type_294585


    @norecursion
    def get_masked_triangles(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_masked_triangles'
        module_type_store = module_type_store.open_function_context('get_masked_triangles', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_function_name', 'Triangulation.get_masked_triangles')
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_param_names_list', [])
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.get_masked_triangles.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.get_masked_triangles', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_masked_triangles', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_masked_triangles(...)' code ##################

        unicode_294586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'unicode', u'\n        Return an array of triangles that are not masked.\n        ')
        
        
        # Getting the type of 'self' (line 116)
        self_294587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'self')
        # Obtaining the member 'mask' of a type (line 116)
        mask_294588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), self_294587, 'mask')
        # Getting the type of 'None' (line 116)
        None_294589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'None')
        # Applying the binary operator 'isnot' (line 116)
        result_is_not_294590 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), 'isnot', mask_294588, None_294589)
        
        # Testing the type of an if condition (line 116)
        if_condition_294591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_is_not_294590)
        # Assigning a type to the variable 'if_condition_294591' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_294591', if_condition_294591)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to compress(...): (line 117)
        # Processing the call arguments (line 117)
        int_294595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'int')
        # Getting the type of 'self' (line 117)
        self_294596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 47), 'self', False)
        # Obtaining the member 'mask' of a type (line 117)
        mask_294597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 47), self_294596, 'mask')
        # Applying the binary operator '-' (line 117)
        result_sub_294598 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 43), '-', int_294595, mask_294597)
        
        # Processing the call keyword arguments (line 117)
        int_294599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 63), 'int')
        keyword_294600 = int_294599
        kwargs_294601 = {'axis': keyword_294600}
        # Getting the type of 'self' (line 117)
        self_294592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'self', False)
        # Obtaining the member 'triangles' of a type (line 117)
        triangles_294593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 19), self_294592, 'triangles')
        # Obtaining the member 'compress' of a type (line 117)
        compress_294594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 19), triangles_294593, 'compress')
        # Calling compress(args, kwargs) (line 117)
        compress_call_result_294602 = invoke(stypy.reporting.localization.Localization(__file__, 117, 19), compress_294594, *[result_sub_294598], **kwargs_294601)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'stypy_return_type', compress_call_result_294602)
        # SSA branch for the else part of an if statement (line 116)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 119)
        self_294603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'self')
        # Obtaining the member 'triangles' of a type (line 119)
        triangles_294604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 19), self_294603, 'triangles')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type', triangles_294604)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_masked_triangles(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_masked_triangles' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_294605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_masked_triangles'
        return stypy_return_type_294605


    @staticmethod
    @norecursion
    def get_from_args_and_kwargs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_from_args_and_kwargs'
        module_type_store = module_type_store.open_function_context('get_from_args_and_kwargs', 121, 4, False)
        
        # Passed parameters checking function
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_type_of_self', None)
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_function_name', 'get_from_args_and_kwargs')
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_param_names_list', [])
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.get_from_args_and_kwargs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'get_from_args_and_kwargs', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_from_args_and_kwargs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_from_args_and_kwargs(...)' code ##################

        unicode_294606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'unicode', u'\n        Return a Triangulation object from the args and kwargs, and\n        the remaining args and kwargs with the consumed values removed.\n\n        There are two alternatives: either the first argument is a\n        Triangulation object, in which case it is returned, or the args\n        and kwargs are sufficient to create a new Triangulation to\n        return.  In the latter case, see Triangulation.__init__ for\n        the possible args and kwargs.\n        ')
        
        
        # Call to isinstance(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining the type of the subscript
        int_294608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'int')
        # Getting the type of 'args' (line 133)
        args_294609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___294610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), args_294609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_294611 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), getitem___294610, int_294608)
        
        # Getting the type of 'Triangulation' (line 133)
        Triangulation_294612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'Triangulation', False)
        # Processing the call keyword arguments (line 133)
        kwargs_294613 = {}
        # Getting the type of 'isinstance' (line 133)
        isinstance_294607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 133)
        isinstance_call_result_294614 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), isinstance_294607, *[subscript_call_result_294611, Triangulation_294612], **kwargs_294613)
        
        # Testing the type of an if condition (line 133)
        if_condition_294615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 8), isinstance_call_result_294614)
        # Assigning a type to the variable 'if_condition_294615' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'if_condition_294615', if_condition_294615)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 134):
        
        # Assigning a Subscript to a Name (line 134):
        
        # Obtaining the type of the subscript
        int_294616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 33), 'int')
        # Getting the type of 'args' (line 134)
        args_294617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'args')
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___294618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 28), args_294617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_294619 = invoke(stypy.reporting.localization.Localization(__file__, 134, 28), getitem___294618, int_294616)
        
        # Assigning a type to the variable 'triangulation' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'triangulation', subscript_call_result_294619)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_294620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 24), 'int')
        slice_294621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 135, 19), int_294620, None, None)
        # Getting the type of 'args' (line 135)
        args_294622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'args')
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___294623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 19), args_294622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_294624 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), getitem___294623, slice_294621)
        
        # Assigning a type to the variable 'args' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'args', subscript_call_result_294624)
        # SSA branch for the else part of an if statement (line 133)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 137):
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_294625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 21), 'int')
        # Getting the type of 'args' (line 137)
        args_294626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'args')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___294627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), args_294626, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_294628 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___294627, int_294625)
        
        # Assigning a type to the variable 'x' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'x', subscript_call_result_294628)
        
        # Assigning a Subscript to a Name (line 138):
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_294629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 21), 'int')
        # Getting the type of 'args' (line 138)
        args_294630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'args')
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___294631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), args_294630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_294632 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), getitem___294631, int_294629)
        
        # Assigning a type to the variable 'y' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'y', subscript_call_result_294632)
        
        # Assigning a Subscript to a Name (line 139):
        
        # Assigning a Subscript to a Name (line 139):
        
        # Obtaining the type of the subscript
        int_294633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'int')
        slice_294634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 139, 19), int_294633, None, None)
        # Getting the type of 'args' (line 139)
        args_294635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'args')
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___294636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 19), args_294635, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_294637 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), getitem___294636, slice_294634)
        
        # Assigning a type to the variable 'args' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'args', subscript_call_result_294637)
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to pop(...): (line 142)
        # Processing the call arguments (line 142)
        unicode_294640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 35), 'unicode', u'triangles')
        # Getting the type of 'None' (line 142)
        None_294641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'None', False)
        # Processing the call keyword arguments (line 142)
        kwargs_294642 = {}
        # Getting the type of 'kwargs' (line 142)
        kwargs_294638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 142)
        pop_294639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 24), kwargs_294638, 'pop')
        # Calling pop(args, kwargs) (line 142)
        pop_call_result_294643 = invoke(stypy.reporting.localization.Localization(__file__, 142, 24), pop_294639, *[unicode_294640, None_294641], **kwargs_294642)
        
        # Assigning a type to the variable 'triangles' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'triangles', pop_call_result_294643)
        
        # Assigning a Name to a Name (line 143):
        
        # Assigning a Name to a Name (line 143):
        # Getting the type of 'False' (line 143)
        False_294644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'False')
        # Assigning a type to the variable 'from_args' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'from_args', False_294644)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'triangles' (line 144)
        triangles_294645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'triangles')
        # Getting the type of 'None' (line 144)
        None_294646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'None')
        # Applying the binary operator 'is' (line 144)
        result_is__294647 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), 'is', triangles_294645, None_294646)
        
        
        
        # Call to len(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'args' (line 144)
        args_294649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'args', False)
        # Processing the call keyword arguments (line 144)
        kwargs_294650 = {}
        # Getting the type of 'len' (line 144)
        len_294648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'len', False)
        # Calling len(args, kwargs) (line 144)
        len_call_result_294651 = invoke(stypy.reporting.localization.Localization(__file__, 144, 37), len_294648, *[args_294649], **kwargs_294650)
        
        int_294652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 49), 'int')
        # Applying the binary operator '>' (line 144)
        result_gt_294653 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 37), '>', len_call_result_294651, int_294652)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_294654 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), 'and', result_is__294647, result_gt_294653)
        
        # Testing the type of an if condition (line 144)
        if_condition_294655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), result_and_keyword_294654)
        # Assigning a type to the variable 'if_condition_294655' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'if_condition_294655', if_condition_294655)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 145):
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        int_294656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 33), 'int')
        # Getting the type of 'args' (line 145)
        args_294657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'args')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___294658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 28), args_294657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_294659 = invoke(stypy.reporting.localization.Localization(__file__, 145, 28), getitem___294658, int_294656)
        
        # Assigning a type to the variable 'triangles' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'triangles', subscript_call_result_294659)
        
        # Assigning a Name to a Name (line 146):
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'True' (line 146)
        True_294660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'True')
        # Assigning a type to the variable 'from_args' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'from_args', True_294660)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 148)
        # Getting the type of 'triangles' (line 148)
        triangles_294661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'triangles')
        # Getting the type of 'None' (line 148)
        None_294662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'None')
        
        (may_be_294663, more_types_in_union_294664) = may_not_be_none(triangles_294661, None_294662)

        if may_be_294663:

            if more_types_in_union_294664:
                # Runtime conditional SSA (line 148)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 150):
            
            # Assigning a Call to a Name (line 150):
            
            # Call to asarray(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'triangles' (line 150)
            triangles_294667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 43), 'triangles', False)
            # Processing the call keyword arguments (line 150)
            # Getting the type of 'np' (line 150)
            np_294668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 60), 'np', False)
            # Obtaining the member 'int32' of a type (line 150)
            int32_294669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 60), np_294668, 'int32')
            keyword_294670 = int32_294669
            kwargs_294671 = {'dtype': keyword_294670}
            # Getting the type of 'np' (line 150)
            np_294665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'np', False)
            # Obtaining the member 'asarray' of a type (line 150)
            asarray_294666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 32), np_294665, 'asarray')
            # Calling asarray(args, kwargs) (line 150)
            asarray_call_result_294672 = invoke(stypy.reporting.localization.Localization(__file__, 150, 32), asarray_294666, *[triangles_294667], **kwargs_294671)
            
            # Assigning a type to the variable 'triangles' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'triangles', asarray_call_result_294672)
            # SSA branch for the except part of a try statement (line 149)
            # SSA branch for the except 'ValueError' branch of a try statement (line 149)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Name (line 152):
            
            # Assigning a Name to a Name (line 152):
            # Getting the type of 'None' (line 152)
            None_294673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'None')
            # Assigning a type to the variable 'triangles' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'triangles', None_294673)
            # SSA join for try-except statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_294664:
                # SSA join for if statement (line 148)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'triangles' (line 154)
        triangles_294674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'triangles')
        # Getting the type of 'None' (line 154)
        None_294675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'None')
        # Applying the binary operator 'isnot' (line 154)
        result_is_not_294676 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), 'isnot', triangles_294674, None_294675)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'triangles' (line 154)
        triangles_294677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'triangles')
        # Obtaining the member 'ndim' of a type (line 154)
        ndim_294678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), triangles_294677, 'ndim')
        int_294679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 60), 'int')
        # Applying the binary operator '!=' (line 154)
        result_ne_294680 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 42), '!=', ndim_294678, int_294679)
        
        
        
        # Obtaining the type of the subscript
        int_294681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 58), 'int')
        # Getting the type of 'triangles' (line 155)
        triangles_294682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'triangles')
        # Obtaining the member 'shape' of a type (line 155)
        shape_294683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 42), triangles_294682, 'shape')
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___294684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 42), shape_294683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_294685 = invoke(stypy.reporting.localization.Localization(__file__, 155, 42), getitem___294684, int_294681)
        
        int_294686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 64), 'int')
        # Applying the binary operator '!=' (line 155)
        result_ne_294687 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 42), '!=', subscript_call_result_294685, int_294686)
        
        # Applying the binary operator 'or' (line 154)
        result_or_keyword_294688 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 42), 'or', result_ne_294680, result_ne_294687)
        
        # Applying the binary operator 'and' (line 154)
        result_and_keyword_294689 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), 'and', result_is_not_294676, result_or_keyword_294688)
        
        # Testing the type of an if condition (line 154)
        if_condition_294690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 12), result_and_keyword_294689)
        # Assigning a type to the variable 'if_condition_294690' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'if_condition_294690', if_condition_294690)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 156):
        
        # Assigning a Name to a Name (line 156):
        # Getting the type of 'None' (line 156)
        None_294691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'None')
        # Assigning a type to the variable 'triangles' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'triangles', None_294691)
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'triangles' (line 158)
        triangles_294692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'triangles')
        # Getting the type of 'None' (line 158)
        None_294693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'None')
        # Applying the binary operator 'isnot' (line 158)
        result_is_not_294694 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), 'isnot', triangles_294692, None_294693)
        
        # Getting the type of 'from_args' (line 158)
        from_args_294695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 41), 'from_args')
        # Applying the binary operator 'and' (line 158)
        result_and_keyword_294696 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), 'and', result_is_not_294694, from_args_294695)
        
        # Testing the type of an if condition (line 158)
        if_condition_294697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 12), result_and_keyword_294696)
        # Assigning a type to the variable 'if_condition_294697' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'if_condition_294697', if_condition_294697)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 159):
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_294698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'int')
        slice_294699 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 23), int_294698, None, None)
        # Getting the type of 'args' (line 159)
        args_294700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'args')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___294701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), args_294700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_294702 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), getitem___294701, slice_294699)
        
        # Assigning a type to the variable 'args' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'args', subscript_call_result_294702)
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to pop(...): (line 162)
        # Processing the call arguments (line 162)
        unicode_294705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'unicode', u'mask')
        # Getting the type of 'None' (line 162)
        None_294706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'None', False)
        # Processing the call keyword arguments (line 162)
        kwargs_294707 = {}
        # Getting the type of 'kwargs' (line 162)
        kwargs_294703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 162)
        pop_294704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), kwargs_294703, 'pop')
        # Calling pop(args, kwargs) (line 162)
        pop_call_result_294708 = invoke(stypy.reporting.localization.Localization(__file__, 162, 19), pop_294704, *[unicode_294705, None_294706], **kwargs_294707)
        
        # Assigning a type to the variable 'mask' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'mask', pop_call_result_294708)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to Triangulation(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'x' (line 164)
        x_294710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 42), 'x', False)
        # Getting the type of 'y' (line 164)
        y_294711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 45), 'y', False)
        # Getting the type of 'triangles' (line 164)
        triangles_294712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 48), 'triangles', False)
        # Getting the type of 'mask' (line 164)
        mask_294713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 59), 'mask', False)
        # Processing the call keyword arguments (line 164)
        kwargs_294714 = {}
        # Getting the type of 'Triangulation' (line 164)
        Triangulation_294709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'Triangulation', False)
        # Calling Triangulation(args, kwargs) (line 164)
        Triangulation_call_result_294715 = invoke(stypy.reporting.localization.Localization(__file__, 164, 28), Triangulation_294709, *[x_294710, y_294711, triangles_294712, mask_294713], **kwargs_294714)
        
        # Assigning a type to the variable 'triangulation' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'triangulation', Triangulation_call_result_294715)
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 165)
        tuple_294716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 165)
        # Adding element type (line 165)
        # Getting the type of 'triangulation' (line 165)
        triangulation_294717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'triangulation')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 15), tuple_294716, triangulation_294717)
        # Adding element type (line 165)
        # Getting the type of 'args' (line 165)
        args_294718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 15), tuple_294716, args_294718)
        # Adding element type (line 165)
        # Getting the type of 'kwargs' (line 165)
        kwargs_294719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'kwargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 15), tuple_294716, kwargs_294719)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', tuple_294716)
        
        # ################# End of 'get_from_args_and_kwargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_from_args_and_kwargs' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_294720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_from_args_and_kwargs'
        return stypy_return_type_294720


    @norecursion
    def get_trifinder(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_trifinder'
        module_type_store = module_type_store.open_function_context('get_trifinder', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_function_name', 'Triangulation.get_trifinder')
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_param_names_list', [])
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.get_trifinder.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.get_trifinder', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_trifinder', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_trifinder(...)' code ##################

        unicode_294721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'unicode', u'\n        Return the default :class:`matplotlib.tri.TriFinder` of this\n        triangulation, creating it if necessary.  This allows the same\n        TriFinder object to be easily shared.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 173)
        # Getting the type of 'self' (line 173)
        self_294722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'self')
        # Obtaining the member '_trifinder' of a type (line 173)
        _trifinder_294723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 11), self_294722, '_trifinder')
        # Getting the type of 'None' (line 173)
        None_294724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'None')
        
        (may_be_294725, more_types_in_union_294726) = may_be_none(_trifinder_294723, None_294724)

        if may_be_294725:

            if more_types_in_union_294726:
                # Runtime conditional SSA (line 173)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 175, 12))
            
            # 'from matplotlib.tri.trifinder import TrapezoidMapTriFinder' statement (line 175)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
            import_294727 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 175, 12), 'matplotlib.tri.trifinder')

            if (type(import_294727) is not StypyTypeError):

                if (import_294727 != 'pyd_module'):
                    __import__(import_294727)
                    sys_modules_294728 = sys.modules[import_294727]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 175, 12), 'matplotlib.tri.trifinder', sys_modules_294728.module_type_store, module_type_store, ['TrapezoidMapTriFinder'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 175, 12), __file__, sys_modules_294728, sys_modules_294728.module_type_store, module_type_store)
                else:
                    from matplotlib.tri.trifinder import TrapezoidMapTriFinder

                    import_from_module(stypy.reporting.localization.Localization(__file__, 175, 12), 'matplotlib.tri.trifinder', None, module_type_store, ['TrapezoidMapTriFinder'], [TrapezoidMapTriFinder])

            else:
                # Assigning a type to the variable 'matplotlib.tri.trifinder' (line 175)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'matplotlib.tri.trifinder', import_294727)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')
            
            
            # Assigning a Call to a Attribute (line 176):
            
            # Assigning a Call to a Attribute (line 176):
            
            # Call to TrapezoidMapTriFinder(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 'self' (line 176)
            self_294730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'self', False)
            # Processing the call keyword arguments (line 176)
            kwargs_294731 = {}
            # Getting the type of 'TrapezoidMapTriFinder' (line 176)
            TrapezoidMapTriFinder_294729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'TrapezoidMapTriFinder', False)
            # Calling TrapezoidMapTriFinder(args, kwargs) (line 176)
            TrapezoidMapTriFinder_call_result_294732 = invoke(stypy.reporting.localization.Localization(__file__, 176, 30), TrapezoidMapTriFinder_294729, *[self_294730], **kwargs_294731)
            
            # Getting the type of 'self' (line 176)
            self_294733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self')
            # Setting the type of the member '_trifinder' of a type (line 176)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_294733, '_trifinder', TrapezoidMapTriFinder_call_result_294732)

            if more_types_in_union_294726:
                # SSA join for if statement (line 173)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 177)
        self_294734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'self')
        # Obtaining the member '_trifinder' of a type (line 177)
        _trifinder_294735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), self_294734, '_trifinder')
        # Assigning a type to the variable 'stypy_return_type' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type', _trifinder_294735)
        
        # ################# End of 'get_trifinder(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_trifinder' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_294736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294736)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_trifinder'
        return stypy_return_type_294736


    @norecursion
    def neighbors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'neighbors'
        module_type_store = module_type_store.open_function_context('neighbors', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.neighbors.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.neighbors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.neighbors.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.neighbors.__dict__.__setitem__('stypy_function_name', 'Triangulation.neighbors')
        Triangulation.neighbors.__dict__.__setitem__('stypy_param_names_list', [])
        Triangulation.neighbors.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.neighbors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.neighbors.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.neighbors.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.neighbors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.neighbors.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.neighbors', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'neighbors', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'neighbors(...)' code ##################

        unicode_294737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'unicode', u'\n        Return integer array of shape (ntri,3) containing neighbor\n        triangles.\n\n        For each triangle, the indices of the three triangles that\n        share the same edges, or -1 if there is no such neighboring\n        triangle.  neighbors[i,j] is the triangle that is the neighbor\n        to the edge from point index triangles[i,j] to point index\n        triangles[i,(j+1)%3].\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 191)
        # Getting the type of 'self' (line 191)
        self_294738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'self')
        # Obtaining the member '_neighbors' of a type (line 191)
        _neighbors_294739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), self_294738, '_neighbors')
        # Getting the type of 'None' (line 191)
        None_294740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'None')
        
        (may_be_294741, more_types_in_union_294742) = may_be_none(_neighbors_294739, None_294740)

        if may_be_294741:

            if more_types_in_union_294742:
                # Runtime conditional SSA (line 191)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 192):
            
            # Assigning a Call to a Attribute (line 192):
            
            # Call to get_neighbors(...): (line 192)
            # Processing the call keyword arguments (line 192)
            kwargs_294748 = {}
            
            # Call to get_cpp_triangulation(...): (line 192)
            # Processing the call keyword arguments (line 192)
            kwargs_294745 = {}
            # Getting the type of 'self' (line 192)
            self_294743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'self', False)
            # Obtaining the member 'get_cpp_triangulation' of a type (line 192)
            get_cpp_triangulation_294744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), self_294743, 'get_cpp_triangulation')
            # Calling get_cpp_triangulation(args, kwargs) (line 192)
            get_cpp_triangulation_call_result_294746 = invoke(stypy.reporting.localization.Localization(__file__, 192, 30), get_cpp_triangulation_294744, *[], **kwargs_294745)
            
            # Obtaining the member 'get_neighbors' of a type (line 192)
            get_neighbors_294747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), get_cpp_triangulation_call_result_294746, 'get_neighbors')
            # Calling get_neighbors(args, kwargs) (line 192)
            get_neighbors_call_result_294749 = invoke(stypy.reporting.localization.Localization(__file__, 192, 30), get_neighbors_294747, *[], **kwargs_294748)
            
            # Getting the type of 'self' (line 192)
            self_294750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'self')
            # Setting the type of the member '_neighbors' of a type (line 192)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), self_294750, '_neighbors', get_neighbors_call_result_294749)

            if more_types_in_union_294742:
                # SSA join for if statement (line 191)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 193)
        self_294751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'self')
        # Obtaining the member '_neighbors' of a type (line 193)
        _neighbors_294752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 15), self_294751, '_neighbors')
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', _neighbors_294752)
        
        # ################# End of 'neighbors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'neighbors' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_294753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'neighbors'
        return stypy_return_type_294753


    @norecursion
    def set_mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_mask'
        module_type_store = module_type_store.open_function_context('set_mask', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangulation.set_mask.__dict__.__setitem__('stypy_localization', localization)
        Triangulation.set_mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangulation.set_mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangulation.set_mask.__dict__.__setitem__('stypy_function_name', 'Triangulation.set_mask')
        Triangulation.set_mask.__dict__.__setitem__('stypy_param_names_list', ['mask'])
        Triangulation.set_mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangulation.set_mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangulation.set_mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangulation.set_mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangulation.set_mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangulation.set_mask.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangulation.set_mask', ['mask'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_mask', localization, ['mask'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_mask(...)' code ##################

        unicode_294754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, (-1)), 'unicode', u'\n        Set or clear the mask array.  This is either None, or a boolean\n        array of shape (ntri).\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 200)
        # Getting the type of 'mask' (line 200)
        mask_294755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'mask')
        # Getting the type of 'None' (line 200)
        None_294756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'None')
        
        (may_be_294757, more_types_in_union_294758) = may_be_none(mask_294755, None_294756)

        if may_be_294757:

            if more_types_in_union_294758:
                # Runtime conditional SSA (line 200)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 201):
            
            # Assigning a Name to a Attribute (line 201):
            # Getting the type of 'None' (line 201)
            None_294759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'None')
            # Getting the type of 'self' (line 201)
            self_294760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'self')
            # Setting the type of the member 'mask' of a type (line 201)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), self_294760, 'mask', None_294759)

            if more_types_in_union_294758:
                # Runtime conditional SSA for else branch (line 200)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_294757) or more_types_in_union_294758):
            
            # Assigning a Call to a Attribute (line 203):
            
            # Assigning a Call to a Attribute (line 203):
            
            # Call to asarray(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'mask' (line 203)
            mask_294763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'mask', False)
            # Processing the call keyword arguments (line 203)
            # Getting the type of 'bool' (line 203)
            bool_294764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'bool', False)
            keyword_294765 = bool_294764
            kwargs_294766 = {'dtype': keyword_294765}
            # Getting the type of 'np' (line 203)
            np_294761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'np', False)
            # Obtaining the member 'asarray' of a type (line 203)
            asarray_294762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 24), np_294761, 'asarray')
            # Calling asarray(args, kwargs) (line 203)
            asarray_call_result_294767 = invoke(stypy.reporting.localization.Localization(__file__, 203, 24), asarray_294762, *[mask_294763], **kwargs_294766)
            
            # Getting the type of 'self' (line 203)
            self_294768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'self')
            # Setting the type of the member 'mask' of a type (line 203)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), self_294768, 'mask', asarray_call_result_294767)
            
            
            # Getting the type of 'self' (line 204)
            self_294769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'self')
            # Obtaining the member 'mask' of a type (line 204)
            mask_294770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 15), self_294769, 'mask')
            # Obtaining the member 'shape' of a type (line 204)
            shape_294771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 15), mask_294770, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 204)
            tuple_294772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 204)
            # Adding element type (line 204)
            
            # Obtaining the type of the subscript
            int_294773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 56), 'int')
            # Getting the type of 'self' (line 204)
            self_294774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'self')
            # Obtaining the member 'triangles' of a type (line 204)
            triangles_294775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 35), self_294774, 'triangles')
            # Obtaining the member 'shape' of a type (line 204)
            shape_294776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 35), triangles_294775, 'shape')
            # Obtaining the member '__getitem__' of a type (line 204)
            getitem___294777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 35), shape_294776, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 204)
            subscript_call_result_294778 = invoke(stypy.reporting.localization.Localization(__file__, 204, 35), getitem___294777, int_294773)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 35), tuple_294772, subscript_call_result_294778)
            
            # Applying the binary operator '!=' (line 204)
            result_ne_294779 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 15), '!=', shape_294771, tuple_294772)
            
            # Testing the type of an if condition (line 204)
            if_condition_294780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 12), result_ne_294779)
            # Assigning a type to the variable 'if_condition_294780' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'if_condition_294780', if_condition_294780)
            # SSA begins for if statement (line 204)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 205)
            # Processing the call arguments (line 205)
            unicode_294782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 33), 'unicode', u'mask array must have same length as triangles array')
            # Processing the call keyword arguments (line 205)
            kwargs_294783 = {}
            # Getting the type of 'ValueError' (line 205)
            ValueError_294781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 205)
            ValueError_call_result_294784 = invoke(stypy.reporting.localization.Localization(__file__, 205, 22), ValueError_294781, *[unicode_294782], **kwargs_294783)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 205, 16), ValueError_call_result_294784, 'raise parameter', BaseException)
            # SSA join for if statement (line 204)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_294757 and more_types_in_union_294758):
                # SSA join for if statement (line 200)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 209)
        self_294785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'self')
        # Obtaining the member '_cpp_triangulation' of a type (line 209)
        _cpp_triangulation_294786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), self_294785, '_cpp_triangulation')
        # Getting the type of 'None' (line 209)
        None_294787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'None')
        # Applying the binary operator 'isnot' (line 209)
        result_is_not_294788 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), 'isnot', _cpp_triangulation_294786, None_294787)
        
        # Testing the type of an if condition (line 209)
        if_condition_294789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), result_is_not_294788)
        # Assigning a type to the variable 'if_condition_294789' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_294789', if_condition_294789)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_mask(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_294793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 45), 'self', False)
        # Obtaining the member 'mask' of a type (line 210)
        mask_294794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 45), self_294793, 'mask')
        # Processing the call keyword arguments (line 210)
        kwargs_294795 = {}
        # Getting the type of 'self' (line 210)
        self_294790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'self', False)
        # Obtaining the member '_cpp_triangulation' of a type (line 210)
        _cpp_triangulation_294791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), self_294790, '_cpp_triangulation')
        # Obtaining the member 'set_mask' of a type (line 210)
        set_mask_294792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), _cpp_triangulation_294791, 'set_mask')
        # Calling set_mask(args, kwargs) (line 210)
        set_mask_call_result_294796 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), set_mask_294792, *[mask_294794], **kwargs_294795)
        
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 213):
        
        # Assigning a Name to a Attribute (line 213):
        # Getting the type of 'None' (line 213)
        None_294797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'None')
        # Getting the type of 'self' (line 213)
        self_294798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self')
        # Setting the type of the member '_edges' of a type (line 213)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_294798, '_edges', None_294797)
        
        # Assigning a Name to a Attribute (line 214):
        
        # Assigning a Name to a Attribute (line 214):
        # Getting the type of 'None' (line 214)
        None_294799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'None')
        # Getting the type of 'self' (line 214)
        self_294800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self')
        # Setting the type of the member '_neighbors' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_294800, '_neighbors', None_294799)
        
        
        # Getting the type of 'self' (line 217)
        self_294801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'self')
        # Obtaining the member '_trifinder' of a type (line 217)
        _trifinder_294802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), self_294801, '_trifinder')
        # Getting the type of 'None' (line 217)
        None_294803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'None')
        # Applying the binary operator 'isnot' (line 217)
        result_is_not_294804 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), 'isnot', _trifinder_294802, None_294803)
        
        # Testing the type of an if condition (line 217)
        if_condition_294805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), result_is_not_294804)
        # Assigning a type to the variable 'if_condition_294805' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_294805', if_condition_294805)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _initialize(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_294809 = {}
        # Getting the type of 'self' (line 218)
        self_294806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self', False)
        # Obtaining the member '_trifinder' of a type (line 218)
        _trifinder_294807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_294806, '_trifinder')
        # Obtaining the member '_initialize' of a type (line 218)
        _initialize_294808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), _trifinder_294807, '_initialize')
        # Calling _initialize(args, kwargs) (line 218)
        _initialize_call_result_294810 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), _initialize_294808, *[], **kwargs_294809)
        
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_mask' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_294811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294811)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_mask'
        return stypy_return_type_294811


# Assigning a type to the variable 'Triangulation' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Triangulation', Triangulation)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

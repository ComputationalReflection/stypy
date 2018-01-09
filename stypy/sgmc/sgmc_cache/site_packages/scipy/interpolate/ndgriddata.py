
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Convenience interface to N-D interpolation
3: 
4: .. versionadded:: 0.9
5: 
6: '''
7: from __future__ import division, print_function, absolute_import
8: 
9: import numpy as np
10: from .interpnd import LinearNDInterpolator, NDInterpolatorBase, \
11:      CloughTocher2DInterpolator, _ndim_coords_from_arrays
12: from scipy.spatial import cKDTree
13: 
14: __all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator',
15:            'CloughTocher2DInterpolator']
16: 
17: #------------------------------------------------------------------------------
18: # Nearest-neighbour interpolation
19: #------------------------------------------------------------------------------
20: 
21: 
22: class NearestNDInterpolator(NDInterpolatorBase):
23:     '''
24:     NearestNDInterpolator(x, y)
25: 
26:     Nearest-neighbour interpolation in N dimensions.
27: 
28:     .. versionadded:: 0.9
29: 
30:     Methods
31:     -------
32:     __call__
33: 
34:     Parameters
35:     ----------
36:     x : (Npoints, Ndims) ndarray of floats
37:         Data point coordinates.
38:     y : (Npoints,) ndarray of float or complex
39:         Data values.
40:     rescale : boolean, optional
41:         Rescale points to unit cube before performing interpolation.
42:         This is useful if some of the input dimensions have
43:         incommensurable units and differ by many orders of magnitude.
44: 
45:         .. versionadded:: 0.14.0
46:     tree_options : dict, optional
47:         Options passed to the underlying ``cKDTree``.
48: 
49:         .. versionadded:: 0.17.0
50: 
51: 
52:     Notes
53:     -----
54:     Uses ``scipy.spatial.cKDTree``
55: 
56:     '''
57: 
58:     def __init__(self, x, y, rescale=False, tree_options=None):
59:         NDInterpolatorBase.__init__(self, x, y, rescale=rescale,
60:                                     need_contiguous=False,
61:                                     need_values=False)
62:         if tree_options is None:
63:             tree_options = dict()
64:         self.tree = cKDTree(self.points, **tree_options)
65:         self.values = y
66: 
67:     def __call__(self, *args):
68:         '''
69:         Evaluate interpolator at given points.
70: 
71:         Parameters
72:         ----------
73:         xi : ndarray of float, shape (..., ndim)
74:             Points where to interpolate data at.
75: 
76:         '''
77:         xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
78:         xi = self._check_call_shape(xi)
79:         xi = self._scale_x(xi)
80:         dist, i = self.tree.query(xi)
81:         return self.values[i]
82: 
83: 
84: #------------------------------------------------------------------------------
85: # Convenience interface function
86: #------------------------------------------------------------------------------
87: 
88: def griddata(points, values, xi, method='linear', fill_value=np.nan,
89:              rescale=False):
90:     '''
91:     Interpolate unstructured D-dimensional data.
92: 
93:     Parameters
94:     ----------
95:     points : ndarray of floats, shape (n, D)
96:         Data point coordinates. Can either be an array of
97:         shape (n, D), or a tuple of `ndim` arrays.
98:     values : ndarray of float or complex, shape (n,)
99:         Data values.
100:     xi : 2-D ndarray of float or tuple of 1-D array, shape (M, D)
101:         Points at which to interpolate data.
102:     method : {'linear', 'nearest', 'cubic'}, optional
103:         Method of interpolation. One of
104: 
105:         ``nearest``
106:           return the value at the data point closest to
107:           the point of interpolation.  See `NearestNDInterpolator` for
108:           more details.
109: 
110:         ``linear``
111:           tesselate the input point set to n-dimensional
112:           simplices, and interpolate linearly on each simplex.  See
113:           `LinearNDInterpolator` for more details.
114: 
115:         ``cubic`` (1-D)
116:           return the value determined from a cubic
117:           spline.
118: 
119:         ``cubic`` (2-D)
120:           return the value determined from a
121:           piecewise cubic, continuously differentiable (C1), and
122:           approximately curvature-minimizing polynomial surface. See
123:           `CloughTocher2DInterpolator` for more details.
124:     fill_value : float, optional
125:         Value used to fill in for requested points outside of the
126:         convex hull of the input points.  If not provided, then the
127:         default is ``nan``. This option has no effect for the
128:         'nearest' method.
129:     rescale : bool, optional
130:         Rescale points to unit cube before performing interpolation.
131:         This is useful if some of the input dimensions have
132:         incommensurable units and differ by many orders of magnitude.
133: 
134:         .. versionadded:: 0.14.0
135: 
136:     Notes
137:     -----
138: 
139:     .. versionadded:: 0.9
140: 
141:     Examples
142:     --------
143: 
144:     Suppose we want to interpolate the 2-D function
145: 
146:     >>> def func(x, y):
147:     ...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
148: 
149:     on a grid in [0, 1]x[0, 1]
150: 
151:     >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
152: 
153:     but we only know its values at 1000 data points:
154: 
155:     >>> points = np.random.rand(1000, 2)
156:     >>> values = func(points[:,0], points[:,1])
157: 
158:     This can be done with `griddata` -- below we try out all of the
159:     interpolation methods:
160: 
161:     >>> from scipy.interpolate import griddata
162:     >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
163:     >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
164:     >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
165: 
166:     One can see that the exact result is reproduced by all of the
167:     methods to some degree, but for this smooth function the piecewise
168:     cubic interpolant gives the best results:
169: 
170:     >>> import matplotlib.pyplot as plt
171:     >>> plt.subplot(221)
172:     >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
173:     >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
174:     >>> plt.title('Original')
175:     >>> plt.subplot(222)
176:     >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
177:     >>> plt.title('Nearest')
178:     >>> plt.subplot(223)
179:     >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
180:     >>> plt.title('Linear')
181:     >>> plt.subplot(224)
182:     >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
183:     >>> plt.title('Cubic')
184:     >>> plt.gcf().set_size_inches(6, 6)
185:     >>> plt.show()
186: 
187:     '''
188: 
189:     points = _ndim_coords_from_arrays(points)
190: 
191:     if points.ndim < 2:
192:         ndim = points.ndim
193:     else:
194:         ndim = points.shape[-1]
195: 
196:     if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
197:         from .interpolate import interp1d
198:         points = points.ravel()
199:         if isinstance(xi, tuple):
200:             if len(xi) != 1:
201:                 raise ValueError("invalid number of dimensions in xi")
202:             xi, = xi
203:         # Sort points/values together, necessary as input for interp1d
204:         idx = np.argsort(points)
205:         points = points[idx]
206:         values = values[idx]
207:         if method == 'nearest':
208:             fill_value = 'extrapolate'
209:         ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
210:                       fill_value=fill_value)
211:         return ip(xi)
212:     elif method == 'nearest':
213:         ip = NearestNDInterpolator(points, values, rescale=rescale)
214:         return ip(xi)
215:     elif method == 'linear':
216:         ip = LinearNDInterpolator(points, values, fill_value=fill_value,
217:                                   rescale=rescale)
218:         return ip(xi)
219:     elif method == 'cubic' and ndim == 2:
220:         ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value,
221:                                         rescale=rescale)
222:         return ip(xi)
223:     else:
224:         raise ValueError("Unknown interpolation method %r for "
225:                          "%d dimensional data" % (method, ndim))
226: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_71161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nConvenience interface to N-D interpolation\n\n.. versionadded:: 0.9\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71162 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_71162) is not StypyTypeError):

    if (import_71162 != 'pyd_module'):
        __import__(import_71162)
        sys_modules_71163 = sys.modules[import_71162]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_71163.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_71162)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.interpolate.interpnd import LinearNDInterpolator, NDInterpolatorBase, CloughTocher2DInterpolator, _ndim_coords_from_arrays' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71164 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.interpnd')

if (type(import_71164) is not StypyTypeError):

    if (import_71164 != 'pyd_module'):
        __import__(import_71164)
        sys_modules_71165 = sys.modules[import_71164]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.interpnd', sys_modules_71165.module_type_store, module_type_store, ['LinearNDInterpolator', 'NDInterpolatorBase', 'CloughTocher2DInterpolator', '_ndim_coords_from_arrays'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_71165, sys_modules_71165.module_type_store, module_type_store)
    else:
        from scipy.interpolate.interpnd import LinearNDInterpolator, NDInterpolatorBase, CloughTocher2DInterpolator, _ndim_coords_from_arrays

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.interpnd', None, module_type_store, ['LinearNDInterpolator', 'NDInterpolatorBase', 'CloughTocher2DInterpolator', '_ndim_coords_from_arrays'], [LinearNDInterpolator, NDInterpolatorBase, CloughTocher2DInterpolator, _ndim_coords_from_arrays])

else:
    # Assigning a type to the variable 'scipy.interpolate.interpnd' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.interpnd', import_71164)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.spatial import cKDTree' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_71166 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.spatial')

if (type(import_71166) is not StypyTypeError):

    if (import_71166 != 'pyd_module'):
        __import__(import_71166)
        sys_modules_71167 = sys.modules[import_71166]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.spatial', sys_modules_71167.module_type_store, module_type_store, ['cKDTree'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_71167, sys_modules_71167.module_type_store, module_type_store)
    else:
        from scipy.spatial import cKDTree

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.spatial', None, module_type_store, ['cKDTree'], [cKDTree])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.spatial', import_71166)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator', 'CloughTocher2DInterpolator']
module_type_store.set_exportable_members(['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator', 'CloughTocher2DInterpolator'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_71168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_71169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'griddata')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_71168, str_71169)
# Adding element type (line 14)
str_71170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', 'NearestNDInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_71168, str_71170)
# Adding element type (line 14)
str_71171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'str', 'LinearNDInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_71168, str_71171)
# Adding element type (line 14)
str_71172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'CloughTocher2DInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_71168, str_71172)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_71168)
# Declaration of the 'NearestNDInterpolator' class
# Getting the type of 'NDInterpolatorBase' (line 22)
NDInterpolatorBase_71173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'NDInterpolatorBase')

class NearestNDInterpolator(NDInterpolatorBase_71173, ):
    str_71174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    NearestNDInterpolator(x, y)\n\n    Nearest-neighbour interpolation in N dimensions.\n\n    .. versionadded:: 0.9\n\n    Methods\n    -------\n    __call__\n\n    Parameters\n    ----------\n    x : (Npoints, Ndims) ndarray of floats\n        Data point coordinates.\n    y : (Npoints,) ndarray of float or complex\n        Data values.\n    rescale : boolean, optional\n        Rescale points to unit cube before performing interpolation.\n        This is useful if some of the input dimensions have\n        incommensurable units and differ by many orders of magnitude.\n\n        .. versionadded:: 0.14.0\n    tree_options : dict, optional\n        Options passed to the underlying ``cKDTree``.\n\n        .. versionadded:: 0.17.0\n\n\n    Notes\n    -----\n    Uses ``scipy.spatial.cKDTree``\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 58)
        False_71175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'False')
        # Getting the type of 'None' (line 58)
        None_71176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 57), 'None')
        defaults = [False_71175, None_71176]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NearestNDInterpolator.__init__', ['x', 'y', 'rescale', 'tree_options'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'rescale', 'tree_options'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'self' (line 59)
        self_71179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'self', False)
        # Getting the type of 'x' (line 59)
        x_71180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'x', False)
        # Getting the type of 'y' (line 59)
        y_71181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'y', False)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'rescale' (line 59)
        rescale_71182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'rescale', False)
        keyword_71183 = rescale_71182
        # Getting the type of 'False' (line 60)
        False_71184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 52), 'False', False)
        keyword_71185 = False_71184
        # Getting the type of 'False' (line 61)
        False_71186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 48), 'False', False)
        keyword_71187 = False_71186
        kwargs_71188 = {'rescale': keyword_71183, 'need_contiguous': keyword_71185, 'need_values': keyword_71187}
        # Getting the type of 'NDInterpolatorBase' (line 59)
        NDInterpolatorBase_71177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'NDInterpolatorBase', False)
        # Obtaining the member '__init__' of a type (line 59)
        init___71178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), NDInterpolatorBase_71177, '__init__')
        # Calling __init__(args, kwargs) (line 59)
        init___call_result_71189 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), init___71178, *[self_71179, x_71180, y_71181], **kwargs_71188)
        
        
        # Type idiom detected: calculating its left and rigth part (line 62)
        # Getting the type of 'tree_options' (line 62)
        tree_options_71190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'tree_options')
        # Getting the type of 'None' (line 62)
        None_71191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'None')
        
        (may_be_71192, more_types_in_union_71193) = may_be_none(tree_options_71190, None_71191)

        if may_be_71192:

            if more_types_in_union_71193:
                # Runtime conditional SSA (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 63):
            
            # Assigning a Call to a Name (line 63):
            
            # Call to dict(...): (line 63)
            # Processing the call keyword arguments (line 63)
            kwargs_71195 = {}
            # Getting the type of 'dict' (line 63)
            dict_71194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'dict', False)
            # Calling dict(args, kwargs) (line 63)
            dict_call_result_71196 = invoke(stypy.reporting.localization.Localization(__file__, 63, 27), dict_71194, *[], **kwargs_71195)
            
            # Assigning a type to the variable 'tree_options' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'tree_options', dict_call_result_71196)

            if more_types_in_union_71193:
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 64):
        
        # Assigning a Call to a Attribute (line 64):
        
        # Call to cKDTree(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_71198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'self', False)
        # Obtaining the member 'points' of a type (line 64)
        points_71199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), self_71198, 'points')
        # Processing the call keyword arguments (line 64)
        # Getting the type of 'tree_options' (line 64)
        tree_options_71200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'tree_options', False)
        kwargs_71201 = {'tree_options_71200': tree_options_71200}
        # Getting the type of 'cKDTree' (line 64)
        cKDTree_71197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'cKDTree', False)
        # Calling cKDTree(args, kwargs) (line 64)
        cKDTree_call_result_71202 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), cKDTree_71197, *[points_71199], **kwargs_71201)
        
        # Getting the type of 'self' (line 64)
        self_71203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'tree' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_71203, 'tree', cKDTree_call_result_71202)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'y' (line 65)
        y_71204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'y')
        # Getting the type of 'self' (line 65)
        self_71205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'values' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_71205, 'values', y_71204)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_localization', localization)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_function_name', 'NearestNDInterpolator.__call__')
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NearestNDInterpolator.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NearestNDInterpolator.__call__', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_71206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        Evaluate interpolator at given points.\n\n        Parameters\n        ----------\n        xi : ndarray of float, shape (..., ndim)\n            Points where to interpolate data at.\n\n        ')
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to _ndim_coords_from_arrays(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'args' (line 77)
        args_71208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'args', False)
        # Processing the call keyword arguments (line 77)
        
        # Obtaining the type of the subscript
        int_71209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 67), 'int')
        # Getting the type of 'self' (line 77)
        self_71210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 49), 'self', False)
        # Obtaining the member 'points' of a type (line 77)
        points_71211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 49), self_71210, 'points')
        # Obtaining the member 'shape' of a type (line 77)
        shape_71212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 49), points_71211, 'shape')
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___71213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 49), shape_71212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_71214 = invoke(stypy.reporting.localization.Localization(__file__, 77, 49), getitem___71213, int_71209)
        
        keyword_71215 = subscript_call_result_71214
        kwargs_71216 = {'ndim': keyword_71215}
        # Getting the type of '_ndim_coords_from_arrays' (line 77)
        _ndim_coords_from_arrays_71207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), '_ndim_coords_from_arrays', False)
        # Calling _ndim_coords_from_arrays(args, kwargs) (line 77)
        _ndim_coords_from_arrays_call_result_71217 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), _ndim_coords_from_arrays_71207, *[args_71208], **kwargs_71216)
        
        # Assigning a type to the variable 'xi' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'xi', _ndim_coords_from_arrays_call_result_71217)
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to _check_call_shape(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'xi' (line 78)
        xi_71220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'xi', False)
        # Processing the call keyword arguments (line 78)
        kwargs_71221 = {}
        # Getting the type of 'self' (line 78)
        self_71218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'self', False)
        # Obtaining the member '_check_call_shape' of a type (line 78)
        _check_call_shape_71219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 13), self_71218, '_check_call_shape')
        # Calling _check_call_shape(args, kwargs) (line 78)
        _check_call_shape_call_result_71222 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), _check_call_shape_71219, *[xi_71220], **kwargs_71221)
        
        # Assigning a type to the variable 'xi' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'xi', _check_call_shape_call_result_71222)
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to _scale_x(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'xi' (line 79)
        xi_71225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'xi', False)
        # Processing the call keyword arguments (line 79)
        kwargs_71226 = {}
        # Getting the type of 'self' (line 79)
        self_71223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'self', False)
        # Obtaining the member '_scale_x' of a type (line 79)
        _scale_x_71224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), self_71223, '_scale_x')
        # Calling _scale_x(args, kwargs) (line 79)
        _scale_x_call_result_71227 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), _scale_x_71224, *[xi_71225], **kwargs_71226)
        
        # Assigning a type to the variable 'xi' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'xi', _scale_x_call_result_71227)
        
        # Assigning a Call to a Tuple (line 80):
        
        # Assigning a Subscript to a Name (line 80):
        
        # Obtaining the type of the subscript
        int_71228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'int')
        
        # Call to query(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'xi' (line 80)
        xi_71232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'xi', False)
        # Processing the call keyword arguments (line 80)
        kwargs_71233 = {}
        # Getting the type of 'self' (line 80)
        self_71229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'self', False)
        # Obtaining the member 'tree' of a type (line 80)
        tree_71230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), self_71229, 'tree')
        # Obtaining the member 'query' of a type (line 80)
        query_71231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), tree_71230, 'query')
        # Calling query(args, kwargs) (line 80)
        query_call_result_71234 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), query_71231, *[xi_71232], **kwargs_71233)
        
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___71235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), query_call_result_71234, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_71236 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), getitem___71235, int_71228)
        
        # Assigning a type to the variable 'tuple_var_assignment_71158' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_71158', subscript_call_result_71236)
        
        # Assigning a Subscript to a Name (line 80):
        
        # Obtaining the type of the subscript
        int_71237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'int')
        
        # Call to query(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'xi' (line 80)
        xi_71241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'xi', False)
        # Processing the call keyword arguments (line 80)
        kwargs_71242 = {}
        # Getting the type of 'self' (line 80)
        self_71238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'self', False)
        # Obtaining the member 'tree' of a type (line 80)
        tree_71239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), self_71238, 'tree')
        # Obtaining the member 'query' of a type (line 80)
        query_71240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), tree_71239, 'query')
        # Calling query(args, kwargs) (line 80)
        query_call_result_71243 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), query_71240, *[xi_71241], **kwargs_71242)
        
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___71244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), query_call_result_71243, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_71245 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), getitem___71244, int_71237)
        
        # Assigning a type to the variable 'tuple_var_assignment_71159' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_71159', subscript_call_result_71245)
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'tuple_var_assignment_71158' (line 80)
        tuple_var_assignment_71158_71246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_71158')
        # Assigning a type to the variable 'dist' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'dist', tuple_var_assignment_71158_71246)
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'tuple_var_assignment_71159' (line 80)
        tuple_var_assignment_71159_71247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'tuple_var_assignment_71159')
        # Assigning a type to the variable 'i' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'i', tuple_var_assignment_71159_71247)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 81)
        i_71248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'i')
        # Getting the type of 'self' (line 81)
        self_71249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'self')
        # Obtaining the member 'values' of a type (line 81)
        values_71250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), self_71249, 'values')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___71251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), values_71250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_71252 = invoke(stypy.reporting.localization.Localization(__file__, 81, 15), getitem___71251, i_71248)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', subscript_call_result_71252)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_71253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_71253


# Assigning a type to the variable 'NearestNDInterpolator' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'NearestNDInterpolator', NearestNDInterpolator)

@norecursion
def griddata(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_71254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 40), 'str', 'linear')
    # Getting the type of 'np' (line 88)
    np_71255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 61), 'np')
    # Obtaining the member 'nan' of a type (line 88)
    nan_71256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 61), np_71255, 'nan')
    # Getting the type of 'False' (line 89)
    False_71257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'False')
    defaults = [str_71254, nan_71256, False_71257]
    # Create a new context for function 'griddata'
    module_type_store = module_type_store.open_function_context('griddata', 88, 0, False)
    
    # Passed parameters checking function
    griddata.stypy_localization = localization
    griddata.stypy_type_of_self = None
    griddata.stypy_type_store = module_type_store
    griddata.stypy_function_name = 'griddata'
    griddata.stypy_param_names_list = ['points', 'values', 'xi', 'method', 'fill_value', 'rescale']
    griddata.stypy_varargs_param_name = None
    griddata.stypy_kwargs_param_name = None
    griddata.stypy_call_defaults = defaults
    griddata.stypy_call_varargs = varargs
    griddata.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'griddata', ['points', 'values', 'xi', 'method', 'fill_value', 'rescale'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'griddata', localization, ['points', 'values', 'xi', 'method', 'fill_value', 'rescale'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'griddata(...)' code ##################

    str_71258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', "\n    Interpolate unstructured D-dimensional data.\n\n    Parameters\n    ----------\n    points : ndarray of floats, shape (n, D)\n        Data point coordinates. Can either be an array of\n        shape (n, D), or a tuple of `ndim` arrays.\n    values : ndarray of float or complex, shape (n,)\n        Data values.\n    xi : 2-D ndarray of float or tuple of 1-D array, shape (M, D)\n        Points at which to interpolate data.\n    method : {'linear', 'nearest', 'cubic'}, optional\n        Method of interpolation. One of\n\n        ``nearest``\n          return the value at the data point closest to\n          the point of interpolation.  See `NearestNDInterpolator` for\n          more details.\n\n        ``linear``\n          tesselate the input point set to n-dimensional\n          simplices, and interpolate linearly on each simplex.  See\n          `LinearNDInterpolator` for more details.\n\n        ``cubic`` (1-D)\n          return the value determined from a cubic\n          spline.\n\n        ``cubic`` (2-D)\n          return the value determined from a\n          piecewise cubic, continuously differentiable (C1), and\n          approximately curvature-minimizing polynomial surface. See\n          `CloughTocher2DInterpolator` for more details.\n    fill_value : float, optional\n        Value used to fill in for requested points outside of the\n        convex hull of the input points.  If not provided, then the\n        default is ``nan``. This option has no effect for the\n        'nearest' method.\n    rescale : bool, optional\n        Rescale points to unit cube before performing interpolation.\n        This is useful if some of the input dimensions have\n        incommensurable units and differ by many orders of magnitude.\n\n        .. versionadded:: 0.14.0\n\n    Notes\n    -----\n\n    .. versionadded:: 0.9\n\n    Examples\n    --------\n\n    Suppose we want to interpolate the 2-D function\n\n    >>> def func(x, y):\n    ...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2\n\n    on a grid in [0, 1]x[0, 1]\n\n    >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]\n\n    but we only know its values at 1000 data points:\n\n    >>> points = np.random.rand(1000, 2)\n    >>> values = func(points[:,0], points[:,1])\n\n    This can be done with `griddata` -- below we try out all of the\n    interpolation methods:\n\n    >>> from scipy.interpolate import griddata\n    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')\n    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')\n    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')\n\n    One can see that the exact result is reproduced by all of the\n    methods to some degree, but for this smooth function the piecewise\n    cubic interpolant gives the best results:\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.subplot(221)\n    >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')\n    >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)\n    >>> plt.title('Original')\n    >>> plt.subplot(222)\n    >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')\n    >>> plt.title('Nearest')\n    >>> plt.subplot(223)\n    >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')\n    >>> plt.title('Linear')\n    >>> plt.subplot(224)\n    >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')\n    >>> plt.title('Cubic')\n    >>> plt.gcf().set_size_inches(6, 6)\n    >>> plt.show()\n\n    ")
    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to _ndim_coords_from_arrays(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'points' (line 189)
    points_71260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 38), 'points', False)
    # Processing the call keyword arguments (line 189)
    kwargs_71261 = {}
    # Getting the type of '_ndim_coords_from_arrays' (line 189)
    _ndim_coords_from_arrays_71259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), '_ndim_coords_from_arrays', False)
    # Calling _ndim_coords_from_arrays(args, kwargs) (line 189)
    _ndim_coords_from_arrays_call_result_71262 = invoke(stypy.reporting.localization.Localization(__file__, 189, 13), _ndim_coords_from_arrays_71259, *[points_71260], **kwargs_71261)
    
    # Assigning a type to the variable 'points' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'points', _ndim_coords_from_arrays_call_result_71262)
    
    
    # Getting the type of 'points' (line 191)
    points_71263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 7), 'points')
    # Obtaining the member 'ndim' of a type (line 191)
    ndim_71264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 7), points_71263, 'ndim')
    int_71265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 21), 'int')
    # Applying the binary operator '<' (line 191)
    result_lt_71266 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 7), '<', ndim_71264, int_71265)
    
    # Testing the type of an if condition (line 191)
    if_condition_71267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 4), result_lt_71266)
    # Assigning a type to the variable 'if_condition_71267' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'if_condition_71267', if_condition_71267)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 192):
    
    # Assigning a Attribute to a Name (line 192):
    # Getting the type of 'points' (line 192)
    points_71268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'points')
    # Obtaining the member 'ndim' of a type (line 192)
    ndim_71269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 15), points_71268, 'ndim')
    # Assigning a type to the variable 'ndim' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'ndim', ndim_71269)
    # SSA branch for the else part of an if statement (line 191)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 194):
    
    # Assigning a Subscript to a Name (line 194):
    
    # Obtaining the type of the subscript
    int_71270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'int')
    # Getting the type of 'points' (line 194)
    points_71271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'points')
    # Obtaining the member 'shape' of a type (line 194)
    shape_71272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), points_71271, 'shape')
    # Obtaining the member '__getitem__' of a type (line 194)
    getitem___71273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), shape_71272, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 194)
    subscript_call_result_71274 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), getitem___71273, int_71270)
    
    # Assigning a type to the variable 'ndim' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'ndim', subscript_call_result_71274)
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ndim' (line 196)
    ndim_71275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'ndim')
    int_71276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 15), 'int')
    # Applying the binary operator '==' (line 196)
    result_eq_71277 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 7), '==', ndim_71275, int_71276)
    
    
    # Getting the type of 'method' (line 196)
    method_71278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'method')
    
    # Obtaining an instance of the builtin type 'tuple' (line 196)
    tuple_71279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 196)
    # Adding element type (line 196)
    str_71280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'str', 'nearest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), tuple_71279, str_71280)
    # Adding element type (line 196)
    str_71281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 43), 'str', 'linear')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), tuple_71279, str_71281)
    # Adding element type (line 196)
    str_71282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 53), 'str', 'cubic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 32), tuple_71279, str_71282)
    
    # Applying the binary operator 'in' (line 196)
    result_contains_71283 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 21), 'in', method_71278, tuple_71279)
    
    # Applying the binary operator 'and' (line 196)
    result_and_keyword_71284 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 7), 'and', result_eq_71277, result_contains_71283)
    
    # Testing the type of an if condition (line 196)
    if_condition_71285 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), result_and_keyword_71284)
    # Assigning a type to the variable 'if_condition_71285' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_71285', if_condition_71285)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 197, 8))
    
    # 'from scipy.interpolate.interpolate import interp1d' statement (line 197)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
    import_71286 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 197, 8), 'scipy.interpolate.interpolate')

    if (type(import_71286) is not StypyTypeError):

        if (import_71286 != 'pyd_module'):
            __import__(import_71286)
            sys_modules_71287 = sys.modules[import_71286]
            import_from_module(stypy.reporting.localization.Localization(__file__, 197, 8), 'scipy.interpolate.interpolate', sys_modules_71287.module_type_store, module_type_store, ['interp1d'])
            nest_module(stypy.reporting.localization.Localization(__file__, 197, 8), __file__, sys_modules_71287, sys_modules_71287.module_type_store, module_type_store)
        else:
            from scipy.interpolate.interpolate import interp1d

            import_from_module(stypy.reporting.localization.Localization(__file__, 197, 8), 'scipy.interpolate.interpolate', None, module_type_store, ['interp1d'], [interp1d])

    else:
        # Assigning a type to the variable 'scipy.interpolate.interpolate' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'scipy.interpolate.interpolate', import_71286)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')
    
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to ravel(...): (line 198)
    # Processing the call keyword arguments (line 198)
    kwargs_71290 = {}
    # Getting the type of 'points' (line 198)
    points_71288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'points', False)
    # Obtaining the member 'ravel' of a type (line 198)
    ravel_71289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), points_71288, 'ravel')
    # Calling ravel(args, kwargs) (line 198)
    ravel_call_result_71291 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), ravel_71289, *[], **kwargs_71290)
    
    # Assigning a type to the variable 'points' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'points', ravel_call_result_71291)
    
    # Type idiom detected: calculating its left and rigth part (line 199)
    # Getting the type of 'tuple' (line 199)
    tuple_71292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'tuple')
    # Getting the type of 'xi' (line 199)
    xi_71293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'xi')
    
    (may_be_71294, more_types_in_union_71295) = may_be_subtype(tuple_71292, xi_71293)

    if may_be_71294:

        if more_types_in_union_71295:
            # Runtime conditional SSA (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'xi' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'xi', remove_not_subtype_from_union(xi_71293, tuple))
        
        
        
        # Call to len(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'xi' (line 200)
        xi_71297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'xi', False)
        # Processing the call keyword arguments (line 200)
        kwargs_71298 = {}
        # Getting the type of 'len' (line 200)
        len_71296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'len', False)
        # Calling len(args, kwargs) (line 200)
        len_call_result_71299 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), len_71296, *[xi_71297], **kwargs_71298)
        
        int_71300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 26), 'int')
        # Applying the binary operator '!=' (line 200)
        result_ne_71301 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), '!=', len_call_result_71299, int_71300)
        
        # Testing the type of an if condition (line 200)
        if_condition_71302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_ne_71301)
        # Assigning a type to the variable 'if_condition_71302' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_71302', if_condition_71302)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 201)
        # Processing the call arguments (line 201)
        str_71304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 33), 'str', 'invalid number of dimensions in xi')
        # Processing the call keyword arguments (line 201)
        kwargs_71305 = {}
        # Getting the type of 'ValueError' (line 201)
        ValueError_71303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 201)
        ValueError_call_result_71306 = invoke(stypy.reporting.localization.Localization(__file__, 201, 22), ValueError_71303, *[str_71304], **kwargs_71305)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 201, 16), ValueError_call_result_71306, 'raise parameter', BaseException)
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 202):
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_71307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'int')
        # Getting the type of 'xi' (line 202)
        xi_71308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'xi')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___71309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), xi_71308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_71310 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), getitem___71309, int_71307)
        
        # Assigning a type to the variable 'tuple_var_assignment_71160' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'tuple_var_assignment_71160', subscript_call_result_71310)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'tuple_var_assignment_71160' (line 202)
        tuple_var_assignment_71160_71311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'tuple_var_assignment_71160')
        # Assigning a type to the variable 'xi' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'xi', tuple_var_assignment_71160_71311)

        if more_types_in_union_71295:
            # SSA join for if statement (line 199)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to argsort(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'points' (line 204)
    points_71314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'points', False)
    # Processing the call keyword arguments (line 204)
    kwargs_71315 = {}
    # Getting the type of 'np' (line 204)
    np_71312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 'np', False)
    # Obtaining the member 'argsort' of a type (line 204)
    argsort_71313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 14), np_71312, 'argsort')
    # Calling argsort(args, kwargs) (line 204)
    argsort_call_result_71316 = invoke(stypy.reporting.localization.Localization(__file__, 204, 14), argsort_71313, *[points_71314], **kwargs_71315)
    
    # Assigning a type to the variable 'idx' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'idx', argsort_call_result_71316)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idx' (line 205)
    idx_71317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'idx')
    # Getting the type of 'points' (line 205)
    points_71318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'points')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___71319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 17), points_71318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_71320 = invoke(stypy.reporting.localization.Localization(__file__, 205, 17), getitem___71319, idx_71317)
    
    # Assigning a type to the variable 'points' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'points', subscript_call_result_71320)
    
    # Assigning a Subscript to a Name (line 206):
    
    # Assigning a Subscript to a Name (line 206):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idx' (line 206)
    idx_71321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'idx')
    # Getting the type of 'values' (line 206)
    values_71322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'values')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___71323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 17), values_71322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_71324 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), getitem___71323, idx_71321)
    
    # Assigning a type to the variable 'values' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'values', subscript_call_result_71324)
    
    
    # Getting the type of 'method' (line 207)
    method_71325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'method')
    str_71326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 21), 'str', 'nearest')
    # Applying the binary operator '==' (line 207)
    result_eq_71327 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), '==', method_71325, str_71326)
    
    # Testing the type of an if condition (line 207)
    if_condition_71328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_eq_71327)
    # Assigning a type to the variable 'if_condition_71328' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_71328', if_condition_71328)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 208):
    
    # Assigning a Str to a Name (line 208):
    str_71329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'str', 'extrapolate')
    # Assigning a type to the variable 'fill_value' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'fill_value', str_71329)
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to interp1d(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'points' (line 209)
    points_71331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 'points', False)
    # Getting the type of 'values' (line 209)
    values_71332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'values', False)
    # Processing the call keyword arguments (line 209)
    # Getting the type of 'method' (line 209)
    method_71333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'method', False)
    keyword_71334 = method_71333
    int_71335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 56), 'int')
    keyword_71336 = int_71335
    # Getting the type of 'False' (line 209)
    False_71337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 72), 'False', False)
    keyword_71338 = False_71337
    # Getting the type of 'fill_value' (line 210)
    fill_value_71339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 33), 'fill_value', False)
    keyword_71340 = fill_value_71339
    kwargs_71341 = {'bounds_error': keyword_71338, 'kind': keyword_71334, 'fill_value': keyword_71340, 'axis': keyword_71336}
    # Getting the type of 'interp1d' (line 209)
    interp1d_71330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'interp1d', False)
    # Calling interp1d(args, kwargs) (line 209)
    interp1d_call_result_71342 = invoke(stypy.reporting.localization.Localization(__file__, 209, 13), interp1d_71330, *[points_71331, values_71332], **kwargs_71341)
    
    # Assigning a type to the variable 'ip' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'ip', interp1d_call_result_71342)
    
    # Call to ip(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'xi' (line 211)
    xi_71344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'xi', False)
    # Processing the call keyword arguments (line 211)
    kwargs_71345 = {}
    # Getting the type of 'ip' (line 211)
    ip_71343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'ip', False)
    # Calling ip(args, kwargs) (line 211)
    ip_call_result_71346 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), ip_71343, *[xi_71344], **kwargs_71345)
    
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', ip_call_result_71346)
    # SSA branch for the else part of an if statement (line 196)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 212)
    method_71347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 9), 'method')
    str_71348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 19), 'str', 'nearest')
    # Applying the binary operator '==' (line 212)
    result_eq_71349 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 9), '==', method_71347, str_71348)
    
    # Testing the type of an if condition (line 212)
    if_condition_71350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 9), result_eq_71349)
    # Assigning a type to the variable 'if_condition_71350' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 9), 'if_condition_71350', if_condition_71350)
    # SSA begins for if statement (line 212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to NearestNDInterpolator(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'points' (line 213)
    points_71352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'points', False)
    # Getting the type of 'values' (line 213)
    values_71353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'values', False)
    # Processing the call keyword arguments (line 213)
    # Getting the type of 'rescale' (line 213)
    rescale_71354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 59), 'rescale', False)
    keyword_71355 = rescale_71354
    kwargs_71356 = {'rescale': keyword_71355}
    # Getting the type of 'NearestNDInterpolator' (line 213)
    NearestNDInterpolator_71351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'NearestNDInterpolator', False)
    # Calling NearestNDInterpolator(args, kwargs) (line 213)
    NearestNDInterpolator_call_result_71357 = invoke(stypy.reporting.localization.Localization(__file__, 213, 13), NearestNDInterpolator_71351, *[points_71352, values_71353], **kwargs_71356)
    
    # Assigning a type to the variable 'ip' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'ip', NearestNDInterpolator_call_result_71357)
    
    # Call to ip(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'xi' (line 214)
    xi_71359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'xi', False)
    # Processing the call keyword arguments (line 214)
    kwargs_71360 = {}
    # Getting the type of 'ip' (line 214)
    ip_71358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'ip', False)
    # Calling ip(args, kwargs) (line 214)
    ip_call_result_71361 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), ip_71358, *[xi_71359], **kwargs_71360)
    
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', ip_call_result_71361)
    # SSA branch for the else part of an if statement (line 212)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 215)
    method_71362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'method')
    str_71363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 19), 'str', 'linear')
    # Applying the binary operator '==' (line 215)
    result_eq_71364 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 9), '==', method_71362, str_71363)
    
    # Testing the type of an if condition (line 215)
    if_condition_71365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 9), result_eq_71364)
    # Assigning a type to the variable 'if_condition_71365' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'if_condition_71365', if_condition_71365)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to LinearNDInterpolator(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'points' (line 216)
    points_71367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'points', False)
    # Getting the type of 'values' (line 216)
    values_71368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 42), 'values', False)
    # Processing the call keyword arguments (line 216)
    # Getting the type of 'fill_value' (line 216)
    fill_value_71369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 61), 'fill_value', False)
    keyword_71370 = fill_value_71369
    # Getting the type of 'rescale' (line 217)
    rescale_71371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 42), 'rescale', False)
    keyword_71372 = rescale_71371
    kwargs_71373 = {'rescale': keyword_71372, 'fill_value': keyword_71370}
    # Getting the type of 'LinearNDInterpolator' (line 216)
    LinearNDInterpolator_71366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'LinearNDInterpolator', False)
    # Calling LinearNDInterpolator(args, kwargs) (line 216)
    LinearNDInterpolator_call_result_71374 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), LinearNDInterpolator_71366, *[points_71367, values_71368], **kwargs_71373)
    
    # Assigning a type to the variable 'ip' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'ip', LinearNDInterpolator_call_result_71374)
    
    # Call to ip(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'xi' (line 218)
    xi_71376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'xi', False)
    # Processing the call keyword arguments (line 218)
    kwargs_71377 = {}
    # Getting the type of 'ip' (line 218)
    ip_71375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'ip', False)
    # Calling ip(args, kwargs) (line 218)
    ip_call_result_71378 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), ip_71375, *[xi_71376], **kwargs_71377)
    
    # Assigning a type to the variable 'stypy_return_type' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', ip_call_result_71378)
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 219)
    method_71379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'method')
    str_71380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'str', 'cubic')
    # Applying the binary operator '==' (line 219)
    result_eq_71381 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 9), '==', method_71379, str_71380)
    
    
    # Getting the type of 'ndim' (line 219)
    ndim_71382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'ndim')
    int_71383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 39), 'int')
    # Applying the binary operator '==' (line 219)
    result_eq_71384 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 31), '==', ndim_71382, int_71383)
    
    # Applying the binary operator 'and' (line 219)
    result_and_keyword_71385 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 9), 'and', result_eq_71381, result_eq_71384)
    
    # Testing the type of an if condition (line 219)
    if_condition_71386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 9), result_and_keyword_71385)
    # Assigning a type to the variable 'if_condition_71386' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'if_condition_71386', if_condition_71386)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to CloughTocher2DInterpolator(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'points' (line 220)
    points_71388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'points', False)
    # Getting the type of 'values' (line 220)
    values_71389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 48), 'values', False)
    # Processing the call keyword arguments (line 220)
    # Getting the type of 'fill_value' (line 220)
    fill_value_71390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 67), 'fill_value', False)
    keyword_71391 = fill_value_71390
    # Getting the type of 'rescale' (line 221)
    rescale_71392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 48), 'rescale', False)
    keyword_71393 = rescale_71392
    kwargs_71394 = {'rescale': keyword_71393, 'fill_value': keyword_71391}
    # Getting the type of 'CloughTocher2DInterpolator' (line 220)
    CloughTocher2DInterpolator_71387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'CloughTocher2DInterpolator', False)
    # Calling CloughTocher2DInterpolator(args, kwargs) (line 220)
    CloughTocher2DInterpolator_call_result_71395 = invoke(stypy.reporting.localization.Localization(__file__, 220, 13), CloughTocher2DInterpolator_71387, *[points_71388, values_71389], **kwargs_71394)
    
    # Assigning a type to the variable 'ip' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'ip', CloughTocher2DInterpolator_call_result_71395)
    
    # Call to ip(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'xi' (line 222)
    xi_71397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'xi', False)
    # Processing the call keyword arguments (line 222)
    kwargs_71398 = {}
    # Getting the type of 'ip' (line 222)
    ip_71396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'ip', False)
    # Calling ip(args, kwargs) (line 222)
    ip_call_result_71399 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), ip_71396, *[xi_71397], **kwargs_71398)
    
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'stypy_return_type', ip_call_result_71399)
    # SSA branch for the else part of an if statement (line 219)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 224)
    # Processing the call arguments (line 224)
    str_71401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 25), 'str', 'Unknown interpolation method %r for %d dimensional data')
    
    # Obtaining an instance of the builtin type 'tuple' (line 225)
    tuple_71402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 225)
    # Adding element type (line 225)
    # Getting the type of 'method' (line 225)
    method_71403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'method', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 50), tuple_71402, method_71403)
    # Adding element type (line 225)
    # Getting the type of 'ndim' (line 225)
    ndim_71404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 58), 'ndim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 50), tuple_71402, ndim_71404)
    
    # Applying the binary operator '%' (line 224)
    result_mod_71405 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 25), '%', str_71401, tuple_71402)
    
    # Processing the call keyword arguments (line 224)
    kwargs_71406 = {}
    # Getting the type of 'ValueError' (line 224)
    ValueError_71400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 224)
    ValueError_call_result_71407 = invoke(stypy.reporting.localization.Localization(__file__, 224, 14), ValueError_71400, *[result_mod_71405], **kwargs_71406)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 8), ValueError_call_result_71407, 'raise parameter', BaseException)
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 212)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'griddata(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'griddata' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_71408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_71408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'griddata'
    return stypy_return_type_71408

# Assigning a type to the variable 'griddata' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'griddata', griddata)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

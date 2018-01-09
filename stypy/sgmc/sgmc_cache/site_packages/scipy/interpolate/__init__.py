
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''========================================
2: Interpolation (:mod:`scipy.interpolate`)
3: ========================================
4: 
5: .. currentmodule:: scipy.interpolate
6: 
7: Sub-package for objects used in interpolation.
8: 
9: As listed below, this sub-package contains spline functions and classes,
10: one-dimensional and multi-dimensional (univariate and multivariate)
11: interpolation classes, Lagrange and Taylor polynomial interpolators, and
12: wrappers for `FITPACK <http://www.netlib.org/dierckx/>`__
13: and DFITPACK functions.
14: 
15: Univariate interpolation
16: ========================
17: 
18: .. autosummary::
19:    :toctree: generated/
20: 
21:    interp1d
22:    BarycentricInterpolator
23:    KroghInterpolator
24:    PchipInterpolator
25:    barycentric_interpolate
26:    krogh_interpolate
27:    pchip_interpolate
28:    Akima1DInterpolator
29:    CubicSpline
30:    PPoly
31:    BPoly
32: 
33: 
34: Multivariate interpolation
35: ==========================
36: 
37: Unstructured data:
38: 
39: .. autosummary::
40:    :toctree: generated/
41: 
42:    griddata
43:    LinearNDInterpolator
44:    NearestNDInterpolator
45:    CloughTocher2DInterpolator
46:    Rbf
47:    interp2d
48: 
49: For data on a grid:
50: 
51: .. autosummary::
52:    :toctree: generated/
53: 
54:    interpn
55:    RegularGridInterpolator
56:    RectBivariateSpline
57: 
58: .. seealso::
59: 
60:     `scipy.ndimage.map_coordinates`
61: 
62: Tensor product polynomials:
63: 
64: .. autosummary::
65:    :toctree: generated/
66: 
67:    NdPPoly
68: 
69: 
70: 1-D Splines
71: ===========
72: 
73: .. autosummary::
74:    :toctree: generated/
75: 
76:    BSpline
77:    make_interp_spline
78:    make_lsq_spline
79: 
80: Functional interface to FITPACK routines:
81: 
82: .. autosummary::
83:    :toctree: generated/
84: 
85:    splrep
86:    splprep
87:    splev
88:    splint
89:    sproot
90:    spalde
91:    splder
92:    splantider
93:    insert
94: 
95: Object-oriented FITPACK interface:
96: 
97: .. autosummary::
98:     :toctree: generated/
99: 
100:    UnivariateSpline
101:    InterpolatedUnivariateSpline
102:    LSQUnivariateSpline
103: 
104: 
105: 
106: 2-D Splines
107: ===========
108: 
109: For data on a grid:
110: 
111: .. autosummary::
112:    :toctree: generated/
113: 
114:    RectBivariateSpline
115:    RectSphereBivariateSpline
116: 
117: For unstructured data:
118: 
119: .. autosummary::
120:    :toctree: generated/
121: 
122:    BivariateSpline
123:    SmoothBivariateSpline
124:    SmoothSphereBivariateSpline
125:    LSQBivariateSpline
126:    LSQSphereBivariateSpline
127: 
128: Low-level interface to FITPACK functions:
129: 
130: .. autosummary::
131:    :toctree: generated/
132: 
133:    bisplrep
134:    bisplev
135: 
136: Additional tools
137: ================
138: 
139: .. autosummary::
140:    :toctree: generated/
141: 
142:    lagrange
143:    approximate_taylor_polynomial
144:    pade
145: 
146: .. seealso::
147: 
148:    `scipy.ndimage.map_coordinates`,
149:    `scipy.ndimage.spline_filter`,
150:    `scipy.signal.resample`,
151:    `scipy.signal.bspline`,
152:    `scipy.signal.gauss_spline`,
153:    `scipy.signal.qspline1d`,
154:    `scipy.signal.cspline1d`,
155:    `scipy.signal.qspline1d_eval`,
156:    `scipy.signal.cspline1d_eval`,
157:    `scipy.signal.qspline2d`,
158:    `scipy.signal.cspline2d`.
159: 
160: Functions existing for backward compatibility (should not be used in
161: new code):
162: 
163: .. autosummary::
164:    :toctree: generated/
165: 
166:    spleval
167:    spline
168:    splmake
169:    spltopp
170:    pchip
171: 
172: '''
173: from __future__ import division, print_function, absolute_import
174: 
175: from .interpolate import *
176: from .fitpack import *
177: 
178: # New interface to fitpack library:
179: from .fitpack2 import *
180: 
181: from .rbf import Rbf
182: 
183: from .polyint import *
184: 
185: from ._cubic import *
186: 
187: from .ndgriddata import *
188: 
189: from ._bsplines import *
190: 
191: from ._pade import *
192: 
193: __all__ = [s for s in dir() if not s.startswith('_')]
194: 
195: from scipy._lib._testutils import PytestTester
196: test = PytestTester(__name__)
197: del PytestTester
198: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_81894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'str', '========================================\nInterpolation (:mod:`scipy.interpolate`)\n========================================\n\n.. currentmodule:: scipy.interpolate\n\nSub-package for objects used in interpolation.\n\nAs listed below, this sub-package contains spline functions and classes,\none-dimensional and multi-dimensional (univariate and multivariate)\ninterpolation classes, Lagrange and Taylor polynomial interpolators, and\nwrappers for `FITPACK <http://www.netlib.org/dierckx/>`__\nand DFITPACK functions.\n\nUnivariate interpolation\n========================\n\n.. autosummary::\n   :toctree: generated/\n\n   interp1d\n   BarycentricInterpolator\n   KroghInterpolator\n   PchipInterpolator\n   barycentric_interpolate\n   krogh_interpolate\n   pchip_interpolate\n   Akima1DInterpolator\n   CubicSpline\n   PPoly\n   BPoly\n\n\nMultivariate interpolation\n==========================\n\nUnstructured data:\n\n.. autosummary::\n   :toctree: generated/\n\n   griddata\n   LinearNDInterpolator\n   NearestNDInterpolator\n   CloughTocher2DInterpolator\n   Rbf\n   interp2d\n\nFor data on a grid:\n\n.. autosummary::\n   :toctree: generated/\n\n   interpn\n   RegularGridInterpolator\n   RectBivariateSpline\n\n.. seealso::\n\n    `scipy.ndimage.map_coordinates`\n\nTensor product polynomials:\n\n.. autosummary::\n   :toctree: generated/\n\n   NdPPoly\n\n\n1-D Splines\n===========\n\n.. autosummary::\n   :toctree: generated/\n\n   BSpline\n   make_interp_spline\n   make_lsq_spline\n\nFunctional interface to FITPACK routines:\n\n.. autosummary::\n   :toctree: generated/\n\n   splrep\n   splprep\n   splev\n   splint\n   sproot\n   spalde\n   splder\n   splantider\n   insert\n\nObject-oriented FITPACK interface:\n\n.. autosummary::\n    :toctree: generated/\n\n   UnivariateSpline\n   InterpolatedUnivariateSpline\n   LSQUnivariateSpline\n\n\n\n2-D Splines\n===========\n\nFor data on a grid:\n\n.. autosummary::\n   :toctree: generated/\n\n   RectBivariateSpline\n   RectSphereBivariateSpline\n\nFor unstructured data:\n\n.. autosummary::\n   :toctree: generated/\n\n   BivariateSpline\n   SmoothBivariateSpline\n   SmoothSphereBivariateSpline\n   LSQBivariateSpline\n   LSQSphereBivariateSpline\n\nLow-level interface to FITPACK functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   bisplrep\n   bisplev\n\nAdditional tools\n================\n\n.. autosummary::\n   :toctree: generated/\n\n   lagrange\n   approximate_taylor_polynomial\n   pade\n\n.. seealso::\n\n   `scipy.ndimage.map_coordinates`,\n   `scipy.ndimage.spline_filter`,\n   `scipy.signal.resample`,\n   `scipy.signal.bspline`,\n   `scipy.signal.gauss_spline`,\n   `scipy.signal.qspline1d`,\n   `scipy.signal.cspline1d`,\n   `scipy.signal.qspline1d_eval`,\n   `scipy.signal.cspline1d_eval`,\n   `scipy.signal.qspline2d`,\n   `scipy.signal.cspline2d`.\n\nFunctions existing for backward compatibility (should not be used in\nnew code):\n\n.. autosummary::\n   :toctree: generated/\n\n   spleval\n   spline\n   splmake\n   spltopp\n   pchip\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 175, 0))

# 'from scipy.interpolate.interpolate import ' statement (line 175)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81895 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 175, 0), 'scipy.interpolate.interpolate')

if (type(import_81895) is not StypyTypeError):

    if (import_81895 != 'pyd_module'):
        __import__(import_81895)
        sys_modules_81896 = sys.modules[import_81895]
        import_from_module(stypy.reporting.localization.Localization(__file__, 175, 0), 'scipy.interpolate.interpolate', sys_modules_81896.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 175, 0), __file__, sys_modules_81896, sys_modules_81896.module_type_store, module_type_store)
    else:
        from scipy.interpolate.interpolate import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 175, 0), 'scipy.interpolate.interpolate', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate.interpolate' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'scipy.interpolate.interpolate', import_81895)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 176, 0))

# 'from scipy.interpolate.fitpack import ' statement (line 176)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81897 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy.interpolate.fitpack')

if (type(import_81897) is not StypyTypeError):

    if (import_81897 != 'pyd_module'):
        __import__(import_81897)
        sys_modules_81898 = sys.modules[import_81897]
        import_from_module(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy.interpolate.fitpack', sys_modules_81898.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 176, 0), __file__, sys_modules_81898, sys_modules_81898.module_type_store, module_type_store)
    else:
        from scipy.interpolate.fitpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy.interpolate.fitpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate.fitpack' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy.interpolate.fitpack', import_81897)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 179, 0))

# 'from scipy.interpolate.fitpack2 import ' statement (line 179)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81899 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 179, 0), 'scipy.interpolate.fitpack2')

if (type(import_81899) is not StypyTypeError):

    if (import_81899 != 'pyd_module'):
        __import__(import_81899)
        sys_modules_81900 = sys.modules[import_81899]
        import_from_module(stypy.reporting.localization.Localization(__file__, 179, 0), 'scipy.interpolate.fitpack2', sys_modules_81900.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 179, 0), __file__, sys_modules_81900, sys_modules_81900.module_type_store, module_type_store)
    else:
        from scipy.interpolate.fitpack2 import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 179, 0), 'scipy.interpolate.fitpack2', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate.fitpack2' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'scipy.interpolate.fitpack2', import_81899)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 181, 0))

# 'from scipy.interpolate.rbf import Rbf' statement (line 181)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81901 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 181, 0), 'scipy.interpolate.rbf')

if (type(import_81901) is not StypyTypeError):

    if (import_81901 != 'pyd_module'):
        __import__(import_81901)
        sys_modules_81902 = sys.modules[import_81901]
        import_from_module(stypy.reporting.localization.Localization(__file__, 181, 0), 'scipy.interpolate.rbf', sys_modules_81902.module_type_store, module_type_store, ['Rbf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 181, 0), __file__, sys_modules_81902, sys_modules_81902.module_type_store, module_type_store)
    else:
        from scipy.interpolate.rbf import Rbf

        import_from_module(stypy.reporting.localization.Localization(__file__, 181, 0), 'scipy.interpolate.rbf', None, module_type_store, ['Rbf'], [Rbf])

else:
    # Assigning a type to the variable 'scipy.interpolate.rbf' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'scipy.interpolate.rbf', import_81901)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 183, 0))

# 'from scipy.interpolate.polyint import ' statement (line 183)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81903 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 183, 0), 'scipy.interpolate.polyint')

if (type(import_81903) is not StypyTypeError):

    if (import_81903 != 'pyd_module'):
        __import__(import_81903)
        sys_modules_81904 = sys.modules[import_81903]
        import_from_module(stypy.reporting.localization.Localization(__file__, 183, 0), 'scipy.interpolate.polyint', sys_modules_81904.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 183, 0), __file__, sys_modules_81904, sys_modules_81904.module_type_store, module_type_store)
    else:
        from scipy.interpolate.polyint import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 183, 0), 'scipy.interpolate.polyint', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate.polyint' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'scipy.interpolate.polyint', import_81903)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 185, 0))

# 'from scipy.interpolate._cubic import ' statement (line 185)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81905 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 185, 0), 'scipy.interpolate._cubic')

if (type(import_81905) is not StypyTypeError):

    if (import_81905 != 'pyd_module'):
        __import__(import_81905)
        sys_modules_81906 = sys.modules[import_81905]
        import_from_module(stypy.reporting.localization.Localization(__file__, 185, 0), 'scipy.interpolate._cubic', sys_modules_81906.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 185, 0), __file__, sys_modules_81906, sys_modules_81906.module_type_store, module_type_store)
    else:
        from scipy.interpolate._cubic import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 185, 0), 'scipy.interpolate._cubic', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate._cubic' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'scipy.interpolate._cubic', import_81905)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 187, 0))

# 'from scipy.interpolate.ndgriddata import ' statement (line 187)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81907 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.interpolate.ndgriddata')

if (type(import_81907) is not StypyTypeError):

    if (import_81907 != 'pyd_module'):
        __import__(import_81907)
        sys_modules_81908 = sys.modules[import_81907]
        import_from_module(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.interpolate.ndgriddata', sys_modules_81908.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 187, 0), __file__, sys_modules_81908, sys_modules_81908.module_type_store, module_type_store)
    else:
        from scipy.interpolate.ndgriddata import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.interpolate.ndgriddata', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate.ndgriddata' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.interpolate.ndgriddata', import_81907)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 189, 0))

# 'from scipy.interpolate._bsplines import ' statement (line 189)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81909 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.interpolate._bsplines')

if (type(import_81909) is not StypyTypeError):

    if (import_81909 != 'pyd_module'):
        __import__(import_81909)
        sys_modules_81910 = sys.modules[import_81909]
        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.interpolate._bsplines', sys_modules_81910.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 189, 0), __file__, sys_modules_81910, sys_modules_81910.module_type_store, module_type_store)
    else:
        from scipy.interpolate._bsplines import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.interpolate._bsplines', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate._bsplines' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.interpolate._bsplines', import_81909)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 191, 0))

# 'from scipy.interpolate._pade import ' statement (line 191)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81911 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.interpolate._pade')

if (type(import_81911) is not StypyTypeError):

    if (import_81911 != 'pyd_module'):
        __import__(import_81911)
        sys_modules_81912 = sys.modules[import_81911]
        import_from_module(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.interpolate._pade', sys_modules_81912.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 191, 0), __file__, sys_modules_81912, sys_modules_81912.module_type_store, module_type_store)
    else:
        from scipy.interpolate._pade import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.interpolate._pade', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.interpolate._pade' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.interpolate._pade', import_81911)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a ListComp to a Name (line 193):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 193)
# Processing the call keyword arguments (line 193)
kwargs_81921 = {}
# Getting the type of 'dir' (line 193)
dir_81920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'dir', False)
# Calling dir(args, kwargs) (line 193)
dir_call_result_81922 = invoke(stypy.reporting.localization.Localization(__file__, 193, 22), dir_81920, *[], **kwargs_81921)

comprehension_81923 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 11), dir_call_result_81922)
# Assigning a type to the variable 's' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 's', comprehension_81923)


# Call to startswith(...): (line 193)
# Processing the call arguments (line 193)
str_81916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 48), 'str', '_')
# Processing the call keyword arguments (line 193)
kwargs_81917 = {}
# Getting the type of 's' (line 193)
s_81914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 193)
startswith_81915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 35), s_81914, 'startswith')
# Calling startswith(args, kwargs) (line 193)
startswith_call_result_81918 = invoke(stypy.reporting.localization.Localization(__file__, 193, 35), startswith_81915, *[str_81916], **kwargs_81917)

# Applying the 'not' unary operator (line 193)
result_not__81919 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 31), 'not', startswith_call_result_81918)

# Getting the type of 's' (line 193)
s_81913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 's')
list_81924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 11), list_81924, s_81913)
# Assigning a type to the variable '__all__' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), '__all__', list_81924)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 195, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 195)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81925 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy._lib._testutils')

if (type(import_81925) is not StypyTypeError):

    if (import_81925 != 'pyd_module'):
        __import__(import_81925)
        sys_modules_81926 = sys.modules[import_81925]
        import_from_module(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy._lib._testutils', sys_modules_81926.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 195, 0), __file__, sys_modules_81926, sys_modules_81926.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy._lib._testutils', import_81925)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a Call to a Name (line 196):

# Call to PytestTester(...): (line 196)
# Processing the call arguments (line 196)
# Getting the type of '__name__' (line 196)
name___81928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), '__name__', False)
# Processing the call keyword arguments (line 196)
kwargs_81929 = {}
# Getting the type of 'PytestTester' (line 196)
PytestTester_81927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 196)
PytestTester_call_result_81930 = invoke(stypy.reporting.localization.Localization(__file__, 196, 7), PytestTester_81927, *[name___81928], **kwargs_81929)

# Assigning a type to the variable 'test' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'test', PytestTester_call_result_81930)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 197, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import threading
4: import numpy as np
5: 
6: from ._ufuncs import _ellip_harm
7: from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm
8: 
9: 
10: def ellip_harm(h2, k2, n, p, s, signm=1, signn=1):
11:     r'''
12:     Ellipsoidal harmonic functions E^p_n(l)
13: 
14:     These are also known as Lame functions of the first kind, and are
15:     solutions to the Lame equation:
16: 
17:     .. math:: (s^2 - h^2)(s^2 - k^2)E''(s) + s(2s^2 - h^2 - k^2)E'(s) + (a - q s^2)E(s) = 0
18: 
19:     where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
20:     returned) corresponding to the solutions.
21: 
22:     Parameters
23:     ----------
24:     h2 : float
25:         ``h**2``
26:     k2 : float
27:         ``k**2``; should be larger than ``h**2``
28:     n : int
29:         Degree
30:     s : float
31:         Coordinate
32:     p : int
33:         Order, can range between [1,2n+1]
34:     signm : {1, -1}, optional
35:         Sign of prefactor of functions. Can be +/-1. See Notes.
36:     signn : {1, -1}, optional
37:         Sign of prefactor of functions. Can be +/-1. See Notes.
38: 
39:     Returns
40:     -------
41:     E : float
42:         the harmonic :math:`E^p_n(s)`
43: 
44:     See Also
45:     --------
46:     ellip_harm_2, ellip_normal
47: 
48:     Notes
49:     -----
50:     The geometric intepretation of the ellipsoidal functions is
51:     explained in [2]_, [3]_, [4]_.  The `signm` and `signn` arguments control the
52:     sign of prefactors for functions according to their type::
53: 
54:         K : +1
55:         L : signm
56:         M : signn
57:         N : signm*signn
58: 
59:     .. versionadded:: 0.15.0
60: 
61:     References
62:     ----------
63:     .. [1] Digital Libary of Mathematical Functions 29.12
64:        http://dlmf.nist.gov/29.12
65:     .. [2] Bardhan and Knepley, "Computational science and
66:        re-discovery: open-source implementations of
67:        ellipsoidal harmonics for problems in potential theory",
68:        Comput. Sci. Disc. 5, 014006 (2012)
69:        :doi:`10.1088/1749-4699/5/1/014006`.
70:     .. [3] David J.and Dechambre P, "Computation of Ellipsoidal
71:        Gravity Field Harmonics for small solar system bodies"
72:        pp. 30-36, 2000
73:     .. [4] George Dassios, "Ellipsoidal Harmonics: Theory and Applications"
74:        pp. 418, 2012
75: 
76:     Examples
77:     --------
78:     >>> from scipy.special import ellip_harm
79:     >>> w = ellip_harm(5,8,1,1,2.5)
80:     >>> w
81:     2.5
82: 
83:     Check that the functions indeed are solutions to the Lame equation:
84: 
85:     >>> from scipy.interpolate import UnivariateSpline
86:     >>> def eigenvalue(f, df, ddf):
87:     ...     r = ((s**2 - h**2)*(s**2 - k**2)*ddf + s*(2*s**2 - h**2 - k**2)*df - n*(n+1)*s**2*f)/f
88:     ...     return -r.mean(), r.std()
89:     >>> s = np.linspace(0.1, 10, 200)
90:     >>> k, h, n, p = 8.0, 2.2, 3, 2
91:     >>> E = ellip_harm(h**2, k**2, n, p, s)
92:     >>> E_spl = UnivariateSpline(s, E)
93:     >>> a, a_err = eigenvalue(E_spl(s), E_spl(s,1), E_spl(s,2))
94:     >>> a, a_err
95:     (583.44366156701483, 6.4580890640310646e-11)
96: 
97:     '''
98:     return _ellip_harm(h2, k2, n, p, s, signm, signn)
99: 
100: 
101: _ellip_harm_2_vec = np.vectorize(_ellipsoid, otypes='d')
102: 
103: 
104: def ellip_harm_2(h2, k2, n, p, s):
105:     r'''
106:     Ellipsoidal harmonic functions F^p_n(l)
107: 
108:     These are also known as Lame functions of the second kind, and are
109:     solutions to the Lame equation:
110: 
111:     .. math:: (s^2 - h^2)(s^2 - k^2)F''(s) + s(2s^2 - h^2 - k^2)F'(s) + (a - q s^2)F(s) = 0
112: 
113:     where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
114:     returned) corresponding to the solutions.
115: 
116:     Parameters
117:     ----------
118:     h2 : float
119:         ``h**2``
120:     k2 : float
121:         ``k**2``; should be larger than ``h**2``
122:     n : int
123:         Degree.
124:     p : int
125:         Order, can range between [1,2n+1].
126:     s : float
127:         Coordinate
128: 
129:     Returns
130:     -------
131:     F : float
132:         The harmonic :math:`F^p_n(s)`
133: 
134:     Notes
135:     -----
136:     Lame functions of the second kind are related to the functions of the first kind:
137: 
138:     .. math::
139: 
140:        F^p_n(s)=(2n + 1)E^p_n(s)\int_{0}^{1/s}\frac{du}{(E^p_n(1/u))^2\sqrt{(1-u^2k^2)(1-u^2h^2)}}
141: 
142:     .. versionadded:: 0.15.0
143: 
144:     See Also
145:     --------
146:     ellip_harm, ellip_normal
147: 
148:     Examples
149:     --------
150:     >>> from scipy.special import ellip_harm_2
151:     >>> w = ellip_harm_2(5,8,2,1,10)
152:     >>> w
153:     0.00108056853382
154: 
155:     '''
156:     with np.errstate(all='ignore'):
157:         return _ellip_harm_2_vec(h2, k2, n, p, s)
158: 
159: 
160: def _ellip_normal_vec(h2, k2, n, p):
161:     return _ellipsoid_norm(h2, k2, n, p)
162: 
163: _ellip_normal_vec = np.vectorize(_ellip_normal_vec, otypes='d')
164: 
165: 
166: def ellip_normal(h2, k2, n, p):
167:     r'''
168:     Ellipsoidal harmonic normalization constants gamma^p_n
169: 
170:     The normalization constant is defined as
171: 
172:     .. math::
173: 
174:        \gamma^p_n=8\int_{0}^{h}dx\int_{h}^{k}dy\frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}
175: 
176:     Parameters
177:     ----------
178:     h2 : float
179:         ``h**2``
180:     k2 : float
181:         ``k**2``; should be larger than ``h**2``
182:     n : int
183:         Degree.
184:     p : int
185:         Order, can range between [1,2n+1].
186: 
187:     Returns
188:     -------
189:     gamma : float
190:         The normalization constant :math:`\gamma^p_n`
191: 
192:     See Also
193:     --------
194:     ellip_harm, ellip_harm_2
195: 
196:     Notes
197:     -----
198:     .. versionadded:: 0.15.0
199: 
200:     Examples
201:     --------
202:     >>> from scipy.special import ellip_normal
203:     >>> w = ellip_normal(5,8,3,7)
204:     >>> w
205:     1723.38796997
206: 
207:     '''
208:     with np.errstate(all='ignore'):
209:         return _ellip_normal_vec(h2, k2, n, p)
210: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import threading' statement (line 3)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503813 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_503813) is not StypyTypeError):

    if (import_503813 != 'pyd_module'):
        __import__(import_503813)
        sys_modules_503814 = sys.modules[import_503813]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_503814.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_503813)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._ufuncs import _ellip_harm' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503815 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs')

if (type(import_503815) is not StypyTypeError):

    if (import_503815 != 'pyd_module'):
        __import__(import_503815)
        sys_modules_503816 = sys.modules[import_503815]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs', sys_modules_503816.module_type_store, module_type_store, ['_ellip_harm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_503816, sys_modules_503816.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import _ellip_harm

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs', None, module_type_store, ['_ellip_harm'], [_ellip_harm])

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs', import_503815)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special._ellip_harm_2 import _ellipsoid, _ellipsoid_norm' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503817 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ellip_harm_2')

if (type(import_503817) is not StypyTypeError):

    if (import_503817 != 'pyd_module'):
        __import__(import_503817)
        sys_modules_503818 = sys.modules[import_503817]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ellip_harm_2', sys_modules_503818.module_type_store, module_type_store, ['_ellipsoid', '_ellipsoid_norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_503818, sys_modules_503818.module_type_store, module_type_store)
    else:
        from scipy.special._ellip_harm_2 import _ellipsoid, _ellipsoid_norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ellip_harm_2', None, module_type_store, ['_ellipsoid', '_ellipsoid_norm'], [_ellipsoid, _ellipsoid_norm])

else:
    # Assigning a type to the variable 'scipy.special._ellip_harm_2' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ellip_harm_2', import_503817)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


@norecursion
def ellip_harm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_503819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 38), 'int')
    int_503820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 47), 'int')
    defaults = [int_503819, int_503820]
    # Create a new context for function 'ellip_harm'
    module_type_store = module_type_store.open_function_context('ellip_harm', 10, 0, False)
    
    # Passed parameters checking function
    ellip_harm.stypy_localization = localization
    ellip_harm.stypy_type_of_self = None
    ellip_harm.stypy_type_store = module_type_store
    ellip_harm.stypy_function_name = 'ellip_harm'
    ellip_harm.stypy_param_names_list = ['h2', 'k2', 'n', 'p', 's', 'signm', 'signn']
    ellip_harm.stypy_varargs_param_name = None
    ellip_harm.stypy_kwargs_param_name = None
    ellip_harm.stypy_call_defaults = defaults
    ellip_harm.stypy_call_varargs = varargs
    ellip_harm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellip_harm', ['h2', 'k2', 'n', 'p', 's', 'signm', 'signn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellip_harm', localization, ['h2', 'k2', 'n', 'p', 's', 'signm', 'signn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellip_harm(...)' code ##################

    str_503821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n    Ellipsoidal harmonic functions E^p_n(l)\n\n    These are also known as Lame functions of the first kind, and are\n    solutions to the Lame equation:\n\n    .. math:: (s^2 - h^2)(s^2 - k^2)E\'\'(s) + s(2s^2 - h^2 - k^2)E\'(s) + (a - q s^2)E(s) = 0\n\n    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not\n    returned) corresponding to the solutions.\n\n    Parameters\n    ----------\n    h2 : float\n        ``h**2``\n    k2 : float\n        ``k**2``; should be larger than ``h**2``\n    n : int\n        Degree\n    s : float\n        Coordinate\n    p : int\n        Order, can range between [1,2n+1]\n    signm : {1, -1}, optional\n        Sign of prefactor of functions. Can be +/-1. See Notes.\n    signn : {1, -1}, optional\n        Sign of prefactor of functions. Can be +/-1. See Notes.\n\n    Returns\n    -------\n    E : float\n        the harmonic :math:`E^p_n(s)`\n\n    See Also\n    --------\n    ellip_harm_2, ellip_normal\n\n    Notes\n    -----\n    The geometric intepretation of the ellipsoidal functions is\n    explained in [2]_, [3]_, [4]_.  The `signm` and `signn` arguments control the\n    sign of prefactors for functions according to their type::\n\n        K : +1\n        L : signm\n        M : signn\n        N : signm*signn\n\n    .. versionadded:: 0.15.0\n\n    References\n    ----------\n    .. [1] Digital Libary of Mathematical Functions 29.12\n       http://dlmf.nist.gov/29.12\n    .. [2] Bardhan and Knepley, "Computational science and\n       re-discovery: open-source implementations of\n       ellipsoidal harmonics for problems in potential theory",\n       Comput. Sci. Disc. 5, 014006 (2012)\n       :doi:`10.1088/1749-4699/5/1/014006`.\n    .. [3] David J.and Dechambre P, "Computation of Ellipsoidal\n       Gravity Field Harmonics for small solar system bodies"\n       pp. 30-36, 2000\n    .. [4] George Dassios, "Ellipsoidal Harmonics: Theory and Applications"\n       pp. 418, 2012\n\n    Examples\n    --------\n    >>> from scipy.special import ellip_harm\n    >>> w = ellip_harm(5,8,1,1,2.5)\n    >>> w\n    2.5\n\n    Check that the functions indeed are solutions to the Lame equation:\n\n    >>> from scipy.interpolate import UnivariateSpline\n    >>> def eigenvalue(f, df, ddf):\n    ...     r = ((s**2 - h**2)*(s**2 - k**2)*ddf + s*(2*s**2 - h**2 - k**2)*df - n*(n+1)*s**2*f)/f\n    ...     return -r.mean(), r.std()\n    >>> s = np.linspace(0.1, 10, 200)\n    >>> k, h, n, p = 8.0, 2.2, 3, 2\n    >>> E = ellip_harm(h**2, k**2, n, p, s)\n    >>> E_spl = UnivariateSpline(s, E)\n    >>> a, a_err = eigenvalue(E_spl(s), E_spl(s,1), E_spl(s,2))\n    >>> a, a_err\n    (583.44366156701483, 6.4580890640310646e-11)\n\n    ')
    
    # Call to _ellip_harm(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'h2' (line 98)
    h2_503823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'h2', False)
    # Getting the type of 'k2' (line 98)
    k2_503824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'k2', False)
    # Getting the type of 'n' (line 98)
    n_503825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'n', False)
    # Getting the type of 'p' (line 98)
    p_503826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'p', False)
    # Getting the type of 's' (line 98)
    s_503827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 's', False)
    # Getting the type of 'signm' (line 98)
    signm_503828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'signm', False)
    # Getting the type of 'signn' (line 98)
    signn_503829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'signn', False)
    # Processing the call keyword arguments (line 98)
    kwargs_503830 = {}
    # Getting the type of '_ellip_harm' (line 98)
    _ellip_harm_503822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), '_ellip_harm', False)
    # Calling _ellip_harm(args, kwargs) (line 98)
    _ellip_harm_call_result_503831 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), _ellip_harm_503822, *[h2_503823, k2_503824, n_503825, p_503826, s_503827, signm_503828, signn_503829], **kwargs_503830)
    
    # Assigning a type to the variable 'stypy_return_type' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type', _ellip_harm_call_result_503831)
    
    # ################# End of 'ellip_harm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellip_harm' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_503832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_503832)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellip_harm'
    return stypy_return_type_503832

# Assigning a type to the variable 'ellip_harm' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'ellip_harm', ellip_harm)

# Assigning a Call to a Name (line 101):

# Call to vectorize(...): (line 101)
# Processing the call arguments (line 101)
# Getting the type of '_ellipsoid' (line 101)
_ellipsoid_503835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), '_ellipsoid', False)
# Processing the call keyword arguments (line 101)
str_503836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 52), 'str', 'd')
keyword_503837 = str_503836
kwargs_503838 = {'otypes': keyword_503837}
# Getting the type of 'np' (line 101)
np_503833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'np', False)
# Obtaining the member 'vectorize' of a type (line 101)
vectorize_503834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), np_503833, 'vectorize')
# Calling vectorize(args, kwargs) (line 101)
vectorize_call_result_503839 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), vectorize_503834, *[_ellipsoid_503835], **kwargs_503838)

# Assigning a type to the variable '_ellip_harm_2_vec' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '_ellip_harm_2_vec', vectorize_call_result_503839)

@norecursion
def ellip_harm_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellip_harm_2'
    module_type_store = module_type_store.open_function_context('ellip_harm_2', 104, 0, False)
    
    # Passed parameters checking function
    ellip_harm_2.stypy_localization = localization
    ellip_harm_2.stypy_type_of_self = None
    ellip_harm_2.stypy_type_store = module_type_store
    ellip_harm_2.stypy_function_name = 'ellip_harm_2'
    ellip_harm_2.stypy_param_names_list = ['h2', 'k2', 'n', 'p', 's']
    ellip_harm_2.stypy_varargs_param_name = None
    ellip_harm_2.stypy_kwargs_param_name = None
    ellip_harm_2.stypy_call_defaults = defaults
    ellip_harm_2.stypy_call_varargs = varargs
    ellip_harm_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellip_harm_2', ['h2', 'k2', 'n', 'p', 's'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellip_harm_2', localization, ['h2', 'k2', 'n', 'p', 's'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellip_harm_2(...)' code ##################

    str_503840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', "\n    Ellipsoidal harmonic functions F^p_n(l)\n\n    These are also known as Lame functions of the second kind, and are\n    solutions to the Lame equation:\n\n    .. math:: (s^2 - h^2)(s^2 - k^2)F''(s) + s(2s^2 - h^2 - k^2)F'(s) + (a - q s^2)F(s) = 0\n\n    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not\n    returned) corresponding to the solutions.\n\n    Parameters\n    ----------\n    h2 : float\n        ``h**2``\n    k2 : float\n        ``k**2``; should be larger than ``h**2``\n    n : int\n        Degree.\n    p : int\n        Order, can range between [1,2n+1].\n    s : float\n        Coordinate\n\n    Returns\n    -------\n    F : float\n        The harmonic :math:`F^p_n(s)`\n\n    Notes\n    -----\n    Lame functions of the second kind are related to the functions of the first kind:\n\n    .. math::\n\n       F^p_n(s)=(2n + 1)E^p_n(s)\\int_{0}^{1/s}\\frac{du}{(E^p_n(1/u))^2\\sqrt{(1-u^2k^2)(1-u^2h^2)}}\n\n    .. versionadded:: 0.15.0\n\n    See Also\n    --------\n    ellip_harm, ellip_normal\n\n    Examples\n    --------\n    >>> from scipy.special import ellip_harm_2\n    >>> w = ellip_harm_2(5,8,2,1,10)\n    >>> w\n    0.00108056853382\n\n    ")
    
    # Call to errstate(...): (line 156)
    # Processing the call keyword arguments (line 156)
    str_503843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'str', 'ignore')
    keyword_503844 = str_503843
    kwargs_503845 = {'all': keyword_503844}
    # Getting the type of 'np' (line 156)
    np_503841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 156)
    errstate_503842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), np_503841, 'errstate')
    # Calling errstate(args, kwargs) (line 156)
    errstate_call_result_503846 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), errstate_503842, *[], **kwargs_503845)
    
    with_503847 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 156, 9), errstate_call_result_503846, 'with parameter', '__enter__', '__exit__')

    if with_503847:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 156)
        enter___503848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), errstate_call_result_503846, '__enter__')
        with_enter_503849 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), enter___503848)
        
        # Call to _ellip_harm_2_vec(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'h2' (line 157)
        h2_503851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'h2', False)
        # Getting the type of 'k2' (line 157)
        k2_503852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'k2', False)
        # Getting the type of 'n' (line 157)
        n_503853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 41), 'n', False)
        # Getting the type of 'p' (line 157)
        p_503854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 44), 'p', False)
        # Getting the type of 's' (line 157)
        s_503855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 47), 's', False)
        # Processing the call keyword arguments (line 157)
        kwargs_503856 = {}
        # Getting the type of '_ellip_harm_2_vec' (line 157)
        _ellip_harm_2_vec_503850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), '_ellip_harm_2_vec', False)
        # Calling _ellip_harm_2_vec(args, kwargs) (line 157)
        _ellip_harm_2_vec_call_result_503857 = invoke(stypy.reporting.localization.Localization(__file__, 157, 15), _ellip_harm_2_vec_503850, *[h2_503851, k2_503852, n_503853, p_503854, s_503855], **kwargs_503856)
        
        # Assigning a type to the variable 'stypy_return_type' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'stypy_return_type', _ellip_harm_2_vec_call_result_503857)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 156)
        exit___503858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), errstate_call_result_503846, '__exit__')
        with_exit_503859 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), exit___503858, None, None, None)

    
    # ################# End of 'ellip_harm_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellip_harm_2' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_503860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_503860)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellip_harm_2'
    return stypy_return_type_503860

# Assigning a type to the variable 'ellip_harm_2' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'ellip_harm_2', ellip_harm_2)

@norecursion
def _ellip_normal_vec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ellip_normal_vec'
    module_type_store = module_type_store.open_function_context('_ellip_normal_vec', 160, 0, False)
    
    # Passed parameters checking function
    _ellip_normal_vec.stypy_localization = localization
    _ellip_normal_vec.stypy_type_of_self = None
    _ellip_normal_vec.stypy_type_store = module_type_store
    _ellip_normal_vec.stypy_function_name = '_ellip_normal_vec'
    _ellip_normal_vec.stypy_param_names_list = ['h2', 'k2', 'n', 'p']
    _ellip_normal_vec.stypy_varargs_param_name = None
    _ellip_normal_vec.stypy_kwargs_param_name = None
    _ellip_normal_vec.stypy_call_defaults = defaults
    _ellip_normal_vec.stypy_call_varargs = varargs
    _ellip_normal_vec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ellip_normal_vec', ['h2', 'k2', 'n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ellip_normal_vec', localization, ['h2', 'k2', 'n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ellip_normal_vec(...)' code ##################

    
    # Call to _ellipsoid_norm(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'h2' (line 161)
    h2_503862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'h2', False)
    # Getting the type of 'k2' (line 161)
    k2_503863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'k2', False)
    # Getting the type of 'n' (line 161)
    n_503864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'n', False)
    # Getting the type of 'p' (line 161)
    p_503865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 38), 'p', False)
    # Processing the call keyword arguments (line 161)
    kwargs_503866 = {}
    # Getting the type of '_ellipsoid_norm' (line 161)
    _ellipsoid_norm_503861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), '_ellipsoid_norm', False)
    # Calling _ellipsoid_norm(args, kwargs) (line 161)
    _ellipsoid_norm_call_result_503867 = invoke(stypy.reporting.localization.Localization(__file__, 161, 11), _ellipsoid_norm_503861, *[h2_503862, k2_503863, n_503864, p_503865], **kwargs_503866)
    
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type', _ellipsoid_norm_call_result_503867)
    
    # ################# End of '_ellip_normal_vec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ellip_normal_vec' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_503868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_503868)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ellip_normal_vec'
    return stypy_return_type_503868

# Assigning a type to the variable '_ellip_normal_vec' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), '_ellip_normal_vec', _ellip_normal_vec)

# Assigning a Call to a Name (line 163):

# Call to vectorize(...): (line 163)
# Processing the call arguments (line 163)
# Getting the type of '_ellip_normal_vec' (line 163)
_ellip_normal_vec_503871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), '_ellip_normal_vec', False)
# Processing the call keyword arguments (line 163)
str_503872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 59), 'str', 'd')
keyword_503873 = str_503872
kwargs_503874 = {'otypes': keyword_503873}
# Getting the type of 'np' (line 163)
np_503869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'np', False)
# Obtaining the member 'vectorize' of a type (line 163)
vectorize_503870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), np_503869, 'vectorize')
# Calling vectorize(args, kwargs) (line 163)
vectorize_call_result_503875 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), vectorize_503870, *[_ellip_normal_vec_503871], **kwargs_503874)

# Assigning a type to the variable '_ellip_normal_vec' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), '_ellip_normal_vec', vectorize_call_result_503875)

@norecursion
def ellip_normal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellip_normal'
    module_type_store = module_type_store.open_function_context('ellip_normal', 166, 0, False)
    
    # Passed parameters checking function
    ellip_normal.stypy_localization = localization
    ellip_normal.stypy_type_of_self = None
    ellip_normal.stypy_type_store = module_type_store
    ellip_normal.stypy_function_name = 'ellip_normal'
    ellip_normal.stypy_param_names_list = ['h2', 'k2', 'n', 'p']
    ellip_normal.stypy_varargs_param_name = None
    ellip_normal.stypy_kwargs_param_name = None
    ellip_normal.stypy_call_defaults = defaults
    ellip_normal.stypy_call_varargs = varargs
    ellip_normal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellip_normal', ['h2', 'k2', 'n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellip_normal', localization, ['h2', 'k2', 'n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellip_normal(...)' code ##################

    str_503876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, (-1)), 'str', '\n    Ellipsoidal harmonic normalization constants gamma^p_n\n\n    The normalization constant is defined as\n\n    .. math::\n\n       \\gamma^p_n=8\\int_{0}^{h}dx\\int_{h}^{k}dy\\frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}\n\n    Parameters\n    ----------\n    h2 : float\n        ``h**2``\n    k2 : float\n        ``k**2``; should be larger than ``h**2``\n    n : int\n        Degree.\n    p : int\n        Order, can range between [1,2n+1].\n\n    Returns\n    -------\n    gamma : float\n        The normalization constant :math:`\\gamma^p_n`\n\n    See Also\n    --------\n    ellip_harm, ellip_harm_2\n\n    Notes\n    -----\n    .. versionadded:: 0.15.0\n\n    Examples\n    --------\n    >>> from scipy.special import ellip_normal\n    >>> w = ellip_normal(5,8,3,7)\n    >>> w\n    1723.38796997\n\n    ')
    
    # Call to errstate(...): (line 208)
    # Processing the call keyword arguments (line 208)
    str_503879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'str', 'ignore')
    keyword_503880 = str_503879
    kwargs_503881 = {'all': keyword_503880}
    # Getting the type of 'np' (line 208)
    np_503877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 208)
    errstate_503878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), np_503877, 'errstate')
    # Calling errstate(args, kwargs) (line 208)
    errstate_call_result_503882 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), errstate_503878, *[], **kwargs_503881)
    
    with_503883 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 208, 9), errstate_call_result_503882, 'with parameter', '__enter__', '__exit__')

    if with_503883:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 208)
        enter___503884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), errstate_call_result_503882, '__enter__')
        with_enter_503885 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), enter___503884)
        
        # Call to _ellip_normal_vec(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'h2' (line 209)
        h2_503887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'h2', False)
        # Getting the type of 'k2' (line 209)
        k2_503888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 37), 'k2', False)
        # Getting the type of 'n' (line 209)
        n_503889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 41), 'n', False)
        # Getting the type of 'p' (line 209)
        p_503890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 44), 'p', False)
        # Processing the call keyword arguments (line 209)
        kwargs_503891 = {}
        # Getting the type of '_ellip_normal_vec' (line 209)
        _ellip_normal_vec_503886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), '_ellip_normal_vec', False)
        # Calling _ellip_normal_vec(args, kwargs) (line 209)
        _ellip_normal_vec_call_result_503892 = invoke(stypy.reporting.localization.Localization(__file__, 209, 15), _ellip_normal_vec_503886, *[h2_503887, k2_503888, n_503889, p_503890], **kwargs_503891)
        
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'stypy_return_type', _ellip_normal_vec_call_result_503892)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 208)
        exit___503893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), errstate_call_result_503882, '__exit__')
        with_exit_503894 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), exit___503893, None, None, None)

    
    # ################# End of 'ellip_normal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellip_normal' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_503895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_503895)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellip_normal'
    return stypy_return_type_503895

# Assigning a type to the variable 'ellip_normal' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'ellip_normal', ellip_normal)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

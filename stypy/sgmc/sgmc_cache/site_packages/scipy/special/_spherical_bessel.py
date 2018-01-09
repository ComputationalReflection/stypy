
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from ._ufuncs import (_spherical_jn, _spherical_yn, _spherical_in,
4:                       _spherical_kn, _spherical_jn_d, _spherical_yn_d,
5:                       _spherical_in_d, _spherical_kn_d)
6: 
7: def spherical_jn(n, z, derivative=False):
8:     r'''Spherical Bessel function of the first kind or its derivative.
9: 
10:     Defined as [1]_,
11: 
12:     .. math:: j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n + 1/2}(z),
13: 
14:     where :math:`J_n` is the Bessel function of the first kind.
15: 
16:     Parameters
17:     ----------
18:     n : int, array_like
19:         Order of the Bessel function (n >= 0).
20:     z : complex or float, array_like
21:         Argument of the Bessel function.
22:     derivative : bool, optional
23:         If True, the value of the derivative (rather than the function
24:         itself) is returned.
25: 
26:     Returns
27:     -------
28:     jn : ndarray
29: 
30:     Notes
31:     -----
32:     For real arguments greater than the order, the function is computed
33:     using the ascending recurrence [2]_.  For small real or complex
34:     arguments, the definitional relation to the cylindrical Bessel function
35:     of the first kind is used.
36: 
37:     The derivative is computed using the relations [3]_,
38: 
39:     .. math::
40:         j_n'(z) = j_{n-1}(z) - \frac{n + 1}{z} j_n(z).
41: 
42:         j_0'(z) = -j_1(z)
43: 
44: 
45:     .. versionadded:: 0.18.0
46: 
47:     References
48:     ----------
49:     .. [1] http://dlmf.nist.gov/10.47.E3
50:     .. [2] http://dlmf.nist.gov/10.51.E1
51:     .. [3] http://dlmf.nist.gov/10.51.E2
52:     '''
53:     if derivative:
54:         return _spherical_jn_d(n, z)
55:     else:
56:         return _spherical_jn(n, z)
57: 
58: 
59: def spherical_yn(n, z, derivative=False):
60:     r'''Spherical Bessel function of the second kind or its derivative.
61: 
62:     Defined as [1]_,
63: 
64:     .. math:: y_n(z) = \sqrt{\frac{\pi}{2z}} Y_{n + 1/2}(z),
65: 
66:     where :math:`Y_n` is the Bessel function of the second kind.
67: 
68:     Parameters
69:     ----------
70:     n : int, array_like
71:         Order of the Bessel function (n >= 0).
72:     z : complex or float, array_like
73:         Argument of the Bessel function.
74:     derivative : bool, optional
75:         If True, the value of the derivative (rather than the function
76:         itself) is returned.
77: 
78:     Returns
79:     -------
80:     yn : ndarray
81: 
82:     Notes
83:     -----
84:     For real arguments, the function is computed using the ascending
85:     recurrence [2]_.  For complex arguments, the definitional relation to
86:     the cylindrical Bessel function of the second kind is used.
87: 
88:     The derivative is computed using the relations [3]_,
89: 
90:     .. math::
91:         y_n' = y_{n-1} - \frac{n + 1}{2} y_n.
92: 
93:         y_0' = -y_1
94: 
95: 
96:     .. versionadded:: 0.18.0
97: 
98:     References
99:     ----------
100:     .. [1] http://dlmf.nist.gov/10.47.E4
101:     .. [2] http://dlmf.nist.gov/10.51.E1
102:     .. [3] http://dlmf.nist.gov/10.51.E2
103:     '''
104:     if derivative:
105:         return _spherical_yn_d(n, z)
106:     else:
107:         return _spherical_yn(n, z)
108: 
109: 
110: def spherical_in(n, z, derivative=False):
111:     r'''Modified spherical Bessel function of the first kind or its derivative.
112: 
113:     Defined as [1]_,
114: 
115:     .. math:: i_n(z) = \sqrt{\frac{\pi}{2z}} I_{n + 1/2}(z),
116: 
117:     where :math:`I_n` is the modified Bessel function of the first kind.
118: 
119:     Parameters
120:     ----------
121:     n : int, array_like
122:         Order of the Bessel function (n >= 0).
123:     z : complex or float, array_like
124:         Argument of the Bessel function.
125:     derivative : bool, optional
126:         If True, the value of the derivative (rather than the function
127:         itself) is returned.
128: 
129:     Returns
130:     -------
131:     in : ndarray
132: 
133:     Notes
134:     -----
135:     The function is computed using its definitional relation to the
136:     modified cylindrical Bessel function of the first kind.
137: 
138:     The derivative is computed using the relations [2]_,
139: 
140:     .. math::
141:         i_n' = i_{n-1} - \frac{n + 1}{2} i_n.
142: 
143:         i_1' = i_0
144: 
145: 
146:     .. versionadded:: 0.18.0
147: 
148:     References
149:     ----------
150:     .. [1] http://dlmf.nist.gov/10.47.E7
151:     .. [2] http://dlmf.nist.gov/10.51.E5
152:     '''
153:     if derivative:
154:         return _spherical_in_d(n, z)
155:     else:
156:         return _spherical_in(n, z)
157: 
158: 
159: def spherical_kn(n, z, derivative=False):
160:     r'''Modified spherical Bessel function of the second kind or its derivative.
161: 
162:     Defined as [1]_,
163: 
164:     .. math:: k_n(z) = \sqrt{\frac{\pi}{2z}} K_{n + 1/2}(z),
165: 
166:     where :math:`K_n` is the modified Bessel function of the second kind.
167: 
168:     Parameters
169:     ----------
170:     n : int, array_like
171:         Order of the Bessel function (n >= 0).
172:     z : complex or float, array_like
173:         Argument of the Bessel function.
174:     derivative : bool, optional
175:         If True, the value of the derivative (rather than the function
176:         itself) is returned.
177: 
178:     Returns
179:     -------
180:     kn : ndarray
181: 
182:     Notes
183:     -----
184:     The function is computed using its definitional relation to the
185:     modified cylindrical Bessel function of the second kind.
186: 
187:     The derivative is computed using the relations [2]_,
188: 
189:     .. math::
190:         k_n' = -k_{n-1} - \frac{n + 1}{2} k_n.
191: 
192:         k_0' = -k_1
193: 
194: 
195:     .. versionadded:: 0.18.0
196: 
197:     References
198:     ----------
199:     .. [1] http://dlmf.nist.gov/10.47.E9
200:     .. [2] http://dlmf.nist.gov/10.51.E5
201:     '''
202:     if derivative:
203:         return _spherical_kn_d(n, z)
204:     else:
205:         return _spherical_kn(n, z)
206: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.special._ufuncs import _spherical_jn, _spherical_yn, _spherical_in, _spherical_kn, _spherical_jn_d, _spherical_yn_d, _spherical_in_d, _spherical_kn_d' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_510948 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs')

if (type(import_510948) is not StypyTypeError):

    if (import_510948 != 'pyd_module'):
        __import__(import_510948)
        sys_modules_510949 = sys.modules[import_510948]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs', sys_modules_510949.module_type_store, module_type_store, ['_spherical_jn', '_spherical_yn', '_spherical_in', '_spherical_kn', '_spherical_jn_d', '_spherical_yn_d', '_spherical_in_d', '_spherical_kn_d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_510949, sys_modules_510949.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import _spherical_jn, _spherical_yn, _spherical_in, _spherical_kn, _spherical_jn_d, _spherical_yn_d, _spherical_in_d, _spherical_kn_d

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs', None, module_type_store, ['_spherical_jn', '_spherical_yn', '_spherical_in', '_spherical_kn', '_spherical_jn_d', '_spherical_yn_d', '_spherical_in_d', '_spherical_kn_d'], [_spherical_jn, _spherical_yn, _spherical_in, _spherical_kn, _spherical_jn_d, _spherical_yn_d, _spherical_in_d, _spherical_kn_d])

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs', import_510948)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


@norecursion
def spherical_jn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 7)
    False_510950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 34), 'False')
    defaults = [False_510950]
    # Create a new context for function 'spherical_jn'
    module_type_store = module_type_store.open_function_context('spherical_jn', 7, 0, False)
    
    # Passed parameters checking function
    spherical_jn.stypy_localization = localization
    spherical_jn.stypy_type_of_self = None
    spherical_jn.stypy_type_store = module_type_store
    spherical_jn.stypy_function_name = 'spherical_jn'
    spherical_jn.stypy_param_names_list = ['n', 'z', 'derivative']
    spherical_jn.stypy_varargs_param_name = None
    spherical_jn.stypy_kwargs_param_name = None
    spherical_jn.stypy_call_defaults = defaults
    spherical_jn.stypy_call_varargs = varargs
    spherical_jn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spherical_jn', ['n', 'z', 'derivative'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spherical_jn', localization, ['n', 'z', 'derivative'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spherical_jn(...)' code ##################

    str_510951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', "Spherical Bessel function of the first kind or its derivative.\n\n    Defined as [1]_,\n\n    .. math:: j_n(z) = \\sqrt{\\frac{\\pi}{2z}} J_{n + 1/2}(z),\n\n    where :math:`J_n` is the Bessel function of the first kind.\n\n    Parameters\n    ----------\n    n : int, array_like\n        Order of the Bessel function (n >= 0).\n    z : complex or float, array_like\n        Argument of the Bessel function.\n    derivative : bool, optional\n        If True, the value of the derivative (rather than the function\n        itself) is returned.\n\n    Returns\n    -------\n    jn : ndarray\n\n    Notes\n    -----\n    For real arguments greater than the order, the function is computed\n    using the ascending recurrence [2]_.  For small real or complex\n    arguments, the definitional relation to the cylindrical Bessel function\n    of the first kind is used.\n\n    The derivative is computed using the relations [3]_,\n\n    .. math::\n        j_n'(z) = j_{n-1}(z) - \\frac{n + 1}{z} j_n(z).\n\n        j_0'(z) = -j_1(z)\n\n\n    .. versionadded:: 0.18.0\n\n    References\n    ----------\n    .. [1] http://dlmf.nist.gov/10.47.E3\n    .. [2] http://dlmf.nist.gov/10.51.E1\n    .. [3] http://dlmf.nist.gov/10.51.E2\n    ")
    
    # Getting the type of 'derivative' (line 53)
    derivative_510952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'derivative')
    # Testing the type of an if condition (line 53)
    if_condition_510953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), derivative_510952)
    # Assigning a type to the variable 'if_condition_510953' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_510953', if_condition_510953)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spherical_jn_d(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'n' (line 54)
    n_510955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'n', False)
    # Getting the type of 'z' (line 54)
    z_510956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'z', False)
    # Processing the call keyword arguments (line 54)
    kwargs_510957 = {}
    # Getting the type of '_spherical_jn_d' (line 54)
    _spherical_jn_d_510954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), '_spherical_jn_d', False)
    # Calling _spherical_jn_d(args, kwargs) (line 54)
    _spherical_jn_d_call_result_510958 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), _spherical_jn_d_510954, *[n_510955, z_510956], **kwargs_510957)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', _spherical_jn_d_call_result_510958)
    # SSA branch for the else part of an if statement (line 53)
    module_type_store.open_ssa_branch('else')
    
    # Call to _spherical_jn(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'n' (line 56)
    n_510960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'n', False)
    # Getting the type of 'z' (line 56)
    z_510961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'z', False)
    # Processing the call keyword arguments (line 56)
    kwargs_510962 = {}
    # Getting the type of '_spherical_jn' (line 56)
    _spherical_jn_510959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), '_spherical_jn', False)
    # Calling _spherical_jn(args, kwargs) (line 56)
    _spherical_jn_call_result_510963 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), _spherical_jn_510959, *[n_510960, z_510961], **kwargs_510962)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', _spherical_jn_call_result_510963)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spherical_jn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spherical_jn' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_510964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510964)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spherical_jn'
    return stypy_return_type_510964

# Assigning a type to the variable 'spherical_jn' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'spherical_jn', spherical_jn)

@norecursion
def spherical_yn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 59)
    False_510965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'False')
    defaults = [False_510965]
    # Create a new context for function 'spherical_yn'
    module_type_store = module_type_store.open_function_context('spherical_yn', 59, 0, False)
    
    # Passed parameters checking function
    spherical_yn.stypy_localization = localization
    spherical_yn.stypy_type_of_self = None
    spherical_yn.stypy_type_store = module_type_store
    spherical_yn.stypy_function_name = 'spherical_yn'
    spherical_yn.stypy_param_names_list = ['n', 'z', 'derivative']
    spherical_yn.stypy_varargs_param_name = None
    spherical_yn.stypy_kwargs_param_name = None
    spherical_yn.stypy_call_defaults = defaults
    spherical_yn.stypy_call_varargs = varargs
    spherical_yn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spherical_yn', ['n', 'z', 'derivative'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spherical_yn', localization, ['n', 'z', 'derivative'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spherical_yn(...)' code ##################

    str_510966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', "Spherical Bessel function of the second kind or its derivative.\n\n    Defined as [1]_,\n\n    .. math:: y_n(z) = \\sqrt{\\frac{\\pi}{2z}} Y_{n + 1/2}(z),\n\n    where :math:`Y_n` is the Bessel function of the second kind.\n\n    Parameters\n    ----------\n    n : int, array_like\n        Order of the Bessel function (n >= 0).\n    z : complex or float, array_like\n        Argument of the Bessel function.\n    derivative : bool, optional\n        If True, the value of the derivative (rather than the function\n        itself) is returned.\n\n    Returns\n    -------\n    yn : ndarray\n\n    Notes\n    -----\n    For real arguments, the function is computed using the ascending\n    recurrence [2]_.  For complex arguments, the definitional relation to\n    the cylindrical Bessel function of the second kind is used.\n\n    The derivative is computed using the relations [3]_,\n\n    .. math::\n        y_n' = y_{n-1} - \\frac{n + 1}{2} y_n.\n\n        y_0' = -y_1\n\n\n    .. versionadded:: 0.18.0\n\n    References\n    ----------\n    .. [1] http://dlmf.nist.gov/10.47.E4\n    .. [2] http://dlmf.nist.gov/10.51.E1\n    .. [3] http://dlmf.nist.gov/10.51.E2\n    ")
    
    # Getting the type of 'derivative' (line 104)
    derivative_510967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'derivative')
    # Testing the type of an if condition (line 104)
    if_condition_510968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), derivative_510967)
    # Assigning a type to the variable 'if_condition_510968' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_510968', if_condition_510968)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spherical_yn_d(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'n' (line 105)
    n_510970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'n', False)
    # Getting the type of 'z' (line 105)
    z_510971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'z', False)
    # Processing the call keyword arguments (line 105)
    kwargs_510972 = {}
    # Getting the type of '_spherical_yn_d' (line 105)
    _spherical_yn_d_510969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), '_spherical_yn_d', False)
    # Calling _spherical_yn_d(args, kwargs) (line 105)
    _spherical_yn_d_call_result_510973 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), _spherical_yn_d_510969, *[n_510970, z_510971], **kwargs_510972)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', _spherical_yn_d_call_result_510973)
    # SSA branch for the else part of an if statement (line 104)
    module_type_store.open_ssa_branch('else')
    
    # Call to _spherical_yn(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'n' (line 107)
    n_510975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'n', False)
    # Getting the type of 'z' (line 107)
    z_510976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'z', False)
    # Processing the call keyword arguments (line 107)
    kwargs_510977 = {}
    # Getting the type of '_spherical_yn' (line 107)
    _spherical_yn_510974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), '_spherical_yn', False)
    # Calling _spherical_yn(args, kwargs) (line 107)
    _spherical_yn_call_result_510978 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), _spherical_yn_510974, *[n_510975, z_510976], **kwargs_510977)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', _spherical_yn_call_result_510978)
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spherical_yn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spherical_yn' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_510979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510979)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spherical_yn'
    return stypy_return_type_510979

# Assigning a type to the variable 'spherical_yn' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'spherical_yn', spherical_yn)

@norecursion
def spherical_in(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 110)
    False_510980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'False')
    defaults = [False_510980]
    # Create a new context for function 'spherical_in'
    module_type_store = module_type_store.open_function_context('spherical_in', 110, 0, False)
    
    # Passed parameters checking function
    spherical_in.stypy_localization = localization
    spherical_in.stypy_type_of_self = None
    spherical_in.stypy_type_store = module_type_store
    spherical_in.stypy_function_name = 'spherical_in'
    spherical_in.stypy_param_names_list = ['n', 'z', 'derivative']
    spherical_in.stypy_varargs_param_name = None
    spherical_in.stypy_kwargs_param_name = None
    spherical_in.stypy_call_defaults = defaults
    spherical_in.stypy_call_varargs = varargs
    spherical_in.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spherical_in', ['n', 'z', 'derivative'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spherical_in', localization, ['n', 'z', 'derivative'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spherical_in(...)' code ##################

    str_510981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, (-1)), 'str', "Modified spherical Bessel function of the first kind or its derivative.\n\n    Defined as [1]_,\n\n    .. math:: i_n(z) = \\sqrt{\\frac{\\pi}{2z}} I_{n + 1/2}(z),\n\n    where :math:`I_n` is the modified Bessel function of the first kind.\n\n    Parameters\n    ----------\n    n : int, array_like\n        Order of the Bessel function (n >= 0).\n    z : complex or float, array_like\n        Argument of the Bessel function.\n    derivative : bool, optional\n        If True, the value of the derivative (rather than the function\n        itself) is returned.\n\n    Returns\n    -------\n    in : ndarray\n\n    Notes\n    -----\n    The function is computed using its definitional relation to the\n    modified cylindrical Bessel function of the first kind.\n\n    The derivative is computed using the relations [2]_,\n\n    .. math::\n        i_n' = i_{n-1} - \\frac{n + 1}{2} i_n.\n\n        i_1' = i_0\n\n\n    .. versionadded:: 0.18.0\n\n    References\n    ----------\n    .. [1] http://dlmf.nist.gov/10.47.E7\n    .. [2] http://dlmf.nist.gov/10.51.E5\n    ")
    
    # Getting the type of 'derivative' (line 153)
    derivative_510982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'derivative')
    # Testing the type of an if condition (line 153)
    if_condition_510983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 4), derivative_510982)
    # Assigning a type to the variable 'if_condition_510983' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'if_condition_510983', if_condition_510983)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spherical_in_d(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'n' (line 154)
    n_510985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'n', False)
    # Getting the type of 'z' (line 154)
    z_510986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'z', False)
    # Processing the call keyword arguments (line 154)
    kwargs_510987 = {}
    # Getting the type of '_spherical_in_d' (line 154)
    _spherical_in_d_510984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), '_spherical_in_d', False)
    # Calling _spherical_in_d(args, kwargs) (line 154)
    _spherical_in_d_call_result_510988 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), _spherical_in_d_510984, *[n_510985, z_510986], **kwargs_510987)
    
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', _spherical_in_d_call_result_510988)
    # SSA branch for the else part of an if statement (line 153)
    module_type_store.open_ssa_branch('else')
    
    # Call to _spherical_in(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'n' (line 156)
    n_510990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'n', False)
    # Getting the type of 'z' (line 156)
    z_510991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'z', False)
    # Processing the call keyword arguments (line 156)
    kwargs_510992 = {}
    # Getting the type of '_spherical_in' (line 156)
    _spherical_in_510989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), '_spherical_in', False)
    # Calling _spherical_in(args, kwargs) (line 156)
    _spherical_in_call_result_510993 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), _spherical_in_510989, *[n_510990, z_510991], **kwargs_510992)
    
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', _spherical_in_call_result_510993)
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spherical_in(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spherical_in' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_510994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510994)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spherical_in'
    return stypy_return_type_510994

# Assigning a type to the variable 'spherical_in' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'spherical_in', spherical_in)

@norecursion
def spherical_kn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 159)
    False_510995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'False')
    defaults = [False_510995]
    # Create a new context for function 'spherical_kn'
    module_type_store = module_type_store.open_function_context('spherical_kn', 159, 0, False)
    
    # Passed parameters checking function
    spherical_kn.stypy_localization = localization
    spherical_kn.stypy_type_of_self = None
    spherical_kn.stypy_type_store = module_type_store
    spherical_kn.stypy_function_name = 'spherical_kn'
    spherical_kn.stypy_param_names_list = ['n', 'z', 'derivative']
    spherical_kn.stypy_varargs_param_name = None
    spherical_kn.stypy_kwargs_param_name = None
    spherical_kn.stypy_call_defaults = defaults
    spherical_kn.stypy_call_varargs = varargs
    spherical_kn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spherical_kn', ['n', 'z', 'derivative'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spherical_kn', localization, ['n', 'z', 'derivative'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spherical_kn(...)' code ##################

    str_510996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', "Modified spherical Bessel function of the second kind or its derivative.\n\n    Defined as [1]_,\n\n    .. math:: k_n(z) = \\sqrt{\\frac{\\pi}{2z}} K_{n + 1/2}(z),\n\n    where :math:`K_n` is the modified Bessel function of the second kind.\n\n    Parameters\n    ----------\n    n : int, array_like\n        Order of the Bessel function (n >= 0).\n    z : complex or float, array_like\n        Argument of the Bessel function.\n    derivative : bool, optional\n        If True, the value of the derivative (rather than the function\n        itself) is returned.\n\n    Returns\n    -------\n    kn : ndarray\n\n    Notes\n    -----\n    The function is computed using its definitional relation to the\n    modified cylindrical Bessel function of the second kind.\n\n    The derivative is computed using the relations [2]_,\n\n    .. math::\n        k_n' = -k_{n-1} - \\frac{n + 1}{2} k_n.\n\n        k_0' = -k_1\n\n\n    .. versionadded:: 0.18.0\n\n    References\n    ----------\n    .. [1] http://dlmf.nist.gov/10.47.E9\n    .. [2] http://dlmf.nist.gov/10.51.E5\n    ")
    
    # Getting the type of 'derivative' (line 202)
    derivative_510997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'derivative')
    # Testing the type of an if condition (line 202)
    if_condition_510998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), derivative_510997)
    # Assigning a type to the variable 'if_condition_510998' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_510998', if_condition_510998)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spherical_kn_d(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'n' (line 203)
    n_511000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'n', False)
    # Getting the type of 'z' (line 203)
    z_511001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), 'z', False)
    # Processing the call keyword arguments (line 203)
    kwargs_511002 = {}
    # Getting the type of '_spherical_kn_d' (line 203)
    _spherical_kn_d_510999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), '_spherical_kn_d', False)
    # Calling _spherical_kn_d(args, kwargs) (line 203)
    _spherical_kn_d_call_result_511003 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), _spherical_kn_d_510999, *[n_511000, z_511001], **kwargs_511002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', _spherical_kn_d_call_result_511003)
    # SSA branch for the else part of an if statement (line 202)
    module_type_store.open_ssa_branch('else')
    
    # Call to _spherical_kn(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'n' (line 205)
    n_511005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'n', False)
    # Getting the type of 'z' (line 205)
    z_511006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'z', False)
    # Processing the call keyword arguments (line 205)
    kwargs_511007 = {}
    # Getting the type of '_spherical_kn' (line 205)
    _spherical_kn_511004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), '_spherical_kn', False)
    # Calling _spherical_kn(args, kwargs) (line 205)
    _spherical_kn_call_result_511008 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), _spherical_kn_511004, *[n_511005, z_511006], **kwargs_511007)
    
    # Assigning a type to the variable 'stypy_return_type' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', _spherical_kn_call_result_511008)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spherical_kn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spherical_kn' in the type store
    # Getting the type of 'stypy_return_type' (line 159)
    stypy_return_type_511009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_511009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spherical_kn'
    return stypy_return_type_511009

# Assigning a type to the variable 'spherical_kn' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'spherical_kn', spherical_kn)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
